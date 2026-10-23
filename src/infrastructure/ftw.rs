// FreeToken Weight (FTW) checkpoint index: parse ``freetoken_weight.json`` and
// resolve tensor byte ranges to per-shard reads. Schema mirrors
// upstream FreeToken's ``python/freetoken/checkpoint/ftw.py`` (not vendored).
//
// On-disk layout (see ftw.py docstring): one logical contiguous byte region of
// all tensors, sliced into shard files of at most ``shard_limit`` bytes. Every
// tensor starts at a 4096-aligned region offset and is padded to 4096; shards
// are cut at aligned boundaries, so a tensor may span shards with each piece
// block-aligned.

use crate::domain::weights::{DType, QuantFormat, TensorDesc, TensorKind, WeightsReader};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;
use std::str::FromStr;
use std::sync::Mutex;

pub const INDEX_NAME: &str = "freetoken_weight.json";
pub const FORMAT_TAG: &str = "freetoken_weight";
pub const FORMAT_VERSION: u64 = 1;
pub const ALIGN: u64 = 4096; // O_DIRECT block alignment
pub const DEFAULT_SHARD_LIMIT: u64 = 8 << 30;

#[derive(Debug, thiserror::Error)]
pub enum FtwError {
    #[error("JSON parse failure: {0}")]
    Json(#[from] serde_json::Error),
    #[error("io failure: {0}")]
    Io(#[from] std::io::Error),
    #[error("not an FTW checkpoint: format={format:?}")]
    NotFtw { format: String },
    #[error("unsupported FTW version {0}")]
    Version(u64),
    #[error("unexpected alignment {0}; FTW requires {ALIGN}")]
    Alignment(u64),
    #[error("invalid FTW index: {0}")]
    Invalid(String),
    #[error("FTW tensor range exceeds shards")]
    OutOfRange,
}

/// One physical shard file and its span of the logical byte region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardDesc {
    pub file: String,
    pub global_off: u64,
    pub nbytes: u64,
}

/// One contiguous (file, dest) copy within a tensor read. All offsets are
/// ALIGN-aligned; ``length`` is the exact piece length, its tail is padded by
/// the caller when rounding up for O_DIRECT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    pub file: String,
    pub file_off: u64,
    pub dest_off: u64,
    pub length: u64,
}

/// Parsed FTW index with the byte-region invariants pre-validated: shards
/// sorted by ``global_off`` and contiguous, tensors aligned and in-range.
#[derive(Debug, Clone)]
pub struct FtwIndex {
    pub shard_limit: u64,
    pub total_bytes: u64,
    pub tensors: Vec<TensorDesc>,
    pub shards: Vec<ShardDesc>,
    pub quant_format: Option<QuantFormat>,
    pub expert_bank_num_layers: Option<u64>,
    /// Anything else in the index top level (per-shard metadata, counts, ...).
    pub meta: HashMap<String, Value>,
}

/// Split a per-layer expert-bank entry name like ``gate_up_packed#L00007`` into
/// its base bank name and layer id (mirrors ``ftw._LAYER_ENTRY_RE``).
pub fn layer_bank_entry(name: &str) -> Option<(String, u64)> {
    let at = name.rfind("#L")?;
    if at == 0 {
        return None;
    }
    let layer = &name[at + 2..];
    if layer.len() != 5 || !layer.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((name[..at].to_string(), layer.parse().ok()?))
}

impl FtwIndex {
    /// Parse and validate a ``freetoken_weight.json`` document.
    pub fn parse(doc: &str) -> Result<Self, FtwError> {
        let v: Value = serde_json::from_str(doc)?;

        let tag = v.get("format").and_then(Value::as_str);
        match tag {
            Some(FORMAT_TAG) => {}
            Some(other) => {
                return Err(FtwError::NotFtw {
                    format: other.to_string(),
                });
            }
            None => return Err(FtwError::Invalid("missing 'format'".into())),
        }
        let version = v
            .get("version")
            .and_then(Value::as_u64)
            .ok_or_else(|| FtwError::Invalid("missing 'version'".into()))?;
        if version != FORMAT_VERSION {
            return Err(FtwError::Version(version));
        }
        let align = v
            .get("align")
            .and_then(Value::as_u64)
            .ok_or_else(|| FtwError::Invalid("missing 'align'".into()))?;
        if align != ALIGN {
            return Err(FtwError::Alignment(align));
        }

        let shard_limit = v
            .get("shard_limit")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_SHARD_LIMIT);
        let total_bytes = v
            .get("total_bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| FtwError::Invalid("missing 'total_bytes'".into()))?;

        let tensors = v
            .get("tensors")
            .and_then(Value::as_array)
            .ok_or_else(|| FtwError::Invalid("missing 'tensors'".into()))?
            .iter()
            .map(parse_tensor)
            .collect::<Result<Vec<_>, FtwError>>()?;

        let shards = v
            .get("shards")
            .and_then(Value::as_array)
            .ok_or_else(|| FtwError::Invalid("missing 'shards'".into()))?
            .iter()
            .map(parse_shard)
            .collect::<Result<Vec<_>, FtwError>>()?;

        let mut prev_end: Option<u64> = None;
        for s in &shards {
            if s.nbytes == 0 {
                return Err(FtwError::Invalid(format!("empty shard {}", s.file)));
            }
            if let Some(end) = prev_end
                && s.global_off != end
            {
                return Err(FtwError::Invalid(format!(
                    "shards not contiguous: {} starts at {} but previous ends at {}",
                    s.file, s.global_off, end
                )));
            }
            prev_end = Some(s.global_off + s.nbytes);
        }
        if let Some(end) = prev_end {
            if end != total_bytes {
                return Err(FtwError::Invalid(format!(
                    "shards cover {end} bytes but total_bytes={total_bytes}"
                )));
            }
        } else if total_bytes != 0 {
            return Err(FtwError::Invalid("no shards but total_bytes > 0".into()));
        }

        for t in &tensors {
            if t.global_off % ALIGN != 0 {
                return Err(FtwError::Invalid(format!(
                    "tensor {} starts at unaligned offset {}",
                    t.name, t.global_off
                )));
            }
            if t.nbytes == 0 {
                return Err(FtwError::Invalid(format!("tensor {} is empty", t.name)));
            }
            if t.global_off + t.nbytes > total_bytes {
                return Err(FtwError::Invalid(format!(
                    "tensor {} range [{}..{}) exceeds total_bytes={total_bytes}",
                    t.name,
                    t.global_off,
                    t.global_off + t.nbytes
                )));
            }
            if t.byte_len() != t.nbytes {
                return Err(FtwError::Invalid(format!(
                    "tensor {} shape*elsize {} != nbytes {}",
                    t.name,
                    t.byte_len(),
                    t.nbytes
                )));
            }
        }

        let mut meta = HashMap::new();
        let quant_format = v
            .get("quant_format")
            .and_then(Value::as_str)
            .map(|s| QuantFormat::from_str(s).unwrap());
        let expert_bank_num_layers = v.get("expert_bank_num_layers").and_then(Value::as_u64);
        for (k, val) in v.as_object().unwrap() {
            if matches!(
                k.as_str(),
                "format"
                    | "version"
                    | "align"
                    | "shard_limit"
                    | "total_bytes"
                    | "tensors"
                    | "shards"
                    | "quant_format"
                    | "expert_bank_num_layers"
            ) {
                continue;
            }
            meta.insert(k.clone(), val.clone());
        }

        Ok(Self {
            shard_limit,
            total_bytes,
            tensors,
            shards,
            quant_format,
            expert_bank_num_layers,
            meta,
        })
    }

    pub fn entry(&self, name: &str) -> Option<&TensorDesc> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Resolve the byte range ``[global_off, global_off + nbytes)`` into aligned
    /// per-shard pieces, mirroring ``FTWReader._pieces``.
    pub fn pieces(&self, global_off: u64, nbytes: u64) -> Result<Vec<Piece>, FtwError> {
        let mut out = Vec::new();
        let mut dest_off = 0u64;
        let mut remaining = nbytes;
        let mut pos = global_off;
        for sh in &self.shards {
            let s0 = sh.global_off;
            let s1 = s0 + sh.nbytes;
            if pos >= s1 || remaining == 0 {
                continue;
            }
            if pos < s0 {
                return Err(FtwError::Invalid("FTW gap / misordered shards".into()));
            }
            let take = remaining.min(s1 - pos);
            out.push(Piece {
                file: sh.file.clone(),
                file_off: pos - s0,
                dest_off,
                length: take,
            });
            pos += take;
            dest_off += take;
            remaining -= take;
        }
        if remaining > 0 {
            return Err(FtwError::OutOfRange);
        }
        Ok(out)
    }
}

fn parse_tensor(v: &Value) -> Result<TensorDesc, FtwError> {
    let name = v
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| FtwError::Invalid("tensor missing 'name'".into()))?
        .to_string();
    let kind = TensorKind::from_str(
        v.get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| FtwError::Invalid(format!("tensor {name} missing 'kind'")))?,
    )
    .unwrap();
    let dtype = DType::from_str(
        v.get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| FtwError::Invalid(format!("tensor {name} missing 'dtype'")))?,
    )
    .map_err(|e| FtwError::Invalid(format!("tensor {name}: {e}")))?;
    let shape = v
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| FtwError::Invalid(format!("tensor {name} missing 'shape'")))?
        .iter()
        .map(|d| d.as_u64().map(|u| u as usize))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| FtwError::Invalid(format!("tensor {name} has non-integer shape")))?;
    let global_off = v
        .get("global_off")
        .and_then(Value::as_u64)
        .ok_or_else(|| FtwError::Invalid(format!("tensor {name} missing 'global_off'")))?;
    let nbytes = v
        .get("nbytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| FtwError::Invalid(format!("tensor {name} missing 'nbytes'")))?;
    Ok(TensorDesc {
        name,
        kind,
        dtype,
        shape,
        global_off,
        nbytes,
    })
}

fn parse_shard(v: &Value) -> Result<ShardDesc, FtwError> {
    let file = v
        .get("file")
        .and_then(Value::as_str)
        .ok_or_else(|| FtwError::Invalid("shard missing 'file'".into()))?
        .to_string();
    let global_off = v
        .get("global_off")
        .and_then(Value::as_u64)
        .ok_or_else(|| FtwError::Invalid(format!("shard {file} missing 'global_off'")))?;
    let nbytes = v
        .get("nbytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| FtwError::Invalid(format!("shard {file} missing 'nbytes'")))?;
    Ok(ShardDesc {
        file,
        global_off,
        nbytes,
    })
}

/// O_DIRECT is a Linux open flag; the value is stable across architectures.
const O_DIRECT: i32 = 0o40000;

/// Payload bytes per `O_DIRECT` request. Reads are chunked to this so the
/// aligned scratch buffer stays a few MiB no matter how large a tensor is.
const DIRECT_CHUNK: usize = 4 << 20;

/// WeightsReader over FTW shard files.
///
/// Reads are `pread` (`read_at`) into the caller's buffer. When the shard
/// filesystem accepts `O_DIRECT`, a read goes straight from disk into an
/// aligned scratch buffer and is copied out, so it never populates the page
/// cache: the dense startup load stays disk-bound (measured ~6x faster cold
/// than an mmap page-in) and streaming expert slabs does not push everything
/// else out of memory. On a filesystem that rejects `O_DIRECT` (tmpfs, some
/// network mounts) the reader falls back to buffered reads.
pub struct FtwReader {
    index: FtwIndex,
    shards: HashMap<String, Shard>,
    direct: bool,
    scratch: Mutex<Scratch>,
}

struct Shard {
    file: File,
    nbytes: u64,
}

/// A `DIRECT_CHUNK + ALIGN` byte buffer whose start is `ALIGN`-aligned, so one
/// allocation can back every `O_DIRECT` read.
struct Scratch {
    buf: Vec<u8>,
    off: usize,
}

impl Scratch {
    fn new() -> Self {
        let size = DIRECT_CHUNK + ALIGN as usize;
        let buf = vec![0u8; size + ALIGN as usize];
        let off = (ALIGN as usize - (buf.as_ptr() as usize % ALIGN as usize)) % ALIGN as usize;
        Self { buf, off }
    }

    fn aligned(&mut self, need: usize) -> &mut [u8] {
        debug_assert!(need <= DIRECT_CHUNK + ALIGN as usize);
        let s = self.off;
        &mut self.buf[s..s + need]
    }
}

impl FtwReader {
    /// Load ``path/freetoken_weight.json`` and open every referenced shard.
    pub fn open(path: &Path) -> Result<Self, FtwError> {
        Self::open_mode(path, None)
    }

    /// ``force_direct`` overrides the `O_DIRECT` probe (tests); ``None`` chooses
    /// `O_DIRECT` when the first shard accepts it and buffered reads otherwise.
    fn open_mode(path: &Path, force_direct: Option<bool>) -> Result<Self, FtwError> {
        let index_path = path.join(INDEX_NAME);
        let doc = std::fs::read_to_string(index_path)?;
        let index = FtwIndex::parse(&doc)?;
        if index.shards.is_empty() {
            return Err(FtwError::Invalid("index has no shards".into()));
        }
        let direct = force_direct.unwrap_or_else(|| {
            OpenOptions::new()
                .read(true)
                .custom_flags(O_DIRECT)
                .open(path.join(&index.shards[0].file))
                .is_ok()
        });
        let mut shards = HashMap::new();
        for sh in &index.shards {
            let file = OpenOptions::new()
                .read(true)
                .custom_flags(if direct { O_DIRECT } else { 0 })
                .open(path.join(&sh.file))?;
            let len = file.metadata()?.len();
            if len < sh.nbytes {
                return Err(FtwError::Invalid(format!(
                    "shard {} is {len} bytes but index claims {}",
                    sh.file, sh.nbytes
                )));
            }
            shards.insert(
                sh.file.clone(),
                Shard {
                    file,
                    nbytes: sh.nbytes,
                },
            );
        }
        Ok(Self {
            index,
            shards,
            direct,
            scratch: Mutex::new(Scratch::new()),
        })
    }

    pub fn index(&self) -> &FtwIndex {
        &self.index
    }

    /// Whether reads bypass the page cache (`O_DIRECT`).
    pub fn direct(&self) -> bool {
        self.direct
    }
}

/// The `[off, off + want)` request widened to a block-aligned `O_DIRECT` read:
/// `(aligned_off, padded_len, delta)`, the wanted bytes starting `delta` into
/// the read. Errors when the request runs past `file_len`.
fn aligned_span(off: u64, want: u64, file_len: u64) -> Result<(u64, usize, usize), String> {
    let aligned_off = off & !(ALIGN - 1);
    let delta = (off - aligned_off) as usize;
    let need = delta as u64 + want;
    let mut padded = need.div_ceil(ALIGN) * ALIGN;
    let available = file_len.saturating_sub(aligned_off);
    if padded > available {
        padded = available & !(ALIGN - 1);
    }
    if padded < need {
        return Err(format!("read past end of shard: {off}+{want} > {file_len}"));
    }
    Ok((aligned_off, padded as usize, delta))
}

fn read_exact_at(file: &File, off: u64, buf: &mut [u8]) -> Result<(), String> {
    let mut filled = 0usize;
    while filled < buf.len() {
        let n = file
            .read_at(&mut buf[filled..], off + filled as u64)
            .map_err(|e| format!("io failure: {e}"))?;
        if n == 0 {
            return Err(format!("short read at {off}+{filled}"));
        }
        filled += n;
    }
    Ok(())
}

fn read_piece_direct(
    shard: &Shard,
    piece: &Piece,
    dst: &mut [u8],
    scratch: &mut Scratch,
) -> Result<(), String> {
    let mut done = 0u64;
    while done < piece.length {
        let want = (piece.length - done).min(DIRECT_CHUNK as u64);
        let (aligned_off, padded, delta) = aligned_span(piece.file_off + done, want, shard.nbytes)?;
        let buf = scratch.aligned(padded);
        read_exact_at(&shard.file, aligned_off, buf)?;
        let d0 = piece.dest_off as usize + done as usize;
        dst[d0..d0 + want as usize].copy_from_slice(&buf[delta..delta + want as usize]);
        done += want;
    }
    Ok(())
}

impl WeightsReader for FtwReader {
    fn read_range(&self, global_off: u64, nbytes: u64, dst: &mut [u8]) -> Result<(), String> {
        if dst.len() < nbytes as usize {
            return Err(format!(
                "read_range buffer too small: {} < {nbytes}",
                dst.len()
            ));
        }
        let pieces = self
            .index
            .pieces(global_off, nbytes)
            .map_err(|e| e.to_string())?;
        let mut scratch = self.scratch.lock().unwrap();
        for piece in &pieces {
            let shard = self
                .shards
                .get(&piece.file)
                .ok_or_else(|| format!("shard {} not opened", piece.file))?;
            if self.direct {
                read_piece_direct(shard, piece, dst, &mut scratch)?;
            } else {
                let d0 = piece.dest_off as usize;
                let n = piece.length as usize;
                read_exact_at(&shard.file, piece.file_off, &mut dst[d0..d0 + n])?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod reader_tests {
    use super::*;
    use std::io::Write;

    fn write_index(dir: &Path, doc: &str) {
        std::fs::write(dir.join(INDEX_NAME), doc).unwrap();
    }

    fn mktemp() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ftw_read_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A scratch dir on the workspace filesystem, which (unlike `/tmp`) supports
    /// `O_DIRECT`, so the direct read path can be exercised.
    fn mktemp_workspace() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp/ftw_read");
        std::fs::create_dir_all(&base).unwrap();
        let dir = base.join(format!(
            "{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A one-shard index over `nbytes` of a byte ramp, written to `dir`.
    fn write_ramp(dir: &Path, nbytes: usize) {
        let data: Vec<u8> = (0..nbytes).map(|i| (i & 0xff) as u8).collect();
        std::fs::write(dir.join("t-00000.ftw"), data).unwrap();
        let doc = format!(
            r#"{{
              "format": "freetoken_weight", "version": 1, "align": 4096,
              "shard_limit": {nbytes}, "total_bytes": {nbytes},
              "tensors": [
                {{"name": "w", "kind": "weight", "dtype": "uint8",
                  "shape": [{nbytes}], "global_off": 0, "nbytes": {nbytes}}}
              ],
              "shards": [
                {{"file": "t-00000.ftw", "global_off": 0, "nbytes": {nbytes}}}
              ]
            }}"#
        );
        write_index(dir, &doc);
    }

    #[test]
    fn aligned_span_rounds_out_and_reports_delta() {
        assert_eq!(aligned_span(4096, 192, 8192).unwrap(), (4096, 4096, 0));
        assert_eq!(aligned_span(4100, 100, 8192).unwrap(), (4096, 4096, 4));
        assert_eq!(aligned_span(4000, 200, 8192).unwrap(), (0, 8192, 4000));
        assert!(aligned_span(8000, 4096, 8192).is_err(), "past end");
    }

    #[test]
    fn read_across_shard_boundary() {
        const N0: usize = 4096;
        const N1: usize = 4096;
        let dir = mktemp();
        let mut f0 = File::create(dir.join("t-00000.ftw")).unwrap();
        let mut f1 = File::create(dir.join("t-00001.ftw")).unwrap();
        let mut a0 = vec![0u8; N0];
        let mut a1 = vec![0u8; N1];
        for (i, x) in a0.iter_mut().enumerate() {
            *x = (i & 0xff) as u8;
        }
        for (i, x) in a1.iter_mut().enumerate() {
            *x = ((i + N0) & 0xff) as u8;
        }
        f0.write_all(&a0).unwrap();
        f1.write_all(&a1).unwrap();
        drop(f0);
        drop(f1);
        let doc = format!(
            r#"{{
              "format": "freetoken_weight", "version": 1, "align": 4096,
              "shard_limit": 4096, "total_bytes": 8192,
              "tensors": [
                {{"name": "w", "kind": "weight", "dtype": "uint8",
                  "shape": [4912], "global_off": 0, "nbytes": 4912}}
              ],
              "shards": [
                {{"file": "t-00000.ftw", "global_off": 0, "nbytes": {N0}}},
                {{"file": "t-00001.ftw", "global_off": 4096, "nbytes": {N1}}}
              ]
            }}"#
        );
        write_index(&dir, &doc);
        let reader = FtwReader::open(&dir).unwrap();
        let mut buf = vec![0u8; 4912];
        reader.read_range(0, 4912, &mut buf).unwrap();
        for (i, x) in buf.iter().enumerate() {
            assert_eq!(*x, (i & 0xff) as u8, "byte {i}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_out_of_range_errors() {
        let dir = mktemp();
        std::fs::write(dir.join("t-00000.ftw"), vec![0xabu8; 4096]).unwrap();
        let doc = r#"{
              "format": "freetoken_weight", "version": 1, "align": 4096,
              "shard_limit": 4096, "total_bytes": 4096,
              "tensors": [
                {"name": "w", "kind": "weight", "dtype": "uint8",
                  "shape": [4096], "global_off": 0, "nbytes": 4096}
              ],
              "shards": [
                {"file": "t-00000.ftw", "global_off": 0, "nbytes": 4096}
              ]
            }"#
        .to_string();
        write_index(&dir, &doc);
        let reader = FtwReader::open(&dir).unwrap();
        let mut buf = vec![0u8; 4096];
        assert!(reader.read_range(0, 8192, &mut buf).is_err());
        assert!(reader.read_range(0, 4096, &mut buf[..100]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unaligned_window_read_returns_the_right_bytes() {
        let dir = mktemp();
        write_ramp(&dir, 8192);
        let reader = FtwReader::open(&dir).unwrap();
        let mut buf = vec![0u8; 5000];
        reader.read_range(1000, 5000, &mut buf).unwrap();
        for (i, x) in buf.iter().enumerate() {
            assert_eq!(*x, ((1000 + i) & 0xff) as u8, "byte {i}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn direct_and_buffered_reads_agree() {
        let dir = mktemp_workspace();
        write_ramp(&dir, 16384);
        let buffered = FtwReader::open_mode(&dir, Some(false)).unwrap();
        assert!(!buffered.direct());
        let Ok(direct) = FtwReader::open_mode(&dir, Some(true)) else {
            eprintln!("skip: O_DIRECT unsupported on the workspace filesystem");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        };
        for (off, len) in [(1000u64, 5000usize), (0, 16384), (4096, 4096), (8191, 2)] {
            let mut a = vec![0u8; len];
            let mut b = vec![0u8; len];
            direct.read_range(off, len as u64, &mut a).unwrap();
            buffered.read_range(off, len as u64, &mut b).unwrap();
            assert_eq!(a, b, "off {off} len {len}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_index() -> &'static str {
        r#"{
          "format": "freetoken_weight",
          "version": 1,
          "align": 4096,
          "shard_limit": 8589934592,
          "total_bytes": 28672,
          "tensors": [
            {"name": "model.embed_tokens.weight", "kind": "weight", "dtype": "bfloat16",
             "shape": [4, 8], "global_off": 0, "nbytes": 64},
            {"name": "model.router.weight", "kind": "weight", "dtype": "float32",
             "shape": [6, 8], "global_off": 4096, "nbytes": 192},
            {"name": "gate_up_packed#L00000", "kind": "experts_bank", "dtype": "uint8",
             "shape": [2, 32], "global_off": 8192, "nbytes": 64},
            {"name": "down_packed#L00001", "kind": "experts_bank", "dtype": "uint8",
             "shape": [2, 64], "global_off": 24576, "nbytes": 128}
          ],
          "shards": [
            {"file": "freetoken-00000.ftw", "global_off": 0, "nbytes": 12288},
            {"file": "freetoken-00001.ftw", "global_off": 12288, "nbytes": 16384}
          ],
          "quant_format": "nvfp4_b12x",
          "expert_bank_num_layers": 40
        }"#
    }

    fn spanning_index() -> &'static str {
        r#"{
          "format": "freetoken_weight",
          "version": 1,
          "align": 4096,
          "shard_limit": 4096,
          "total_bytes": 8192,
          "tensors": [
            {"name": "big.weight", "kind": "weight", "dtype": "bfloat16",
             "shape": [2456], "global_off": 0, "nbytes": 4912}
          ],
          "shards": [
            {"file": "freetoken-00000.ftw", "global_off": 0, "nbytes": 4096},
            {"file": "freetoken-00001.ftw", "global_off": 4096, "nbytes": 4096}
          ]
        }"#
    }

    #[test]
    fn parse_sample_index() {
        let ix = FtwIndex::parse(sample_index()).unwrap();
        assert_eq!(ix.total_bytes, 28672);
        assert_eq!(ix.shard_limit, 8589934592);
        assert_eq!(ix.tensors.len(), 4);
        assert_eq!(ix.shards.len(), 2);
        assert_eq!(ix.quant_format, Some(QuantFormat::Nvfp4));
        assert_eq!(ix.expert_bank_num_layers, Some(40));
        let emb = ix.entry("model.embed_tokens.weight").unwrap();
        assert_eq!(emb.kind, TensorKind::Weight);
        assert_eq!(emb.dtype, DType::Bf16);
        assert_eq!(emb.shape, vec![4, 8]);
        assert!(ix.entry("model.embed.weight").is_none());
    }

    #[test]
    fn reject_wrong_format_tag() {
        let doc = sample_index().replace("freetoken_weight", "safetensors");
        match FtwIndex::parse(&doc) {
            Err(FtwError::NotFtw { format }) => assert_eq!(format, "safetensors"),
            other => panic!("expected NotFtw, got {other:?}"),
        }
    }

    #[test]
    fn reject_unsupported_version() {
        let doc = sample_index().replace("\"version\": 1", "\"version\": 2");
        assert!(matches!(FtwIndex::parse(&doc), Err(FtwError::Version(2))));
    }

    #[test]
    fn reject_wrong_align() {
        let doc = sample_index().replace("\"align\": 4096", "\"align\": 512");
        assert!(matches!(
            FtwIndex::parse(&doc),
            Err(FtwError::Alignment(512))
        ));
    }

    #[test]
    fn reject_unaligned_tensor() {
        let doc = sample_index().replace(r#""global_off": 8192"#, r#""global_off": 8193"#);
        assert!(matches!(FtwIndex::parse(&doc), Err(FtwError::Invalid(_))));
    }

    #[test]
    fn reject_shard_gap() {
        let doc = sample_index().replace(r#""global_off": 12288"#, r#""global_off": 13312"#);
        assert!(matches!(FtwIndex::parse(&doc), Err(FtwError::Invalid(_))));
    }

    #[test]
    fn reject_tensor_overruns_total() {
        let doc = sample_index().replace(r#""global_off": 24576"#, r#""global_off": 28672"#);
        assert!(matches!(FtwIndex::parse(&doc), Err(FtwError::Invalid(_))));
    }

    #[test]
    fn reject_shape_nbytes_mismatch() {
        let doc = sample_index().replace(r#""shape": [4, 8]"#, r#""shape": [4, 9]"#);
        assert!(matches!(FtwIndex::parse(&doc), Err(FtwError::Invalid(_))));
    }

    #[test]
    fn piece_single_shard() {
        let ix = FtwIndex::parse(sample_index()).unwrap();
        let desc = ix.entry("model.router.weight").unwrap();
        let pieces = ix.pieces(desc.global_off, desc.nbytes).unwrap();
        assert_eq!(
            pieces,
            vec![Piece {
                file: "freetoken-00000.ftw".into(),
                file_off: 4096,
                dest_off: 0,
                length: 192,
            }]
        );
    }

    #[test]
    fn piece_spans_shard_boundary() {
        let ix = FtwIndex::parse(spanning_index()).unwrap();
        let pieces = ix.pieces(0, 4912).unwrap();
        assert_eq!(
            pieces,
            vec![
                Piece {
                    file: "freetoken-00000.ftw".into(),
                    file_off: 0,
                    dest_off: 0,
                    length: 4096
                },
                Piece {
                    file: "freetoken-00001.ftw".into(),
                    file_off: 0,
                    dest_off: 4096,
                    length: 816
                },
            ]
        );
        // shard pieces share the 4096 boundary
        assert_eq!(pieces[0].file_off + pieces[0].length, 4096);
        assert_eq!(pieces[1].file_off, 0);
        assert_eq!(pieces[1].dest_off, 4096);
    }

    #[test]
    fn piece_range_beyond_eof_errors() {
        let ix = FtwIndex::parse(spanning_index()).unwrap();
        assert!(matches!(ix.pieces(0, 9000), Err(FtwError::OutOfRange)));
    }

    #[test]
    fn layer_bank_entry_names() {
        assert_eq!(
            layer_bank_entry("gate_up_packed#L00007"),
            Some(("gate_up_packed".to_string(), 7))
        );
        assert_eq!(
            layer_bank_entry("down_packed#L00040"),
            Some(("down_packed".to_string(), 40))
        );
        assert_eq!(layer_bank_entry("gate_up_alpha"), None);
        assert!(layer_bank_entry("foo#L12").is_none());
        assert!(layer_bank_entry("#L00007").is_none());
        assert!(layer_bank_entry("foo#L0000X").is_none());
    }
}
