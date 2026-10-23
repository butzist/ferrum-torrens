// FTW writer mirroring FreeToken's `FTWWriter`: one logical byte region sliced
// into files at `shard_limit`, every tensor starting at an ALIGN boundary and
// padded to ALIGN so any tensor (or shard-local slice of one) is block-aligned.

use crate::domain::weights::DType;
use crate::infrastructure::ftw::{ALIGN, FORMAT_TAG, FORMAT_VERSION, INDEX_NAME};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum FtwError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid shard_limit {0}: must be a positive multiple of {ALIGN}")]
    ShardLimit(u64),
    #[error("FTW index already exists at {0}")]
    AlreadyExists(PathBuf),
    #[error("tensor {name:?}: {msg}")]
    Tensor { name: String, msg: String },
    #[error("duplicate tensor {0:?}")]
    Duplicate(String),
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorEntry {
    pub name: String,
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub global_off: u64,
    pub nbytes: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ShardEntry {
    pub file: String,
    pub global_off: u64,
    pub nbytes: u64,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Counts {
    pub weight: usize,
    pub experts_bank: usize,
}

#[derive(Debug, Clone)]
pub struct FtwMeta {
    pub quant_format: Option<String>,
    pub expert_bank_num_layers: Option<usize>,
    pub counts: Counts,
    pub source_model_path: String,
    pub fingerprint: String,
    pub copied_metadata: Vec<String>,
    pub side_files: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct Finalized {
    pub total_bytes: u64,
    pub shard_count: usize,
    pub tensor_count: usize,
}

#[derive(serde::Serialize)]
struct IndexOut<'a> {
    format: &'static str,
    version: u64,
    align: u64,
    shard_limit: u64,
    total_bytes: u64,
    tensors: &'a [TensorEntry],
    shards: &'a [ShardEntry],
    quant_format: Option<&'a str>,
    expert_bank_num_layers: Option<usize>,
    counts: Counts,
    source_model_path: &'a str,
    fingerprint: &'a str,
    copied_metadata: &'a [String],
    side_files: &'a [String],
}

pub struct FtwWriter {
    out_dir: PathBuf,
    shard_limit: u64,
    tensors: Vec<TensorEntry>,
    names: HashSet<String>,
    shards: Vec<ShardEntry>,
    global: u64,
    file: Option<BufWriter<File>>,
    shard_idx: Option<usize>,
    shard_start: u64,
    cur: u64,
}

impl FtwWriter {
    pub fn new(out_dir: &Path, shard_limit: u64) -> Result<Self, FtwError> {
        if shard_limit == 0 || !shard_limit.is_multiple_of(ALIGN) {
            return Err(FtwError::ShardLimit(shard_limit));
        }
        if out_dir.join(INDEX_NAME).exists() {
            return Err(FtwError::AlreadyExists(out_dir.to_path_buf()));
        }
        std::fs::create_dir_all(out_dir)?;
        Ok(Self {
            out_dir: out_dir.to_path_buf(),
            shard_limit,
            tensors: Vec::new(),
            names: HashSet::new(),
            shards: Vec::new(),
            global: 0,
            file: None,
            shard_idx: None,
            shard_start: 0,
            cur: 0,
        })
    }

    pub fn total_bytes(&self) -> u64 {
        self.global
    }

    /// Append one tensor; ``shape`` and ``dtype`` must describe exactly ``bytes``.
    pub fn add_tensor(
        &mut self,
        name: &str,
        kind: &str,
        dtype: DType,
        shape: &[usize],
        bytes: &[u8],
    ) -> Result<(), FtwError> {
        if !self.names.insert(name.to_string()) {
            return Err(FtwError::Duplicate(name.to_string()));
        }
        let elements = shape.iter().try_fold(1u64, |acc, &d| {
            u64::try_from(d).ok().and_then(|d| acc.checked_mul(d))
        });
        let expected = elements
            .and_then(|n| n.checked_mul(dtype.element_size() as u64))
            .ok_or_else(|| FtwError::Tensor {
                name: name.to_string(),
                msg: "shape overflows".into(),
            })?;
        if expected != bytes.len() as u64 {
            return Err(FtwError::Tensor {
                name: name.to_string(),
                msg: format!(
                    "shape/dtype imply {expected} bytes but {} were given",
                    bytes.len()
                ),
            });
        }
        if bytes.is_empty() {
            return Err(FtwError::Tensor {
                name: name.to_string(),
                msg: "empty tensors cannot be stored".into(),
            });
        }
        let nbytes = bytes.len() as u64;
        if self.file.is_none()
            || (nbytes <= self.shard_limit && self.cur + nbytes > self.shard_limit)
        {
            self.roll()?;
        }
        let global_off = self.global;
        if !global_off.is_multiple_of(ALIGN) {
            return Err(FtwError::Tensor {
                name: name.to_string(),
                msg: format!("start offset {global_off} is not aligned"),
            });
        }
        self.write_data(bytes)?;
        self.tensors.push(TensorEntry {
            name: name.to_string(),
            kind: kind.to_string(),
            dtype: dtype.as_str().to_string(),
            shape: shape.to_vec(),
            global_off,
            nbytes,
        });
        let pad = align_up(self.global) - self.global;
        if pad > 0 {
            self.write_zeros(pad)?;
        }
        Ok(())
    }

    pub fn finalize(mut self, meta: FtwMeta) -> Result<Finalized, FtwError> {
        if self.file.is_some() {
            self.close_shard()?;
        }
        let index = IndexOut {
            format: FORMAT_TAG,
            version: FORMAT_VERSION,
            align: ALIGN,
            shard_limit: self.shard_limit,
            total_bytes: self.global,
            tensors: &self.tensors,
            shards: &self.shards,
            quant_format: meta.quant_format.as_deref(),
            expert_bank_num_layers: meta.expert_bank_num_layers,
            counts: meta.counts,
            source_model_path: &meta.source_model_path,
            fingerprint: &meta.fingerprint,
            copied_metadata: &meta.copied_metadata,
            side_files: &meta.side_files,
        };
        let tmp = self.out_dir.join(format!("{INDEX_NAME}.tmp"));
        let mut file = File::create(&tmp)?;
        serde_json::to_writer(&mut file, &index)?;
        file.flush()?;
        std::fs::rename(&tmp, self.out_dir.join(INDEX_NAME))?;
        Ok(Finalized {
            total_bytes: self.global,
            shard_count: self.shards.len(),
            tensor_count: self.tensors.len(),
        })
    }

    fn roll(&mut self) -> Result<(), FtwError> {
        if self.file.is_some() {
            self.close_shard()?;
        }
        let idx = self.shard_idx.map_or(0, |i| i + 1);
        let path = self.out_dir.join(format!("freetoken-{idx:05}.ftw"));
        let file = File::create(path)?;
        self.file = Some(BufWriter::with_capacity(8 << 20, file));
        self.shard_idx = Some(idx);
        self.shard_start = self.global;
        self.cur = 0;
        Ok(())
    }

    fn close_shard(&mut self) -> Result<(), FtwError> {
        if let Some(mut file) = self.file.take() {
            file.flush()?;
        }
        let idx = self.shard_idx.expect("a shard is open");
        self.shards.push(ShardEntry {
            file: format!("freetoken-{idx:05}.ftw"),
            global_off: self.shard_start,
            nbytes: self.cur,
        });
        Ok(())
    }

    fn write_data(&mut self, data: &[u8]) -> Result<(), FtwError> {
        let mut off = 0;
        while off < data.len() {
            if self.cur == self.shard_limit {
                self.roll()?;
            }
            let take = (data.len() - off).min((self.shard_limit - self.cur) as usize);
            let file = self.file.as_mut().expect("a shard is open");
            file.write_all(&data[off..off + take])?;
            off += take;
            self.cur += take as u64;
            self.global += take as u64;
        }
        Ok(())
    }

    fn write_zeros(&mut self, mut len: u64) -> Result<(), FtwError> {
        const ZEROS: [u8; ALIGN as usize] = [0u8; ALIGN as usize];
        while len > 0 {
            let take = len.min(ALIGN) as usize;
            self.write_data(&ZEROS[..take])?;
            len -= take as u64;
        }
        Ok(())
    }
}

fn align_up(n: u64) -> u64 {
    n.div_ceil(ALIGN) * ALIGN
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::ftw::{DEFAULT_SHARD_LIMIT, FtwIndex};
    use std::os::unix::fs::FileExt;

    fn temp() -> crate::converter::testing::TempDir {
        crate::converter::testing::TempDir::new("ftw-writer")
    }

    #[test]
    fn writes_aligned_tensors_and_spans_shards() {
        let dir = temp();
        let shard_limit = 3 * ALIGN;
        let mut writer = FtwWriter::new(dir.path(), shard_limit).unwrap();
        let small = vec![0xabu8; ALIGN as usize];
        let big: Vec<u8> = (0..shard_limit + ALIGN).map(|i| i as u8).collect();
        writer
            .add_tensor("small", "weight", DType::U8, &[ALIGN as usize], &small)
            .unwrap();
        writer
            .add_tensor("big", "experts_bank", DType::U8, &[big.len()], &big)
            .unwrap();
        let tail = vec![0x5au8; 5];
        writer
            .add_tensor("tail", "weight", DType::U8, &[5], &tail)
            .unwrap();
        let fin = writer
            .finalize(FtwMeta {
                quant_format: Some("nvfp4".into()),
                expert_bank_num_layers: Some(1),
                counts: Counts {
                    weight: 2,
                    experts_bank: 1,
                },
                source_model_path: "/src".into(),
                fingerprint: "0123456789abcdef".into(),
                copied_metadata: vec!["config.json".into()],
                side_files: vec![],
            })
            .unwrap();
        assert_eq!(fin.tensor_count, 3);

        let doc = std::fs::read_to_string(dir.path().join(INDEX_NAME)).unwrap();
        let index = FtwIndex::parse(&doc).unwrap();
        assert_eq!(index.total_bytes, fin.total_bytes);
        assert_eq!(index.shard_limit, shard_limit);
        assert_eq!(
            index.quant_format,
            Some(crate::domain::weights::QuantFormat::Nvfp4)
        );
        assert_eq!(index.expert_bank_num_layers, Some(1));
        assert!(index.tensors.iter().all(|t| t.global_off % ALIGN == 0));
        assert!(index.shards.len() > 1);

        fn read(root: &Path, index: &FtwIndex, name: &str) -> Vec<u8> {
            let t = index.entry(name).unwrap();
            let mut out = vec![0u8; t.nbytes as usize];
            let mut done = 0usize;
            for p in index.pieces(t.global_off, t.nbytes).unwrap() {
                let f = File::open(root.join(&p.file)).unwrap();
                f.read_exact_at(&mut out[done..done + p.length as usize], p.file_off)
                    .unwrap();
                done += p.length as usize;
            }
            out
        }
        assert_eq!(read(dir.path(), &index, "small"), small);
        assert_eq!(read(dir.path(), &index, "big"), big);
        assert_eq!(read(dir.path(), &index, "tail"), tail);
    }

    #[test]
    fn small_tensors_never_split_across_a_shard() {
        let dir = temp();
        let shard_limit = 2 * ALIGN;
        let mut writer = FtwWriter::new(dir.path(), shard_limit).unwrap();
        let one = vec![1u8; ALIGN as usize];
        writer
            .add_tensor("a", "weight", DType::U8, &[ALIGN as usize], &one)
            .unwrap();
        writer
            .add_tensor("b", "weight", DType::U8, &[ALIGN as usize], &one)
            .unwrap();
        writer
            .add_tensor("c", "weight", DType::U8, &[ALIGN as usize], &one)
            .unwrap();
        writer
            .finalize(FtwMeta {
                quant_format: None,
                expert_bank_num_layers: None,
                counts: Counts::default(),
                source_model_path: "/src".into(),
                fingerprint: "x".into(),
                copied_metadata: vec![],
                side_files: vec![],
            })
            .unwrap();
        let index = FtwIndex::parse(&std::fs::read_to_string(dir.path().join(INDEX_NAME)).unwrap())
            .unwrap();
        assert_eq!(index.tensors.len(), 3);
        assert_eq!(index.shards.len(), 2);
        for t in &index.tensors {
            let pieces = index.pieces(t.global_off, t.nbytes).unwrap();
            assert_eq!(pieces.len(), 1, "{} was split", t.name);
        }
    }

    #[test]
    fn refuses_bad_limits_and_overwrites() {
        let dir = temp();
        assert!(matches!(
            FtwWriter::new(dir.path(), 0),
            Err(FtwError::ShardLimit(0))
        ));
        assert!(matches!(
            FtwWriter::new(dir.path(), ALIGN + 1),
            Err(FtwError::ShardLimit(_))
        ));
        let mut writer = FtwWriter::new(dir.path(), DEFAULT_SHARD_LIMIT).unwrap();
        writer
            .add_tensor("a", "weight", DType::U8, &[1], &[1])
            .unwrap();
        writer
            .finalize(FtwMeta {
                quant_format: None,
                expert_bank_num_layers: None,
                counts: Counts::default(),
                source_model_path: "/src".into(),
                fingerprint: "x".into(),
                copied_metadata: vec![],
                side_files: vec![],
            })
            .unwrap();
        assert!(matches!(
            FtwWriter::new(dir.path(), DEFAULT_SHARD_LIMIT),
            Err(FtwError::AlreadyExists(_))
        ));
    }

    #[test]
    fn rejects_a_shape_that_disagrees_with_the_bytes() {
        let dir = temp();
        let mut writer = FtwWriter::new(dir.path(), DEFAULT_SHARD_LIMIT).unwrap();
        assert!(matches!(
            writer.add_tensor("a", "weight", DType::F32, &[2], &[0u8; 4]),
            Err(FtwError::Tensor { .. })
        ));
    }
}
