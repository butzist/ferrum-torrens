// CPU-only converter gates against the real HF snapshot and the golden FTW.
// Run one at a time, e.g.:
//   cargo test --test converter_matches_golden full_conversion -- --ignored --nocapture
//
// These do not touch the GPU or load the engine. The full gate writes ~22 GiB
// into `.model-cache/` and compares every tensor byte-for-byte.

use ferrum_torrens::converter::{ConvertOptions, convert};
use ferrum_torrens::domain::weights::{QuantFormat, TensorKind};
use ferrum_torrens::infrastructure::ftw::FtwIndex;
use serial_test::serial;
use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const HF_DIR: &str = ".model-cache/hf/Qwen3.6-35B-A3B-NVFP4";
const GOLDEN_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(prefix: &str) -> Self {
        let unique = format!(
            "{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = Path::new(".model-cache").join(format!("converter-{prefix}-{unique}"));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn inputs() -> Option<(PathBuf, PathBuf)> {
    let hf = Path::new(HF_DIR);
    let golden = Path::new(GOLDEN_DIR);
    if !hf.join("model.safetensors.index.json").is_file()
        || !golden.join("freetoken_weight.json").is_file()
    {
        eprintln!("skip: {HF_DIR} and {GOLDEN_DIR} are required");
        return None;
    }
    Some((hf.to_path_buf(), golden.to_path_buf()))
}

fn parse_index(dir: &Path) -> FtwIndex {
    FtwIndex::parse(&std::fs::read_to_string(dir.join("freetoken_weight.json")).unwrap()).unwrap()
}

struct Shards {
    files: HashMap<String, std::fs::File>,
}

impl Shards {
    fn new(dir: &Path, index: &FtwIndex) -> Self {
        let files = index
            .shards
            .iter()
            .map(|s| {
                (
                    s.file.clone(),
                    std::fs::File::open(dir.join(&s.file)).unwrap(),
                )
            })
            .collect();
        Self { files }
    }

    fn read(&mut self, index: &FtwIndex, name: &str) -> Vec<u8> {
        let t = index.entry(name).unwrap();
        let mut out = vec![0u8; t.nbytes as usize];
        let mut done = 0usize;
        for piece in index.pieces(t.global_off, t.nbytes).unwrap() {
            let file = self.files.get(&piece.file).unwrap();
            read_exact_at(
                file,
                piece.file_off,
                &mut out[done..done + piece.length as usize],
            );
            done += piece.length as usize;
        }
        out
    }
}

fn read_exact_at(file: &std::fs::File, mut offset: u64, mut dst: &mut [u8]) {
    while !dst.is_empty() {
        let n = file.read_at(dst, offset).unwrap();
        assert!(n > 0, "short FTW read");
        offset += n as u64;
        dst = &mut dst[n..];
    }
}

fn compare_produced(out: &Path, new: &FtwIndex, golden_dir: &Path, golden: &FtwIndex) -> usize {
    let mut new_shards = Shards::new(out, new);
    let mut golden_shards = Shards::new(golden_dir, golden);
    let mut mismatches = Vec::new();
    for t in &new.tensors {
        let Some(g) = golden.entry(&t.name) else {
            mismatches.push(format!("{}: not in the golden FTW", t.name));
            continue;
        };
        if g.kind != t.kind {
            mismatches.push(format!("{}: kind {:?} vs {:?}", t.name, t.kind, g.kind));
        }
        if g.dtype != t.dtype {
            mismatches.push(format!("{}: dtype {} vs {}", t.name, t.dtype, g.dtype));
        }
        if g.shape != t.shape {
            mismatches.push(format!("{}: shape {:?} vs {:?}", t.name, t.shape, g.shape));
        }
        if g.nbytes != t.nbytes {
            mismatches.push(format!("{}: nbytes {} vs {}", t.name, t.nbytes, g.nbytes));
            continue;
        }
        if new_shards.read(new, &t.name) != golden_shards.read(golden, &t.name) {
            mismatches.push(format!("{}: bytes differ", t.name));
        }
    }
    if !mismatches.is_empty() {
        for line in mismatches.iter().take(20) {
            eprintln!("MISMATCH {line}");
        }
        panic!(
            "{} produced tensors disagree with the golden FTW",
            mismatches.len()
        );
    }
    new.tensors.len()
}

#[test]
#[ignore = "CPU-only golden gate; needs the HF snapshot and the golden FTW under .model-cache"]
#[serial]
fn subset_conversion_matches_golden() {
    let Some((hf, golden_dir)) = inputs() else {
        return;
    };
    let out = TempDir::new("subset");
    let mut opts = ConvertOptions::new(&hf, out.path());
    opts.layers = Some(4);
    opts.limit = Some(vec![
        "lm_head.*".into(),
        "model.embed_tokens.weight".into(),
        "model.norm.weight".into(),
        "model.layers.0.*".into(),
        "model.layers.3.*".into(),
        "*#L00000".into(),
    ]);
    let start = Instant::now();
    let report = convert(&opts).unwrap();
    eprintln!(
        "subset: {} weights + {} banks, {} bytes in {:.1}s",
        report.weight_count,
        report.experts_bank_count,
        report.total_bytes,
        start.elapsed().as_secs_f64()
    );
    let new = parse_index(out.path());
    let golden = parse_index(&golden_dir);
    let compared = compare_produced(out.path(), &new, &golden_dir, &golden);
    eprintln!("subset: compared {compared} tensors byte-for-byte");
    for bank in [
        "gate_up_packed#L00000",
        "gate_up_scale#L00000",
        "gate_up_global#L00000",
        "down_packed#L00000",
        "down_scale#L00000",
        "down_global#L00000",
    ] {
        assert!(new.entry(bank).is_some(), "{bank} missing");
    }
    assert!(
        new.entry("model.layers.3.self_attn.qkv_proj.weight")
            .is_some()
    );
    assert!(
        new.entry("model.layers.0.linear_attn.in_proj_qkvz.weight")
            .is_some()
    );
    assert_eq!(report.experts_bank_count, 6);
}

#[test]
#[ignore = "CPU-only golden gate; full ~22 GiB run, several minutes"]
#[serial]
fn full_conversion_matches_golden() {
    let Some((hf, golden_dir)) = inputs() else {
        return;
    };
    let out = TempDir::new("full");
    let opts = ConvertOptions::new(&hf, out.path());
    let start = Instant::now();
    let report = convert(&opts).unwrap();
    let elapsed = start.elapsed();
    eprintln!(
        "full: {} weights + {} banks, {} bytes, {} shards in {:.1}s",
        report.weight_count,
        report.experts_bank_count,
        report.total_bytes,
        report.shard_count,
        elapsed.as_secs_f64()
    );

    let new = parse_index(out.path());
    let golden = parse_index(&golden_dir);
    assert_eq!(report.weight_count, 1149);
    assert_eq!(report.experts_bank_count, 246);
    assert_eq!(new.tensors.len(), 1394);
    assert_eq!(new.tensors.len(), golden.tensors.len());
    assert_eq!(new.quant_format, Some(QuantFormat::Nvfp4));
    assert_eq!(new.expert_bank_num_layers, Some(40));
    assert_eq!(new.has_mtp, Some(true));
    assert_eq!(new.max_draft_depth, Some(1));
    assert_eq!(
        new.tensors
            .iter()
            .filter(|t| t.kind == TensorKind::Weight)
            .count(),
        1148
    );
    assert_eq!(
        new.tensors
            .iter()
            .filter(|t| t.kind == TensorKind::ExpertsBank)
            .count(),
        240
    );
    assert_eq!(new.total_bytes, golden.total_bytes);
    assert_eq!(new.shard_limit, golden.shard_limit);
    assert_eq!(
        new.meta.get("counts"),
        Some(&serde_json::json!({"weight": 1148, "experts_bank": 240}))
    );
    assert_eq!(new.meta.get("side_files"), Some(&serde_json::json!([])));
    let fingerprint = new.meta["fingerprint"].as_str().unwrap();
    assert_eq!(fingerprint.len(), 16);
    assert!(fingerprint.chars().all(|c| c.is_ascii_hexdigit()));
    let mut copied: Vec<String> =
        serde_json::from_value(new.meta["copied_metadata"].clone()).unwrap();
    let mut want = vec![
        "generation_config.json",
        "tokenizer.json",
        "configuration.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "video_preprocessor_config.json",
        "config.json",
        "vocab.json",
        "hf_quant_config.json",
        "preprocessor_config.json",
    ];
    copied.sort();
    want.sort();
    assert_eq!(copied, want);

    let compared = Instant::now();
    let n = compare_produced(out.path(), &new, &golden_dir, &golden);
    let names: std::collections::HashSet<&str> =
        new.tensors.iter().map(|t| t.name.as_str()).collect();
    let missing: Vec<&str> = golden
        .tensors
        .iter()
        .filter(|t| !names.contains(t.name.as_str()))
        .map(|t| t.name.as_str())
        .take(10)
        .collect();
    assert!(missing.is_empty(), "missing golden tensors: {missing:?}");
    eprintln!(
        "full: compared {n} tensors byte-for-byte in {:.1}s (convert {:.1}s)",
        compared.elapsed().as_secs_f64(),
        elapsed.as_secs_f64()
    );
}

#[test]
#[ignore = "CPU-only; parses a freshly converted FTW through the engine index"]
#[serial]
fn converted_index_parses() {
    let Some((hf, _)) = inputs() else {
        return;
    };
    let out = TempDir::new("parse");
    let mut opts = ConvertOptions::new(&hf, out.path());
    opts.layers = Some(1);
    let report = convert(&opts).unwrap();
    let index = parse_index(out.path());
    assert_eq!(
        index.tensors.len(),
        report.weight_count + report.experts_bank_count
    );
    assert_eq!(index.quant_format, Some(QuantFormat::Nvfp4));
    assert_eq!(index.expert_bank_num_layers, Some(1));
    assert_eq!(report.experts_bank_count, 6);
    assert!(index.entry("model.embed_tokens.weight").is_some());
    assert!(
        index
            .entry("model.layers.0.linear_attn.in_proj_qkvz.weight")
            .is_some()
    );
    eprintln!(
        "parse: {} tensors, quant {:?}, bank layers {:?}",
        index.tensors.len(),
        index.quant_format,
        index.expert_bank_num_layers
    );
}
