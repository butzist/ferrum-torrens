//! M7.2 gate: the FTW dense tower loads through cudarc with the serving
//! storage policy (bf16 widened, FP8 packed, NVFP4 K-major) and without
//! libtorch in the process.
//!
//!   cargo test --release --features cudarc --test cudarc_load -- --ignored --nocapture
//!
//! One process at a time; the loader reads the dense set only, so it does not
//! pin the routed-expert banks.
#![cfg(feature = "cudarc")]

use std::sync::Arc;

use ferrum_torrens::domain::dequant::{bf16_array, nvfp4_block};
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::cudarc::Runtime;
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::models::qwen3_6::load_cudarc::{DenseWeight, HeadWeight, Mixer, Tower};
use serial_test::serial;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";

fn raw(index: &FtwIndex, reader: &dyn WeightsReader, name: &str) -> Vec<u8> {
    let t = index
        .entry(name)
        .unwrap_or_else(|| panic!("missing {name}"));
    let mut buf = vec![0u8; t.nbytes as usize];
    reader.read_range(t.global_off, t.nbytes, &mut buf).unwrap();
    buf
}

fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn maps_has(needle: &str) -> bool {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    maps.lines().any(|l| l.contains(needle))
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
        / scale
}

#[test]
#[ignore = "needs the converted checkpoint and a CUDA device"]
#[serial]
fn the_dense_tower_loads_through_cudarc() {
    let dir = std::path::Path::new(FTW_DIR);
    let index = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(dir.join("freetoken_weight.json"))
                .expect("read the FTW index"),
        )
        .expect("parse the FTW index"),
    );
    let reader: Arc<dyn WeightsReader> = Arc::new(FtwReader::open(dir).expect("open the FTW"));
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let params = ferrum_torrens::models::qwen3_6::config::parse_text_config(
        config.get("text_config").expect("text_config"),
    )
    .expect("parse text_config");

    let rt = Runtime::open(0).expect("open cudarc runtime");
    let (free0, total) = rt.driver().mem_info().unwrap();
    let tower = Tower::load(params.clone(), rt.clone(), index.clone(), reader.clone())
        .expect("load the dense tower through cudarc");
    let (free1, _) = rt.driver().mem_info().unwrap();

    assert_eq!(tower.layers.len(), params.num_layers);
    assert_eq!(tower.embed.rows, params.vocab_size);
    assert_eq!(tower.embed.cols, params.hidden_size);
    assert_eq!(tower.lm_head.rows, params.vocab_size);

    // The three storage policies: bf16 stays bf16, FP8 keeps its e4m3 bytes,
    // the NVFP4 head keeps its K-major banks.
    assert!(matches!(tower.embed.w, DenseWeight::Bf16(_)));
    assert!(matches!(tower.lm_head.weight, HeadWeight::Nvfp4(_)));

    let h = params.hidden_size;

    // Embedding row 0 is the checkpoint's bf16 row, bit for bit.
    let want = bf16_array(
        &raw(&index, reader.as_ref(), "model.embed_tokens.weight"),
        h,
    );
    let got = match &tower.embed.w {
        DenseWeight::Bf16(t) => t.narrow(0, 0, 1).unwrap().to_host_f32().unwrap(),
        _ => unreachable!(),
    };
    assert_eq!(got, want, "embedding row 0");

    // Final norm is the checkpoint's bf16 vector.
    let want = bf16_array(&raw(&index, reader.as_ref(), "model.norm.weight"), h);
    assert_eq!(tower.final_norm.to_host_f32().unwrap(), want);

    // Layer 0 is GDN: its qkvz is FP8 and stays packed with the f32 row scale.
    let Mixer::Linear { qkvz, .. } = &tower.layers[0].mixer else {
        panic!("layer 0 must be a GDN layer");
    };
    let name = "model.layers.0.linear_attn.in_proj_qkvz.weight";
    let bytes = raw(&index, reader.as_ref(), name);
    let scale = f32s(&raw(&index, reader.as_ref(), &format!("{name}_scale")));
    let DenseWeight::Fp8 {
        bytes: got_bytes,
        scale: got_scale,
    } = &qkvz.w
    else {
        panic!("qkvz must stay FP8");
    };
    assert_eq!(got_bytes.to_host_u8().unwrap(), bytes, "qkvz e4m3 bytes");
    assert_eq!(got_scale.to_host_f32().unwrap(), scale, "qkvz row scale");

    // The router is f32 in the forward even though the checkpoint is bf16.
    let want = bf16_array(
        &raw(&index, reader.as_ref(), "model.layers.0.mlp.gate.weight"),
        params.num_experts * h,
    );
    assert_eq!(tower.layers[0].gate.to_host_f32().unwrap(), want);

    // The shared expert keeps both the widened bf16 weight (prefill) and the
    // K-major NVFP4 banks (decode).
    let name = "model.layers.0.mlp.shared_expert.gate_up_proj.weight";
    let cols = params.hidden_size;
    let banks = tower.layers[0]
        .shared
        .gate_up
        .nvfp4
        .as_ref()
        .expect("banks");
    let packed_raw = raw(&index, reader.as_ref(), name);
    let words: Vec<i32> = packed_raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    let k_words = cols / 8;
    let got_words = banks.packed.to_host_i32().unwrap();
    assert_eq!(got_words[0], words[0], "shared packed word");
    assert_eq!(
        got_words[k_major(1, 0, k_words)],
        words[k_words],
        "shared packed transpose"
    );

    let scale_raw = raw(&index, reader.as_ref(), &format!("{name}_scale"));
    let got_scale = banks.scale.to_host_u8().unwrap();
    assert_eq!(got_scale[0], scale_raw[0], "shared scale word");

    let widened = match &tower.layers[0].shared.gate_up.w {
        DenseWeight::Bf16(t) => t.narrow(0, 0, 1).unwrap().to_host_f32().unwrap(),
        _ => panic!("shared expert must be widened for prefill"),
    };
    let row0 = nvfp4_block(
        &packed_raw[..(cols / 2) * 4],
        1,
        cols,
        &scale_raw[..cols / 16],
    );
    assert!(
        max_rel(&widened, &row0) < 1e-3,
        "shared widened row drifted: {}",
        max_rel(&widened, &row0)
    );

    // The head's K-major packed word at [k_word, n] is the checkpoint's word
    // at [n, k_word].
    let HeadWeight::Nvfp4(head) = &tower.lm_head.weight else {
        unreachable!()
    };
    let head_raw = raw(&index, reader.as_ref(), "lm_head.weight");
    let head_words: Vec<i32> = head_raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    let head_k_words = params.hidden_size / 8;
    let got = head.packed.to_host_i32().unwrap();
    assert_eq!(got[0], head_words[0]);
    assert_eq!(
        got[k_major(1, 0, params.vocab_size)],
        head_words[head_k_words]
    );

    let used = (total - free1) >> 20;
    eprintln!(
        "dense tower: {:.2} GiB logical, {} MiB VRAM used (free {} -> {} MiB)",
        tower.resident_bytes() as f64 / (1u64 << 30) as f64,
        used,
        free0 >> 20,
        free1 >> 20
    );
    assert!(
        tower.resident_bytes() < 6 << 30,
        "the dense tower must fit the 8 GiB target: {} bytes",
        tower.resident_bytes()
    );
    // A cudarc-only process must not map libtorch. This also catches an
    // accidental `cuda` feature being enabled alongside `cudarc`.
    for needle in ["libtorch", "libc10", "libpython"] {
        assert!(!maps_has(needle), "{needle} is mapped");
    }
    std::hint::black_box(&tower);
}

/// The K-major bank index of checkpoint element `[row, k]`.
fn k_major(row: usize, k: usize, rows: usize) -> usize {
    k * rows + row
}
