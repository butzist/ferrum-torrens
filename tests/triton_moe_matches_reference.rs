// Gated GPU validation of the Triton NVFP4 grouped-MoE kernel against the
// scalar oracle on the real converted checkpoint. Run with:
//   cargo test --release --features triton --test triton_moe_matches_reference -- --ignored --nocapture
// (needs a CUDA device with native fp8 (sm_89+) and .model-cache/Qwen3.6-35B-A3B-NVFP4.ftw)
//
// This is the gate for the whole D4 path: real expert slabs out of the mmap
// banks, staged H2D through `FtwBanks` -> `TritonMoE`, planned and copied by
// `run_experts`, the kernel over the slot routes, and `grouped_moe_reference`
// over the same host bytes. The tolerance is the scalar path's own (decision
// 23): the same math accumulated in a different order, so agreement is to ~1e-3
// of the output scale, not bitwise.
//
// The kernel adapter is pyo3-embedded, so `python_path` points at this repo's
// `python/` and `site_packages` at the FreeToken venv's site-packages
// (decision 29). Set FERRUM_KERNEL_SITE_PACKAGES to override the default.

#![cfg(feature = "triton")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ferrum_torrens::application::run_experts::{ExpertStep, run_experts};
use ferrum_torrens::domain::expert_cache::ExpertCache;
use ferrum_torrens::domain::expert_kernel::{ExpertSlabSource, HostBanks, MoEGroup, Part, Slab};
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::banks::FtwBanks;
use ferrum_torrens::infrastructure::ftw::{FtwIndex, MmapFtwReader};
use ferrum_torrens::infrastructure::triton::{ExpertShapes, TritonMoE, TritonRuntime};

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";
const LAYER: usize = 17;
const TOP_K: usize = 8;
const TOKENS: usize = 4;
const SLOTS: usize = 64;
const TOLERANCE: f32 = 2e-3;

/// `(rows, cols)` of one part's bank, straight from the index.
fn bank_shape(index: &FtwIndex, part: Part, layer: usize) -> (usize, usize) {
    let name = match part {
        Part::GateUp => format!("gate_up_packed#L{layer:05}"),
        Part::Down => format!("down_packed#L{layer:05}"),
    };
    let t = index.entry(&name).unwrap_or_else(|| panic!("{name}"));
    (t.shape[1], t.shape[2] * 2)
}

fn shapes(index: &FtwIndex) -> ExpertShapes {
    let (gu_rows, hidden) = bank_shape(index, Part::GateUp, LAYER);
    let (dn_rows, dn_cols) = bank_shape(index, Part::Down, LAYER);
    assert_eq!(dn_rows, hidden, "down slab rows are the hidden dim");
    let shapes = ExpertShapes::new(hidden, gu_rows / 2).expect("expert shapes from the index");
    assert_eq!(shapes.inter, dn_cols, "down slab cols are moe_intermediate");
    shapes
}

/// Every gate failure prints its port error on **stdout** before panicking:
/// cargo shows a failing test's captured stdout even without `--nocapture`, so
/// the log identifies the step and the reason whether or not stderr was teed.
fn gate<T>(what: &str, r: Result<T, impl std::fmt::Display>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => {
            println!("GATE FAILED at {what}: {e}");
            panic!("gate failed at {what}")
        }
    }
}

fn site_packages() -> Vec<PathBuf> {
    match std::env::var("FERRUM_KERNEL_SITE_PACKAGES") {
        Ok(v) => v
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect(),
        // The FreeToken venv is the only place the torch/triton pair lives.
        Err(_) => vec![PathBuf::from(
            "FreeToken/.venv/lib/python3.12/site-packages",
        )],
    }
}

/// The banks own their bytes; the domain view borrows them.
struct OwnedBanks {
    gu_packed: Vec<u8>,
    gu_scale: Vec<u8>,
    gu_global: Vec<u8>,
    dn_packed: Vec<u8>,
    dn_scale: Vec<u8>,
    dn_global: Vec<u8>,
    shapes: ExpertShapes,
    slots: usize,
}

impl OwnedBanks {
    fn as_host(&self) -> HostBanks<'_> {
        HostBanks {
            gu_packed: &self.gu_packed,
            gu_scale: &self.gu_scale,
            gu_global: &self.gu_global,
            dn_packed: &self.dn_packed,
            dn_scale: &self.dn_scale,
            dn_global: &self.dn_global,
            slots: self.slots,
            gu_rows: self.shapes.gate_up_rows(),
            gu_cols: self.shapes.hidden,
            dn_rows: self.shapes.hidden,
            dn_cols: self.shapes.inter,
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn scale_of(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
}

/// Slot-major banks over the experts the plan made resident, read straight from
/// the checkpoint. Slots the plan never wrote read as zeros, exactly what the
/// device cache holds for them.
fn owned_banks(
    source: &dyn ExpertSlabSource,
    shapes: ExpertShapes,
    slots: usize,
    routed: &[u32],
    routes: &[i32],
) -> OwnedBanks {
    let mut slabs: Vec<(Slab, Slab)> = vec![(Slab::default(), Slab::default()); slots];
    for (expert, &slot) in routed.iter().zip(routes) {
        if slot < 0 {
            continue;
        }
        slabs[slot as usize] = (
            source.slab(LAYER, *expert, Part::GateUp).unwrap(),
            source.slab(LAYER, *expert, Part::Down).unwrap(),
        );
    }
    let mut banks = OwnedBanks {
        gu_packed: Vec::new(),
        gu_scale: Vec::new(),
        gu_global: Vec::new(),
        dn_packed: Vec::new(),
        dn_scale: Vec::new(),
        dn_global: Vec::new(),
        shapes,
        slots,
    };
    for (gu, dn) in &slabs {
        banks.gu_packed.extend(&gu.packed);
        banks.gu_scale.extend(&gu.scale);
        banks.gu_global.extend(&gu.global);
        banks.dn_packed.extend(&dn.packed);
        banks.dn_scale.extend(&dn.scale);
        banks.dn_global.extend(&dn.global);
    }
    banks
}

#[test]
#[ignore = "needs a CUDA device and the converted checkpoint in .model-cache"]
fn the_kernel_matches_the_scalar_oracle_on_real_expert_slabs() {
    let sites = site_packages();
    println!(
        "GATE env: cwd={} ftw={} python_path=python site_packages={:?}",
        std::env::current_dir().unwrap().display(),
        FTW_DIR,
        sites,
    );
    println!(
        "GATE loader: LD_LIBRARY_PATH={}",
        std::env::var("LD_LIBRARY_PATH").unwrap_or_else(|_| "<unset>".into()),
    );
    let index_text = gate(
        "freetoken_weight.json",
        std::fs::read_to_string(format!("{FTW_DIR}/freetoken_weight.json")),
    );
    let index = Arc::new(gate("FtwIndex::parse", FtwIndex::parse(&index_text)));
    let reader: Arc<dyn WeightsReader> =
        Arc::new(gate("mmap open", MmapFtwReader::open(Path::new(FTW_DIR))));
    let shapes = shapes(&index);
    let runtime = Arc::new(gate(
        "embedded python + torch",
        TritonRuntime::new(Path::new("python"), &sites),
    ));
    let source: Arc<dyn ExpertSlabSource> = Arc::new(FtwBanks::new(reader, index));
    let moe = Arc::new(Mutex::new(TritonMoE::new(
        runtime,
        source.clone(),
        shapes,
        SLOTS,
    )));

    // Deterministic inputs: a seeded activation and a spread of expert ids, so
    // a failure is reproducible. Experts stay inside the slot cache, so the
    // routes the kernel sees are all resident.
    let hidden = shapes.hidden;
    let x: Vec<f32> = (0..TOKENS * hidden)
        .map(|i| ((i * 2654435761usize % 1009) as f32 / 1009.0) - 0.5)
        .collect();
    let routed: Vec<u32> = (0..TOKENS * TOP_K)
        .map(|i| ((i * 7 + 3) % SLOTS) as u32)
        .collect();
    let weights: Vec<f32> = (0..TOKENS * TOP_K)
        .map(|i| 1.0 / TOP_K as f32 + i as f32 * 1e-3)
        .collect();

    let step = ExpertStep {
        hidden,
        inter: shapes.inter,
        top_k: TOP_K,
        x: &x,
        experts: &routed,
        weights: &weights,
    };
    let mut cache = ExpertCache::new(LAYER + 1, SLOTS, SLOTS);
    let mut got = vec![0.0f32; TOKENS * hidden];
    let t0 = Instant::now();
    {
        // One shared handle is both ports: the plan copies through the store,
        // the kernel runs through the same banks.
        let ports = moe.as_ref();
        gate(
            "run_experts through the triton adapter",
            run_experts(&mut cache, ports, ports, LAYER, &step, &mut got),
        );
    }
    let kernel_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Everything routed is resident now, so this plan copies nothing and its
    // rewrites are the very routes the kernel used.
    let routes = cache.ensure_experts(LAYER, &routed).rewrites;
    assert!(
        routes.iter().all(|&r| r >= 0),
        "the gate keeps every routed expert resident"
    );
    let oracle = owned_banks(&*source, shapes, SLOTS, &routed, &routes);
    let req = MoEGroup {
        hidden,
        inter: shapes.inter,
        x: &x,
        top_k: TOP_K,
        routes: &routes,
        weights: &weights,
    };
    let mut want = vec![0.0f32; TOKENS * hidden];
    gate(
        "scalar oracle",
        ferrum_torrens::domain::expert_kernel::grouped_moe_reference(
            &oracle.as_host(),
            &req,
            &mut want,
        ),
    );

    let diff = max_abs_diff(&got, &want);
    let scale = scale_of(&want);
    let rel = diff / scale.max(1e-6);
    println!(
        "layer {LAYER}: {TOKENS} tokens x top_k {TOP_K}, {SLOTS} slots staged, \
         max|kernel - scalar| = {diff:.3e} of {scale:.3e} ({rel:.2e} rel) in {kernel_ms:.1} ms"
    );
    if diff > TOLERANCE * scale.max(1.0) {
        println!(
            "GATE FAILED at the numeric comparison: {diff:.3e} > {:.3e}",
            TOLERANCE * scale.max(1.0)
        );
        panic!("kernel drifted from the scalar oracle");
    }
    println!("PASS");
}
