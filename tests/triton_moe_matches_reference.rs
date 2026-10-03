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

/// A slot nobody routed still has to occupy its bytes: the oracle checks the
/// bank lengths the way the device cache holds them, all `slots` experts wide,
/// so an unwritten slot is zeros rather than nothing at all.
fn zero_slab(rows: usize, cols: usize) -> Slab {
    Slab {
        packed: vec![0u8; rows * cols / 2],
        scale: vec![0u8; rows * cols / 16],
        global: vec![0u8; rows * 2],
    }
}

fn owned_banks(
    source: &dyn ExpertSlabSource,
    shapes: ExpertShapes,
    slots: usize,
    routed: &[u32],
    routes: &[i32],
) -> OwnedBanks {
    let gu = shapes.slab_shape(Part::GateUp);
    let dn = shapes.slab_shape(Part::Down);
    let mut slabs: Vec<(Slab, Slab)> = vec![(zero_slab(gu.0, gu.1), zero_slab(dn.0, dn.1)); slots];
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

/// One deterministic case: seeded activations, a spread of expert ids, the
/// routes a plan hands out, and host banks holding the same slabs the slot cache
/// would hold. Shared by the oracle-only test and the GPU gate so a harness bug
/// cannot masquerade as a kernel bug.
struct Case {
    shapes: ExpertShapes,
    x: Vec<f32>,
    routed: Vec<u32>,
    weights: Vec<f32>,
    routes: Vec<i32>,
    banks: OwnedBanks,
}

impl Case {
    fn request<'a>(&'a self, routes: &'a [i32]) -> MoEGroup<'a> {
        MoEGroup {
            hidden: self.shapes.hidden,
            inter: self.shapes.inter,
            x: &self.x,
            top_k: TOP_K,
            routes,
            weights: &self.weights,
        }
    }

    fn step(&self) -> ExpertStep<'_> {
        ExpertStep {
            hidden: self.shapes.hidden,
            inter: self.shapes.inter,
            top_k: TOP_K,
            x: &self.x,
            experts: &self.routed,
            weights: &self.weights,
        }
    }
}

fn plan(index: &FtwIndex, source: &dyn ExpertSlabSource) -> Case {
    let shapes = shapes(index);
    let hidden = shapes.hidden;
    // Deterministic inputs: a seeded activation and a spread of expert ids, so
    // a failure is reproducible. Experts stay inside the slot cache, so every
    // route the kernel sees is resident and no rewrite happens.
    let x: Vec<f32> = (0..TOKENS * hidden)
        .map(|i| ((i * 2654435761usize % 1009) as f32 / 1009.0) - 0.5)
        .collect();
    let routed: Vec<u32> = (0..TOKENS * TOP_K)
        .map(|i| ((i * 7 + 3) % SLOTS) as u32)
        .collect();
    let weights: Vec<f32> = (0..TOKENS * TOP_K)
        .map(|i| 1.0 / TOP_K as f32 + i as f32 * 1e-3)
        .collect();
    let mut cache = ExpertCache::new(LAYER + 1, SLOTS, SLOTS);
    let routes = cache.ensure_experts(LAYER, &routed).rewrites;
    assert!(
        routes.iter().all(|&r| r >= 0),
        "the gate keeps every routed expert resident"
    );
    let banks = owned_banks(source, shapes, SLOTS, &routed, &routes);
    Case {
        shapes,
        x,
        routed,
        weights,
        routes,
        banks,
    }
}

fn open_checkpoint() -> Option<(Arc<FtwIndex>, Arc<dyn WeightsReader>)> {
    if !Path::new(FTW_DIR).is_dir() {
        println!("skipping: no checkpoint in {FTW_DIR}");
        return None;
    }
    let index = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(format!("{FTW_DIR}/freetoken_weight.json")).unwrap(),
        )
        .unwrap(),
    );
    let reader: Arc<dyn WeightsReader> = Arc::new(MmapFtwReader::open(Path::new(FTW_DIR)).unwrap());
    Some((index, reader))
}

/// The oracle half of the gate, with no GPU and no python: the real slabs out of
/// the real checkpoint, through the scalar reference. Shape and length mistakes
/// in the harness show up here instead of costing a GPU run.
#[test]
fn the_scalar_oracle_runs_on_the_real_expert_slabs() {
    let Some((index, reader)) = open_checkpoint() else {
        return;
    };
    let source: Arc<dyn ExpertSlabSource> = Arc::new(FtwBanks::new(reader, index.clone()));
    let case = plan(&index, &*source);
    let mut want = vec![0.0f32; TOKENS * case.shapes.hidden];
    ferrum_torrens::domain::expert_kernel::grouped_moe_reference(
        &case.banks.as_host(),
        &case.request(&case.routes),
        &mut want,
    )
    .expect("scalar oracle");
    let scale = scale_of(&want);
    println!(
        "layer {LAYER}: {TOKENS} tokens x top_k {TOP_K} over {} experts, |out|max {scale:.4e}",
        case.routes
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    assert!(
        want.iter().all(|v| v.is_finite()),
        "oracle produced non-finite output"
    );
    assert!(scale > 0.0, "oracle produced all zeros");
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
    let Some((index, reader)) = open_checkpoint() else {
        panic!("gate needs the checkpoint in {FTW_DIR}");
    };
    let runtime = Arc::new(gate(
        "embedded python + torch",
        TritonRuntime::new(Path::new("python"), &sites),
    ));
    let source: Arc<dyn ExpertSlabSource> = Arc::new(FtwBanks::new(reader, index.clone()));
    let case = plan(&index, &*source);
    let moe = Arc::new(Mutex::new(TritonMoE::new(
        runtime,
        source,
        case.shapes,
        SLOTS,
    )));

    let step = case.step();
    let mut cache = ExpertCache::new(LAYER + 1, SLOTS, SLOTS);
    let mut got = vec![0.0f32; TOKENS * case.shapes.hidden];
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

    // The plan is deterministic and every routed expert was already resident in
    // the oracle case too, so these are the routes the kernel saw.
    let routes = cache.ensure_experts(LAYER, &case.routed).rewrites;
    assert_eq!(routes, case.routes, "the plan must be deterministic");

    let mut want = vec![0.0f32; TOKENS * case.shapes.hidden];
    gate(
        "scalar oracle",
        ferrum_torrens::domain::expert_kernel::grouped_moe_reference(
            &case.banks.as_host(),
            &case.request(&routes),
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
