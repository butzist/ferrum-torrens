// Gated isolation harness for the device-staged MoE path: one step over the
// real checkpoint, exercising the LRU victim selection and the index_copy
// launch. Smaller and faster than the whole-step capture gate, so a staging
// fault shows here first. Run with:
//
//   FT_MEM_CAP_GIB=80 cargo test --release --features cudarc \
//     --test staged_moe_repro -- --ignored --nocapture
#![cfg(feature = "cudarc")]

use std::path::Path;
use std::sync::{Arc, Mutex};

use ferrum_torrens::application::run_experts::CachedExperts;
use ferrum_torrens::domain::expert_kernel::{
    ExpertShapes, ExpertSlabSource, MoEDeviceRequest, MoEExecutor,
};
use ferrum_torrens::domain::weights::WeightsReader;
use ferrum_torrens::infrastructure::banks::{FtwBanks, FtwPinnedBanks};
use ferrum_torrens::infrastructure::cuda_moe::CudaMoE;
use ferrum_torrens::infrastructure::cudarc::TensorDType;
use ferrum_torrens::infrastructure::ftw::{FtwIndex, FtwReader};
use ferrum_torrens::infrastructure::torch::Device;

const FTW_DIR: &str = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw";
const HIDDEN: usize = 2048;
const INTER: usize = 512;
const TOP_K: usize = 8;
const LAYERS: usize = 40;
const EXPERTS: usize = 256;
const SLOTS: usize = 64;

#[test]
#[ignore = "needs a CUDA device and the checkpoint"]
fn staged_moe_step_does_not_fault() {
    let dev = Device::Cuda(0);
    let index = Arc::new(
        FtwIndex::parse(
            &std::fs::read_to_string(format!("{FTW_DIR}/freetoken_weight.json")).unwrap(),
        )
        .unwrap(),
    );
    let reader: Arc<dyn WeightsReader> = Arc::new(FtwReader::open(Path::new(FTW_DIR)).unwrap());
    let shapes = ExpertShapes::new(HIDDEN, INTER).unwrap();

    let source: Arc<dyn ExpertSlabSource> = Arc::new(FtwBanks::new(reader.clone(), index.clone()));
    let mut adapter = CudaMoE::new(dev, source, shapes, SLOTS).unwrap();
    let rt = adapter.runtime().clone();
    let pinned =
        FtwPinnedBanks::load(&adapter.driver(), &reader, &index, shapes, LAYERS, EXPERTS).unwrap();
    adapter
        .enable_device_staging(Box::new(pinned), LAYERS, EXPERTS)
        .unwrap();
    eprintln!("staging enabled; cap {}", adapter.staged_route_cap());
    let moe = Arc::new(Mutex::new(adapter));
    let exec = CachedExperts::with_device_staging(moe.clone(), moe, LAYERS, EXPERTS, SLOTS);

    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 2654435761usize % 1009) as f32 / 1009.0) - 0.5)
        .collect();
    let ids: Vec<i32> = (0..TOP_K).map(|i| ((i * 7 + 3) % EXPERTS) as i32).collect();
    let weights: Vec<f32> = (0..TOP_K)
        .map(|i| 1.0 / TOP_K as f32 + i as f32 * 1e-3)
        .collect();
    let xd = rt.from_slice_f32(&x, &[1, HIDDEN]).unwrap();
    let ed = rt.from_slice_i32(&ids, &[1, TOP_K]).unwrap();
    let wd = rt.from_slice_f32(&weights, &[1, TOP_K]).unwrap();
    let mut out = rt.zeros(&[1, HIDDEN], TensorDType::F32).unwrap();

    for layer in 0..8 {
        eprintln!("layer {layer}");
        exec.execute_device(
            &MoEDeviceRequest {
                layer,
                hidden: HIDDEN,
                inter: INTER,
                top_k: TOP_K,
                x: &xd,
                experts: &ed,
                weights: &wd,
            },
            &mut out,
        )
        .unwrap();
    }
    let values = out.to_host_f32().unwrap();
    let finite = values.iter().all(|v| v.is_finite());
    eprintln!("finite: {finite}; first {}", values[0]);
    assert!(finite);
}
