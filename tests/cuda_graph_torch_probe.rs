// Gated proof that a graph of torch ops can be captured on a driver-created
// side stream and replayed over freshly written inputs, matching an uncaptured
// run -- the mempool question TODO.md decision 40 step 2 must settle before the
// whole decode forward can be captured. Run with:
//   cargo test --release --features cuda --test cuda_graph_torch_probe -- --ignored --nocapture
//
// `tch` 0.22 exposes no way to (a) put the torch runtime's current stream on a
// stream this crate made through the driver API, or (b) route a capture's
// allocations into a private mempool, so doing either without a `cudaMalloc`
// between begin and end capture. Both live in `shim/c10_graph_shim.cpp`; this
// gate is what proves they work together and replay correctly. No checkpoint is
// needed: the arithmetic is a plain matmul + bias so the comparison is between
// the captured and uncaptured code paths, not against a model.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use ferrum_torrens::infrastructure::cuda_driver::{CAPTURE_MODE_GLOBAL, Driver};
use ferrum_torrens::infrastructure::cuda_graph::{self, GraphPool};
use ferrum_torrens::infrastructure::torch::to_f32_vec;
use tch::{Kind, Tensor};

const ROWS: i64 = 32;
const HIDDEN: i64 = 96;

/// Deterministic, well-scaled host data: `(i * 7 % 251 - 125) / 125` lands in
/// `[-1, 1)` so a bf16-able step stays finite.
fn data(seed: i64, n: i64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i + seed) * 7).rem_euclid(251) - 125;
            x as f32 / 125.0
        })
        .collect()
}

fn host_to(t: &mut Tensor, values: &[f32]) {
    let src = Tensor::from_slice(values)
        .reshape(t.size())
        .to_kind(t.kind())
        .to_device(t.device());
    t.copy_(&src);
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "shape mismatch");
    let scale = b.iter().fold(0.0f32, |acc, v| acc.max(v.abs())).max(1e-6);
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst / scale < 1e-5,
        "captured and uncaptured runs differ by {:.3e} of scale",
        worst / scale
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn a_captured_torch_op_replays_equal_to_an_uncaptured_run() {
    let lib = Arc::new(Driver::open().expect("libcuda"));
    lib.init().expect("cuInit");
    lib.attach_primary_context(0).expect("primary context");
    let stream = unsafe { lib.create_stream() }.expect("side stream");
    assert!(!stream.is_null(), "the side stream must not be stream 0");

    // torch launches its ops on the very stream the capture records.
    unsafe { cuda_graph::set_current_stream(stream, 0) };
    let dev = tch::Device::Cuda(0);

    // Fixed-address inputs and output: created once, only their contents change
    // between replays, so the graph's baked-in addresses stay valid.
    let mut a = Tensor::zeros([ROWS, HIDDEN], (Kind::Float, dev));
    let mut w = Tensor::zeros([HIDDEN, HIDDEN], (Kind::Float, dev));
    let mut b = Tensor::zeros([HIDDEN], (Kind::Float, dev));
    let mut out = Tensor::zeros([ROWS, HIDDEN], (Kind::Float, dev));
    host_to(&mut a, &data(0, ROWS * HIDDEN));
    host_to(&mut w, &data(1, HIDDEN * HIDDEN));
    host_to(&mut b, &data(2, HIDDEN));

    // Warmup: run the step once uncaptured so every kernel and workspace is
    // realised before the capture records them.
    out.copy_(&(a.matmul(&w) + &b));
    unsafe { lib.synchronize(stream) }.expect("warmup flush");

    // Capture: the pool holds the matmul's temporaries, so no `cudaMalloc`
    // happens between begin and end capture. The pool must outlive the graph.
    let pool = unsafe { GraphPool::begin(stream, 0) };
    unsafe { lib.begin_capture(stream, CAPTURE_MODE_GLOBAL) }.expect("begin capture");
    out.copy_(&(a.matmul(&w) + &b));
    let graph = unsafe { lib.end_capture(stream) }.expect("end capture");
    let exec = unsafe { lib.graph_instantiate(graph) }.expect("instantiate");
    unsafe { lib.graph_destroy(graph) }.expect("drop the source graph");

    // Replay over new values, then the same step uncaptured. Taking the
    // difference first keeps an accidental no-op from passing as equal.
    host_to(&mut a, &data(99, ROWS * HIDDEN));
    unsafe { lib.graph_launch(exec, stream) }.expect("replay");
    unsafe { lib.synchronize(stream) }.expect("replay flush");
    let replayed = to_f32_vec(&out);

    let twin = to_f32_vec(&(a.matmul(&w) + &b));
    close(&replayed, &twin);
    assert!(
        replayed.iter().any(|v| v.abs() > 1e-3),
        "the step must produce a nonzero result"
    );

    unsafe { lib.graph_exec_destroy(exec) }.expect("drop the instance");
    drop(pool);
    unsafe { lib.destroy_stream(stream) }.expect("destroy stream");
    println!(
        "PASS: replayed a captured torch step over {}x{} and matched the uncaptured twin",
        ROWS, HIDDEN
    );
}
