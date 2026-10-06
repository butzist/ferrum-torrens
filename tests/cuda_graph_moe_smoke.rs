// Gated proof that the grouped-MoE launches can be captured into a CUDA graph
// on a side stream and replayed over freshly staged buffers, byte for byte the
// same as an uncaptured run. Run with:
//   cargo test --release --features cuda --test cuda_graph_moe_smoke -- --ignored --nocapture
//
// This is decision 39's gate. The split it exercises is the one FreeToken's
// `engine/graph.py` makes: everything data-dependent (activations, routes,
// weights -- and in this engine, the expert staging too) is written into
// persistent buffers OUTSIDE the capture, and only the launches are recorded.
// `CudaMoE::stage_into` rewrites the same buffers between replays, so the
// graph's baked-in addresses stay valid while the data under them changes.
//
// No checkpoint is needed: a synthetic slab source fills the banks with
// deterministic, finite bytes, so the comparison is between two code paths
// (replayed vs uncaptured) over identical inputs, not against a model.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use ferrum_torrens::domain::expert_kernel::{
    ExpertShapes, ExpertSlabSource, MoEError, MoEGroup, Part, Slab,
};
use ferrum_torrens::infrastructure::cuda_driver::{CAPTURE_MODE_GLOBAL, Driver};
use ferrum_torrens::infrastructure::cuda_moe::CudaMoE;
use ferrum_torrens::infrastructure::kernel_abi::Abi;
use ferrum_torrens::infrastructure::torch::Device;

const SLOTS: usize = 8;
const TOKENS: usize = 4;

/// Same bytes for the same `(layer, expert, part)`, and only finite ones: the
/// scales are e4m3fn `1.0` and the globals f16 `1.0`, so every nibble the fp4
/// weights decode to lands in the silu without an inf or a NaN that would make
/// the equality below vacuous.
struct Synthetic {
    shapes: ExpertShapes,
}

impl ExpertSlabSource for Synthetic {
    fn slab(&self, layer: usize, expert: u32, part: Part) -> Result<Slab, MoEError> {
        let (rows, cols) = self.shapes.slab_shape(part);
        let salt = layer * 131 + expert as usize * 17 + (part as usize) * 7;
        let packed: Vec<u8> = (0..rows * cols / 2)
            .map(|i| (i.wrapping_mul(37).wrapping_add(salt) % 251) as u8)
            .collect();
        let scale = vec![0x38u8; rows * cols / 16];
        let global: Vec<u8> = (0..rows).flat_map(|_| 0x3c00u16.to_le_bytes()).collect();
        Ok(Slab {
            packed,
            scale,
            global,
        })
    }
}

/// Deterministic activations in a range bf16 keeps a full step finite over.
fn activations(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 2048;
            (x as f32 - 1024.0) / 1024.0
        })
        .collect()
}

fn request<'a>(
    shapes: ExpertShapes,
    top_k: usize,
    x: &'a [f32],
    routes: &'a [i32],
    weights: &'a [f32],
) -> MoEGroup<'a> {
    MoEGroup {
        hidden: shapes.hidden,
        inter: shapes.inter,
        x,
        top_k,
        routes,
        weights,
    }
}

/// Every failure prints on stdout before panicking, so the log names the step
/// even when cargo captured the panic text for a plain `| tee`.
fn gate<T>(what: &str, r: Result<T, impl std::fmt::Display>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => {
            println!("GATE FAILED at {what}: {e}");
            panic!("gate failed at {what}");
        }
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn a_replayed_grouped_moe_is_byte_for_byte_an_uncaptured_run() {
    let abi = gate("committed manifest", Abi::committed());
    let shapes = gate("abi shapes", ExpertShapes::new(abi.hidden, abi.inter));
    let top_k = abi.top_k;
    let source: Arc<dyn ExpertSlabSource> = Arc::new(Synthetic { shapes });

    // The side stream has to exist before the adapter: `cuStreamBeginCapture`
    // rejects the legacy stream (decision 39), and everything the graph records
    // is launched on this one.
    let lib = Arc::new(gate("libcuda", Driver::open()));
    gate("cuInit", lib.init());
    gate("primary context", lib.attach_primary_context(0));
    let stream = gate("side stream", unsafe { lib.create_stream() });
    assert!(!stream.is_null(), "the side stream must not be stream 0");

    let mut moe = gate(
        "CudaMoE on the side stream",
        CudaMoE::new_with_stream(Device::Cuda(0), source, shapes, SLOTS, stream),
    );
    for slot in 0..SLOTS {
        gate(
            &format!("stage expert {slot} into slot {slot}"),
            moe.load(0, slot as u32, slot),
        );
    }

    let routes: Vec<i32> = (0..TOKENS * top_k).map(|i| (i % SLOTS) as i32).collect();
    let flip: Vec<i32> = routes.iter().map(|s| (s + 1) % SLOTS as i32).collect();
    let weights: Vec<f32> = (0..TOKENS * top_k)
        .map(|i| 1.0 / top_k as f32 + i as f32 * 1e-3)
        .collect();
    let other_weights: Vec<f32> = (0..TOKENS * top_k)
        .map(|i| 1.0 / top_k as f32 - i as f32 * 1e-3)
        .collect();
    let x1 = activations(1, TOKENS * shapes.hidden);
    let x2 = activations(2, TOKENS * shapes.hidden);

    // The split's own consistency: `grouped_moe` is stage + launch + collect,
    // and hand-running the three over the same request must agree with it.
    let mut want1 = vec![0.0f32; TOKENS * shapes.hidden];
    gate(
        "the unsplit path",
        moe.grouped_moe(&request(shapes, top_k, &x1, &routes, &weights), &mut want1),
    );
    let mut group = gate(
        "stage the first step",
        moe.stage(&request(shapes, top_k, &x1, &routes, &weights)),
    );
    gate("launch", moe.launch(&group));
    let mut got1 = vec![0.0f32; want1.len()];
    gate("collect", moe.collect(&group, &mut got1));
    assert_eq!(got1, want1, "stage/launch/collect must equal grouped_moe");
    assert!(
        got1.iter().any(|v| *v != 0.0),
        "the synthetic banks must produce a nonzero step"
    );

    // Capture the four launches over `group`'s addresses. Nothing else: the
    // legacy stream is flushed so the capture inherits no cross-stream
    // dependency, and no torch op runs while the stream is capturing.
    gate("legacy flush", unsafe {
        lib.synchronize(std::ptr::null_mut())
    });
    gate("side flush", unsafe { lib.synchronize(stream) });
    gate("begin capture", unsafe {
        lib.begin_capture(stream, CAPTURE_MODE_GLOBAL)
    });
    gate("launch during capture", moe.launch(&group));
    let graph = gate("end capture", unsafe { lib.end_capture(stream) });
    let exec = gate("instantiate", unsafe { lib.graph_instantiate(graph) });
    gate("drop the source graph", unsafe { lib.graph_destroy(graph) });

    // The next decode step: same shapes, new data. `stage_into` rewrites the
    // buffers the graph baked in, which is exactly what a replay reads.
    gate(
        "stage the second step",
        moe.stage_into(
            &request(shapes, top_k, &x2, &flip, &other_weights),
            &mut group,
        ),
    );
    gate("replay", unsafe { lib.graph_launch(exec, stream) });
    let mut got2 = vec![0.0f32; want1.len()];
    gate("collect the replay", moe.collect(&group, &mut got2));

    // The same step, uncaptured: fresh buffers, identical inputs.
    let mut want2 = vec![0.0f32; want1.len()];
    gate(
        "the uncaptured twin",
        moe.grouped_moe(
            &request(shapes, top_k, &x2, &flip, &other_weights),
            &mut want2,
        ),
    );
    assert_eq!(
        got2, want2,
        "a replay must be byte-for-byte an uncaptured run"
    );
    assert_ne!(
        got2, got1,
        "the second step must differ from the first, or the gate proves nothing"
    );

    // A step with a different batch shape cannot ride the captured grid.
    let short = activations(3, (TOKENS - 1) * shapes.hidden);
    let short_routes = &routes[..(TOKENS - 1) * top_k];
    let short_weights = &weights[..(TOKENS - 1) * top_k];
    let err = moe
        .stage_into(
            &request(shapes, top_k, &short, short_routes, short_weights),
            &mut group,
        )
        .expect_err("a captured group pins the batch shape");
    assert!(err.to_string().contains("captured"), "{err}");

    gate("drop the instance", unsafe { lib.graph_exec_destroy(exec) });
    gate("drop the side stream", unsafe {
        lib.destroy_stream(stream)
    });
    println!("PASS: replay and uncaptured runs agree on all {TOKENS} tokens");
}
