# Torch-less runtime: candle re-evaluation (2026-10-21)

Status: evaluation + spike plan. No code from this plan is on the serving path yet.
Decision: `DONE.md` 114.

## Why this is a re-evaluation

`DONE.md` 17 rejected candle-core as the compute backend because it had no
NVFP4, no gated-delta-net, and no public CUDA-graph capture API. Two of those
still hold, but they no longer decide the question:

- NVFP4 and GDN are already hand-written Triton cubins launched over the CUDA
  driver API. A tensor runtime does not need to implement them; it only needs to
  hold the packed banks, expose device pointers, and leave the quantized/GDN ops
  to those kernels.
- Graph capture is now engine-owned (`infrastructure/cuda_driver.rs` and
  `decode_graph.rs`), and `cudarc` (which candle uses) exposes
  `CudaStream::begin_capture`/`end_capture`. Candle has capture-aware guards
  already (`cuda_graph_capture_active`, the HTOD cache), so the missing public
  candle API is no longer a blocker: capture is ours either way.

So the question changes from "can candle run the model?" to "can candle replace
`tch` as the tensor plumbing under our kernels?".

## Audited facts (candle 0.11.0, 2026-06-26; cudarc 0.19.10)

- `CudaContext::new` retains the **primary** context (`cudarc` safe/core.rs:77).
  The engine's `Driver::attach_primary_context` attaches the same context, so
  candle tensors and our driver launches share one context.
- `CudaDevice::cuda_stream() -> Arc<CudaStream>` and
  `CudaStream::cu_stream()` are public. Our launchers take a raw `CUstream`, so
  a candle tensor's kernels and our cubins can be ordered on one stream.
- `Tensor::storage_and_layout()` is public, as are `Storage::Cuda`,
  `CudaStorage`, `CudaStorageSlice` and `cudarc`'s `device_ptr`. A candle tensor
  can be handed to a cubin without a copy.
- `DType` has U8/U32/I16/I32/I64/BF16/F16/F32/F64/F8E4M3/F4/F8E8M0. `F4` is MX4
  (microscaling), **not** NVFP4: the NVFP4 banks stay our own U8/I32 tensors, as
  they already are for the Rust MMA/GEMV kernels.
- BF16/F16/F32 `matmul` goes through cuBLAS (`gemm_strided_batched_*`), with
  layouts and batching, so the prefill GEMM remains cuBLAS-class.
- `embedding`, `index_select`, `gather`, `copy`, `narrow`/`slice`, `reshape`,
  `rope`, `conv1d`, `softmax` and reductions exist in `candle-core`/`candle-nn`
  for the remaining non-quantized ops.
- Candle allocates with `cuMemAllocAsync` when the device supports memory pools
  (`cudarc` safe/core.rs:1623). That is stream-ordered and capturable; candle
  also has an explicit `disable_event_tracking` for capture and a cached HTOD
  path for host uploads during capture.
- No NVFP4 and no GDN in candle-core/candle-nn/candle-transformers. Neither is
  needed: the engine's NVFP4 MoE, dense NVFP4 head, FP8 projections, GDN,
  paged attention, staging and norms are the existing cubins.

## Proposed shape

Keep the model boundary (`models/qwen3_6/`) and the domain unchanged. Replace
only the `cuda`-feature tensor adapter:

- A `ComputeRuntime`-shaped adapter over candle (device, tensor, stream, matmul,
  casts, slicing, gather) in `infrastructure/`, mirroring the current
  `torch.rs` surface. The forward talks to it, not to `tch` directly.
- Our driver launchers take the candle stream (`cu_stream()`) and raw pointers
  from candle storage, so the cubin ABI does not change.
- The decode graph is captured over the candle stream by the engine's existing
  capture/replay path. Candle event tracking is disabled for capture, and the
  fixed-address buffers stay engine-owned (do not rely on candle's allocator for
  the captured step).
- `forward.rs` remains the scalar CPU oracle; a CPU candle backend is optional,
  not a requirement.

This is a port of the tensor plumbing, not a rewrite of the algorithm. The
quantized and GDN/attention kernels, the FTW reader, the MoE cache, the
scheduler and the HTTP layer do not move.

## Spike (M7.0), before any migration commit

Add `candle-core`/`candle-nn` behind a non-default feature and prove four
things on the target GPU. If any fails, the fallback is a minimal `cudarc`
tensor layer (same shape, less code) rather than abandoning the milestone.

1. **GEMM parity and speed.** Candle BF16 `matmul` versus the current `tch`
   path for `M in {1, 512, 2057}` on the real projections (`H=2048`,
   `I=512`, `vocab=248320`, including the transposed weight layout). Gate:
   argmax-equal forward and no more than 5% slower on prefill shapes; M=1 is
   not decisive because decode uses our kernels.
2. **Graph capture.** Capture a sequence of candle matmul/add/rms_norm and one
   of our cubin launches on a cudarc side stream, replay it, and match the
   uncaptured result. No host synchronization inside the captured region, and
   the captured allocations must replay at stable addresses.
3. **Pointer/stream interop.** Run one cubin (`add_rmsnorm` or
   `nvfp4_dense_gemv`) on a candle tensor's pointer and stream, and match the
   current tch-fed result bit for bit where the kernel is deterministic.
4. **Memory/load.** Load the FTW dense weights into candle tensors, attach the
   existing expert banks, run `FT_SLOTS_REPORT`, and confirm libtorch is absent
   from the process and the VRAM budget is not worse.

## Migration slices after a green spike

- **M7.1 Runtime adapter.** `ComputeRuntime` over candle; port
  `infrastructure/torch.rs` helpers and `Device`.
- **M7.2 Weights and load.** FTW dense widening/upload through candle; keep
  `domain::dequant` as the host transform.
- **M7.3 Forward.** Port `forward_torch.rs` off `tch::Tensor`; keep the scalar
  oracle as the gate.
- **M7.4 Kernels and capture.** Point every launcher at the candle stream and
  raw storage; disable event tracking for capture; move the C++ graph shim out
  (`shim/c10_graph_shim.cpp` goes away with libtorch).
- **M7.5 Remove libtorch.** Drop `tch`, `torch-sys`, the `cuda` C++ build
  script, the shim and the libtorch closure; keep `--features cuda` as the
  runtime gate or rename it to match the new adapter.

Gates per slice: the existing unit suite, the kernel interpreter selftest, and
one at a time of the real-checkpoint GPU gates. The end state is the same gate
set on a process that links neither libtorch nor Python.

## Risks

- Candle's per-op Rust overhead and allocation pattern may lose against ATen on
  the non-quantized prefill ops. M7.0 decides this before the port.
- Capture + `cuMemAllocAsync` replay needs the spike's memory-stability check;
  the engine's fixed buffers remain the fallback.
- Candle's API is pre-1.0 and has churned; pin the exact version.
- The `forward_torch` port is mechanical but large (~2.7k lines, several hundred
  tensor calls). Do not transliterate; keep the scalar oracle and the existing
  kernel boundary.

## Non-goals

- Re-implementing NVFP4, GDN, or paged attention in candle.
- Replacing the Triton/Triton-free kernel pipeline in this milestone.
- Dropping `forward.rs` or the CPU-only unit path.
