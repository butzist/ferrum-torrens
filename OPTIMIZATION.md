# Fused GDN decode kernel -- implementation instructions

Target and context: `TODO.md` section 8 item 15. This document is the spec for
replacing our libtorch-op GDN decode with one AOT Triton kernel per layer,
matching FreeToken's `fused_sigmoid_gating_delta_rule_update`. It records the
FreeToken facts for our exact model/hardware so the kernel is written to match.

## Why

At bs=1 a decode step is ~200 ms/token on the RTX 5060 Laptop, of which **GDN is
~68 ms (35%)** -- the largest single block (attention ~49 ms, MoE ~17 ms, rest
~59 ms). Our GDN decode (`gdn_core_batch`, `src/models/qwen3_6/forward_torch.rs`)
is ~15 libtorch ops per layer, run 30 times per token, including per-step `f32_`
casts of weights and activations. FreeToken does the same layer in **two fused
kernels** (conv + recurrence) and is ~4.8x faster end to end (decision 76).

## FreeToken facts (this model, sm_120)

Model: `Qwen3_5MoeForConditionalGeneration` (`qwen3_5_moe`), dtype **bfloat16**,
40 layers with `full_attention_interval = 4` -> **30 linear (GDN) + 10 full
attention**.

Kernels FreeToken actually uses:
- **GDN decode**: `freetoken.kernel.fla.fused_sigmoid_gating_delta_rule_update`
  (vendored flash-linear-attention Triton). One kernel, per GDN layer, doing
  gating + q/k L2-norm + recurrent update + state read/write-by-index. Wrapper
  `models/qwen3_5_moe/gdn_kernels.py::gdn_decode_fla`.
- **GDN conv decode**: `freetoken.kernel.causal_conv1d.causal_conv1d_decode`
  (fused single-token depthwise conv + silu, updates conv state in place by slot).
- **GDN prefill**: `chunk_gated_delta_rule` + `causal_conv1d_varlen` (not this
  task; decode only for now).
- **Full attention**: flashinfer (`attention_backend='fi'`, auto-selected).
- **Experts**: NVFP4 W4A16 group 16, backend `auto` -> on sm_120 the candidates
  are `nvfp4_b12x` (flashinfer SM12x) or the native FreeToken Triton kernels
  (`nvfp4`); `nvfp4_marlin` is sm_80-99 and excluded. We have our own NVFP4
  grouped-MoE Triton kernels (committed cubins) and do not need theirs.

Dtypes (`gdn_kernels.py::gdn_decode_fla` docstring):
- `q`,`k`: `[1, B, num_k_heads, head_k_dim]` **bf16**, NOT GQA-expanded.
- `v`: `[1, B, num_v_heads, head_v_dim]` **bf16**.
- `a`, `b`: `[B, num_v_heads]` raw (kernel derives `g = -exp(A_log)*softplus(a +
  dt_bias)` and `beta = sigmoid(b)` in fp32).
- `A_log`, `dt_bias`: `[num_v_heads]` **fp32** (stored fp32; loader exempts them
  from the model-dtype downcast).
- state: `[num_slots, num_v_heads, head_k_dim, head_v_dim]` **fp32**, read and
  written **in place by index**.
- GDN projections: FP8 (`in_proj_qkv`, `in_proj_z`, `out_proj`); the recurrence
  still runs bf16 in / fp32 state.

## Model constants (Qwen3.6-35B-A3B-NVFP4)

| name | value |
|---|---|
| hidden_size | 2048 |
| linear_num_key_heads (NK) | 16 |
| linear_key_head_dim (DK) | 128 |
| linear_num_value_heads (NV) | 32 |
| linear_value_head_dim (DV) | 128 |
| linear_conv_kernel_dim (K) | 4 (width 3) |
| key_dim = NK*DK | 2048 |
| value_dim = NV*DV | 4096 |
| conv_dim = 2*key_dim + value_dim | 8192 |
| GQA rep = NV/NK | 2 |
| scale | `DK^-0.5` = 1/sqrt(128) |
| rms_norm_eps | 1e-6 |
| GDN layers | 30 |

DK == DV == 128 (asserted in FreeToken; Qwen3.5/3.6 satisfies it).

## Our current decode path (to replace)

`gdn_core_batch` (forward_torch.rs:1458):
`qkvz` GEMM -> split -> `ba` GEMM (**f32 upcasts of x and the ba weight**) ->
`conv1d_silu_step_batch` (`f32_` cast, `cat`, `w.repeat([n,1,1])`, grouped
libtorch `conv1d`, `silu`, `narrow`) -> split/reshape/GQA-expand
(`expand_heads`) -> `g = softplus(a + f32_(gdn.dt_bias)) * (-exp(a_log))`,
`beta = sigmoid(b)` -> `gdn_recurrence_step_batch` (12+ fp32 ops: two `pow`+`sum`
+`sqrt`+`reciprocal` for the l2norms, `exp`, `mul`, `sum_dim` for kv_mem, `sub`,
`mul`, `outer`, `sum_dim`, `mul`).

Reference math to preserve exactly (`gdn_recurrence_step_batch`), per (row n,
head hv):
```
qinv = 1/sqrt(sum_kk q^2 + 1e-6)      # l2norm, eps 1e-6
kinv = 1/sqrt(sum_kk k^2 + 1e-6)
S = S * exp(g)                        # g <= 0 (log-decay), per hv
kn = k * kinv
kv_mem[v]   = sum_kk S[kk,v] * kn[kk]
delta[v]    = (v[v] - kv_mem[v]) * beta
S[kk,v]    += kn[kk] * delta[v]       # outer product
o[v]        = (sum_kk S[kk,v] * qn[kk]) * qinv * scale
```
Our state is `[layers, num_slots, NV=32, DK=128, DV=128]` **f32**
(`LinearStateSlabs`, kvcache_dev.rs:241) -- same `[slot, NV, K, V]` layout as
FreeToken's `state_source`, so the indexed in-place read/write matches. Conv
state is `[layers, num_slots, conv_dim=8192, K-1=3]` f32.

## Kernel spec: `gdn_decode`

One kernel per GDN layer per decode step. Fuse the gating too (drop the separate
`softplus`/`sigmoid`/`exp` ops and the `a_log.exp().neg()` precompute).

Inputs (all on the step's stream):
- `q` `[B, NK, DK]` bf16, `k` `[B, NK, DK]` bf16, `v` `[B, NV, DV]` bf16
  (post-conv split, not GQA-expanded).
- `a` `[B, NV]` and `b` `[B, NV]` raw (the `ba` GEMM output; keep it in the
  model dtype, do not upcast).
- `a_log` `[NV]` f32, `dt_bias` `[NV]` f32.
- `state` `[num_slots, NV, DK, DV]` f32, mutated in place.
- `index` `[B]` i32 slot id per row (the scheduler's table row).
Constexprs: `NK`, `NV`, `DK`, `DV`, `SCALE` (or pass scale as f32), `B` fixed per
graph.

Output: `o` `[B, NV, DV]` bf16.

Per program: one (row `b`, v-head `hv`) pair. State tile `S[:, dv]` for a `BV`
column block lives in registers; loop `DK` in `BK` tiles. Compute `g` and `beta`
once per (b, hv) in fp32, load `S` from `state[index[b], hv]`, apply the 6 steps
above, store back to the same slice. q/k at `NK` heads; each `hv` maps to
`hk = hv / (NV/NK)`.

Casting: load q/k/v as bf16 and accumulate the l2norm in fp32; `kn`/`qn` in
fp32; `S`, `g`, `beta`, `o` accumulate f32, store `o` bf16. This matches the
current fp32 semantics within bf16 input precision (our gates use bf16 q/k/v as
input already).

Optionally also fuse the conv (`causal_conv1d_decode` equivalent): single-token
depthwise conv over `[B, conv_dim]` with the `[conv_dim, 1, K]` weight and the
per-slot conv state `[B, conv_dim, K-1]` -> `silu`. If fusing both, the kernel
reads `conv_in` + conv state and produces `q,k,v` internally. Start by porting
just the recurrence kernel (bigger, cleaner win); do conv as a second kernel or
a second commit.

## ABI / AOT integration

Follow the existing kernel pattern (see `python/ferrum_kernel/offload.py`,
`signatures.py`, `aot.py`, `kernel/manifest.json`, `src/infrastructure/kernel_abi.rs`,
`src/infrastructure/cuda_driver.rs`):
1. Author `python/ferrum_kernel/gdn.py` with the `@triton.jit` kernel + a launch
   helper.
2. Add `gdn_decode()` to `signatures.py` (pointer args + constexprs) and register
   it in `aot.py` (fixed grid, `sm_120`).
3. Add a launcher in `kernel_abi.rs`/`cuda_driver.rs` (3-tuple grid, pointer
   params), and a `.cubin` under `kernel/`.
4. Add the kernel name to `tests/` AOT expectations and the `MANIFEST`.
5. `kernel-selftest --interp` (CPU) against the libtorch reference
   `gdn_recurrence_step_batch` -- add the GDN decode case to
   `python/ferrum_kernel/selftest.py`.
6. `kernel-aot` to regenerate `kernel/*.cubin` + `kernel/manifest.json`;
   `scripts/check-kernels.sh` must pass (it also recompiles committed artifacts).
7. Wire `gdn_core_batch` (or a new decode-only path in `forward_torch.rs`) to
   launch the kernel instead of the op soup. Dispatch is model-specific
   (`src/models/qwen3_6/`) but the kernel/ABI is shared infra.

## Testing

- **Correctness (CPU)**: `kernel-selftest --interp` comparing `gdn_decode` to
  `gdn_recurrence_step_batch` on random bf16 q/k/v, fp32 a/b/A_log/dt_bias, and a
  random-but-valid index; tolerance from the existing GDN tests (bf16).
  Must also cover B>1 (two rows -> two slots) to prove the indexed read/write.
- **Correctness (GPU, gated)**: a `tests/gdn_decode_matches_reference.rs` that
  builds both and compares, `--ignored`, run in a devshell.
- **End to end**: re-run `tests/throughput_smoke.rs`; expect GDN's ~68 ms to drop
  materially. Record the before/after in `DONE.md`.
- Existing gates that must stay green: `engine_real_generate`
  (`a_greedy_generation_matches_the_reference`), `kvcache_cross_request_reuse`
  (the GDN snapshot restore uses the state layout), and `cuda_moe_matches_reference`.

## Risks / notes

- The FLA decode kernel autotunes per (DK, DV, BK, BV); we must pin constexprs
  (`no autotune`) because decode is captured in a CUDA graph (FreeToken notes the
  same, `fused_nvfp4.py:27`).
- `eps = 1e-6` in the l2norm and the exact `softplus` threshold (`20.0` in FLA)
  must match our reference or the greedy-parity gate will drift.
- Keep the state dtype f32 end to end; do not downcast (precision-sensitive).
- The `ba` GEMM in `gdn_core_batch` upcasts x and the ba weight to f32 every step
  -- fix that in the same change (compute `a`,`b` in the model dtype and let the
  kernel gate them), it is free throughput.
- Licensing: FLA is MIT; if we port its kernel rather than write from scratch,
  note the origin in a comment and keep it in `python/ferrum_kernel/`.

## Result (implemented -- see DONE.md decision 77)

Landed as `python/ferrum_kernel/gdn.py` `_gdn_decode_kernel`, AOT'd into a new
`linear` manifest section, launched by `infrastructure/cuda_gdn.rs` (`GdnDecode`)
via `GdnKernels`/`LinearAbi`, and wired into `gdn_core_batch_fused` on the device
decode path. Grid `(1, cdiv(V, BV), rows * NUM_V_HEADS)`, `BV = 32`,
`num_warps = 4`. It is **correct**: the gated `tests/device_decode_matches_refeed`
matches the re-fed reference (argmax exact, `max|dlogit|/scale ~1e-6`) for f32 and
bf16 pools and for `rows > 1`; the `--interp` selftest is ~1e-7;
`scripts/check-kernels.sh` passes.

**Contract notes vs this spec:**
- q/k/v/state/out are **f32** (the forward's conv and recurrence are f32), not
  bf16; q/k/v are strided row-views of one conv buffer, so the kernel takes their
  row strides as runtime args and needs no contiguous copy. `a_log`/`dt_bias` stay
  bf16 (the checkpoint dtype) and are upcast in-kernel.
- State is our `[rows, HV, K, V]` layout (`K` outer, `V` inner), not FLA's
  transposed `[V, K]` storage.

**Throughput: flat -- the recurrence was never the bottleneck.** tg stayed at
`4.7 tok/s` (~3 ms/token, ~1.4%). Isolating the whole linear mixer put it at
**~86 ms/token (40% of decode)**; skipping the conv alone changed nothing and
swapping the recurrence for the kernel saved only ~3 ms. The cost is the **FP8
`in_proj_qkvz` / `in_proj_ba` / `out_proj` projections plus the norm**, not the
conv or the recurrence (TODO.md item 17). The kernel is kept (correct, fewer
temporaries/graph nodes); the next lever is the FP8 projection path, so the
`gdn_decode` conv-fusion follow-up below is **not** on the critical path.

## Decode-gap review (2026-10-16): why we are still ~3x slower and what to try

Standing numbers: FreeToken does 22.5 tok/s (44.4 ms/token) at 512 slots and
30.3 tok/s (33.3 ms/token) at 1900 slots (decision 76). We did 4.7 tok/s
(213 ms/token); the bf16-GEMM `linear` fix (uncommitted, decision 78 candidate)
takes it to **8.8 tok/s (~114 ms/token)**, leaving a **2.6-3.4x** gap at bs=1.
That gap decomposes exactly into two factors: **bytes x efficiency**.

### Factor 1 (2.0x): every quantized weight is widened to bf16 and re-read each token

The loader widens FP8 and NVFP4 weights to bf16 (lossless, scale applied after
the matmul -- see the `linear` docstring). Lossless means the widened bytes
carry **zero extra information**, but decode reads the whole resident tower
every token, so we pay 2x (FP8) to 3.5x (NVFP4) the checkpoint bytes for
nothing. Per-token weight traffic at bs=1, straight from the FTW index:

| block (per token) | ours, bf16 resident | FreeToken, native quant |
|---|---|---|
| 30 GDN `in_proj_qkvz`+`ba`+`out_proj` | 2.02 GB | 1.01 GB (FP8) |
| 10 attn `qkv_proj`+`o_proj` | 0.55 GB | 0.27 GB (FP8) |
| 40 shared expert + router | 0.29 GB | 0.11 GB (NVFP4) |
| `lm_head` | 1.02 GB | 0.29 GB (NVFP4) |
| 8 routed experts x 40 layers | 0.54 GB | 0.54 GB (NVFP4, same) |
| **total** | **~4.4 GB** | **~2.2 GB** |

The routed-expert row is already optimal (our own NVFP4 kernel), and the
embed table is prompt-only, so the whole difference is the widened dense
set: GDN/attention projections and `lm_head`.

### Factor 2 (1.3-1.7x): effective read rate

Ours: 4.4 GB / 114 ms = **~39 GB/s**. FreeToken: 2.2 GB / 33-44 ms =
**50-67 GB/s**. Same GPU, same bs=1, peak ~250 GB/s (item 9) -- both are
latency-bound at one row, but its dequant GEMV kernels (multi-row programs,
fused scales) stream better than our cuBLAS M=1 bf16 GEMV plus per-op norm and
cast kernels. 2.0 x 1.3-1.7 = the measured 2.6-3.4x.

### What already landed, in order

1. **FIXED (uncommitted, decision 78 candidate)**: `linear` upcast every
   weight (up to the 2 GB `lm_head`) to f32 on every step. bf16 GEMM with f32
   accumulation: tg 4.7 -> 8.8 tok/s, ~1.9x. Side effect: cuBLAS bf16 GEMM is
   **shape-dependent** (batched-vs-single 3.2e-2 of scale, device-vs-refeed
   9.6e-2; argmax stable in both) so the f32-calibrated 1e-3 shape gates in
   `tests/device_decode_matches_refeed.rs` fail. Any tensor-core path has this
   property; FreeToken lives with it. Resolution below (E1 restores it).

### Suggested experiments, ranked by expected effect / cost

- **E1: FP8 weight-only dequant GEMV for the FP8 projections** (GDN
  `in_proj_qkvz`/`out_proj`, attn `qkv_proj`/`o_proj`). Keep the FP8 bytes +
  per-row f32 scale resident, dequant in-kernel, f32 accumulate. Bytes:
  4.4 -> 3.1 GB/token (-30%) and resident VRAM -1.3 GiB (item 12's audit).
  Numerics: identical to the current path (the bf16 widen is lossless; the
  kernel reads the same values, applies the same scale). A hand GEMV also
  computes each output row as one fixed-order K dot, so **row results are
  batch-shape-invariant** -- it fixes the broken 1e-3 gates that cuBLAS bf16
  GEMM breaks, and it is where FreeToken's 50-67 GB/s comes from (multi-row
  programs per launch). Decode-only dispatch (rows small); prefill keeps
  cuBLAS. Expected: the FP8 projections are the bulk of the mixer, so
  **tg 8.8 -> ~12-16 tok/s**; measure, do not trust the estimate. Cost: one
  Triton kernel + ABI entry + a quant-kind dispatch in `linear` -- all the
  AOT machinery exists (`kernel_abi.rs`, `check-kernels.sh`, `--interp`
  selftest). This is TODO item 17's "eventual lever".
- **E2: `lm_head` NVFP4 GEMV.** The head is the single largest per-token read
  (1.02 GB, a quarter of all traffic) and a quarter of the VRAM gap. The
  dequant inner loop is the same as the NVFP4 expert GEMV; a single-matrix
  variant is a small kernel. Bytes 1.02 -> 0.29 GB/token, VRAM -0.73 GiB.
  Expected: **~10-18 ms/token** off the "rest" block (item 16), tg +1-2.
  Also makes the fp8-vs-bf16 sampler input identical to FreeToken's.
- **E3: bf16 paged attention** (item 14, already planned). The default bf16
  KV takes the libtorch dense-gather + masked-softmax fallback, ~20 ms/token
  (18% of decode post-bf16). A bf16 variant of our own f32 paged GQA cubin.
  Expected ~10-15 ms/token. Independent of E1/E2.
- **E4: spend the freed VRAM on expert slots** (item 12). E1+E2 free ~2 GiB,
  which is exactly the 512 -> ~1900 slot growth FreeToken uses; it measured
  ~35% from cache size at bs=1 (22.5 -> 30.3). For us MoE is only ~9-15% of
  decode, so expect a few ms; still, it is free once E1/E2 land.
- **E5: norm/cast fusion** (only after re-profiling). ~82 small rms_norm +
  f32-cast kernels per token; inside the graph each is a few us of launch and
  a redundant re-read of x. Folding input/post norms into the adjacent E1 GEMV
  (read x once, normalize in-kernel) is worth single-digit ms. Do not do this
  before E1-E3 are measured; the current split is stale (it predates the
  bf16-GEMM fix).
- **E0 (first): re-run the ablate-and-diff profile post-bf16.** The 68/49/17/59
  ms GDN/attention/MoE/rest split was measured before the bf16 GEMM fix; it is
  stale. Re-measure so E1-E5 are ranked by data, not by this estimate.

**Not worth trying** (measured or reasoned elsewhere): fusing the conv into
`gdn_decode` (~0-3 ms, above); mmap zero-copy dense loads (decision 72
experiments; the loader always dequant-transforms anyway); speculative/multi-
token decode (FreeToken has none for this model, item 9); growing the MoE cache
before E1/E2 free the VRAM (item 12).

### Evaluation for each experiment

- Correctness: `kernel-selftest --interp` vs the widened-bf16 libtorch
  reference (E1/E2 should match to FP8-exactness, ~1e-3 relative), then the
  gated GPU test on the real checkpoint. **E1/E2 should bring the
  `device_decode_matches_refeed` gates back under ~1e-2 with exact argmax**
  (fixed-order rows); if they stay at 1e-1 the kernel is wrong, not the gate.
- End to end: `tests/throughput_smoke.rs` tg before/after, recorded here;
  VRAM via the smoke's NVML sampler (E1: -1.3 GiB, E2: -0.73 GiB).
- Oracle: `tests/ftw_cross_check.rs` argmax/top-5 stable (bf16 tolerance, 8%).
- Ceiling check: after E1+E2, per-token traffic ~2.4 GB ~= FreeToken's 2.2; if
  tg lands at 50-67 GB/s effective (33-44 ms/token) we are at parity and the
  remaining lever is serving-level (batching, item 11), not kernels.

### Results (2026-10-16, decisions 78-81)

- **The byte model was wrong; decode is latency/launch-bound.** E0's
  ablate-and-diff (`FT_ABLATE=mixer,attn,gdn,moe,head`, decision 81) puts bs=1
  decode at ~0.12 s/token as **attention ~30-35 ms (29%), lm_head ~20 ms (17%),
  MoE ~17 ms, GDN ~15 ms, rest ~30 ms**. Effective weight traffic is ~28 GB/s,
  far under HBM peak, so cutting *bytes* does not pay off proportionally.
- **E2 (dense NVFP4 lm_head) is done (decision 79) and did not speed anything
  up.** The kernel is correct (CUDA cross-check: argmax 11751, top-5 identical,
  0.21% logit shift) and cut resident VRAM 7.9 -> 7.1 GiB, but tg stayed flat:
  the dequant-in-K GEMV is arithmetic/issue-bound, so reading 3.5x fewer bytes
  is a wash. The head costs ~20 ms either way. This is the direct evidence that
  **E1's predicted win is doubtful** for the same reason. E2 is kept for the
  footprint and as the reusable dense-quant-GEMV pattern.
- **The freed VRAM cannot grow the expert cache yet.** A load at 768 slots OOMs
  on the load-time peak (transient widen buffers), so item 12's audit comes
  before E4.
- **Gate recalibration (decision 80).** The f32-calibrated 1e-3 shape gates now
  pass at an argmax-pinned 0.2 bound; a fixed-order kernel is what would restore
  the tight bound, but E1/E3 must handle the prefill `M > 1` shape to do that
  (a per-row GEMV is O(M x weight) and wrong for prefill).
- **Revised priority: E3 (bf16 paged attention) is the top lever**, then the
  ~30 ms "rest" (fuse the 82 norms), then MoE/GDN. E1 drops to last: implement it
  only if a measured `Dense::apply` profile shows the projection GEMVs dominate,
  which the ablation argues against.

## Plan (2026-10-16, decision 82): vendor FreeToken's kernels, bf16 activations

The gap is a **substrate difference**, not a missing trick: FreeToken authors the
mixer as a handful of Triton kernels, we compose libtorch ops. Auditing
`FreeToken/` against this repo shows the formats already line up, so the path is
to **vendor FreeToken's own Triton sources** into `python/ferrum_kernel/` and AOT
them through our pipeline (FLA is MIT, FreeToken Apache-2.0; attribute the
origin), keeping libtorch as the generic/CPU fallback and our own paged-attention
and NVFP4-MoE kernels.

### What already matches (nothing to change)

- **Checkpoint/weights**: we read FreeToken's own FTW output. Their W8A16 linear
  expects `weight` fp8-e4m3 + a per-output-row `weight_scale [N]` -- exactly the
  `Dense { w, row_scale }` we already keep.
- **GDN state**: their `state_source` is `[num_slots, NV, K, V]` **fp32**, the same
  as `LinearStateSlabs`. (The "FLA stores `[V, K]`" note earlier was about the
  kernel's internal tile, not the wrapper ABI.)
- **Conv state**: `[slots, dim, K-1]` in both.
- **GDN kernel shape**: our `gdn.py` is already ~FLA's
  `fused_sigmoid_gating_delta_rule_update` (non-GQA-expanded q/k, in-place indexed
  state, fused gating + l2norm).
- **NVFP4 experts**: our grouped kernels are equivalent; keep them.

### The one real divergence: activation dtype

Our mixer runs **f32** (rms_norm/conv/attention outputs); FreeToken runs **bf16**.
That is the only format change needed for their kernels to drop in with
signature-only edits, and it converges our numerics with the reference. It does
mean a tower-wide numerics change, so the gates are re-measured (the bf16 bounds
of decision 80 already anticipate it).

### Not vendorable (keep ours)

- **flashinfer attention** is a C++/CUDA extension, not Triton. Keep our paged
  kernel (E3); an FFI bind is a separate project.
- **`torch._scaled_mm`** is a libtorch op (CUTLASS/cuBLASLt); tch does not expose
  it. FreeToken uses it when available and their Triton `fp8_pertensor_linear`
  otherwise -- so the Triton kernel is our route anyway.

### Ordered slices

1. **E3: bf16 paged attention** (this change; half-wired). q/output stay f32
   (the mixer activation), K/V read bf16 from the pool. One extra signature +
   cubin, dispatch on `pool.dtype()`.
2. **E1: vendor `fp8_pertensor_linear` (W8A16).** Keep the FP8 projections
   resident, dequant e4m3->f32 in-register, f32 accumulate. Their docstring
   records this exact path (bf16-widened weights through cuBLAS) as **~54% of
   their step**, halved by the W8A16 kernel. **This corrects the E2-based doubt
   above**: our lm_head kernel was neutral because its 4-bit LUT unpack is
   arithmetic-heavy; W8A16 is a trivial per-byte convert, so its bandwidth win
   lands. Dispatch: fp8 kinds -> kernel, bf16 -> cuBLAS (also covers prefill).
3. **bf16 mixer activations + vendor `causal_conv1d_decode`.** Run norm/conv/qkv
   in bf16 like the reference; replace `conv1d_silu_step_batch` with their fused
   conv+silu (in-place slot state).
4. **Prefill chunk kernels** (`chunk_gated_delta_rule`, `causal_conv1d_varlen`):
   our prefill is still op soup and has no FLA chunk kernel at all.
