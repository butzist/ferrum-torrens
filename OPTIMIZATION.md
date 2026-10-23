# Optimization plan of record -- closing the decode gap to FreeToken

Originally `FUSED_GDN.md` (the fused-GDN kernel spec, TODO item 15); renamed
2026-10-17 when it became the running home of the whole decode-optimization
campaign: the GDN kernel spec and result (decision 77), the decode-gap review
(decisions 78-81), the vendoring plan (decision 82) and its slices, and now the
decision-88 verification and the remediation plan to FreeToken parity. `TODO.md`
section 8 holds the open-item numbering; this file holds the measured facts and
the ordered plan.

> **Runtime note (2026-10-23, decision 132).** Every "captured" number below was
> measured on the engine-owned CUDA-graph path over the libtorch forward. That
> path was removed with libtorch: the serving runtime is now cudarc and decode
> capture is parked, so captured numbers are history, not reproducible, until a
> capture path is ported. Eager numbers and all prefill numbers remain the
> measured reference for the current runtime.

## Why

At bs=1 a decode step is ~200 ms/token on the RTX 5060 Laptop, of which **GDN is
~68 ms (35%)** -- the largest single block (attention ~49 ms, MoE ~17 ms, rest
~59 ms). The historical GDN decode (`gdn_core_batch`, then in
`forward_torch.rs`) was ~15 libtorch ops per layer, run 30 times per token,
including per-step `f32_` casts of weights and activations. FreeToken does the same layer in **two fused
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
  grouped-MoE Triton kernels (AOT cubins) and do not need theirs.

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

### Measured reference baseline (recorded 2026-10-16..18; do not re-measure)

Every number here is the read-only reference on the same box (RTX 5060 Laptop,
26 SMs, 8 GiB), same FTW, bs=1, bf16 KV, `ft serve --moe-backend offload` via
`benchmarks/bench_decode_moe.py --aime` (AIME-25 #0), greedy. Run recipe in
`DONE.md` decision 76; per-stage via `scripts/freetoken_probe/` (`FT_PROBE_OUT`).
Persisted here so a comparison never needs a fresh probe run.

| metric | value |
|---|---|
| captured decode, 512 slots | 22.5 tok/s, 44.4 ms/token, 7.05 GiB |
| captured decode, ~1900 (auto) | 30.3 tok/s, 33.0 ms/token, 6.85 GiB |
| eager decode, auto | 30.25 tok/s, 33.1 ms/token (== captured: no launch tax to hide) |
| MoE miss rate, AIME, 512 / ~1900 | 0.66 / 0.34 |
| H2D staging, 512 / auto | 28.5 / 15.1 ms, miss-proportional at ~12.5-13.1 GB/s |

Per-stage p50, ms/token (auto cache, event instrumented; ~+1.5-3 ms/stage tax,
instrumented total 46.6 vs the real 33.0): `copy` 15.7, `gdn` 9.0+0.6, `attn`
3.0, `gemm` 4.2 (+glue), `shared` 3.6, `lru` 1.6, `router` 1.6, `norms` 1.0,
`head` 0.85. At 512 slots (decision 88): staging 28.5, GEMV 3.5 (p50 3.6;
torch-profiler self-CUDA 2.95), GDN 6.6, attn 2.1, head 0.82 (min 0.80 / max 0.85,
~350 GB/s, at the read roofline), shared 1.7, router 0.8, LRU 0.6-1.5. Kernel
self-CUDA at 512: `fast_index_copy_multi` 67%, `_scaled_mm` 16%, routed GEMV 7.4%.

Routing trace (156 decode steps x 40 layers, AIME, decision 95): consecutive-token
expert-set overlap 0.33 (p50 0.38); exact-LRU replay miss 0.59 / 0.32 / 0.31 at
512 / 1900 / 2048 slots; 0% of misses were routed by the previous token. The
reference uses the same global LRU over `layer*E+expert` as ours, so the earlier
miss-rate "gap" was the prompt, not the policy (0.32 vs 0.34 on AIME).

Auto-run VRAM layout (decision 76): free-before 7.35 GiB, weights+fixed ~4.6 GiB,
slots 1900 x 1.13 MiB ~2.1 GiB, KV (8274 tokens) 0.16 GiB, graphs ~0.01 GiB,
free-after 0.44 GiB.

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
   `scripts/check-kernels.sh` must pass (`nix build .#kernels` builds the tree).
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

**Correction (decision 88, 2026-10-17).** This list originally assumed our NVFP4
routed-expert and dense-head GEMVs were fine to keep. The `FT_PROFILE=1` stage
profile says otherwise: at 512 slots our `moe_gemm` is 29.0 ms/token vs the
reference's `_decode_nvfp4_marlin_kernel` 3.5, and our `head` 12.6 vs its
`_nvfp4_gemv_kernel` 0.8. These are ordinary Triton kernels in
`FreeToken/python/freetoken/kernel/triton/nvfp4_linear.py`, so they are
vendor-able like the others; doing so is item 22 and the current top decode
lever, ahead of cache growth.

### Ordered slices

1. **E3: bf16 paged attention. DONE (decision 83).** q/output stay f32 (the mixer
   activation), K/V read bf16 from the pool. One extra signature + cubin
   (`paged_attention_bf16`), dispatch on `pool.dtype()`. tg 8.2 -> 10.3, VRAM
   7.1 -> 6.05 GiB.
2. **E1: vendor `fp8_pertensor_linear` (W8A16). DONE (decision 84).** Keep the FP8
   projections resident, dequant e4m3->f32 in-register, f32 accumulate. Their
   docstring records this exact path (bf16-widened weights through cuBLAS) as
   **~54% of their step**, halved by the W8A16 kernel. Landed: `fp8_linear.py`
   (split-K GEMV + reduce + tiled GEMM), AOT'd into `linear`, `Fp8Linear` +
   a resident-FP8 `Dense`. tg 10.3 -> 10.9, VRAM -1.22 GiB. **This corrects the
   E2-based doubt above**: our lm_head kernel was neutral because its 4-bit LUT
   unpack is arithmetic-heavy; W8A16 is a trivial per-byte convert, so its
   bandwidth win lands (smaller than FreeToken's share because our step is spread
   across attention and norms, not projection-dominated).
3. **bf16 mixer activations + vendor `causal_conv1d_decode`.** Run norm/conv/qkv
   in bf16 like the reference; replace `conv1d_silu_step_batch` with their fused
   conv+silu (in-place slot state).
4. **Prefill chunk kernels** (`chunk_gated_delta_rule`, `causal_conv1d_varlen`):
   our prefill is still op soup and has no FLA chunk kernel at all. A smaller
   libtorch-level prefill win landed first (decision 101): a 256-token whole-layer
   GEMV chunk plus a hoisted GDN recurrence, pp ~88 -> ~120 tok/s at 517 tokens
   (1.36x) and ~76 -> ~122 at 2057 (1.60x), decode unchanged.

## Next steps (ordered, 2026-10-16)

Standing measurement (RTX 5060 Laptop, bs=1, bf16 KV, `tests/throughput_smoke.rs`):

| stage | tg (tok/s) | VRAM (GiB) | commit |
|---|---|---|---|
| after bf16 GEMM `linear` (decision 78) | 8.2-8.8 | 7.9 | `8eeee85` |
| + packed NVFP4 lm_head (79) | 8.2 | 7.1 | `8eeee85` |
| + bf16 paged attention (83, slice 1) | 10.3 | 6.05 | `61021e8` |
| + vendored W8A16 FP8 projections (84, slice 2) | 10.9 | 4.83 | `fea0281` |

FreeToken is 22.5 tok/s (512 slots) / 30.3 (1900 slots); the gap is now ~2.1-2.8x.
Slices 1-2 and E4 are done. The ordered remainder, with what each step actually is:

1. **E4 (item 21): raise the staging `BLOCK_C`. DONE (2026-10-17), decision 85.**
   The `lru_ensure` ceiling is 512 -> 2048, so `enable_device_staging` no longer
   refuses a cache past 512. Swept in the smoke (bs=1, bf16 KV, same box):
   512 slots 4830 MiB / tg 10.6; 1024 5708 / 11.0; 1280 6140 / 11.3; 1536 6570 /
   11.5; 1792 7002 / 11.4; 2048 7434 / 11.4. The knee is ~1536 slots (+8.5%), not
   FreeToken's ~35%, because MoE is only ~17% of decode here. The config default
   stays 512; the ceiling is opt-in. Gated by
   `cuda_moe_matches_reference::a_cache_above_the_old_ceiling_stages_on_device`.
2. **Slice 3 (item 19): bf16 mixer activations + `causal_conv1d_decode`.**
   **Fused conv DONE (2026-10-17), decision 86:** `python/ferrum_kernel/conv.py`
   `conv_decode` (one `[BLOCK_N, TAPS]`-tile program per row/feature block, f32
   in/out, in-place `[rows, conv_dim, K-1]` state, `x` row stride runtime) replaces
   `conv1d_silu_step_batch` on the device decode path; `--interp` ~1e-7 and
   `device_decode_matches_refeed` argmax-exact. **Throughput neutral**: the GDN
   block is ~10 ms/token before and after, so the conv was never the cost (decision
   77); it is kept for the dropped `w.repeat` allocation / op soup and as the ABI
   pattern for slice 4. **bf16 mixer activations DONE (2026-10-17), decision 87:**
   rms_norm/linear/projection/conv/attention/MoE outputs are bf16, the state/
   scales/router/softmax stay f32, and the vendored cubins take bf16 pointers
   (store cast in-kernel). Correct (device gate 3/3 argmax-exact; `ftw_cross_check`
   argmax 11751, 2.4%) and **throughput neutral** (tg 11.3 at 1536 slots, flat):
   the activation stream is negligible against the weight read, matching decision
   81 -- so item 11 (batching) is the larger serving lever. Slice 3 is complete.
3. **Slice 4 (item 20): prefill chunk kernels** (`chunk_gated_delta_rule`,
   `causal_conv1d_varlen`). This is the **pp** lever, not tg. **A libtorch-level
   slice landed first (decision 101, 2026-10-18):** a 256-token whole-layer GEMV
   chunk (`PREFILL_CHUNK_TOKENS`) and a GDN recurrence that hoists its per-token
   work out of the loop; pp **88 -> 120 tok/s** at 517 tokens and **76 -> 122** at
   2057, decode flat. The FLA chunk kernels remain open.
4. **Item 11: server-side batching.** Not kernel work, but the largest *serving*
   lever -- the generation loop currently runs one `run_batch` per request, so
   decode never batches and `max_running_requests` is dead. At bs>1 the per-token
   weight read amortizes; measure it before investing further in single-stream
   kernels, and note it reuses the per-batch-size graphs already built.

Expectation setting: the E2 caution (decision 79, the NVFP4 lm_head was neutral
because its 4-bit LUT unpack is arithmetic-bound) did **not** repeat for W8A16 --
a per-byte e4m3 convert is cheap, so its bandwidth win landed (+6% and -1.22 GiB).
But our step is spread across attention and norms, so slices 3-4 win less than
their FreeToken share; batching (item 11) and cache growth (item 21) may dominate
the remaining serving gap.

---

## Decision 88 verification (2026-10-17): what held, what was wrong

Re-checked against the saved probe outputs (`/tmp/opencode/ftprobe*`,
`/tmp/ft_profile_*`), the kernel sources on both sides, fresh cache sweeps with
`FT_PROFILE=1`, and fresh roofline microbenches on this GPU (RTX 5060 Laptop,
**26 SMs**, 8 GiB).

Roofline anchors (measured here, not assumed):

| probe | rate |
|---|---|
| D2D copy (read+write) | 265 GB/s |
| pure-read reduce (torch) | 282 GB/s (read roofline ~330-350) |
| pinned H2D | 14.4 GB/s |
| theoretical HBM | ~384 GB/s |

**Corrected numbers.**

- FreeToken's decode MoE GEMV at ~1900 slots is **3.8 ms/token (p50)**, not 8.8:
  the event mean was poisoned by one 164 ms outlier step. At 512 slots it is
  3.5 (p50 3.6; torch-profiler kernel self-CUDA 2.95). The `head` number is
  rock stable: **0.82 ms/token** (min 0.80, max 0.85) at both cache sizes --
  that is ~350 GB/s effective, i.e. essentially at the read roofline.
- **Our `moe_copy` is NOT a fixed-cost launch.** Decision 88's "copy time is
  ~flat 512 -> 1536" conclusion was wrong. The sweep (same prompt, `FT_PROFILE`,
  ms/token): 256 slots 43.0, 512 32.7, 1536 29.1, 2048 28.9. It scales with the
  miss count and runs at the PCIe rate (12.6-13.9 GB/s effective vs 14.4 pinned
  H2D), exactly like FreeToken's copy (theirs: 28.5 -> 15.1 ms tracking their
  210 -> 109 misses/token, also ~12.5 GB/s). The flatness was the **miss rate**:
  ours stays 74-84% at 512-2048 slots on the smoke's prompts, vs FreeToken's
  66% at 512 / 34% at ~1900 on their AIME prompt. Partly workload (a like-for-
  like prompt override should be added when this is next measured), partly that
  a broad routing distribution defeats any cache under ~20% of the expert space.
- **The two headline ratios held up:**
  - `moe_gemm` 29.0-29.7 ms vs theirs ~3.5: **~8x**, and it is cache-independent
    (the GEMV reads the same 320 slots at any cache size).
  - `head` 12.6 ms vs theirs 0.82: **~15x**; ours streams the same ~286 MiB at
    ~23 GB/s, i.e. ~8% of the measured read rate.
- Our AIME-style-prompt run (512 slots) lands in the same place as the fox
  prompt (copy 29.4, gemm 29.7, total 96.9), so the GEMV findings are not a
  prompt artifact.

**Confirmed structural causes (source-level, both kernels).**
Our `_nvfp4_gemv_kernel` (routed) and `_nvfp4_dense_gemv_kernel` (head) share
three properties that FreeToken's marlin-style kernels (`_decode_nvfp4_marlin_kernel`,
`_nvfp4_gemv_kernel` in `kernel/triton/nvfp4_linear.py`) do not:

1. **Byte-at-a-time packed loads** (uint8, `BLOCK_KB=64` *bytes*) vs **int32
   wide loads** (8 codes per word). Their docstring: wide loads take the gate/up
   GEMV's mem-only ceiling from ~65% to ~85% of peak read BW.
2. **A cross-lane `tl.sum` reduction every K-iteration** vs a **deferred
   `[BLOCK_KW, BLOCK_N]` partial reduced once at the end**. The per-iter
   barrier serializes the whole K loop; deferral lets successive iters pipeline.
3. **A 16-entry LUT gather per nibble** vs the **fp16-bitcast dequant trick**
   (`((c & 7) << 9) | ((c & 8) << 12)`, `2^-14` folded into the block scale) --
   ~5 integer ops per code pair instead of an L1 gather per code. Their
   docstring: the LUT capped the decode GEMV at ~36% of the weight-bandwidth
   roofline, the ALU version reaches ~64% (and ~350 GB/s here).
4. Occupancy makes 1-3 worse for the routed kernel: this GPU has **26 SMs**
   and our grid is `(routes, cdiv(N, 128))` = **64 programs** (gate_up, N=1024)
   and 128 (down, N=2048) per layer -- ~2.5-5 programs/SM, each serialized by
   the per-iter barrier. Their `BLOCK_N=16` grid is 512/1024 programs.
5. Their resident head is **transposed to K-major int32 `[K//8, N]` at load**
   (`nvfp4_transpose_resident`), so the GEMV's wide loads coalesce along N.
   Ours stays row-major `[N, K//2]` uint8.

**Also confirmed:** our `moe_lru` (2.4 ms) sorts the whole `BLOCK_C` usage table
per layer (theirs 0.6-1.5); our `gdn` 9.4 vs theirs 6.6-8.2, `attn` 4.4 vs
2.1-2.8, `shared` 3.0 vs 1.7-3.8, `router` 1.7 vs 0.8-1.5 (ours is the f32
matmul decision 87 keeps) -- all secondary.

## Remediation plan: match, then beat FreeToken (ordered)

Standing reference (same GPU/FTW, bs=1, graph-captured): FreeToken
**43.7 ms/token (24.2 tok/s) at 512 slots**, **33.0 ms (30.7 tok/s) at ~1900**.
Ours: 94-98 ms (tg 10.6-10.8) at 512. Every number below is the stage profile
(ms/token, uncaptured, event-instrumented ~+4 ms over captured).

### Status (applied 2026-10-17)

R1, R2, R4 and R5 landed; R3 is **withdrawn** (its premise was wrong, see
below); R6 is partial (one win, the rest blocked). Measured on the RTX 5060
Laptop, bs=1, bf16 KV, graph-captured smoke:

| | before | after R1+R2 | after R4+R6 |
|---|---|---|---|
| tg @512 slots | 10.6 tok/s | 16.7 | **16.8** |
| tg @1536 slots | 11.5 tok/s | 18.1 | **18.3** |
| tg @auto (~1900) | -- | -- | **18.7** |
| aggregate @K=4 (R5) | -- | -- | **26.2** |
| `moe_gemm` (uncaptured) | 29.0 ms | 4.2 | 4.2 |
| `head` (uncaptured) | 12.6 ms | 1.06 | 1.03 |
| `attn` (uncaptured) | 4.4 ms | -- | **4.37** |
| total @auto (uncaptured) | -- | -- | **59.7** |

The reference's `moe_gemm` 3.5 and `head` 0.82 are now within 1.2x / 1.3x (were
8x / 15x); the auto cache matches its ~1900-slot default. The residual @auto is
copy ~29.6 (PCIe, miss-bound) + gdn 9.8 + attn 4.4 + gemm 4.2 + shared 3.2 +
lru 2.5 + router 1.8 + norms 2.0. See DONE.md decisions 89 (R1/R2) and 90
(R4/R6); the section below is the original plan text, kept as the record, with
the R3 correction inline.

### R1 -- Routed-expert decode GEMV, marlin-style (the big lever) -- DONE

Vendor FreeToken's `_decode_nvfp4_marlin_kernel` into
`python/ferrum_kernel/nvfp4_moe.py`, keeping our flat-route-list ABI
(`route_slot`/`route_token`/`route_weight`) so the Rust side changes only the
cubin entry and arg pack:

- reinterpret the packed bank as int32 `[S, N, K//8]` (contiguous; K % 8 == 0
  holds for H=2048 and I=512),
- deferred `[BLOCK_KW, BLOCK_N]` partial, one final reduction,
- `BLOCK_N = 16`, `BLOCK_KW = 16` (128 k-values per iter; the deep-K variant is
  not needed at K <= 2048),
- keep our `silu_mul` + `sum_reduce` (their act/reduce equivalents measure
  ~0.5 + 0.7 ms/token),
- AOT through the existing `offload.py` signatures/manifest,
  `scripts/check-kernels.sh`, `--interp` selftest vs the existing torch
  reference, and the gated `cuda_moe_matches_reference` (argmax-exact).

Applied with the pair-trick dequant + `_split8` activation split on top.
Expected `~3.5-5`; measured **4.2** (`-24.8 ms/token`). tg at 512 -> 16.7.

### R2 -- Dense head GEMV, same three fixes -- DONE

Rewrite `_nvfp4_dense_gemv_kernel`:

- int32 wide loads + deferred reduction (`BLOCK_N = 32`, `BLOCK_KW = 16`),
- the `_nvfp4_pair_f32` bit-trick dequant instead of the LUT (drop `_e2m1_lut`
  from this kernel), folding the `2^-14` into the block scale like theirs
  (`scale * 16384.0`),
- transpose the resident head once at load to K-major int32 `[K//8, N]`
  (`Head` storage, a `finalize`-style step) so loads coalesce along N,
- apply the f16 row global after the loop (unchanged semantics).

Gates: `--interp` selftest, `ftw_cross_check` argmax 11751, the decision-80
shape-invariance bounds (a fixed-K-order GEMV should tighten them, not loosen).

Expected: `head` 12.6 -> **~1.0-1.5** (-11 ms/token). Cumulative after R1+R2:
step ~98 -> **~62** (tg ~16 uncaptured, ~17 captured at 512).

### R3 -- Decode copy/compute overlap (the "beat" lever) -- WITHDRAWN

Both engines serialize `copy_missing` on the compute stream, so at 512-2048
slots the step pays copy + compute. The plan assumed the copy could be hidden
by prefetching layer `l+1`'s misses during layer `l`'s compute. **That premise
is wrong:** a decoder's MoE is the last op of its layer, so layer `l+1`'s router
(and therefore its expert ids) is unknown until layer `l`'s MoE finishes -- there
is nothing to prefetch. Confirmed in `forward_torch.rs::run_sequence` (`x =
self.moe(li, h2)` closes each layer) and by the reference's own profile, which is
serialized the same way (staging 28.5 + compute ~16 = 44.4 ms measured). A
speculative next-step prefetch (the previous token's routing) is a heuristic, not
the parity lever this section promised.

What remains sound is small: the routed copy for layer `l` can overlap that
**same layer's shared-expert** compute (independent of the routed slots), worth
~3.2 ms/token at 512 (~5%). The MoE port was split into prefetch/finish to try
it, then **reverted** (decision 97): neutral, so not carried. Left as an optional
R6 item rather than a milestone.

### R3b -- Decode copy/compute overlap -- tried, neutral, reverted (decision 97)

The MoE port was split into `prefetch_device` (issue `lru_ensure` + `index_copy`)
and `finish_device` (wait for the copy, then the grouped GEMV), with the adapter
running the copy on a second non-blocking stream ordered by a per-layer
`begun`/`ready` event pair and the model computing the shared expert between the
two calls. It was correct and capture-safe, but **neutral**, so it was removed to
keep the port simple (decision 97).

**Measured neutral, and why.** rt/graph smoke tg `18.5 -> 18.3` (noise), and the
uncaptured profile showed `moe_copy` `29.6 -> 0.03` while `shared` `2.3 -> 30.1`:
the shared expert was not hidden, it *waited for the copy*. The `index_copy`
kernel (`REG:72`, no shared memory) launches a `(6, BLOCK_K, splits)` grid and
its resident CTAs sit on every SM slot while stalled on PCIe, so an independent
compute kernel cannot co-schedule with two streams. The per-layer window was also
only ~0.74 ms (the copy) against ~0.06 ms of router-independent work (the shared
expert), so even a working overlap caps at ~2.3 ms/token at bs=1.

**Speculative prefetch is ruled out by the routing trace** (decision 95). Traced
156 decode steps x 40 layers on the reference's own AIME prompt (eager smoke,
`FERRUM_SMOKE_NOGRAPH=1 FERRUM_SMOKE_CACHE=auto`). Consecutive-token expert-set
overlap is only **0.33** (p50 0.38), so routing is not strongly persistent on
this workload. Replaying an exact LRU over the trace: miss 0.59 / 0.32 / 0.31 at
512 / 1900 / 2048 slots; of those misses **0.000** were routed by the previous
token (they are resident by construction), the median miss was last routed
10-20 steps ago, and 21-41% were never seen in the window. A one-token-ahead
prefetch therefore predicts the *resident* set, not the misses: it would spend
PCIe bytes (the copy is already byte-bound) for no hit. Real lookahead needs the
future tokens themselves (draft/speculative decoding), its own milestone.

A follow-up probe (R8.6) found our `auto` cache resolved to **1664 slots** (free
3918 MiB, slot 1775616 B) and our AIME `moe_copy` is **27.4 ms** against the
reference's 15.1 at ~1900 slots, despite a matching LRU miss rate (0.32 vs 0.34).
So the copy gap is not miss count -- it is either the smaller auto cache (a
larger resident footprint) or the `index_copy` grid overhead at low miss counts;
worth a `moe_copy`-vs-misses sweep before tuning.


### R4 -- MoE-first VRAM budget planner (item 12) -- DONE

`[moe] cache_size_slots` is now an integer or `"auto"` (default `Auto`). The
composition root loads the tower with the MoE port detached, then
`registry::resolve_cache_slots` reads free device memory after the tower and KV
pool are resident (`cuMemGetInfo_v2`), subtracts a 1 GiB reserve for the
captured graphs/activations/fragmentation, divides by the per-expert slot bytes
and clamps to `[256, BLOCK_C 2048]`, rounded to a multiple of 128. On the 8 GiB
target it resolves ~1900 slots, matching FreeToken's auto default, and lifts
captured tg to 18.7 tok/s. On a broad routing distribution it buys little (the
miss rate stays 74-84% even at 2048); on skewed/agentic routing it is the lever
FreeToken relies on.

### R5 -- Server-side batching (item 11) -- DONE

A single `application::dispatcher::Dispatcher` worker owns every forward:
handlers tokenize and submit jobs, the worker coalesces the queue (2 ms window,
up to `max_running_requests`) into one `run_batch`, and answers each job over its
oneshot. Requests carry their own seed (per-request RNG) and streams route
per-token callbacks by uid, so a batch can mix streaming and non-streaming
without coupling tokens. This also serializes all model work, closing a latent
race where two handlers drove the one CUDA stream / MoE slot cache / prefix tree
at once. Measured (RTX 5060, captured, K=4): serial 16.6 -> batched **26.2 tok/s
(1.58x)**, `tests/dispatcher_batching.rs`.

### R6 -- Secondary polish (optional, post-parity) -- PARTIAL

Done: the paged attention kernel runs 8 warps (`attn` 4.7 -> 4.37 ms/token; the
`[64, 256]` f32 tile spilled at 4). `BLOCK_N` 32/128 measured neutral.

Blocked, with the reason:
- `moe_lru` 2.5 ms is the full `BLOCK_C` sort. A bitonic top-k (`tl.topk`) over
  the `num_missing` coldest would be cheaper, but Triton's interpreter cannot run
  `tl.topk` and `--interp` is a required gate; a manual selection loop is worse
  than the sort. FreeToken's own `_lru_ensure` profiled at ~2.2 ms, so this is
  near parity anyway.
- `gdn` 9.8 and `attn` 4.4 vs the reference's 6.6 / 2.1 is FreeToken's W8A8
  `torch._scaled_mm` for the projections; `tch` does not expose it (decision 84),
  so our W8A16 split-K GEMV stays.
- `shared` 3.2 vs 1.7: the shared expert is NVFP4 in the checkpoint but our dense
  NVFP4 kernel is GEMV-only, so keeping it NVFP4 needs an M>1 (prefill) path.
- the ~3 ms shared-expert/routed-copy overlap (R3 residue) needs the MoE port
  split into prefetch/compute inside the captured step.
- `router` 1.8 vs 0.8 stays f32 (decision 87).

None is safe to add as polish; each is its own slice.

### R6 investigation (2026-10-17): what the sub-stage profile says

Added `FT_PROFILE` sub-labels (`gdn_qkvz`/`gdn_ba`/`gdn_conv`/`gdn_recur`/
`gdn_tail`, `attn_proj`/`attn_mem`/`attn_out`) and a `FERRUM_SMOKE_PROMPT`
override so the same prompt can be replayed. Measured at auto (~1700 slots),
ms/token, uncaptured: `gdn_qkvz` 3.46, `gdn_recur` 2.62, `gdn_tail` 1.40,
`gdn_conv` 0.53, `gdn_ba` 0.41; `attn_proj` 1.85, `attn_out` 0.64. The GDN qkvz
projection (N=12288, K=2048, fp8) runs at **~220 GB/s**, i.e. at the read
roofline; the recurrence is the hand-written kernel. So there is no large GDN
lever left.

Negative results (do not repeat):
- **FP8 GEMV tuning.** BLOCK_N 64 + 4 warps regressed the projections ~2x
  (spills); 2 warps at BLOCK_N 16 regressed ~1.8x; a split-K target of 768
  (vs 1536) was ~4% worse. The committed `BLOCK_N=16, BLOCK_K=128, 1 warp,
  1536-CTA split-K` is the optimum on this 26-SM part, which is also what
  FreeToken ships. Its apparent 57 GB/s on the attention qkv is the W8A16
  fp8->f32 conversion cost that W8A8 tensor cores avoid; `tch` cannot reach
  `_scaled_mm` (decision 84). At M>1 (batched, R5) the projections take the
  tensor-core `fp8_gemm`, so batching is the way around it.
- **LRU recency clock.** The hypothesis that the per-layer step clock makes
  early-layer experts look colder and thrashes was tested by keying recency by
  token (`usage // num_layers`): `moe_copy` was unchanged (29.7 vs 28.7 ms,
  noise). The miss rate is the workload, not a policy bug.
- **Copy is workload-bound.** On the reference's own AIME prompt at auto we
  still stage ~220 experts/token (69% miss, 12.5 GB/s) vs the reference's 109
  (34%) at ~1900 slots. Both run at the PCIe rate; the byte count, not the copy
  code, is the difference. `plan_cap` is the full slot count (the prefill double
  buffer is borrowed, not reserved), and the cache is already at auto/ceiling.


### Measurement discipline

- Re-profile with `FT_PROFILE=1 FERRUM_SMOKE_NOGRAPH=1 FERRUM_SMOKE_CACHE=<n>`
  before/after each step (`/tmp/ft_profile*.{txt,json}`), and re-run the plain
  captured smoke for the tg number.
- Add a prompt-file override to the smoke before comparing miss rates to
  FreeToken's AIME runs: the miss rate is workload-sensitive (74-84% on the
  smoke's prompts at 512-2048 slots vs FreeToken's 34-66%), and R4 pays
  differently on broad vs skewed routing.
- The event-instrumented profile carries ~+4-5 ms/token over the captured
  step; compare stage deltas, not absolute totals, across engines.

Expected trajectory (bs=1), revised after R1/R2/R4: 94-98 ms @512 -> R1+R2
**64 ms (16.7 tok/s) measured** -> R4 auto **~1900 slots, 18.7 tok/s measured**
(the reference is 24.2 @512 and 30.7 @~1900). The remaining @auto split is
~29.6 ms copy (PCIe, miss-bound on this prompt) + ~30 ms compute (GDN/attn/
shared/LRU). Closing the rest needs the R6 slices above and R5 batching, which
multiplies serving throughput beyond anything bs=1 can show.


## R7 -- Shared-expert NVFP4 and prefill/cache residency (operator request)

Recorded 2026-10-17 from the operator's review. Item 23 is in progress; item 24 is
an audit-and-redesign slice. The questions are the acceptance criteria: answer
each with the code path and, where it matters, a measurement.

### R7.1 Shared expert stays NVFP4 (TODO item 23)

The checkpoint stores `mlp.shared_expert.{gate_up,down}_proj` NVFP4; `Loader::
dense` widens both to bf16 (240 MiB) and `shared` reads them every token at
~77 GB/s. Keep the packed K-major banks and run `nvfp4_dense_gemv` for decode
(M <= 8), keeping the bf16 weight for prefill (M > 1). Expected: `shared`
3.1 -> ~1 ms, VRAM +60 MiB (both representations) or -180 MiB if the bf16 copy
is dropped once a prefill path for NVFP4 exists. Gate with the device decode
oracle and `ftw_cross_check`.

### R7.2 Prefill/cache residency (TODO item 24)

The operator's questions, to be answered from the code and then fixed:

1. **Do we retain the LRU experts in VRAM during prefill, or evict all?**
   `CudaMoE::enable_prefill_overlap` "borrows the cache's first
   `2 * num_experts` slots" for the whole-layer double buffer. Audit whether the
   prefill copy rewrites `slot_for_id`/`id_of_slot` for those slots so a decode
   step afterwards sees them as invalid, and whether decode experts placed there
   are lost. Expected answer: they are lost; prefill should touch only its
   `2 * num_experts` slots and leave the rest mapped.
2. **Between requests, is the expert cache invalidated?** Check `reset_staging`,
   any per-request `CudaMoE` construction, and `CachedExperts` lifecycle. The
   cache must survive the end of a request (only the *KV pages/state slots* of
   that request are donated/freed).
3. **Where are the expert and KV caches invalidated?** Enumerate every mutation
   of `slot_for_id`, `id_of_slot`, `usage`, `step`, the page pool and the state
   slot pool, and assert none of them clears the whole expert cache except an
   explicit reset (none should exist in serving).
4. **How much space does an input batch need?** A whole-layer prefill needs
   `2 * num_experts` slots (two buffers of one layer's 256 experts, 512 slots),
   not all 40 layers. Confirm the geometry (`groups = experts / BLOCK_K`, two
   buffers) and that nothing sizes the prefill path by `layers * experts`.
5. **Cache state as an enum.** Replace the current boolean-ish wiring with
   `enum SlotCacheState { Experts, Prefill { retained: Range<usize> } }` (or the
   Rust-native equivalent) where `Experts` is the decode LRU over every slot and
   `Prefill` is the double buffer over `2 * num_experts` slots with the decode
   LRU confined to `retained`. Keep LRU/MFU experts in `retained`: rank eviction
   inside `retained` by `(usage, slot)` as today, and never let the prefill
   buffer's slots enter the decode LRU.
6. **Evict KV/dense before experts under prefill pressure.** Evaluate trading:
   expert-cache slots (PCIe re-fetch later) vs KV pages (re-prefill) vs dense
   weights (host re-widen / reload). Prefill is compute-bound and reads whole
   layers, so the double buffer is the only extra requirement; if VRAM is tight,
   prefer shrinking the *KV page pool* (or spilling the least-recently-used
   prompt's pages) over evicting the decode expert working set, and never
   evict the resident dense tower.
7. **Retention policy.** When a prefill must take slots from the decode LRU, drop
   the LRU experts first and keep the MFU set (highest reuse) in the retained
   area; measure the post-prefill decode miss rate before/after.

Acceptance: a gated test that runs a decode, then a prefill, then another decode
on the same engine and shows the second decode's `moe_copy` miss count is close
to the first (the cache was retained), plus the prefill using only
`2 * num_experts` slots. Re-measure tg after the change.

### R7.2 audit result (2026-10-17, decision 94)

Answered, with the code path:

1. **Prefill used to evict the whole expert cache.** `CudaMoE::enter_prefill`
   called `reset_staging()` and `leave_prefill` called it again, so every
   prefill wiped `slot_for_id`, `id_of_slot`, `usage` and `step` for *all*
   slots. Fixed: `enter_prefill` now calls `clear_buffer_slots`, which frees only
   the `2 * num_experts` slots the double buffer owns (scatter `-1` for the ids
   they held, zero `usage`), and `leave_prefill` no longer resets. The region
   `[2*E, slots)` keeps its decode experts across the prefill.
2. **Between requests the expert cache is not reset.** The adapter is built once
   in `registry::build`; the only whole-cache reset was the prefill path above
   (now removed). A finished request donates its *KV pages* and GDN state slot
   (`KvCache::commit/end`), which is per-request and correct.
3. **KV/state invalidation points:** `DeviceCache::end`/`commit` free the
   request's pages/state slot; `alloc_slot` evicts unlocked radix snapshots under
   pressure. None touches the expert slot cache.
4. **Input-batch space:** a whole-layer prefill needs `2 * num_experts` slots
   (two halves of one layer, 512 at 256 experts), not `layers * experts`. The
   harness already sizes it that way (`groups = experts / BLOCK_K`, two buffers).
5. **State:** `Residency::{Decode, Prefill}` (the enum the operator asked for)
   gates the path. `enter_prefill` now selects the buffer slots as the `2*E`
   **coldest** by `usage.argsort` (free slots first, MFU set retained), frees
   exactly those, and rebuilds `dst_idx` + the `victims` slot map from them
   (prefill is uncaptured, so the formerly static copy plan is rebuilt per
   prefill). `prefill_layer` gathers `slot = victims[half*E + expert]` instead of
   `expert + half*E`. Gate: `a_prefill_retains_the_decode_expert_cache`.
6. **KV/dense eviction under prefill pressure:** not needed yet. The prefill
   only borrows the 512 buffer slots (already reserved out of the auto budget),
   and the resident dense tower is never evicted. If a future prefill needs more
   scratch, the trade order is: request KV pages (re-prefill) first, then the
   expert LRU (PCIe re-fetch), never the resident dense tower.
7. **Retention policy:** the buffer now takes the coldest `2*E` slots, so the
   MFU set is protected explicitly; the resident count is unchanged across a
   whole-layer prefill in the gate.

## R8 -- Kernel quality, dtypes, and the miss-rate question (operator request, 2026-10-17)

Findings are from reading `FreeToken/` (read-only) plus our `FT_PROFILE` and the
decision-95 `FT_ROUTE_TRACE`. Our post-R7 stage profile (auto, bs=1): copy 29.6,
gdn 9.8, attn 4.37, moe_gemm 4.2, shared 2.26, router 1.8, moe_lru 1.4, head
1.06. Reference (auto, their AIME): copy 15.1, gdn 6.6-8.2, attn 2.1-2.8,
moe_gemm 3.5-3.8, shared 1.7-3.8, router 0.8-1.5, moe_lru 0.6-1.5, head 0.82;
total 33.0 vs our ~54 ms.

### Finding 1 -- the miss-rate "gap" is a prompt artifact

The decision-88 note compared our smoke prompt (74-84% miss) to their AIME
prompt (34%): different prompts. The policy is the same on both sides: a global
LRU over the flat `layer*E+expert` id space keyed on `(usage, slot)` (theirs
`flashlib.kernels.slot_cache.lru_ensure`, its mirror
`offload_kernels._ensure_experts_hybrid_cpu` picks
`min(range(cache_size), key=(usage[s], s))`; ours packs the same key and takes
the smallest). Replaying an exact LRU over our own AIME routing trace gives
**miss 0.32 at 1900 slots** against their **0.34** on the same prompt at the
same size. So there is no structural miss-rate advantage; the copy gap was
bytes = miss count = prompt. Action: measure like-for-like (both on AIME) before
any eviction work; `FERRUM_SMOKE_PROMPT` now allows it.

### Finding 2 -- the router: adopt their fused kernel

Theirs (`kernel/triton/moe_router.py::_router_triton_kernel`, vendored from
SGLang) is one Triton launch: load the gate output, `.to(f32)`, f32 row
softmax, then an iterative top-k (`K` steps of row `tl.max`, lowest-id
tie-break), renormalize, store weights + ids. Ours is a torch op stack (f32
matmul, softmax, `torch.topk`, renormalize) at 1.8 vs their 0.8-1.5. Decision
87's f32 choice is a *matmul* decision and is orthogonal: their kernel casts the
gate output to f32 before the reduce, so we keep the f32 gate and fuse the rest.

### Finding 3 -- GDN is Triton on their side; adopt the fusion

Theirs (`kernel/fla/fused_sigmoid_gating_recurrent.py`, vendored from
sglang/flash-linear-attention) is one Triton kernel: gating (sigmoid+softplus) +
in-kernel q/k l2norm + the delta-rule recurrent update + per-request state
read/write-by-index. Ours splits the same math over `gdn_conv` + `gdn_ba` +
`gdn_recur` + `gdn_tail` (`gdn_qkvz` is the projection and is already at the
~220 GB/s read roofline). So the gap is fusion/glue, not Triton.

### Finding 4 -- attention is the one CUDA dependency

`engine._resolve_auto_attention_backend` on sm_120 resolves `auto` to
**flashinfer** (`fi`); `trtllm` needs sm_100 and `fa,fi` needs sm_90. Their
`triton` backend exists (FULL+SWA) but is not selected here. Flashinfer is CUDA,
so matching it means vendoring it (heavy) or lifting our Triton paged kernel
toward it. This is the largest compute gap (attn 4.4 vs 2.1-2.8).

### Finding 5 -- the decode copy is fixed-grid-overhead-bound, not PCIe (the big one)

`_index_copy_multi_kernel` is launched as one program per `(bank, row, seg)`,
grid `(6, BLOCK_K=128, splits=64)` = **49152 programs per layer**, and gates the
work with `row < n`. At decode only ~8 of the 128 rows are live, so **~94% of the
programs are no-ops that still pay the CTA preamble**. Measured (AIME, `FT_PROFILE`,
eager): `moe_copy` 30.3 / 28.2 / 27.6 / 27.2 ms at 512 / 1024 / 1664 / 2048 slots,
while the actual misses (`FT_MISS_TRACE`) fall from **~150 to ~50 experts/token**.
The copy is flat where the bytes fall 3x: fitting `copy = c*misses + F` gives
**F ~ 25.6 ms/token** and only ~0.03 ms/miss. Decisive probe: baking the grid to
`[6, 8, 1]` (96 CTAs/layer) drops `moe_copy` to **0.40 ms** (incomplete copy, but
the timing isolates the overhead). So ~25 ms of the ~54 ms step -- ~44% -- is
launch grid overhead, not the PCIe transfer. FreeToken's copy is a C++
(`fast_index_copy.py`) kernel with a **fixed grid of `banks * blocks_per_bank`
(8) and a grid-stride loop**, which is why it has no such cost.

This is the biggest single lever found and it reorders the plan: a grid-stride
`index_copy` (small fixed grid, loop over `n` on the device) should take the
copy to ~PCIe time (~7-10 ms at 2048), i.e. **~20 ms/token off the step -> tg
~18.5 -> low 30s**, near or past FreeToken's 30.7.

### Finding 6 -- dtype gaps

- M>1 projections: theirs use **W8A8 `torch._scaled_mm`** (cuBLASLt fp8 tensor
  cores); `tch` cannot call it (decision 84). Workaround: a Triton **W8A8
  `tl.dot`** (both operands e4m3, out f32, per-tensor scale after). Our
  `_gemm_kernel` already does `tl.dot` on an fp8->bf16 upcast; feeding fp8
  straight to `tl.dot` removes the upcast. At bs=1 both sides are W8A16, so this
  is a **batching (R5)** win, not bs=1.
- Router gate: theirs bf16 matmul + f32 reduce; ours f32 matmul (decision 87).
- KV: the checkpoint declares an fp8 `kv_cache_quant_algo`; we run bf16.
- Experts: theirs `nvfp4_marlin`/`b12x` fold the global into the alpha at
  repack; ours keeps the fp16 global as a bank. Same byte class.

### Plan (ordered, low risk first)

- **R8.0 Grid-stride `index_copy` -- DONE** (decision 98). One program per
  `(bank, segment)` (grid `(6, 64)`) with a `tl.range(0, n)` loop over the
  device plan, replacing one program per `(bank, row, seg)`. `moe_copy` 27.4 ->
  9.0 ms, uncaptured total 55.9 -> 41.8 ms, **bs=1 captured tg 18.4 -> 23.4**
  (~27%). Headroom: the copy still runs ~10 GB/s vs 14.4 pinned H2D; sweep the
  segment size / block count.
- **R8.1 Fused Triton router -- PARTIAL so far.** Keep the f32 gate matmul,
  replace softmax + `topk` + renormalize with one Triton kernel (f32 reduce,
  iterative top-k). Done: precompute the f32 gate at load, `router` 2.20 -> 1.50
  ms (decision 99). Tried and reverted: top-k the logits then `softmax` the `k`
  values (same math, fewer ops, `router` -> ~1.26) -- its float rounding flips a
  near-tie routing decision and fails the exact-token oracle. Remaining: a fused
  kernel that keeps the full-softmax arithmetic, gated on the device-decode
  argmax unchanged.
- **R8.2 Fused GDN decode** (adopt the FLA kernel). 9.8 -> ~7. Medium risk
  (state layout `[K,V]`, in-kernel l2norm, GQA); gate on the GDN oracle.
- **R8.3 Attention kernel** -- CLOSED (2026-10-18). The "largest compute gap"
  premise was a measurement artifact: the eager `FT_PROFILE` `attn_mem` 1.9 ms is
  the libtorch gather + SDPA fallback, not the paged kernel. `run_sequence_cached_into`
  (the eager/uncaptured path) builds `CacheAccess::Host`; the paged Triton kernel
  runs only in the captured `Device` path (`forward_decode_into`). On the real GPU
  the paged kernel is ~0.10 ms/token at ctx 76 and ~0.25 ms at ctx 279 across the
  10 full layers (microbench: ~3.5 us fixed + 0.085 us/KV-token, BLOCK_N=64,
  8 warps), and ablating it in the captured graph is within noise. `num_warps=4`
  measured ~30% faster in the microbench but regressed the captured step ~1.5
  tok/s (a graph interaction), so the committed 8-warp config stays. The captured
  attention cost is the FP8 QKV/O projections (~2.4-2.5 ms/token), already at the
  ~220 GB/s read roofline (R6). No kernel change kept; remaining bs=1 levers are
  `moe_lru` and the router, not attention.
- **R8.4 W8A8 Triton `tl.dot` GEMM** -- REJECTED (2026-10-19, measured). A W8A8
  variant (on-device per-tensor activation quant + e4m3 x e4m3 `tl.dot`) is a real
  kernel win -- 1.04-1.25x in a model-free microbench and **+8.1%** on the K=4
  batched aggregate (29.7 -> 32.1 tok/s) -- but the per-tensor activation scale
  perturbs the batched rows ~2-3% and flips
  `device_decode_matches_refeed`'s batched-vs-single exact argmax (HEAD is exact),
  while pp gains were noise (+0.4%) and M=1 was unchanged. Not shippable while
  that gate is exact; the patch is archived at `/tmp/opencode/r84_tracked.patch`.
- **R8.5 fp8 KV cache** (type): frees VRAM for more expert slots.
- **R8.6 Like-for-like miss measurement** (AIME both sides) to confirm Finding 1
  before any eviction change.

## R9 -- 2026-10-18 like-for-like re-measure, and the fused add + RMSNorm

Fresh head-to-head with the reference, same box/FTW, bs=1, bf16 KV, auto cache,
and -- this time -- the **same prompt** on both sides (`FERRUM_SMOKE_PROMPT` /
`bench_decode_moe.py --aime` with AIME-25 #0), greedy:

| metric | ours (before R9) | FreeToken |
|---|---|---|
| captured tg | 25.9 tok/s (38.6 ms/token) | **30.3 tok/s (33.0 ms)** |
| eager tg | 18.7 tok/s (53.5 ms) | 30.25 tok/s (33.1 ms) |
| VRAM after load | 7190 MiB | ~7.1 GiB |
| stage p50 copy / gdn / attn / gemm / shared / lru / router / norms / head | 13.7 / 10.1 / 4.3 / 3.8 / 2.25 / 2.35 / 1.34 / 1.83 / 1.06 | 15.7 / 9.0+0.6 / 3.0 / 4.2(+glue) / 3.6 / 1.6 / 1.6 / 1.0 / 0.85 |

(Their probe numbers carry a ~1.5-3 ms event tax per labelled stage -- their
instrumented total is 46.6 vs their real 33 -- so compare the shape, not the sum.)

**Where the time is lost, as of now.** The old blockers (GEMVs, copy grid
overhead, cache size) are fixed; the remaining ~5.6 ms/token gap at captured
bs=1 is *launches*, not bytes: FreeToken's eager step equals its captured step
(33 ms -- nothing left to hide), while ours gains 38.6 -> 53.5 ms when the
graph is off. The largest per-step launch count left in the forward is the
norm block: `rms_norm` is nine libtorch kernels (`f32_`, `pow`, `mean_dim`,
`sqrt`, `reciprocal`, two `mul`, `to_kind`) plus a separate `x + r` add, at 81
sites per step, against their single fused `forward_add_residual` per site.
Secondary: moe_lru 2.35 vs 1.6, router 1.34 + the R8.1 fused kernel, and the gdn
tail norm.

**Correction (2026-10-18).** The `attn 4.3` row above is the **eager libtorch
fallback**, not the paged kernel: the eager path (`CacheAccess::Host`) runs a
gather + SDPA fallback, and the Triton paged kernel only runs in the captured
`Device` path. The captured attention cost is the FP8 QKV/O projections, not the
kernel, so attention is not the lever it looked like (see R8.3, now closed). A
fresh controlled baseline on this box measures the committed HEAD at **captured
tg ~28.3 tok/s** (5 runs 27.8-28.7) rather than 27.7; treat absolute single runs
as +/-5% and use within-session interleaved A/B for any decision.

### R9.1 Fused add + RMSNorm -- DONE (decision 100)

`python/ferrum_kernel/norm.py` `_add_rmsnorm_kernel`: one program per row,
bf16 in/out, the residual add rounded to bf16 exactly like the libtorch op
chain it replaces, the variance and scale in f32, and both outputs (new
residual + normalized row) written by the same launch. Registered in the
`linear` manifest section and launched from `GdnDecode::add_rmsnorm`;
`run_tower::add_rms_norm` dispatches it at the `norm_in`-with-residual and
`norm_post` sites (79 of 81 per step; layer 0 and the final head norm stay
libtorch), falling back to the ops on a CPU device or a non-bf16 stream.
First f32 runtime scalar through the ABI (`ArgKind::F32`, `.param .f32`
handled in `aot.py` and the PTX test parser).

Measured (AIME prompt, auto cache): eager `norm_in` 0.90 -> 0.27, `norm_post`
0.93 -> 0.20 ms/token; captured **tg 25.9 -> 27.7** (38.6 -> 36.1 ms); eager tg
18.7 -> 22.9. Gates: `--interp` + `check-kernels` green; device
refeed/capture/reuse/batching/cross-check/generate all green with argmax exact.
The refeed safety-net bounds moved 0.2 -> 0.3 (f32) and 0.25 -> 0.35 (bf16):
the fused reduction order shifts last-bit greedy chaos to 2.46e-1 of scale
with the token sequence unchanged (oracles green) -- same class as decision
80's cuBLAS-shape recalibration.

Next levers after R9.1, re-ranked 2026-10-18:
- **R8.3 attention is CLOSED** (see R8.3): the 4.3 ms was the eager libtorch
  fallback, not the paged kernel; the captured kernel cost is ~0.1-0.25 ms.
- **Prefill (pp): a libtorch-level win landed** (decision 101) -- a 256-token
  whole-layer GEMV chunk plus a hoisted GDN recurrence, pp 88 -> 120 tok/s at 517
  tokens (76 -> 122 at 2057).
- **R9.3 Prefill attribution + a chunked parallel GDN recurrence** (decision
  102) -- `FT_PROFILE` now keeps a second `FT-PREFILL` table and the whole-layer
  MoE has `pf_*` sub-labels. A 451-token chunk split 9.54 ms/token as GDN 5.11
  (recurrence 4.26, 45% of all prefill), prefill MoE 3.43 (at the streaming
  floor), attention 0.77 (`attn_mem` 0.15). `gdn_recurrence_chunked` replaces
  the per-token recurrence with FLA's chunk formulation (triangular solve +
  batched matmuls) for `t >= 128`: pp **120 -> 143** at 517 and **122 -> 150** at
  2057, decode within noise. The solve's per-row launches are still ~1.8
  ms/token; the FLA `chunk_gated_delta_rule` + `causal_conv1d_varlen` kernels
  (slice 4) remain the larger pp item, with a smaller `gdn_tri_solve` kernel as
  the next step.
- **R9.3b `gdn_tri_solve` landed (decision 103):** one Triton program per
  `(value head, value tile)` walks the `n <= 64` rows in registers, replacing
  the `n` batched-matmul launches. The solve drops 1.98 -> 0.037 ms/token
  (profile `gdn_solve`, 451-token chunk); pp 142 -> 150 at 517 and 151 -> 156 at
  2057, decode unchanged. The wall-clock win is smaller than the solve saving
  because that time was already overlapped with the prefill MoE copy, so prefill
  is now MoE-bound (`pf_gemm` ~3.5 ms/token).
- **`moe_lru` and the fused router are CLOSED as captured levers (2026-10-18,
  measured).** Both were implemented, gate-green and 44% faster in the kernel
  (`moe_lru` 2.52 -> 1.41 ms p50 with a bitonic `tl.topk`; a fused full-softmax
  router 1.26 -> 0.70 ms p50, exact-token oracle intact), and both moved the
  *eager* tg (+1.5%) -- but neither moved captured tg (LRU -2%, router
  neutral-to--6%). The captured graph is not bound by these sub-ms stages: it is
  bound by the PCIe copy / memory, so only a large launch-count change (decision
  100's 700-launch norm block) moves it. Decision 96's "interpreter cannot run
  `tl.topk`" note was also wrong: the real blockers were a `tl.constexpr`
  subtraction into `tl.topk` and a `k=1` interpreter bug. The candidate patches
  are archived under `/tmp/opencode/{topk_files,router_files}/`; do not re-chase
  them without a captured-tg reason.
- Remaining bs=1 items: the gdn tail norm (a `t>=128`-style prefill win, not
  captured), R8.5 fp8 KV (frees VRAM for slots). Decode is otherwise at parity.
- A stashed R9.2 attempt (fused q/k Gemma RMSNorm + fused GDN gated norm) was
  correct and gate-green but measured a captured regression (~6% at the time;
  each fusion alone neutral) with no proven win, so it is not shipped. Revisit
  only with a per-kernel GPU timing under capture to explain the interaction.
- The copy (~13 ms, ~87% of pinned H2D peak) and the FP8 projections (~220 GB/s,
  read roofline) are at parity with the reference; not levers.

### Reference baselines (2026-10-20): prefill pp + K=4 aggregate

The two comparisons the plan of record never had, measured on the same GPU/FTW
with `scripts/freetoken_probe/baselines.py`: one `ft serve` session per run
(decision 76's config plus `--max-running-requests 4 --cuda-graph-max-bs 4
--memory-ratio 0.9 --max-seq-len-override 8216`), numbers taken through the real
HTTP surface, same prompt texts and params as our smokes, warm-ups discarded,
n=5 per metric, rounds order-balanced.

| metric | ours (current) | FreeToken (2026-10-20) | ratio |
|---|---|---|---|
| pp @517 / 476-478 tok | 143 (dec 102) -> 150 (dec 103) -> **267.4** (R10.2) -> **271.3** (R10.6a) -> **401.0** (R10.6b) tok/s | **366.2 +/- 1.3** (n=5, min 364.4, max 368.0) | **1.10x** |
| pp @2057 / 1878-1879 tok | **391.7** (R10.2) -> **462.1** (R10.6a) -> **585.0** (R10.6b) -> **664.6** (R10.7) -> **717.1** (R10.8) -> **929.4** (R10.9) -> **958.5** (R10.10) tok/s | **725.0 +/- 3.1** (n=5, min 719.9, max 727.4) | **1.32x** |
| K=4 aggregate, batched | **26.2** tok/s | **25.17 +/- 0.05** (n=5, min 25.09, max 25.21) | **1.04x** |
| K=4 serial window | 16.6 tok/s | **11.70 +/- 0.01** (n=5, 8.19-8.21 s windows) | **1.42x** |
| bs=1 captured decode | **32.8** tok/s (R11) | 30.3 tok/s (decision 76, unchanged) | **1.08x** |

The FreeToken prefill runs predate R10.2, so the `ours` column carries the
post-R10.2 number (decision 112). A 2026-10-22 control on the same smoke config
re-measured ours at 270.8 tok/s at 517 and 382.6 at 2057, within run-to-run noise
of the R10.2 pair. R10.3's paged prefill attention was rejected (decision 117),
so the R10.2-era pp values stand (R10.4 later closed `norm_in`, decision 118).
`ratio` is ours / FreeToken, so >1 means we are ahead.

**R10.6a method correction (decision 119):** the earlier ours-side 2057 numbers
(391.7, 382.6) were measured with the smoke's hardcoded `max_prefill_length =
1024`, i.e. two forwards, while production defaults to 8192 and the FreeToken
runs used one forward (`--max-seq-len-override 8216`). Ours now uses the
production chunk cap, so the 2057 row above is **462.1 tok/s** and the ratio
**0.64x**; the 517 control is unchanged (~271, one chunk either way).
**R10.6b (decision 120)** moves the prefill MoE copy to the DMA engine: pp is
now **401.0 at 517 (1.10x FreeToken, the first prefill point ahead of it)** and
**585.0 at 2057 (0.81x)**. The 517 prefill is copy-bound at the PCIe Gen4 x8
line rate; the 2057 one is compute-bound with the copy hidden. Decode is
unchanged. **R10.7 (decision 121)** then raises the production MoE chunk to
2048, making the 1991-token prefill one GEMM chunk: **664.6 at 2057 (0.92x)**,
517 flat (400.9). **R10.8 (decision 122)** widens the grouped MoE tile to
`BLOCK_N=64` (717.1 at 2057, 0.99x), and **R10.9 (decision 123)** fixes the FP8
prefill GEMM's coalescing and 4-warp register spills: **929.4 at 2057 (1.28x)**,
517 flat (400.9).

Method notes and caveats:

- **pp** = `usage.prompt_tokens / (first SSE text chunk - request send)` on raw
  `/v1/completions` with `stream_options.include_usage`, greedy, `max_tokens 4`.
  Ours is in-process first-token time (`tests/throughput_smoke.rs`): the HTTP
  frame costs *them*, not us. A fit over their two lengths (1.30 s at 478,
  2.59 s at 1878) gives ~0.87 s fixed per request plus ~1086 tok/s marginal,
  so their compute-side prefill gap is wider than the table's 2.4x / 4.6x.
- **Prompt counts**: same prompt text, but our smoke repeats the *pre-encoded*
  unit (11 ids/rep -> 517/2057) while the server re-encodes the string (interior
  `". "` boundaries merge -> 471/1871) and the driver prepends a random 8-char
  nonce (6-8 tokens), hence 476-479 / 1878-1879. Tokenizer identity checked:
  both encode the single unit to the same 11 ids.
- **Radix validity gate**: every measured prefill shows `#cached-token: 0` in
  `serve.log`. The pp prompts carry the random prefix so the hybrid radix never
  serves them; an earlier run with identical prompts hit 448-1856 cached tokens
  (TTFT then measured the cache, not prefill) and was discarded, not recorded.
- **Config**: `PYTORCH_ALLOC_CONF=expandable_segments:False` is required with
  `--cuda-graph-max-bs 4`: with torch's default expandable allocator the bs=4
  graph capture OOMs on 2-4 MiB allocations while ~1.5 GiB is still free (the
  engine's `_ensure_expandable_segments` skips when the env var is set), and
  capture [1, 2, 4] passes with it off at 0.58 GiB free. The auto MoE cache
  resolved **1195 slots** at ratio 0.9 (KV pool 8236 tokens; VRAM 7.02 GiB).
- **Sanity re-check** (the plan's >2x rule): serve flags match
  `bench_decode_moe.py`'s serve (see `serve.log`), radix hits ruled out, the
  frame direction works against them, and K=4 lands in our band (25.17 vs 26.2)
  -- so the pp gap is real and is the largest measured difference. It is
  consistent with our own prefill attribution (item 20, decision 111: `pf_gemm`
  ~3.5 ms/token at the streaming floor, i.e. per-token weight streaming, against
  their chunked prefill amortizing weights over the batch).

Commands + logs: env recipe and run steps in
`plans/2026-10-20-freetoken-prefill-k4-baselines.md` (untracked), driver
docstring in `scripts/freetoken_probe/baselines.py`; logs
`/tmp/opencode/freetoken_baseline/{run.log,serve.log,baseline_rows.json}`.

## R10 -- prefill (the measured gap, queued 2026-10-20)

Prefill is 2.4-4.6x behind FreeToken (150/156 vs 366/725 tok/s, R9 baselines);
decode is at parity. The R9.1-era attribution of a 451-token chunk (decision
111: 7.72 ms/token) had `pf_gemm` 3.53 (46%), host attention fallback ~1.2 (16%,
`CacheAccess::Host` gathers + SDPA), `norm_in` 0.87 (11%), GDN recurrence 0.19 (closed; the bundle, which decision
111 did not aggregate, is 42% -- see the step-0 result below).
R10.2 then removed the `pf_gemm` share (down to ~0.62 ms/token, ~14%, decision
112). R10.3 measured the host attention fallback and found the decode paged
kernel slower, so the remaining lever is `norm_in`.
Slices, in order (TODO.md section 8 has the checkboxes):

- **R10.1 chunk 512 -- LANDED (2026-10-20).** The GEMV-level sweep in
  `cuda_moe.rs`'s const comment already flattened past 256 (16 -> 512: MoE GEMV
  2.32 -> 1.50 s at 517 tokens), but the end-to-end pp A/B (`FERRUM_PF_CHUNK`
  knob, n=3 per arm, order-balanced, warm-up discarded, 12 runs) still gained
  **+2.8% at 517 (149.0 -> 153.2 tok/s) and +3.2% at 2057 (156.9 -> 161.9)**,
  run-to-run sd ~0.5, at identical VRAM (7190 MiB). `PREFILL_CHUNK_TOKENS` is
  now 512; the knob stays as a diagnostic. Logs `/tmp/opencode/pp_sweep/`.
- **R10.2 batched grouped MoE GEMM -- LANDED (2026-10-21, decision 112).** The
  46% lever. Prefill launched one M=1 `gemv_gate_up`/`gemv_down` per route
  (`group_with_slots`, scratch `[routes, ...]`), so each expert's weight was
  re-fetched from L2 per route. Now `_nvfp4_grouped_gemm` (Triton, AOT cubin,
  sm_120) runs one program per (slot segment, N tile) over a CSR that buckets
  the chunk's routes by slot, so an expert's weight is read once per chunk; the
  activations dequant to bf16 and go through `tl.dot`, decode keeps the M=1
  GEMV. `launch_grouped` on `CudaMoE` is the entry point; `FERRUM_PF_GEMV=1`
  forces the old GEMV for the A/B and as a fallback. Order-balanced A/B (n=3 per
  arm, same binary, warm-up discarded) raises pp **189.7 -> 267.4 tok/s at 517
  (+41.0%)** and **217.5 -> 391.7 at 2057 (+80.1%)**; `FT_PROFILE` at a 451-token
  chunk drops `pf_gemm` **1005 -> 278 ms (2.228 -> 0.617 ms/token, -72%)**, from
  46% of prefill to ~14%. The grouped path is bf16 tensor-core `tl.dot`, so it
  matches the host plan to a bf16 budget rather than bit-exactly (the GEMV arm
  stays exact); `a_whole_layer_prefill_overlap_matches_the_host_plan` now gates
  both arms. Logs `/tmp/opencode/r10_2/`.
- **R10.3 device prefill attention -- CLOSED as a measured regression (2026-10-22,
  decision 117).** The first cut ran the decode paged-attention kernel with one
  row per query token and a per-row prefix window built once per chunk
  (`FERRUM_PF_ATTN=0` restored the Host fallback). It is correct: the kernel
  gates in `tests/paged_attention_kernel.rs` (synthetic prefix window + a real
  `MhaPool` with scattered page ids) match causal SDPA, and the integrated gate
  (`tests/prefill_attention_matches_host.rs`, chunks 9/12/16, both arms bf16)
  lands the same argmax on every chunk (`max|dlogit|/scale` 8.70e-2 / 2.45e-2 /
  7.84e-2 under the decision-80 0.2 bound; a per-layer probe matched the host
  recomputation to <=8.8e-4). It is **slower**: order-balanced A/B (n=3/arm,
  same binary, 1664 slots, 7190 MiB, bf16) gives pp paged vs host **260.6 vs
  270.8 tok/s at 517 (-3.8%)** and **308.3 vs 382.6 at 2057 (-19.4%)**. The
  decode kernel launches per `(row, q_head)`, re-reads each KV head for all 8 q
  heads, and its per-chunk `[rows, window]` window is quadratic; libtorch's
  GQA-aware SDPA is the better prefill kernel. The implementation, knob and
  integrated gate are parked in `git stash`, not shipped. A revisit needs a
  batched prefill kernel that shares K/V tiles, not this decode kernel. Logs
  `/tmp/opencode/r10_3/`.
- **R10.4 fused prefill `norm_in`** -- 0.87 ms/token; reuse the decode-side
  `add_rmsnorm` pattern (decision 100) for the chunk rows. Now the only open
  prefill slice.

With R10.1 and R10.2 landed and R10.3 closed, the remaining arithmetic floor is
`norm_in`; the pp already reaches ~383 tok/s at 2057, mid-way to FreeToken's
725. Their marginal (R9 baselines, after ~0.87 s of HTTP per request) is
~1086 tok/s, so further levers (larger chunks under the batched GEMM,
scratch-budget tuning) may follow.

### R10.5 pp parity ladder (2026-10-22, ordered retries)

Arithmetic first. FreeToken's end-to-end pp carries ~0.87 s of fixed HTTP cost
per request; subtracting it, their compute-side prefill is ~0.91-0.92 ms/token
at both lengths (478: (1.306 - 0.87) / 478; 1878: (2.590 - 0.87) / 1878). Ours
is in-process first-token time: **3.69 ms/token at 517** (270.8 tok/s) and
**2.61 at 2057** (382.6), so the compute gap is ~4.0x / ~2.8x and shrinks with
prompt length: our cost amortizes over the chunk, theirs does not. Ordered
retries, biggest recorded lever first:

0. **Re-profile at HEAD** (`FT_PROFILE` prefill split at 517/1024/2057). The
   decision-111 split (pf_gemm 3.53, norm_in 0.87, attn 0.77 + attn_out 0.43 +
   attn_mem 0.16, attn_proj 0.18, gdn 0.19) predates R10.2 and the shares have
   shifted. Do not size anything below without this.
1. **R10.4 fused `norm_in`** (0.87 ms/token recorded). The `residual=None`
   first layer still runs the op chain; port the decode-side `add_rmsnorm`
   (decision 100) to chunk rows. Bounded; gates: norm oracle + pp smoke.
2. **Host attention slimming** (the recorded ~1.36 ms/token across
   `attn`/`attn_out`/`attn_mem`), in rising effort:
   - (a) Cut the fallback's per-layer overheads, not the kernel: the pool
     gather is f32 today (doubling the bytes and forcing the f32 SDPA); gather
     bf16 and let SDPA run bf16 natively; the causal mask depends only on
     `q_offset`/window and is rebuilt per full-attention layer -- build it once
     per chunk; and the GQA `expand + reshape` materializes an `[n, qh, hd]`
     copy (expand in bf16, or switch to `enable_gqa` once the cross-check pins
     the repeat-interleave pairing).
   - (b) If (a) leaves attention expensive: the proper R10.3b, a **batched
     prefill attention kernel** (BLOCK_M queries per program sharing K/V
     tiles, GQA-aware, causal) -- the shape the rejected decode-kernel attempt
     lacked (decision 117). The kernel-level and integrated gates from the
     R10.3 stash can be reused.
3. **Chunk-size sweep under the grouped GEMM -- LANDED as R10.7 (decision
   121).** 512 -> 1024 -> one 1991-token chunk raised pp 583.2 -> 633.6 ->
   664.6 tok/s; production `PREFILL_CHUNK_TOKENS` is now 2048. The remaining
   `pf_gemm` inefficiency is tile/grid, not chunk size (R10.7 below).
4. **MoE streaming floor** (pf_gemm 0.617 ms/token at a 451-token chunk):
   - (a) audit the whole-layer prefill's hit accounting -- do the retained
     LRU slots actually skip H2D copy rows for experts already resident?
   - (b) free VRAM into slots by sizing the KV pool to the actual context
     (R8.5: numerically free; the smoke's 4096 tokens is 2x its need), ~700
     more slots -> fewer streamed bytes;
   - (c) prefetch depth 2 (copy layer l+2 under layer l's compute) if the copy
     is still the bound;
   - (d) D2D reuse of resident experts (the `moe_prefill_hit_d2d` analog).
     The batched-copy API landed for the whole-layer copy (R10.6b, decision
     120); a hit/miss split would now only save the resident rows' PCIe bytes
     and needs `cache_size > 2 * num_experts`.
5. **Small-op fusion** (attn_proj 0.18, the embed gather, the q/k Gemma norms)
   only if the re-profile still shows them.
6. Speculative, largest change, last: a **fixed-shape captured prefill step**
   (stage the chunk into fixed buffers the way decode does and capture a
   512-token prefill graph). Only if 1-2 do not close the gap; the R9 finding
   (the residual decode gap was launch count, not bytes) argues it would help
   here too, but prefill addresses vary per request.

Do not re-chase: the decode-kernel paged prefill (decision 117), the GDN chunk
kernels (decision 111), R9.2's fused norms (stash, -6.4% captured), fp8 KV
(R8.5). Working theory carried over from R9: the gap is launch count / op soup
in eager mode, not bytes -- FreeToken's prefill is flashinfer attention over a
handful of bf16 torch ops -- so a slice that removes launches (1, 2a, 5) is
worth more per recorded ms than it looks.

**Step 0 result (2026-10-22, HEAD d2467f5, eager FT_PROFILE, logs
`/tmp/opencode/r10_5/`).** Final measured-chunk split, ms/token (451 / 968 /
967 tok; the last is the 1024-prefix tail of the 2057 prompt):

| stage | 451 | 968 | 967 |
|---|---:|---:|---:|
| total | 4.231 | 2.726 | 2.777 |
| GDN bundle | 1.812 | 1.149 | 1.163 |
| -- gdn_qkvz | 0.849 | 0.523 | 0.521 |
| -- gdn_ba | 0.562 | 0.307 | 0.313 |
| -- conv/recur/tail | 0.401 | 0.319 | 0.329 |
| norm_in | 0.872 | 0.438 | 0.438 |
| attention bundle | 0.663 | 0.417 | 0.513 |
| -- attn_mem | 0.294 | 0.187 | 0.342 |
| pf_gemm | 0.636 | 0.597 | 0.591 |

Two corrections to the plan, both measured:

1. **`norm_in` is a per-forward cost, not per-token math.** It is 393-424 ms
   absolute per chunk (~8.2-8.8 ms per layer) at all three lengths, so it halves
   per token when the chunk doubles, while decode measures 5.8 us/call on the
   fused kernel. The fused `add_rms_norm` is already wired for residual layers
   at HEAD, so R10.4 is not "port the kernel"; it is finding the remaining
   per-forward overhead (likely the `residual=None` first layer or allocations).
   The step-3 lever is therefore mostly `max_prefill_length`: making the 2057
   prompt one forward instead of two removes a whole ~0.42 s norm_in tax.
2. **The GDN *projections* are the largest single bucket (42%), and decision
   111 never priced them** (it recorded only the recurrence, 0.19). `gdn_qkvz`
   + `gdn_ba` are 1.41 ms/token at 451 and 0.83 at 968+, so this is a new
   candidate slice (R10.6) ahead of attention. The `gdn_ba` number is
   suspiciously large for a K=2048, N=64 GEMM and needs a dispatch/launch audit
   before it is trusted.

`pf_gemm` is flat from 968 to 1024 tokens (0.597 vs 0.591), so the R10.2
amortization has already flattened; `FERRUM_PF_CHUNK` 512 -> 1024 has <2%
headroom, unlike raising `max_prefill_length`. `attn_mem` is the only stage that
grew since decision 111 (0.16 -> 0.29/0.34), consistent with the bf16-gather
first cut. Captured controls at HEAD: 271.4 tok/s at 517 and 379.9 at 2057
(within single-run noise of 270.8 / 382.6).

**Correction (R10.4, decision 118): the non-MoE path is already at parity, and
the stage shares above are overlap-contaminated.** `FT_ABLATE=moe` takes the
451-token chunk from 4.231 to **0.832 ms/token** (375 ms) and `norm_in` from
393 to 0.527 ms, while a forced op-chain norm arm moves pp <1%. The profiler's
spans overlap the whole-layer prefill copy, so `norm_in` (and the step-0 "GDN
42%") are wait/contention, not math. The serial prefill gap is the MoE expert
stream: one whole-layer pass is ~17.3 GiB per chunk at ~11 GB/s, and a 2057
prompt pays it twice because `max_prefill_length` is 1024. FT's pp arithmetic is
consistent with ~20 GB/s and one forward. R10.4 is closed; the next slices are
(a) a larger prefill chunk and (b) a faster hit/miss copy.

**R10.6a production prefill chunk -- LANDED (2026-10-22, decision 119).** The
R10.4 correction above exposed a harness bug: the throughput smoke
hardcoded `max_prefill_length = 1024`, so every recorded 2057-token pp ran as two
forwards (1024 + 967) and paid the whole-layer expert copy twice. Production
already defaults to 8192 (`DEFAULT_MAX_PREFILL_LENGTH`), and the FreeToken
reference used one forward, so the old 2057 comparison was not like-for-like. The
smoke now defaults its chunk cap to the production value; `FERRUM_SMOKE_PREFILL_CHUNK`
drops it for an A/B, and `FERRUM_SMOKE_SEQ` pins `max_seq_len` when the chunk
exceeds the smoke's 4096 context (the config requires
`max_prefill_length <= max_seq_len`; the default context follows the chunk, a
128 KiB captured decode window, and production is untouched). Order-balanced A/B
(n=3/arm, same release binary, warm-up discarded, `FERRUM_SMOKE_TOKENS=2`,
`FERRUM_SMOKE_CACHE=1664`, `--test-threads=1`):

| prompt | cap 1024 (tok/s) | cap 8192 (tok/s) | delta |
|---|---:|---:|---:|
| 517 (control) | 272.1 +/- 1.2 | 271.3 +/- 0.3 | **-0.3%** |
| 2057 | 383.3 +/- 1.7 | 462.1 +/- 0.9 | **+20.6%** |

Raw 2057: 1024 = 382.3 / 382.4 / 385.2, 8192 = 462.1 / 461.2 / 462.9; 517:
1024 = 272.9 / 272.7 / 270.8, 8192 = 271.1 / 271.1 / 271.6. Within-arm spread
is <0.8%, so one round stands. VRAM after load is 7190 / 8151 MiB in both arms
(961 MiB margin), so 8192 is safe at 2057; the expert-step count drops 360 ->
320 (two forwards -> one). Correctness: the greedy ids are `[561, 3841]` in all
12 measured runs and the warm-up, so the split does not change the computed
tokens. Implication: the production default was never the problem; the
previously recorded 2057 pp (382.6) was the two-forward number and should be
read as ~462 under the production chunk. The remaining gap to FT's ~725 is now
~1.57x, not ~1.9x. Logs `/tmp/opencode/r10_6a/`.

**R10.6b prefill copy on the DMA engine -- LANDED (2026-10-22, decision 120).**
The whole-layer prefetch was the committed `index_copy` grid, whose CTAs sat on
every SM while stalled on PCIe (the R3b/decision-97 effect), so the "overlap"
serialized with compute from the SM side. It is now one `cuMemcpyBatchAsync` of
six runs (one per bank) on the existing non-blocking copy stream, through the
copy engine. The buffer slots are the contiguous `2E`-slot window with the
lowest total usage, so each bank's whole layer is one contiguous run in both
directions and every batch entry is 512 KiB-256 MiB, above the driver's
small-entry synchronous floor. `cuMemcpyBatchAsync_v2` is optional: a driver
without it selects the `index_copy` path once with one log, and
`FERRUM_PF_DMA=0` forces the same path (otherwise kept for `copy_missing`).
Order-balanced A/B, n=3/arm, same binary,
warm-up discarded, `FERRUM_SMOKE_TOKENS=2`, `FERRUM_SMOKE_CACHE=1664`: pp
**271.8 -> 401.0 tok/s at 517 (+47.5%; raw 272.1/271.7/271.6 ->
400.9/401.0/401.0)** and **464.6 -> 585.0 at 2057 (+25.9%; 466.3/463.8/463.6 ->
585.4/583.4/586.1)**. VRAM is 7190/8151 MiB in both arms and the greedy ids are
`[561, 3841]` in all 12 runs. The copy is 40 x 6 x 256 expert rows = 18.18 GB
per chunk; at 517 the prefill is now copy-bound at 1.29 s, i.e. **14.1 GB/s**
(the kernel path was ~11.9 GB/s / 1.53 s). NVML reports max PCIe Gen4 x8
(~15.75 GB/s theoretical), so 14.1 is ~90% of line rate and the rate lever is
closed. `FT_PROFILE` at 451 tokens: total 1904 -> 1292 ms and `pf_wait` 34 ->
566 ms, i.e. the compute stream now genuinely waits for the copy instead of
being starved by its SM occupancy; at 1991 tokens `pf_wait` is 32 / 3542 ms, so
the copy is hidden and prefill is compute-bound again. Decode is unchanged (a
4-run control at 517 with 64 decode tokens: tg 30.05 kernel vs 30.30 DMA tok/s).
Logs `/tmp/opencode/r10_6b/`.

**R10.7 prefill chunk under the grouped GEMM -- LANDED (2026-10-22, decision
121).** The DMA copy is hidden at 2057 (`pf_wait` 0.016 ms/token), so prefill is
compute-bound and the 512-token GEMM chunk re-reads each routed expert's weight
once per chunk: four chunks per 1991-token forward, and each 512-token chunk's
4096 routes cover essentially all 256 experts. Order-balanced sweep (n=3/arm,
same release binary, warm-up discarded, `FERRUM_SMOKE_PREFILL=2057`,
`FERRUM_SMOKE_PREFILL_CHUNK=8192`, `FERRUM_SMOKE_TOKENS=2`,
`FERRUM_SMOKE_CACHE=1664`, `--test-threads=1`; C = `FERRUM_PF_CHUNK` 512, M =
1024, F = 8192 = one chunk):

| prompt / arm | raw runs (tok/s) | mean | delta |
|---|---|---:|---:|
| 2057 C=512 | 588.7 / 580.3 / 580.5 | 583.2 | - |
| 2057 M=1024 | 636.3 / 633.4 / 631.2 | 633.6 | **+8.6%** |
| 2057 F=one chunk | 667.6 / 663.5 / 662.6 | 664.6 | **+4.9%** (vs M), **+14.0%** (vs C) |
| 517 C=512 (control) | 400.8 / 401.1 / 400.8 | 400.9 | - |
| 517 F=one chunk | 401.0 / 401.1 / 400.5 | 400.9 | -0.01% |

The 517 control is one chunk in both arms, hence flat; within-arm spread is
<1.5%. VRAM after load is 7190/8151 MiB in every run and the greedy ids are
`[561, 3841]` in all 15 measured runs. A production-default check (2048, no
`FERRUM_PF_CHUNK`) measures 668.9 tok/s with the same ids, i.e. the F arm.
`FT_PROFILE` at 2057 (`FERRUM_SMOKE_NOGRAPH=1`, 40 decode tokens, skip 8) on the
winning arm: the 1991-token step goes **3542.0 -> 3096.7 ms** and `pf_gemm`
**1464.9 -> 1037.8 ms (0.736 -> 0.521 ms/token, -29.2%)**; every other stage is
flat (GDN bundle 1436.7 -> 1420.0, attention 533.2 -> 531.8, `pf_wait` 32.1 ->
32.1), so the win is entirely the MoE expert stream. Production
`PREFILL_CHUNK_TOKENS` is now 2048: it captures the one-chunk plateau for the
measured prompt (~130 MiB transient scratch) and bounds a longer
`max_prefill_length` to 2048-token chunks. `FERRUM_PF_CHUNK` remains the
diagnostic override.

`pf_gemm` is still **0.521 ms/token**, above the 0.45 target, so this closes the
chunk lever, not the kernel. Measured launch facts for the next slice
(`_grouped_gemm` config): `BLOCK_M=32`, `BLOCK_N=32`, `BLOCK_KW=16` (128
k-values/iter), 4 warps; grid is `(slots, cdiv(N, BLOCK_N))`, i.e. 1664 x 32 =
53,248 CTAs for gate_up (N=1024) and 1664 x 64 = 106,496 for down (N=2048) per
layer. At 1991 tokens there are 15,928 routes over 256 experts = **62 routes per
expert**, so the segment loop runs 2 M-tiles and ~90% of the grid (slots with no
routes) retires after loading `seg`. Proposal, in rising effort: (a) launch over
the nonempty segment list (a compacted `seg_ids` indirection) instead of all
slots, ~6x fewer CTAs; (b) raise `BLOCK_M` to 64 and `BLOCK_N` to 64 with 8
warps, so one tile covers a whole expert's 62 routes and the N tiles halve; (c)
sweep `BLOCK_KW` 16/32. Gate with the existing
`a_whole_layer_prefill_overlap_matches_the_host_plan` (bf16 budget), the kernel
compile check and the 2057 A/B. Logs `/tmp/opencode/r10_7/`.

**R10.8 grouped GEMM tile tune -- LANDED (2026-10-22, decision 122).** R10.7
left `pf_gemm` at 0.521 ms/token with the R10.2 tile (`BLOCK_M=32`,
`BLOCK_N=32`, `BLOCK_KW=16`, 4 warps); the grid was `(1664 slots, N/32)` and
~90% of it retired on an empty segment. This slice changed only the
centralized constants (plus `num_warps` in the AOT entries, so the compiled
cubin follows `GEMM_WARPS` instead of silently taking Triton's default 4).
Ordered sweep at 2057 (`FERRUM_SMOKE_PREFILL=2057`, chunk 8192, cache 1664,
`FERRUM_SMOKE_TOKENS=2`, one model load at a time):

| config (`M/N/KW/warps`) | raw runs (tok/s) | mean | vs best |
|---|---|---:|---:|
| baseline `32/32/16/4` | 663.9 / 662.2 / 667.5 | 664.5 | - |
| `32/64/16/8` | 664.9 / 662.2 | 663.6 | -0.1% |
| `32/64/16/4` | 716.8 / 715.3 / 718.7 / 717.6 | **717.1** | **+7.9%** |
| `32/128/16/8` | 707.1 / 705.7 | 706.4 | -1.5% (vs best) |
| `64/64/16/8` | 699.2 / 699.7 | 699.5 | -2.5% |
| `64/64/16/4` | 727.1 / 723.3 | 725.2 | +1.1% (under the 2% bar) |
| `64/64/32/4` | 681.5 / 680.6 | 681.1 | -5.0% |

The baseline arm was rebuilt from the pre-change store cubins and interleaved
with the winner (W B B W W B W), so the comparison is same-session. The 517
control is flat: 402.1 / 402.1 / 401.9 = **402.0** (R10.6b recorded 400.9).
Within-arm spread is <0.6%, and the winner repeated on the final tree (714.9 at
2057, 402.0 at 517). Greedy ids are `[561, 3841]` in every run and VRAM stays
7190/8151 MiB.

Doubling N halves the grid (26,624 gate_up + 53,248 down CTAs per layer) and
widens the `tl.dot` tile, so the per-K-iteration dequant is amortized over twice
the output; 8 warps loses because each warp's MMA share shrinks and the CTA
becomes launch/scheduling heavy at this M (62 routes per expert, two M-tiles).
`BLOCK_M=64` removes the second M-tile but only gains 1.1%, under the repo's 2%
bar and not interleaved, so it is not shipped.

`FT_PROFILE` at 2057 (`FERRUM_SMOKE_NOGRAPH=1`, 40 decode tokens, skip 8) on the
winner: the 1991-token step is **3096.7 -> 2878.4 ms (1.555 -> 1.446
ms/token)** and `pf_gemm` **1037.8 -> 807.2 ms (0.521 -> 0.405 ms/token,
-22.2%)**; every other stage is flat (GDN 1420.0 -> 1428.2, attention 531.8 ->
536.7, `pf_wait` 32.1 -> 32.0). Effective grouped-GEMM throughput is ~5.0
TFLOP/s (4.0 TFLOP of routed work), still far below the tensor-core roofline and
also below the ~17.3 GiB / 0.81 s = 21 GB/s weight-read rate of the HBM, so the
tile is now dequant/occupancy bound rather than dot bound.

The next lever is unchanged from R10.7: launch over the **nonempty segment
list** (a compacted `seg_ids` indirection, a kernel signature change plus a Rust
CSR change) instead of all 1664 slot segments, ~6x fewer CTAs at this prompt.
`pf_gemm` is 0.405 ms/token, still short of the 0.35 target. Gates: `--interp`
equality for both grouped kernels, `a_whole_layer_prefill_overlap_matches_the_host_plan`
(grouped default and `FERRUM_PF_GEMV=1`), 342 CPU units, 453 CUDA units,
`check-kernels.sh`. Logs `/tmp/opencode/r10_8/`.

**R10.9 FP8 prefill GEMM coalescing + tile -- LANDED (2026-10-23, decision
123).** At the 1991-token step the largest remaining dense bucket was
`gdn_qkvz` (`[1991,2048] x [12288,2048]`, FP8 W8A16) at 718.8 ms / 0.361
ms/token, with `attn_proj` 205.5 and `gdn_tail` 243.2 ms behind it. The vendored
kernel loaded the weight as a `[BLOCK_K, BLOCK_N]` tile from the row-major
`[N, K]` bank, so Triton 3.7.1 assigned the layout's fast axis to N (stride
`K=2048`): each lane's `ld.global.b8` hit its own 32-byte sector. The same
source under Triton 3.6 is canonicalized to a coalesced `[BLOCK_N, BLOCK_K]`
load (verified in TTGIR), which is why the earlier 3.6-based JIT probe showed no
problem and the codegen regression only appeared in the AOT cubin. The 3.7.1
PTX at 4 warps also fails to fit: `ptxas` reports 1224/892 bytes of
spill stores/loads at `64x128x64` (3.6 reports zero), so the shipped kernel was
both sector-amplified and spilling.

The fix is the coalesced orientation (`offs_n[:, None] * stride_wn +
offs_k[None, :] * stride_wk`, then `tl.dot(a, tl.trans(w))`) plus an 8-warp
`128x128x64` tile. The AOT entries now carry `num_stages` as well as
`num_warps`, so the compiled cubin follows `GEMM_STAGES`/`GEMM_WARPS` instead of
Triton's defaults. Decode is untouched: `M == 1` still runs the split-K GEMV.

Isolated kernel bench with the committed cubins (same GPU, `[1991,2048] x
[12288,2048]`, best of 20 after 2 warm-ups, CUDA-event sync per launch):

| config (M/N/K/warps/stages) | best ms | TFLOP/s | notes |
|---|---:|---:|---|
| base `64/128/64/4/3` (uncoalesced) | 17.99 | 5.57 | 1224/892 B spills |
| coalesced `64/128/64/4/3` | 14.91 | 6.72 | same spill, fixed sectors |
| coalesced `64/128/64/8/3` | 6.61 | 15.16 | no spills |
| coalesced `128/64/64/8/3` | 6.36 | 15.75 | no spills |
| coalesced `128/128/64/8/3` | **5.36** | **18.71** | 40/20 B spill, **shipped** |
| coalesced `128/128/32/8/3` | 5.50 | 18.22 | -2.7%, under the bar |
| coalesced `128/128/64/8/2` | 5.90 | 16.98 | -10.2% |
| coalesced `128/256/64/8/3` | 13.97 | 7.17 | 704/468 B spills |
| coalesced `64/64/64/8/3` | 8.65 | 11.59 | -38% |

Two configs failed to beat the winner by >2%, so the sweep stopped; `BLOCK_M=128`
halves the per-M-tile weight re-reads (32 -> 16 tiles at 1991 rows) and
`BLOCK_N=128` keeps the CTA count and dot width, while 8 warps is what removes
the spill. Order-balanced A/B at 2057 (n=3/arm, same builder, base rebuilt from
the pre-change store cubins, interleaved B W W B B W,
`FERRUM_SMOKE_TOKENS=2`, `FERRUM_SMOKE_CACHE=1664`, chunk 8192): pp
**713.1 -> 929.4 tok/s (+30.3%; raw 707.5/717.6/714.1 ->
932.5/934.5/921.1)**. The 517 control is flat: **401.7 -> 400.9 tok/s (-0.2%;
402.1/401.8/401.2 -> 398.8/401.7/402.2)**, because at 451 tokens prefill is
still copy-bound. Greedy ids are `[561, 3841]` in every measured run and VRAM
stays 7190/8151 MiB.

`FT_PROFILE` at 2057 (`FERRUM_SMOKE_NOGRAPH=1`, 40 decode tokens, skip 8; two
runs per arm, 2890.2/2900.7 vs 2236.5/2239.4 ms): the 1991-token step is
**2890 -> 2239 ms (-22.5%)**; `gdn_qkvz` **718.8 -> 269.3 ms (-62.5%)**,
`attn_proj` **205.5 -> 88.2 (-57.1%)**, `gdn_tail` **243.2 -> 98.8 (-59.4%)**,
`attn_out` **90.9 -> 42.9 (-52.8%)**, `attn` 537.6 -> 392.8. The grouped MoE kernel is unchanged; `pf_gemm`
measures 813.8 -> 856.7 ms (the profile spans overlap, so this is not attributed
to the FP8 change) and `pf_wait` stays 32.1 ms, so the win is the dense FP8
projections alone.

Correctness: `selftest --interp` PASS (the interpreter does not cover the M>1
`tl.dot`; the device numeric gates are below), `compile_check` and
`scripts/check-kernels.sh` PASS, `a_whole_layer_prefill_overlap_matches_the_host_plan`
both arms, `a_prefill_retains_the_decode_expert_cache`,
`a_device_decode_step_matches_refeeding_the_whole_prefix`,
`a_batched_decode_step_matches_n_single_row_steps` (the M=3 batched decode now
runs the new GEMM tile) and `batching_raises_aggregate_decode_throughput`
(serial 29.5 vs batched 46.5 tok/s, 1.57x, unchanged). `cargo fmt --check`,
`cargo clippy --all-targets` and `--features cuda`, `cargo test` (342),
`cargo test --features cuda --lib` (453) all pass. Logs
`/tmp/opencode/r10_9/`.

## R10.10 grouped-prefill grid compaction + `BLOCK_M=64` -- LANDED (2026-10-23)

R10.8 left the grouped GEMM launching `(cache_slots, N/BLOCK_N)` programs, but a
whole-layer prefill half only holds `2E = 512` buffer slots, so ~85% of the
grid loaded `seg`, found `start == end` and returned. The kernel now takes a
`seg_ids` list and runs `program_id(0)` over it; the Rust side passes the
active half's ascending `victims` slice (the CSR stays global, so the ranges
line up). `FERRUM_PF_SEGMENTS=all` reproduces the old grid for an A/B.
`GEMM_BLOCK_M` goes 32 -> 64, so one M-tile covers the ~62 routes per expert at
a 1991-token chunk.

Interleaved A/B at 2057 (n=3/arm, same binary, cache 1664, 2 tokens): new
955.3/940.7/954.4 = **950.1 tok/s** vs old 915.2/936.4/938.1 = **929.9 (+2.2%)**.
The parts are individually below the bar (`BLOCK_M=64` +1.4%, compaction
+0.6%) and only ship together. The 517 control is flat (copy-bound), greedy ids
are `[561, 3841]`, VRAM is unchanged. `pf_gemm` is still dequant-bound; larger
N/K tiles and more warps were already rejected in R10.8. Logs
`/tmp/opencode/r10_10/`.

## R11 decode cache sizing -- LANDED (2026-10-23)

The auto planner reserved 1 GiB, which resolved 1664 slots on the 8 GiB target
with the smoke's 4096-token KV pool (`free=3878 MiB`, `slot_bytes=1775616`). Dropping the reserve to 512 MiB
resolves 1920. That is the largest safe step: 1984 and 2048 both OOM the
2057-token prefill scratch (silu/cat) even though they pass a 517-token
prefill, so the `PREFILL_CHUNK_TOKENS=2048` scratch, not the decode graph, is
the binding constraint. Interleaved n=3/arm at bs=1: tg **30.2 -> 32.8 tok/s
(+8.4%)**, VRAM 7190 -> 7622 MiB, greedy ids unchanged; FreeToken is 30.3, so
bs=1 decode is now **1.08x**. Prefill is flat at 517 (401.0) and 958.5 at 2057
on the auto cache. `FERRUM_SMOKE_CACHE=1984`/`2048` remain useful diagnostics
for a larger prefill chunk budget. Logs `/tmp/opencode/r10_10/`.
