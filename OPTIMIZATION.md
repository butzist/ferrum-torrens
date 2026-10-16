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
