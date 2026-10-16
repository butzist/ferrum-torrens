"""Fused GDN (gated-delta-net) decode kernel (TODO.md section 8 item 15).

The libtorch decode path ran the linear-attention mixer as a dozen-plus ops per
layer: separate `softplus`/`sigmoid`/`exp` gating, a `pow`/`sum`/`sqrt`/`recip`
L2-norm on q and k, `exp` decay, a `sum_dim` for `kv_mem`, an outer product, a
second `sum_dim` for the output, each with an `f32_` cast of a bf16 operand. Over
30 layers that dominated decode (~35%). This kernel is the FreeToken-equivalent
swap: one launch per layer folds gating (sigmoid + softplus decay), the in-kernel
q/k L2-norm, the delta-rule recurrent update and the output into a single
program, reading and writing the per-slot state in place.

Math, per (row, value-head), matching `gdn_recurrence_step_batch`:
    g    = -exp(A_log) * softplus(a + dt_bias)      (<= 0, the log-decay)
    beta = sigmoid(b)
    qn   = q / sqrt(sum(q^2) + 1e-6) * (K ** -0.5)
    kn   = k / sqrt(sum(k^2) + 1e-6)
    h    = h * exp(g)
    h    = h + kn[:, None] * ((v - sum(h * kn[:, None], 0)) * beta)[None, :]
    o    = sum(h * qn[:, None], 0)

q/k carry `NUM_K_HEADS` heads and v `NUM_V_HEADS`; each value-head reads
`i_hv // (NUM_V_HEADS // NUM_K_HEADS)` of the key heads, the repeat-interleave
mapping `expand_heads` bakes in. The state is `[rows, NUM_V_HEADS, K, V]` f32 with
`K` outer and `V` inner (`LinearStateSlabs`'s recurrent layout), written back to
the same rows.

Grid is `(1, cdiv(V, BV), rows * NUM_V_HEADS)`: `NK` is always 1 because
`BK = next_power_of_2(K)` covers every key head width this model uses.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The value tile. A constexpr baked into the cubin and recorded in the manifest
# (signatures.SERVING_GDN_BV).
GDN_BV = 32
# The L2-norm epsilon, the same 1e-6 `gdn_recurrence_step_batch` uses. Inlined in
# the kernel: triton only reads module globals through `tl.constexpr`.
_L2NORM_EPS = tl.constexpr(1e-6)


@triton.jit
def _gdn_decode_kernel(
    ba_ptr,
    a_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    out_ptr,
    stride_q,
    stride_k,
    stride_v,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_nh = tl.program_id(2)
    i_n = i_nh // NUM_V_HEADS
    i_hv = i_nh % NUM_V_HEADS
    # Repeat-interleave GQA: value-head `i_hv` reads key-head `i_hv // rep`.
    i_h = i_hv // (NUM_V_HEADS // NUM_K_HEADS)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    q_row = q_ptr + i_n * stride_q
    k_row = k_ptr + i_n * stride_k
    v_row = v_ptr + i_n * stride_v
    ba_row = ba_ptr + i_n * (2 * NUM_V_HEADS)
    o_row = out_ptr + i_n * (NUM_V_HEADS * V)
    h_row = state_ptr + i_n * (NUM_V_HEADS * K * V)

    b_q = tl.load(q_row + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(k_row + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(v_row + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
    # `ba` is `[b | a]` along the head axis: b at `i_hv`, a at `NUM_V_HEADS + i_hv`.
    b_b = tl.load(ba_row + i_hv).to(tl.float32)
    b_a = tl.load(ba_row + NUM_V_HEADS + i_hv).to(tl.float32)
    b_A_log = tl.load(a_log_ptr + i_hv).to(tl.float32)
    b_dt = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    # g = -exp(A_log) * softplus(a + dt_bias), the branch-free form the forward
    # uses (`max(x, 0) + log1p(exp(-|x|))`), and beta = sigmoid(b).
    x = b_a + b_dt
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    b_g = -tl.exp(b_A_log) * sp
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + _L2NORM_EPS) * (K ** -0.5)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + _L2NORM_EPS)

    p_h = h_row + i_hv * (K * V) + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    b_h = b_h * tl.exp(b_g)
    b_v = (b_v - tl.sum(b_h * b_k[:, None], 0)) * b_beta
    b_h = b_h + b_k[:, None] * b_v[None, :]
    b_o = tl.sum(b_h * b_q[:, None], 0)

    tl.store(p_h, b_h.to(state_ptr.dtype.element_ty), mask=mask_h)
    tl.store(o_row + i_hv * V + o_v, b_o.to(out_ptr.dtype.element_ty), mask=mask_v)


def gdn_decode(
    ba: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    bv: int = GDN_BV,
) -> None:
    """Launch the fused GDN decode step.

    `ba` is `[rows, 2 * v_heads]` f32 (`b` then `a`), `a_log`/`dt_bias` are
    `[v_heads]` in the model dtype, `q`/`k` are `[rows, k_heads, K]`, `v` is
    `[rows, v_heads, V]`, `state` is `[rows, v_heads, K, V]` f32 (updated in
    place) and `out` is `[rows, v_heads, V]`. q/k/v/state/out are f32, matching
    the forward's f32 conv and recurrence; q/k/v may be strided views (the
    forward slices them out of one conv buffer), so their row strides are passed
    in.
    """
    rows, k_heads, kd = q.shape
    v_heads = v.shape[1]
    vdim = v.shape[2]
    bk = triton.next_power_of_2(kd)
    grid = (triton.cdiv(kd, bk), triton.cdiv(vdim, bv), rows * v_heads)
    _gdn_decode_kernel[grid](
        ba,
        a_log,
        dt_bias,
        q,
        k,
        v,
        state,
        out,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        NUM_K_HEADS=k_heads,
        NUM_V_HEADS=v_heads,
        K=kd,
        V=vdim,
        BK=bk,
        BV=bv,
        num_warps=4,
    )


def gdn_decode_reference(
    ba: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch oracle: the same recurrence, one (row, value-head) at a time.

    `state` is not mutated; the new state and `o` are returned. Uses the same
    branch-free softplus and L2-norm the kernel does.
    """
    rows, k_heads, kd = q.shape
    v_heads, vdim = v.shape[1], v.shape[2]
    rep = v_heads // k_heads
    scale = kd ** -0.5
    out = torch.zeros((rows, v_heads, vdim), dtype=torch.float32, device=q.device)
    new_state = state.to(torch.float32).clone()
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    for n in range(rows):
        for hv in range(v_heads):
            h = hv // rep
            qi = qf[n, h]
            ki = kf[n, h]
            x = (ba[n, v_heads + hv] + dt_bias[hv]).to(torch.float32)
            sp = torch.clamp(x, min=0.0) + torch.log1p(torch.exp(-x.abs()))
            g = -torch.exp(a_log[hv].to(torch.float32)) * sp
            beta = torch.sigmoid(ba[n, hv].to(torch.float32))
            qn = qi / torch.sqrt((qi * qi).sum() + 1e-6) * scale
            kn = ki / torch.sqrt((ki * ki).sum() + 1e-6)
            s = new_state[n, hv] * torch.exp(g)
            s = s + kn[:, None] * ((vf[n, hv] - torch.sum(s * kn[:, None], 0)) * beta)[None, :]
            out[n, hv] = torch.sum(s * qn[:, None], 0)
            new_state[n, hv] = s
    return out, new_state


__all__ = ["GDN_BV", "gdn_decode", "gdn_decode_reference"]
