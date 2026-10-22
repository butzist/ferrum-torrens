# ALGORITHM -- FerrumTorrens inference algorithm and expert caching

This document explains, for a Rust port, (1) the exact math of the Qwen3.5/3.6
hybrid-attention MoE forward pass, (2) how FreeToken serves it (prefill vs decode,
paged KV + radix reuse, GDN state), and (3) the expert memory hierarchy and caching
algorithms of the `offload` strategy. It mirrors the behaviour in `./FreeToken`
and the design section of the paper (arXiv:2608.16157, §3). Line references are to
`./FreeToken/python/freetoken/...` for wherever the exact semantics are defined.

## 0. Reference model: Qwen3.6-35B-A3B (NVFP4)

- 2048 hidden, 40 layers, vocab 248320 (untied), Gemma norms (`1 + w`), no biases.
- `layer_types` (explicit in config, or interval 4): 30 `linear_attention`
  (GatedDeltaNet) layers, 10 `full_attention` GQA layers at ids {3,7,...,39}.
- Every layer runs the same hybrid decoder block and then an MoE block.

Compute/space talk (NVFP4): 256 routed experts x 40 layers, ~1.69 MiB per expert
(~17.3 GiB total host side), ~433 MiB per full layer; the GPU keeps a slot cache
of `cache_size_slots` experts. Non-expert weights (embeddings, norms, routers,
attention/GDN projections, shared experts, lm_head) are GPU-resident.

Target hardware: **RTX 5060, 8 GB VRAM, sm_120 (Blackwell)**. On this card the
resident non-expert weights total roughly 2.5-3 GB (untied bf16 embeddings ~1 GB +
fp8/nvfp4 lm_head and projections ~1.4 GB + routers/norms). An NVFP4 expert cache of
256 slots = 0.43 GB (512 slots = 0.87 GB) leaves ~4-5 GB, of which KV needs only
~20 KB/token (full-attention layers only), so tens of thousands of KV tokens fit and
the context size is not the binding constraint on this card. Per-expert and per-token
figures let us recompute for other cards.

## 1. The serving loop: two phases, one forward per step

FreeToken serves two batch phases (`scheduler.py`, `engine/engine.py`):

- **prefill**: a batch of requests with new tokens (extend_len can be thousands).
  Token budget per step caps at `max_prefill_length` (chunked prefill). Any one
  batch may be a mix of full prompts, continuation chunks, and decode requests
  that join only after their prompt is done.
- **decode**: one new token per batch member per step. `Batch.size = len(reqs)`,
  `extend_len == 1`. Runs inside a CUDA graph when `size <= max_graph_bs`.

The scheduler builds, per step, a `ForwardInput` that the engine turns into a single
`model.forward()` over all 40 layers:

```
input_ids    : gathered from token_pool[input_mapping]        (int32)
positions    : concat arange(cached_len, device_len) per req  (int32 [N])
out_loc      : page_table[input_mapping]                      (flat token slots)
linear_table_idx : GDN state slot per request (before typing; each req also has a
                   live state slot; decode re-reads/restores it)
```

One forward:
1. `x = embed_tokens[input_ids]` (VocabParallelEmbedding, no scale).
2. For layer 0..39: `x, residual = layer.forward(x, residual)`.
3. `x = final_norm(norm(x + residual))`, then `logits = lm_head(x)`.
4. sample -> `next_tokens`; scatter into `token_pool[output_mapping]`;
   D2H copy for detokenize/stop checks; each `Req.complete_one()` advances
   `cached_len/device_len`.

The per-request `table_idx`/`token_pool` indirection is how input and output
tensors are gathered/scattered on-device without host round-trips (engine
`page_table`, `scheduler/table.py`).

## 2. The decoder block (equations)

Semantics: `x = x + mixer(input_norm(x)); x = x + mlp(post_norm(x))`; implemented in
"residual-stream" form so add + norm is one fused kernel per sublayer
(`models/qwen3_5_moe/model.py:27-74`).

### 2.1 Gemma RMSNorm (all norms except the GDN output norm)

```
rms(x) = x / sqrt(mean(x^2) + eps),  eps = 1e-6,  fp32 accumulation
GemmaRMSNorm:  y = rms(x) * (w)             # w already holds 1+w (baked at load)
forward_add_residual:  x <- rms(x + residual) * w;  return x, x_as_new_residual
```
`layers/norm.py:29-76`. GDN output `linear_attn.norm` is plain `y = rms(x) * w`
(no +1 bake, `weight.py:38-44`).

### 2.2 Full-attention layer (10 of 40)

Dims: `q = 16`, `kv = 2`, `head_dim = 256` (`models/qwen3_5_moe/attention.py`).

```
qkv      = qkv_proj(x)                                   # bf16/fp8 linear, no bias
qg, k, v = split(qkv, [q*2*hd, kv*hd, kv*hd])
q, gate  = qg[..., :hd], qg[..., hd:]                    # per-head scalar gate
q        = q_norm(q);  k = k_norm(k)                     # Gemma 1+w RMSNorm
q, k     = rope(q, k, positions)                          # partial NeoX, 64/256 dims
o        = paged_attention(q, k, v)                       # GQA, NHD, page_size=1
out      = o_proj(o * sigmoid(gate))
```

RoPE is applied **before** attention (the paged backend runs with
`pos_encoding_mode=none`). Partial-rope: only the first `rotary_dim = 64` elements
per head rotate; `inv_freq[i] = theta^(-2i/rotary_dim)`, `theta = 1e7`, NeoX
half-pair convention (`layers/rotary.py:40-59`, `qwen3_5_moe/config.py:49-56`).

### 2.3 Gated DeltaNet (linear) layer (30 of 40)

Dims: `key = 16 x 128 = 2048`, `value = 32 x 128 = 4096`, conv kernel 4
(`models/qwen3_5_moe/gdn.py`, reference oracle `gdn_reference.py` -- reproduce this
math exactly for correctness tests).

```
proj     = in_proj(x)                        # split [conv_dim | value_dim | vh | vh]
                                               # conv_dim = 2*key_dim + value_dim = 8192
conv_in, z, b, a = split(proj)
mixed    = silu(conv1d(conv_in))             # causal depthwise conv1d, kernel 4, left-pad 3
q, k, v  = split(mixed, [key_dim, key_dim, value_dim])
beta     = sigmoid(b)                        # [., vh]
g        = -exp(A_log) * softplus(a + dt_bias)   # log-decay, fp32, softplus_threshold 20.0

# recurrence (per value-head, heads grouped by GQA rep = 2):
state = 0                                    # [Hv, 128, 128] fp32
for t in tokens:
    q_t, k_t, v_t  = l2norm(q_t), l2norm(k_t), v_t     # eps 1e-6, cast fp32
    q_t *= 1/sqrt(128)
    g_t = exp(g_t); beta_t = sigmoid(b_t)     # decode kernel fuses this
    state = state * g_t
    kv_mem = einsum(state, k_t)               # [Hv, dv]
    delta = (v_t - kv_mem) * beta_t
    state += outer(k_t, delta)
    out_t = einsum(state, q_t)                # accumulate output

out      = GatedRMSNorm(out, silu(z))         # rmsnorm(out) * silu(z), norm before gate
out      = out_proj(out)
```

- q/k are l2-normed; the recurrent state is always fp32.
- Prefill uses a chunked implementation (state-independent chunks + carry), decode
  uses a fused single-token update (`fused_sigmoid_gating_delta_rule_update`):
  `beta = sigmoid(b)`, `g = -exp(A_log)*softplus(a+dt_bias)` computed in-kernel.
- Per-request state lives in the **linear state pool** (`kvcache/linear_state_pool.py`):
  conv tail `[Lg, slots, conv_dim, 3]` + recurrent `[Lg, slots, 32, 128, 128]` fp32.
  Decode restores the slot at `chapter` start and writes the updated state back by
  `linear_slot_idx`.

#### 2.3.1 Fast chunked GDN kernels (documented for later; the MVP runs the reference)

The chunked GatedDeltaNet forward splits the sequence into chunks (FreeToken:
`CHUNK_SIZE = 64`) so each position costs O(64) work instead of O(T), plus a chunk
carry recursion:

1. **Intra-chunk term** (parallel over the chunk): a causal "gated linear attention"
   kernel evaluates the contributions of keys/values *within* the chunk as a small
   set of batched matmuls over the chunk matrix, applying the log-decay factors and
   the lower-triangular causal mask -- no per-position sequential update.
2. **Inter-chunk carry**: a chunk's end-state (the piece only its own prefix probes)
   is the next chunk's start-state. The recursion is re-expressed so a chunk's
   contribution reduces to a `q/k-l2norm`-weighted `kv_mem` and `delta` accumulation
   (the chunk's last inner product) handed forward as one small tensor. Chunks are
   otherwise independent -> parallel over contiguous chunks.
3. **Decode update** stays the single-token fusion from the reference: `l2norm`,
   `beta = sigmoid(b)`, `g = -exp(A_log)*softplus(a+dt_bias)`,
   `state = state*g + outer(k, (v - state@k)*beta)`, `out = state@q`; state is read
   and written back by pool slot (`linear_slot_idx`).
4. **Fusions to chase** (what the fla-based kernels in
   `models/qwen3_5_moe/gdn_kernels.py` actually do): inline l2norm+scale on q/k,
   inline sigmoid/softplus gating, recycling the state write-back into the output
   norm, and keeping `h[last]` + the trailing conv inputs as the periodic snapshot
   that `hybrid_radix` later donates.

The MVP's reference recurrence is the golden oracle for tuning (1)-(2).

### 2.4 MoE block (every layer)

Router (`models/qwen3_5_moe/moe.py`, bf16 always, no bias):

```
logits  = gate(x)                  # [N, E=256] LinearReplicated 2048->256
p       = softmax(logits, dim=-1)  # over ALL 256 experts
w, ids  = topk(p, k=8, dim=-1)
w       = w / w.sum(-1, keepdim=True)        # norm_topk_prob=true
```

Routed experts (NVFP4 on disk/in cache, see §4.3 for dequant):

```
x_e      = permute x to per-expert groups (moe-align, padded to BLOCK_SIZE_M)
ic1      = x_e @ gate_up[e]^T               # top-8 group, [N, k, 2*I=1024]
ic2      = silu(ic1[:, :I]) * ic1[:, I:]    # gated_act_and_mul
ic3      = ic2 @ down[e]^T                  # [N, k, H]
routed   = sum_k w_k * ic3_k                # router weight applied AFTER down (apply on input = off)
```

Shared expert (per layer):

```
shared_ffn = silu(gate_up_s(x)) then down_s                    # SwiGLU, I=512
shared = sigmoid(shared_expert_gate(x)) * shared_ffn(x)
out    = routed + shared
```

### 2.5 Final norm + lm_head

`x = final_norm(x + residual)`; `logits = lm_head(x)` (untied, NVFP4 or bf16 per
checkpoint dialect).

## 3. KV cache, radix prefix reuse, GDN state

### 3.1 Paged KV pool (full-attention layers)

`MHAKVCache` (`kvcache/mha_pool.py`): one flat buffer
`[2, num_full_layers, num_pages, page_size, kv_heads, head_dim]`; only the 10 full
layers get slabs (global layer id -> dense index remap). `page_size = 1`.

- `page_table` is `[max_running_req + 1, width]` int32 and stores **raw token
  locations** (not page ids). Row `max_running_req` is the dummy request pointing at
  the sentinel page.
- `CacheManager.free_slots` is a host free-list of page bases; allocate pops `k`
  bases, expand to per-token indices, write into the request's page-table row.
- Attention backend stores K/V at `out_loc` per token per layer (the `store_kv`
  step), then runs paged attention (FlashInfer-equivalent: kv tensor viewed as
  `(num_tokens, 1, kv_heads, head_dim)`, indices = one entry per token).

### 3.2 Radix prefix cache (`kvcache/radix_cache.py`)

A tree of prefixes sharing KV pages across requests:

- Node = token span key + `(value: per-token out_locs)`, `ref_count`, `timestamp`,
  children keyed by the first `page_size` tokens of each child's span.
- **Match** (`match_req`): walk children by the next key; compare with a
  `std::mismatch`; align match length down to `page_size`; on a partial match,
  `split_at` breaks the node so the matched prefix is reusable; return
  `cached_len` + the concatenated out_locs up the chain.
- **Admission**: charge `(div_ceil(total,page) - div_ceil(cached,page)) * page`
  pages against `evictable + free`; `lock(handle)` protects matched pages.
- **Allocate**: re-point the request's page-table tail at matched pages, pop free
  pages for the rest, run forward, `store_kv`.
- **Commit** (`cache_req`): insert the request's computed prefix into the tree
  (page-aligned); cloned duplicate pages go back to `free_slots`; an unfinished
  request's row is re-pointed at the tree's canonical pages and re-locked.
- **Eviction**: LRU min-heap over **unlocked leaf nodes only** (internal nodes'
  KV is a prefix dependency of descendants); evicted out_locs return to the free
  list. Runs only when `_allocate` exhausts the free list.

### 3.3 GDN linear state + hybrid_radix

- Rock-solid baseline: states are unique per request; nothing is shared. Radix-KV
  reuse still applies to the full-attention layers' pages.
- `hybrid_radix` (paper's "semantic anchor"/sglang MambaRadixCache inspiration)
  additionally **donates** a GDN recurrent-state snapshot at page-aligned chunk
  boundaries (x64) into radix nodes, with a second refcount + separate eviction
  (internal-node tombstones free the slot but keep KV). The domain core is
  implemented (`domain/kvcache.rs::HybridRadixIndex`, decision 68): a snapshot is
  point-attached at a node's END boundary, so a strict prefix of a snapshot node
  is not reusable and `match_prefix` truncates to the deepest node on the matched
  path with a live snapshot; `split_at` keeps the snapshot on the suffix; the two
  refcounts are `full_ref` (KV pages) and `mamba_ref` (the slot), with
  `full_ref >= mamba_ref`; `evict_mamba` counts snapshots (tombstone an internal
  node, free a leaf's KV too) and `evict_full` counts tokens; donation is a
  caller-owned slot id (donate-not-copy). The device primitive
  `LinearStateSlabs::copy_slot` does the COW donate/restore.
- The wiring is implemented (decision 69). `DeviceCache` owns the tree and drives
  the commit boundaries: at **prompt end** `commit(finished=false)` COWs the
  running GDN state into a tree-owned slot and locks the node; at **finish**
  `commit(finished=true)` inserts the full sequence and donates the request's
  final live slot zero-copy. `adopt_prefix` matches the prompt minus its last
  token, and on a live-snapshot match copy-restores the snapshot into the
  request's live slot and re-points the request at the tree's canonical KV pages.
  Pressure runs the evictions: `reserve` reclaims pages via `evict_full`, slot
  allocation reclaims GDN slots via `evict_mamba` (the two pools are independent,
  so no partitioning). Prefill is chunked (decision 71): every non-final chunk
  `commit(finished=false)`s its boundary too, so a prefix that ends mid-prompt is
  reusable, not just prompt-end and completed-sequence prefixes. A snapshot
  carries both the recurrent state and the causal-conv tail (`conv_state`), which
  is why the boundary is page-aligned: the conv tail must line up with the tokens
  the KV pages hold.

## 4. Expert memory hierarchy (the `offload` strategy)

### 4.1 The two-level layout

- **Level 1 (host, source of truth):** per-layer expert banks in pinned host RAM,
  loaded once (`expert_load = "serial"` reads layer by layer) from the FTW shards.
  One bank tensor per role with a leading `[num_experts, ...]` dim, e.g. NVFP4:
  `gate_up_packed [E, 2I, H/2]`, `gate_up_scale [E, 2I, H/16]`, `gate_up_global [E, 2I]`,
  `down_packed [E, H, I/2]`, `down_scale [E, H, I/16]`, `down_global [E, H]`.
  The tiny per-expert `gate_up_global/down_global` fp16 alphas are the exception:
  they are GPU-resident (`[L*E]`), not streamed.
- **Level 2 (GPU, LRU slot cache):** `cache_size_slots` slots
  (`[cache_size, ...]` per bank role). A slot holds one expert across every role.

### 4.2 Indexing

Flat id space: `flat = layer * num_experts + expert` (one shared index across all
banks of a layer). Three tables (`moe/offload_cache.py:169-201`):

```
slot_for_id[L*E] -> int32   # expert -> GPU slot, -1 if not resident
id_of_slot[C]    -> int32   # slot -> flat id,  -1 if free
usage[C]         -> int64   # last-use step (one global clock bumped per layer per step)
```

### 4.3 Ensure + copy (decode, per layer)

```
routed_ids            # int32 [N*k] expert ids from the router
ensure_experts(layer, routed_ids):
    for each id: slot = slot_for_id[id]
    misses = ids with slot == -1
    victims = LRU picks from usage (argmin), protected set = ids already touched
              earlier IN THIS STEP (owner_active): never evict under an in-flight route
    rewrite routed_ids in place: id -> slot id (resident or newly assigned)
    emit (evict_slots, src_indices)
copy_missing(misses):
    one fused launch over all banks: for each (slot, host_src) pair, copy the
    expert's rows  # 16-byte aligned; one kernel, not one per bank
    (compute stream, serial with the following GEMM; no overlap in plain offload)
```

The sequence per decode step per layer is therefore:
`route -> ensure_experts -> copy_missing(on-stream) -> grouped GEMM from slot cache`
(`layers/moe.py:403-435`). Misses are always batched -- serial means one copy plan
per layer-step, not per expert, which is what keeps PCIe saturated.

### 4.4 Prefill (double-buffered layer streaming)

Routing is dense over the batch during prefill, so FreeToken streams **whole
expert layers** instead of per-expert misses:

```
if layer_id == 0: begin_prefill()      # fence copy stream behind compute stream
prefetch_prefill_layer(layer_id)       # issue H2D of this layer on copy_stream
prefetch_prefill_layer(layer_id + 1)   # next layer's H2D queued
wait_prefill_layer(layer_id)           # compute waits only on THIS layer's ready event
   ... run expert GEMMs (routing passes through unmapped: id == expert) ...
release_prefill_layer(layer_id)        # record release event so the copy stream
                                       # can reuse the buffer next parity pass
```

- Uses `2 * num_experts` cache slots as a `[2, num_experts]` double buffer
  (parity = layer_id % 2), so overlap needs `cache_size >= 2 * num_experts`.
  FreeToken borrows the first `2E` slots; our engine picks the coldest
  contiguous window of that size and copies each bank's whole layer as one DMA
  run (decision 120).
- No overlap fallback: `materialize_layer(layer_id)` + synchronous copy.
- Decode never overlaps; its GEMMs must not read slots the prefill copy stream is
  about to overwrite -- `begin_prefill` orders the boundaries.

### 4.5 What the other strategies mean (context; not in MVP)

| strategy | decode compute | miss handling | MVP? |
|---|---|---|---|
| fused | GPU | all experts resident, no cache | no |
| **offload** | **GPU** | **misses streamed over PCIe into LRU slots** | **yes** |
| cpu | CPU | misses computed on CPU from host banks, not fetched | no |
| hybrid | GPU+CPU | per step, fetch <=k misses over PCIe, compute the rest on CPU, merge (q\* policy from measured B_P / B_H) | no |

So "inference on GPU, stream required experts to GPU" **is** `offload`. The `cpu`
strategy is the alternative you asked about: it trades PCIe weight movement for CPU
compute on the same host memory.

### 4.6 Sizing constraints (no memory_ratio; explicit)

- `cache_size_slots >= num_experts` (one full layer) is the floor.
- `cache_size_slots >= 2 * num_experts` enables the prefill double buffer.
- All experts resident: `cache_size = L * E = 10240` slots would fully fit on most
  consumer cards at NVFP4 (only if you want no misses).
- Host RAM floor: banks (~17.3 GiB) + working set for pins.

## 5. Scheduling and batching

- Strict prefill-priority: `prefill or decode` (FreeToken never runs both in one
  step). Known behaviour to preserve: a long chunked prefill blocks decode.
- Prefill admission checks page tables, prefix match, KV reservation
  (`div_ceil` alignment), then locks; chunked prompts reuse their table/state slots
  across chunks. Our implementation commits a `finished=false` snapshot at every
  chunk boundary (decision 71); a continuation chunk samples nothing and is
  re-queued. FreeToken differs: it skips `ChunkedReq` (no intermediate commit) and
  snapshots on a fixed x64 mamba-track grid within one whole-prompt prefill, so our
  reuse points are budget-dependent rather than on a x64 grid.
- Decode batch is sorted by request `uid` for determinism.
- Stop priority: EOS / stop-string wins over max-tokens; token budgets clamp
  `max_device_len`.

## 6. Sampling

Greedy when `temperature <= 0` or `top_k == 1` (pure argmax, no softmax kernel).
Otherwise temperature-softmax -> top_k / top_p filter -> renormalize. Mixed batches:
probabilistic sampling runs on all rows, greedy rows are overwritten by argmax
(deterministic ties). No min-p, repetition penalty, beam, or logit processors.

## 7. End-to-end request lifecycle (pseudocode)

```
tokenize(prompt)                          # byte-level BPE + chat template, in-process
req = {input_ids, sampling, max_tokens}
prefill_chunks = chunk(input_ids, max_prefill_length)

while not done:
    if prefill work available and budget:
        batch = next_prefill_chunk()      # match radix, allocate pages, lock
    elif running requests:
        batch = decode_batch()            # uid-sorted
    else: break

    # one model step
    fwd = build_forward_input(batch)      # positions, input_mapping, out_loc, states
    logits = model.forward(fwd)           # 40 layers, KV stores + radix/GDN writes
    next_tokens = sample(logits, batch)   # + greedy argmax fast path
    scatter token_pool; D2H next_tokens

    for req in batch: req.complete_one()  # advance lengths AFTER the step
    stop: eos / stop_str / max_device_len -> finish, else keep decoding
    prefill chunk done -> cache_req (commit KV prefix + boundary GDN snapshot,
                          finished=false); a non-final chunk re-queues, no sample
    request done   -> free pages, table slot, GDN slots (hybrid: donate the final
                      GDN slot to the tree first, idempotent)
    detokenize next_tokens -> SSE delta to client
```