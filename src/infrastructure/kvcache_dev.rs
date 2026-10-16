// The device side of the KV cache: the paged MHA pool and the per-request GDN
// linear state from ALGORITHM.md 3.1 / 3.3. Infrastructure, so the tensors stay
// out of the domain core, which only ever handles page ids and slot numbers
// (`domain::kvcache`).
//
// `page_size` is 1 everywhere (ALGORITHM.md 3.1), so a token location *is* its
// page index: the numbers a `SequenceCache` holds are exactly the numbers these
// slabs are indexed with.
//
// Storage is bf16 for KV and f32 for the recurrent state, which is what the
// stateless forward already used for each: the cached path stores the very
// values the re-fed forward would have recomputed, so a decode step matches it
// instead of merely approximating it. fp8 KV is a later memory optimisation and
// would break that equivalence, so it is not here yet.

use std::sync::Mutex;

use tch::{Kind, Tensor};

use crate::domain::kvcache::{
    EvictResult, HybridRadixIndex, KvCache, KvcacheError, LinearStatePool, PagePool, SequenceCache,
};
use crate::domain::model::{LayerType, ModelParams};
use crate::infrastructure::torch::Device;

/// ALGORITHM.md 3.1: one token per page.
pub const PAGE_SIZE: i64 = 1;

/// What the KV pool stores a token in.
///
/// f32 is the default because the cached forward's job is to reproduce the
/// re-fed forward exactly, and the re-fed forward keeps K/V in f32 after the
/// norm and the rope. bf16 is the cheaper serving option and is lossy by design
/// (FreeToken's reference serves fp8 KV, coarser still); reach for it only once
/// the decode path is proven and VRAM is the binding constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDType {
    F32,
    BFloat16,
}

impl KvDType {
    fn kind(self) -> Kind {
        match self {
            KvDType::F32 => Kind::Float,
            KvDType::BFloat16 => Kind::BFloat16,
        }
    }
}

/// Global layer id -> slab index, `None` for a layer that keeps no state of this
/// kind. Hybrid towers interleave the two kinds, so the remap is what keeps the
/// slabs dense.
fn dense_map(params: &ModelParams, kind: LayerType) -> Vec<Option<usize>> {
    let mut next = 0usize;
    params
        .layer_types
        .iter()
        .map(|l| {
            (l == &kind).then(|| {
                let d = next;
                next += 1;
                d
            })
        })
        .collect()
}

/// `[2, full_layers, num_pages, PAGE_SIZE, kv_heads, head_dim]`, bf16.
///
/// Index 0 of the leading dim is K, 1 is V. Only the full-attention layers get a
/// slab, so a 40-layer tower with 10 full layers pays for 10 layers of KV.
pub struct MhaPool {
    kv: Tensor,
    dense_of: Vec<Option<usize>>,
    kv_heads: i64,
    head_dim: i64,
    dtype: KvDType,
}

impl MhaPool {
    pub fn new(params: &ModelParams, num_pages: usize, dtype: KvDType, device: Device) -> Self {
        let dense_of = dense_map(params, LayerType::FullAttention);
        let layers = dense_of.iter().filter(|d| d.is_some()).count() as i64;
        let (kv_heads, head_dim) = (params.num_kv_heads as i64, params.head_dim as i64);
        let kv = Tensor::zeros(
            [2, layers, num_pages as i64, PAGE_SIZE, kv_heads, head_dim],
            (dtype.kind(), device.to_tch()),
        );
        Self {
            kv,
            dense_of,
            kv_heads,
            head_dim,
            dtype,
        }
    }

    /// The slab index for a global layer id, or `None` for a linear-attention
    /// layer, which has no KV slab.
    pub fn dense_of(&self, layer: usize) -> Option<usize> {
        self.dense_of.get(layer).copied().flatten()
    }

    pub fn num_pages(&self) -> usize {
        self.kv.size()[2] as usize
    }

    /// What the pool stores a token in. The paged attention kernel is compiled
    /// for one pointer element type, so the forward picks the paged path only
    /// when this matches what the cubin was built for.
    pub fn dtype(&self) -> KvDType {
        self.dtype
    }

    /// Device addresses of this layer's K and V slabs, each viewed as
    /// `[num_pages * PAGE_SIZE, kv_heads, head_dim]`; PAGE_SIZE is 1, so a
    /// token's byte stride is `kv_heads * head_dim * elt`. The driver-API
    /// attention launcher takes raw addresses, and this accessor is the one
    /// place that knows the pool's physical layout.
    pub fn kv_slab_ptrs(&self, layer: usize) -> (u64, u64) {
        let slab = self.slab(layer);
        let k = self.kv.select(0, 0).select(0, slab);
        let v = self.kv.select(0, 1).select(0, slab);
        (k.data_ptr() as u64, v.data_ptr() as u64)
    }

    /// Bytes the pool holds, for the config's memory arithmetic.
    pub fn bytes(&self) -> usize {
        self.kv.numel() * self.dtype.kind().elt_size_in_bytes()
    }

    /// Write this step's `k`/`v` (`[t, kv_heads, head_dim]`) at `locs`.
    pub fn store(&self, layer: usize, locs: &[u32], k: &Tensor, v: &Tensor) {
        let idx = self.index(locs);
        self.store_indexed(layer, &idx, locs.len(), k, v);
    }

    /// `store`, but the locations are already a device `[t]` index tensor.
    ///
    /// A captured decode step bakes the address of the index tensor, so the
    /// locations live on the device rather than being rebuilt host-side each
    /// step (decision 51).
    pub fn store_at(&self, layer: usize, locs: &Tensor, k: &Tensor, v: &Tensor) {
        let idx = locs.to_kind(Kind::Int64);
        self.store_indexed(layer, &idx, locs.numel(), k, v);
    }

    fn store_indexed(&self, layer: usize, idx: &Tensor, t: usize, k: &Tensor, v: &Tensor) {
        let slab = self.slab(layer);
        debug_assert_eq!(k.size(), vec![t as i64, self.kv_heads, self.head_dim]);
        debug_assert_eq!(v.size(), vec![t as i64, self.kv_heads, self.head_dim]);
        for (half, tt) in [(0i64, k), (1, v)] {
            // `select` hands back a view, so the in-place write lands in the
            // pool's storage rather than in a copy.
            let mut view = self.kv.select(0, half).select(0, slab).shallow_clone();
            let _ = view.index_copy_(0, idx, &tt.unsqueeze(1).to_kind(self.dtype.kind()));
        }
    }

    /// Read the K/V at `locs` back, in the order asked.
    pub fn gather(&self, layer: usize, locs: &[u32]) -> (Tensor, Tensor) {
        let idx = self.index(locs);
        self.gather_indexed(layer, &idx)
    }

    /// Read one fixed-length window per row: `locs` is `[rows, window]` and the
    /// result is `[rows, window, kv_heads, head_dim]`, so a batched decode
    /// gathers every request's window in one op (decision 51).
    pub fn gather_rows(&self, layer: usize, locs: &Tensor) -> (Tensor, Tensor) {
        let rows = locs.size()[0];
        let window = locs.size()[1];
        let idx = locs.to_kind(Kind::Int64).reshape([-1]);
        let (k, v) = self.gather_indexed(layer, &idx);
        (
            k.reshape([rows, window, self.kv_heads, self.head_dim]),
            v.reshape([rows, window, self.kv_heads, self.head_dim]),
        )
    }

    fn gather_indexed(&self, layer: usize, idx: &Tensor) -> (Tensor, Tensor) {
        let slab = self.slab(layer);
        // Gathers are always f32: the attention math and the paged-kernel
        // fallback read f32, and a narrower pool stores bf16 only to save bytes.
        // Upcasting bf16 -> f32 is exact.
        let half = |h: i64| -> Tensor {
            self.kv
                .select(0, h)
                .select(0, slab)
                .index_select(0, idx)
                .squeeze_dim(1)
                .to_kind(Kind::Float)
        };
        (half(0), half(1))
    }

    fn index(&self, locs: &[u32]) -> Tensor {
        let data: Vec<i64> = locs.iter().map(|&l| l as i64).collect();
        Tensor::from_slice(&data).to(self.kv.device())
    }

    fn slab(&self, layer: usize) -> i64 {
        self.dense_of[layer].expect("layer has no kv slab") as i64
    }
}

/// GDN state for every linear-attention layer and every live request slot.
///
/// The conv state is `[conv_dim, K-1]` per layer per slot in f32 -- the dtype
/// `causal_conv1d_silu` reads -- and the recurrent state is `[v_heads, k_dim,
/// v_dim]` in f32, which ALGORITHM.md 3.3 requires (`mamba_ssm_dtype=float32`).
pub struct LinearStateSlabs {
    conv: Tensor,
    recurrent: Tensor,
    dense_of: Vec<Option<usize>>,
    slots: i64,
    conv_dim: i64,
    width: i64,
    v_heads: i64,
    k_dim: i64,
    v_dim: i64,
}

impl LinearStateSlabs {
    pub fn new(params: &ModelParams, num_slots: usize, device: Device) -> Self {
        let dense_of = dense_map(params, LayerType::LinearAttention);
        let layers = dense_of.iter().filter(|d| d.is_some()).count() as i64;
        let (k_heads, k_dim) = (
            params.linear_num_key_heads as i64,
            params.linear_key_head_dim as i64,
        );
        let (v_heads, v_dim) = (
            params.linear_num_value_heads as i64,
            params.linear_value_head_dim as i64,
        );
        let width = params.linear_conv_kernel_dim as i64 - 1;
        let conv_dim = 2 * k_heads * k_dim + v_heads * v_dim;
        let kind = (Kind::Float, device.to_tch());
        Self {
            conv: Tensor::zeros([layers, num_slots as i64, conv_dim, width], kind),
            recurrent: Tensor::zeros([layers, num_slots as i64, v_heads, k_dim, v_dim], kind),
            dense_of,
            slots: num_slots as i64,
            conv_dim,
            width,
            v_heads,
            k_dim,
            v_dim,
        }
    }

    pub fn dense_of(&self, layer: usize) -> Option<usize> {
        self.dense_of.get(layer).copied().flatten()
    }

    pub fn num_slots(&self) -> usize {
        self.slots as usize
    }

    /// Bytes the state slabs hold, for the config's memory arithmetic.
    pub fn bytes(&self) -> usize {
        (self.conv.numel() + self.recurrent.numel()) * Kind::Float.elt_size_in_bytes()
    }

    pub fn conv_size(&self) -> Vec<i64> {
        vec![self.conv_dim, self.width]
    }

    pub fn recurrent_size(&self) -> Vec<i64> {
        vec![self.v_heads, self.k_dim, self.v_dim]
    }

    /// The slot's conv state for this layer, zero before anything is written.
    pub fn conv_state(&self, slot: u32, layer: usize) -> Tensor {
        self.conv
            .select(0, self.slab(layer))
            .select(0, slot as i64)
            .shallow_clone()
    }

    pub fn set_conv_state(&self, slot: u32, layer: usize, state: Tensor) {
        let mut view = self
            .conv
            .select(0, self.slab(layer))
            .select(0, slot as i64)
            .shallow_clone();
        view.copy_(&state.view(&self.conv_size()[..]));
    }

    /// The slot's recurrent state for this layer, zero before anything is
    /// written.
    pub fn recurrent_state(&self, slot: u32, layer: usize) -> Tensor {
        self.recurrent
            .select(0, self.slab(layer))
            .select(0, slot as i64)
            .shallow_clone()
    }

    pub fn set_recurrent_state(&self, slot: u32, layer: usize, state: Tensor) {
        let mut view = self
            .recurrent
            .select(0, self.slab(layer))
            .select(0, slot as i64)
            .shallow_clone();
        view.copy_(&state.view(&self.recurrent_size()[..]));
    }

    /// The rows' conv state, `[rows, conv_dim, width]`, addressed by a device
    /// `[rows]` slot index: a batched decode step reads the slots it staged
    /// rather than host numbers (decision 51).
    pub fn conv_state_rows(&self, slots: &Tensor, layer: usize) -> Tensor {
        self.conv
            .select(0, self.slab(layer))
            .index_select(0, &slots.to_kind(Kind::Int64))
    }

    /// `set_conv_state` for a device `[rows]` slot index; `state` is
    /// `[rows, conv_dim, width]`.
    pub fn set_conv_state_rows(&self, slots: &Tensor, layer: usize, state: &Tensor) {
        let mut view = self.conv.select(0, self.slab(layer)).shallow_clone();
        let _ = view.index_copy_(0, &slots.to_kind(Kind::Int64), &state.to_kind(Kind::Float));
    }

    /// The rows' recurrent state, `[rows, v_heads, k_dim, v_dim]`, addressed by a
    /// device `[rows]` slot index.
    pub fn recurrent_state_rows(&self, slots: &Tensor, layer: usize) -> Tensor {
        self.recurrent
            .select(0, self.slab(layer))
            .index_select(0, &slots.to_kind(Kind::Int64))
    }

    /// `set_recurrent_state` for a device `[rows]` slot index; `state` is
    /// `[rows, v_heads, k_dim, v_dim]`.
    pub fn set_recurrent_state_rows(&self, slots: &Tensor, layer: usize, state: &Tensor) {
        let mut view = self.recurrent.select(0, self.slab(layer)).shallow_clone();
        let _ = view.index_copy_(0, &slots.to_kind(Kind::Int64), &state.to_kind(Kind::Float));
    }

    /// Zero a whole slot, so a recycled slot cannot leak the previous request's
    /// state into its first prefill chunk.
    pub fn reset_slot(&self, slot: u32) {
        for layer in 0..self.dense_of.len() {
            if self.dense_of[layer].is_some() {
                self.set_conv_state(
                    slot,
                    layer,
                    Tensor::zeros(self.conv_size(), (Kind::Float, self.conv.device())),
                );
                self.set_recurrent_state(
                    slot,
                    layer,
                    Tensor::zeros(
                        self.recurrent_size(),
                        (Kind::Float, self.recurrent.device()),
                    ),
                );
            }
        }
    }

    /// Copy a whole slot's conv + recurrent state (all linear layers) from `src`
    /// to `dst`. The hybrid-radix currency: a donation snapshots a live slot into
    /// a tree-owned slot, and a restore copies a snapshot back into a fresh live
    /// slot (copy-on-write, so the donated snapshot stays intact).
    pub fn copy_slot(&self, src: u32, dst: u32) {
        debug_assert_ne!(src, dst, "copy_slot is not a reset");
        for layer in 0..self.dense_of.len() {
            if self.dense_of[layer].is_some() {
                self.conv
                    .select(0, self.slab(layer))
                    .select(0, dst as i64)
                    .shallow_clone()
                    .copy_(&self.conv.select(0, self.slab(layer)).select(0, src as i64));
                self.recurrent
                    .select(0, self.slab(layer))
                    .select(0, dst as i64)
                    .shallow_clone()
                    .copy_(
                        &self
                            .recurrent
                            .select(0, self.slab(layer))
                            .select(0, src as i64),
                    );
            }
        }
    }

    fn slab(&self, layer: usize) -> i64 {
        self.dense_of[layer].expect("layer has no linear state") as i64
    }
}

/// The device cache the engine hands out from: the two slabs plus the page and
/// state-slot free lists that make them a cache rather than scratch.
///
/// The free lists are behind mutexes and the tensors behind shared borrows, so
/// the engine can hold one `DeviceCache` for every session while each session's
/// forward borrows the slabs for the duration of its step.
///
/// It also owns the hybrid prefix tree (`HybridRadixIndex`): the pure radix
/// bookkeeping that lets a request reuse another's KV pages and GDN state. The
/// tree stores page and slot ids only; this adapter is what moves the actual
/// device bytes (`copy_slot`) and drives the two pools.
pub struct DeviceCache {
    pages: Mutex<PagePool>,
    slots: Mutex<LinearStatePool>,
    tree: Mutex<HybridRadixIndex>,
    slabs: SharedSlabs,
}

/// `tch::Tensor` is `Send` but not `Sync`, and the engine holds the model (and
/// through it this cache) behind an `Arc` on its own thread pool. libtorch is
/// thread-safe; the only thing being shared is the handle, and the engine runs
/// one forward at a time. Same newtype the dense weights already use.
struct SharedSlabs {
    mha: MhaPool,
    state: LinearStateSlabs,
}

unsafe impl Sync for SharedSlabs {}

impl DeviceCache {
    pub fn new(
        params: &ModelParams,
        num_pages: usize,
        num_slots: usize,
        dtype: KvDType,
        device: Device,
    ) -> Self {
        Self {
            pages: Mutex::new(PagePool::new(PAGE_SIZE as usize, num_pages)),
            slots: Mutex::new(LinearStatePool::new(num_slots)),
            tree: Mutex::new(HybridRadixIndex::new(PAGE_SIZE as usize)),
            slabs: SharedSlabs {
                mha: MhaPool::new(params, num_pages, dtype, device),
                state: LinearStateSlabs::new(params, num_slots, device),
            },
        }
    }

    pub fn mha(&self) -> &MhaPool {
        &self.slabs.mha
    }

    pub fn state(&self) -> &LinearStateSlabs {
        &self.slabs.state
    }

    /// Bytes the whole cache holds, for the config's memory arithmetic.
    pub fn bytes(&self) -> usize {
        self.slabs.mha.bytes() + self.slabs.state.bytes()
    }

    /// Return an eviction's spoils to the two pools: KV locations to the page
    /// pool, GDN state slots to the slot pool.
    fn free_evicted(&self, ev: EvictResult) {
        if !ev.kv.is_empty() {
            self.pages.lock().expect("kvcache lock").free_pages(ev.kv);
        }
        if !ev.mamba_slots.is_empty() {
            let mut slots = self.slots.lock().expect("kvcache lock");
            for slot in ev.mamba_slots {
                slots.release(slot);
            }
        }
    }

    /// Allocate a GDN state slot, evicting unlocked tree snapshots (LRU) until
    /// one is free. Panics only if the pool is sized below the live
    /// concurrency, which is a config error.
    fn alloc_slot(&self) -> u32 {
        loop {
            let free = { self.slots.lock().expect("kvcache lock").allocate() };
            if let Some(slot) = free {
                return slot;
            }
            let ev = self.tree.lock().expect("kvcache lock").evict_mamba(1);
            if ev.mamba_slots.is_empty() && ev.kv.is_empty() {
                panic!("state slots exhausted: size the cache for the request concurrency");
            }
            self.free_evicted(ev);
        }
    }

    /// Reserve `n` pages for `cache`, evicting the tree's unlocked KV leaves when
    /// the page pool runs short. Eviction also returns each evicted leaf's
    /// snapshot slot.
    fn reserve_pages(&self, cache: &mut SequenceCache, n: usize) -> Result<(), KvcacheError> {
        {
            let mut pages = self.pages.lock().expect("kvcache lock");
            if cache.grow(n, &mut pages).is_ok() {
                return Ok(());
            }
        }
        loop {
            let have = self.pages.lock().expect("kvcache lock").available();
            if have >= n {
                break;
            }
            let ev = self.tree.lock().expect("kvcache lock").evict_full(n - have);
            if ev.kv.is_empty() && ev.mamba_slots.is_empty() {
                break;
            }
            self.free_evicted(ev);
        }
        cache.grow(n, &mut self.pages.lock().expect("kvcache lock"))
    }
}

impl KvCache for DeviceCache {
    fn begin(&self) -> SequenceCache {
        let slot = self.alloc_slot();
        // A recycled slot still holds the last request's GDN state; zero it, or
        // the new request's first chunk would continue someone else's. A reused
        // prefix overwrites it with the restored snapshot in `adopt_prefix`.
        self.slabs.state.reset_slot(slot);
        SequenceCache::new(slot)
    }

    fn reserve(&self, cache: &mut SequenceCache, n: usize) -> Result<(), KvcacheError> {
        self.reserve_pages(cache, n)
    }

    fn end(&self, cache: &mut SequenceCache) {
        cache.release(&mut self.pages.lock().expect("kvcache lock"));
        if let Some(node) = cache.take_lock() {
            self.tree.lock().expect("kvcache lock").dec_lock(node);
        }
        if !cache.donated() {
            self.slots
                .lock()
                .expect("kvcache lock")
                .release(cache.state_slot());
        }
    }

    fn available(&self) -> usize {
        self.pages.lock().expect("kvcache lock").available()
    }

    fn num_pages(&self) -> usize {
        self.pages.lock().expect("kvcache lock").total_pages()
    }

    fn adopt_prefix(&self, cache: &mut SequenceCache, ids: &[u32]) -> usize {
        if ids.is_empty() {
            return 0;
        }
        let (node, snapshot, canonical) = {
            let mut tree = self.tree.lock().expect("kvcache lock");
            let m = tree.match_prefix(ids);
            match m.mamba_value {
                Some(snapshot) if m.cached_len > 0 => {
                    tree.inc_lock(m.node);
                    (m.node, snapshot, m.kv_indices)
                }
                _ => return 0,
            }
        };
        // Copy-on-write the snapshot into this request's live slot: the tree
        // keeps its copy, the request resumes the GDN recurrence from here.
        self.slabs.state.copy_slot(snapshot, cache.state_slot());
        cache.adopt_prefix(&canonical);
        cache.set_lock(Some(node));
        canonical.len()
    }

    fn commit(&self, cache: &mut SequenceCache, ids: &[u32], finished: bool) {
        let cached_len = cache.cached();
        if cached_len == 0 || ids.len() < cached_len {
            if let Some(node) = cache.take_lock() {
                self.tree.lock().expect("kvcache lock").dec_lock(node);
            }
            return;
        }
        let old_reused = cache.reused();
        let keys = &ids[..cached_len];
        // The donated slot: the live one for a finished request (zero-copy, the
        // request is done), else a COW copy so the live slot keeps decoding.
        let donated_slot = if finished {
            cache.state_slot()
        } else {
            let slot = self.alloc_slot();
            self.slabs.state.copy_slot(cache.state_slot(), slot);
            slot
        };
        let (prefix_len, mamba_exist, node, canonical) = {
            let mut tree = self.tree.lock().expect("kvcache lock");
            let ins = tree.insert(keys, cache.filled(), donated_slot);
            let m = tree.match_prefix(keys);
            (ins.prefix_len, ins.mamba_exist, m.node, m.kv_indices)
        };
        if mamba_exist {
            // The boundary already owns a snapshot (dedup): drop the donated
            // copy, or, when finished, let `end` return the live slot.
            if !finished {
                self.slots
                    .lock()
                    .expect("kvcache lock")
                    .release(donated_slot);
            }
        } else if finished {
            // The tree owns the live slot; `end` must not return it.
            cache.set_donated(true);
        }
        // The request's own locations in [old_reused, prefix_len) already exist
        // in the tree. Free them; if the request keeps decoding, re-point the
        // row at the tree's canonical pages first (attention still reads it).
        if prefix_len > old_reused {
            let dup = cache.filled()[old_reused..prefix_len].to_vec();
            if !finished {
                cache.repoint(old_reused, prefix_len, &canonical);
            }
            self.pages.lock().expect("kvcache lock").free_pages(dup);
        }
        if let Some(old) = cache.take_lock() {
            self.tree.lock().expect("kvcache lock").dec_lock(old);
        }
        if finished {
            cache.clear();
        } else {
            self.tree.lock().expect("kvcache lock").inc_lock(node);
            cache.set_lock(Some(node));
            // The insert adopted the whole committed span, so the tree now owns
            // every leading location, not just the pre-existing prefix.
            cache.set_reused(cached_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::kvcache::KvcacheError;
    use crate::infrastructure::torch::{auto_device, f32_tensor, to_f32_vec};

    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 9) as f32 / (1 << 23) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Two full-attention layers and two linear ones, so the remap is exercised
    /// on a hybrid tower without a checkpoint.
    fn params() -> ModelParams {
        ModelParams {
            hidden_size: 8,
            num_layers: 4,
            vocab_size: 16,
            num_experts: 4,
            top_k: 2,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            rotary_dim: 4,
            rope_theta: 1e7,
            linear_key_head_dim: 4,
            linear_num_key_heads: 2,
            linear_value_head_dim: 4,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 4,
            gdn_state_recurrent_dtype: crate::domain::weights::DType::F32,
            rms_norm_eps: 1e-6,
            layer_types: vec![
                LayerType::FullAttention,
                LayerType::LinearAttention,
                LayerType::LinearAttention,
                LayerType::FullAttention,
            ],
            max_position_embeddings: 4096,
            eos_token_id: 0,
            bos_token_id: 0,
            expert_slot_bytes: 1,
        }
    }

    /// Values already rounded to bf16, so a store/gather round trip is exact
    /// rather than close: the pool promises it hands back what it was given.
    fn kv_pair(t: usize, kvh: i64, hd: i64, seed: u32) -> (Tensor, Tensor) {
        let d = auto_device();
        let n = t * kvh as usize * hd as usize;
        let shape = [t as i64, kvh, hd];
        (
            f32_tensor(&data(n, seed), &shape, d).to_kind(Kind::BFloat16),
            f32_tensor(&data(n, seed + 1), &shape, d).to_kind(Kind::BFloat16),
        )
    }

    #[test]
    fn only_full_attention_layers_get_a_slab() {
        let pool = MhaPool::new(&params(), 16, KvDType::F32, auto_device());
        assert_eq!(pool.dense_of(0), Some(0));
        assert_eq!(pool.dense_of(1), None, "a linear layer has no kv");
        assert_eq!(pool.dense_of(3), Some(1));
        assert_eq!(pool.num_pages(), 16);
    }

    #[test]
    fn a_stored_location_gathers_back_exactly() {
        let pool = MhaPool::new(&params(), 16, KvDType::F32, auto_device());
        let (k, v) = kv_pair(3, 2, 4, 61);
        pool.store(0, &[5, 2, 9], &k, &v);
        let (gk, gv) = pool.gather(0, &[5, 2, 9]);
        assert_eq!(to_f32_vec(&gk), to_f32_vec(&k));
        assert_eq!(to_f32_vec(&gv), to_f32_vec(&v));
    }

    #[test]
    fn gather_reads_locations_in_the_order_asked() {
        let pool = MhaPool::new(&params(), 16, KvDType::F32, auto_device());
        let (k, v) = kv_pair(3, 2, 4, 63);
        pool.store(0, &[5, 2, 9], &k, &v);
        let (gk, _) = pool.gather(0, &[9, 5, 2]);
        let rows = |t: &Tensor| {
            let f = to_f32_vec(t);
            f[..8].to_vec()
        };
        assert_eq!(rows(&gk), rows(&k.select(0, 2)));
    }

    #[test]
    fn a_later_store_replaces_the_earlier_one_at_the_same_location() {
        let pool = MhaPool::new(&params(), 16, KvDType::F32, auto_device());
        let (k, v) = kv_pair(2, 2, 4, 65);
        pool.store(0, &[1, 2], &k, &v);
        let (k2, v2) = kv_pair(1, 2, 4, 67);
        pool.store(0, &[2], &k2, &v2);
        let (gk, gv) = pool.gather(0, &[2]);
        assert_eq!(to_f32_vec(&gk), to_f32_vec(&k2));
        assert_eq!(to_f32_vec(&gv), to_f32_vec(&v2));
    }

    #[test]
    fn one_layers_slab_does_not_disturb_another() {
        let pool = MhaPool::new(&params(), 16, KvDType::F32, auto_device());
        let (k, v) = kv_pair(2, 2, 4, 69);
        pool.store(0, &[3, 4], &k, &v);
        let (k2, v2) = kv_pair(2, 2, 4, 71);
        pool.store(3, &[3, 4], &k2, &v2);
        assert_eq!(to_f32_vec(&pool.gather(0, &[3, 4]).0), to_f32_vec(&k));
        assert_eq!(to_f32_vec(&pool.gather(3, &[3, 4]).0), to_f32_vec(&k2));
    }

    #[test]
    fn an_unwritten_location_reads_as_zero() {
        let pool = MhaPool::new(&params(), 8, KvDType::F32, auto_device());
        let (gk, gv) = pool.gather(0, &[7]);
        assert_eq!(to_f32_vec(&gk), vec![0.0; 8]);
        assert_eq!(to_f32_vec(&gv), vec![0.0; 8]);
    }

    #[test]
    fn a_bf16_pool_hands_back_the_value_rounded_to_bf16() {
        let pool = MhaPool::new(&params(), 8, KvDType::BFloat16, auto_device());
        let n = 2 * 2 * 4;
        let want = data(n, 83);
        let k = f32_tensor(&want, &[2, 2, 4], auto_device());
        pool.store(0, &[4, 6], &k, &k);
        let (gk, _) = pool.gather(0, &[4, 6]);
        assert_eq!(
            gk.kind(),
            Kind::Float,
            "a bf16 pool must gather back as f32 for the attention math"
        );
        assert_eq!(
            to_f32_vec(&gk),
            to_f32_vec(&k.to_kind(Kind::BFloat16)),
            "a lossy pool must round, not truncate"
        );
        assert_ne!(to_f32_vec(&gk), want);
    }

    #[test]
    fn the_kv_slab_pointers_step_by_one_layer_of_tokens() {
        let p = params();
        let pool = MhaPool::new(&p, 7, KvDType::F32, auto_device());
        let (k0, v0) = pool.kv_slab_ptrs(0);
        let (k1, _) = pool.kv_slab_ptrs(3);
        // Two dense slabs in the tower, K then V, so V starts two layers in.
        let per = 7 * PAGE_SIZE * p.num_kv_heads as i64 * p.head_dim as i64 * 4;
        assert_eq!(v0 - k0, 2 * per as u64, "V follows both K slabs");
        assert_eq!(k1 - k0, per as u64, "slab 1 is one layer after slab 0");
        assert_eq!(pool.dtype(), KvDType::F32);
    }

    #[test]
    fn a_device_cache_hands_out_pages_and_state_slots() {
        let cache = DeviceCache::new(&params(), 8, 2, KvDType::F32, auto_device());
        assert_eq!(cache.num_pages(), 8);
        assert_eq!(cache.available(), 8);

        let mut a = cache.begin();
        cache.reserve(&mut a, 3).unwrap();
        assert_eq!(a.cached(), 3);
        assert_eq!(cache.available(), 5);
        assert_eq!(a.state_slot(), 0);

        let mut b = cache.begin();
        cache.reserve(&mut b, 2).unwrap();
        assert_eq!(
            b.state_slot(),
            1,
            "a second request gets its own state slot"
        );

        cache.end(&mut a);
        assert_eq!(a.cached(), 0);
        assert_eq!(
            cache.available(),
            6,
            "only a's pages come back; b still holds its two"
        );

        cache.end(&mut b);
        assert_eq!(
            cache.available(),
            8,
            "every ended request returns its pages"
        );

        let mut c = cache.begin();
        cache.reserve(&mut c, 1).unwrap();
        assert!(
            c.state_slot() < 2,
            "an ended request's slot goes back in play"
        );
        assert_eq!(c.cached(), 1);
    }

    #[test]
    fn a_device_cache_refuses_to_reserve_past_its_pages() {
        let cache = DeviceCache::new(&params(), 4, 1, KvDType::F32, auto_device());
        let mut a = cache.begin();
        cache.reserve(&mut a, 3).unwrap();
        assert_eq!(
            cache.reserve(&mut a, 2),
            Err(KvcacheError::OutOfPages { want: 2, have: 1 })
        );
        assert_eq!(a.cached(), 3, "a refused reserve must not reserve anything");
    }

    #[test]
    fn a_recycled_state_slot_is_zeroed() {
        let cache = DeviceCache::new(&params(), 8, 1, KvDType::F32, auto_device());
        let a_slot = {
            let a = cache.begin();
            cache.state().set_recurrent_state(
                a.state_slot(),
                1,
                f32_tensor(&[1.0; 32], &[2, 4, 4], auto_device()),
            );
            a.state_slot()
        };
        let mut a = SequenceCache::new(a_slot);
        cache.end(&mut a);
        let b = cache.begin();
        assert_eq!(b.state_slot(), a.state_slot());
        assert!(
            to_f32_vec(&cache.state().recurrent_state(b.state_slot(), 1))
                .iter()
                .all(|v| *v == 0.0),
            "a recycled slot must not leak the previous request's state"
        );
    }

    #[test]
    fn only_linear_layers_hold_linear_state() {
        let s = LinearStateSlabs::new(&params(), 4, auto_device());
        assert_eq!(s.dense_of(0), None);
        assert_eq!(s.dense_of(1), Some(0));
        assert_eq!(s.dense_of(2), Some(1));
        assert_eq!(s.dense_of(3), None);
        assert_eq!(s.num_slots(), 4);
    }

    #[test]
    fn a_linear_state_slot_starts_at_zero_and_round_trips() {
        let s = LinearStateSlabs::new(&params(), 2, auto_device());
        let conv = s.conv_state(0, 1);
        assert_eq!(conv.size(), vec![2 * 2 * 4 + 2 * 4, 3]);
        assert!(to_f32_vec(&conv).iter().all(|v| *v == 0.0));
        let rec = s.recurrent_state(0, 1);
        assert_eq!(rec.size(), vec![2, 4, 4]);

        let want_conv = data(conv.numel(), 73);
        s.set_conv_state(
            0,
            1,
            f32_tensor(&want_conv, &conv.size()[..], auto_device()),
        );
        assert_eq!(to_f32_vec(&s.conv_state(0, 1)), want_conv);

        let want_rec = data(rec.numel(), 75);
        s.set_recurrent_state(0, 1, f32_tensor(&want_rec, &rec.size()[..], auto_device()));
        assert_eq!(to_f32_vec(&s.recurrent_state(0, 1)), want_rec);
    }

    #[test]
    fn linear_state_slots_are_independent_and_resettable() {
        let s = LinearStateSlabs::new(&params(), 2, auto_device());
        let one = vec![1.0f32; 32];
        s.set_recurrent_state(0, 1, f32_tensor(&one, &[2, 4, 4], auto_device()));
        assert_eq!(to_f32_vec(&s.recurrent_state(1, 1)), vec![0.0; 32]);
        assert_eq!(to_f32_vec(&s.recurrent_state(0, 1)), one);
        s.reset_slot(0);
        assert!(
            to_f32_vec(&s.recurrent_state(0, 1))
                .iter()
                .all(|v| *v == 0.0),
            "a recycled slot must not leak the previous request's state"
        );
    }

    #[test]
    fn a_linear_state_slot_copies_conv_and_recurrent_to_another() {
        let s = LinearStateSlabs::new(&params(), 2, auto_device());
        let cv = s.conv_size();
        let rv = s.recurrent_size();
        // Both linear layers carry distinct state on slot 0.
        for layer in [1usize, 2] {
            s.set_conv_state(
                0,
                layer,
                f32_tensor(
                    &data((cv[0] * cv[1]) as usize, 81 + layer as u32),
                    &cv[..],
                    auto_device(),
                ),
            );
            s.set_recurrent_state(
                0,
                layer,
                f32_tensor(
                    &data((rv[0] * rv[1] * rv[2]) as usize, 91 + layer as u32),
                    &rv[..],
                    auto_device(),
                ),
            );
        }
        s.copy_slot(0, 1);
        for layer in [1usize, 2] {
            assert_eq!(
                to_f32_vec(&s.conv_state(1, layer)),
                to_f32_vec(&s.conv_state(0, layer)),
                "the copy matches the source conv"
            );
            assert_eq!(
                to_f32_vec(&s.recurrent_state(1, layer)),
                to_f32_vec(&s.recurrent_state(0, layer)),
                "the copy matches the source recurrent"
            );
        }
        // Copy-on-write: writing the live slot afterwards leaves the snapshot.
        let before = to_f32_vec(&s.recurrent_state(1, 1));
        s.set_recurrent_state(
            0,
            1,
            f32_tensor(&vec![7.0f32; before.len()], &rv[..], auto_device()),
        );
        assert_eq!(
            to_f32_vec(&s.recurrent_state(1, 1)),
            before,
            "a snapshot slot is not aliased by a later live write"
        );
    }

    /// A device index tensor, as a captured step would stage one.
    fn dev_index(values: &[i64]) -> Tensor {
        Tensor::from_slice(values).to(auto_device().to_tch())
    }

    #[test]
    fn a_device_index_store_and_gather_match_the_host_path() {
        let pool = MhaPool::new(&params(), 16, KvDType::F32, auto_device());
        let (k, v) = kv_pair(2, 2, 4, 91);
        let locs = dev_index(&[5, 2]);
        pool.store_at(0, &locs, &k, &v);
        let (gk, gv) = pool.gather_rows(0, &locs.reshape([1, 2]));
        assert_eq!(to_f32_vec(&gk), to_f32_vec(&k));
        assert_eq!(to_f32_vec(&gv), to_f32_vec(&v));
        // The host accessor reads back the very values the device path wrote.
        assert_eq!(to_f32_vec(&pool.gather(0, &[5, 2]).0), to_f32_vec(&k));
    }

    #[test]
    fn a_batched_gather_reads_each_rows_window_in_order() {
        let pool = MhaPool::new(&params(), 16, KvDType::F32, auto_device());
        let (k, v) = kv_pair(4, 2, 4, 97);
        // Token `r` lands at pool location `r`, so the window can name rows back.
        pool.store_at(0, &dev_index(&[0, 1, 2, 3]), &k, &v);
        let windows = dev_index(&[2, 0, 3, 1]).reshape([2, 2]);
        let (gk, _) = pool.gather_rows(0, &windows);
        assert_eq!(gk.size(), vec![2, 2, 2, 4]);
        let flat = to_f32_vec(&gk);
        let want: Vec<f32> = [2i64, 0, 3, 1]
            .iter()
            .flat_map(|&r| to_f32_vec(&k.select(0, r)))
            .collect();
        assert_eq!(flat, want, "row 0 then row 1, each window in order");
    }

    #[test]
    fn a_device_slot_reads_and_writes_the_same_state_as_the_host_slot() {
        let s = LinearStateSlabs::new(&params(), 3, auto_device());
        let slot = dev_index(&[1]);
        let cv = s.conv_size();
        let conv = f32_tensor(&data((cv[0] * cv[1]) as usize, 93), &cv[..], auto_device());
        s.set_conv_state_rows(&slot, 1, &conv.unsqueeze(0));
        assert_eq!(
            to_f32_vec(&s.conv_state_rows(&slot, 1).select(0, 0)),
            to_f32_vec(&conv)
        );
        assert_eq!(
            to_f32_vec(&s.conv_state(1, 1)),
            to_f32_vec(&conv),
            "the host accessor sees the device write"
        );

        let rv = s.recurrent_size();
        let rec = f32_tensor(
            &data((rv[0] * rv[1] * rv[2]) as usize, 95),
            &rv[..],
            auto_device(),
        );
        s.set_recurrent_state_rows(&slot, 1, &rec.unsqueeze(0));
        assert_eq!(
            to_f32_vec(&s.recurrent_state_rows(&slot, 1).select(0, 0)),
            to_f32_vec(&rec)
        );
        assert_eq!(to_f32_vec(&s.recurrent_state(1, 1)), to_f32_vec(&rec));
        assert!(
            to_f32_vec(&s.recurrent_state_rows(&dev_index(&[0]), 1))
                .iter()
                .all(|v| *v == 0.0),
            "the device accessor never aliases another slot"
        );
    }

    /// Mark a slot's first linear layer so a restore can be told apart from a
    /// zeroed fresh slot.
    fn stamp(s: &LinearStateSlabs, slot: u32, value: f32) {
        let n = s.recurrent_size().iter().product::<i64>() as usize;
        s.set_recurrent_state(
            slot,
            1,
            f32_tensor(&vec![value; n], &s.recurrent_size()[..], auto_device()),
        );
    }

    fn stamp_of(cache: &DeviceCache, slot: u32) -> Vec<f32> {
        to_f32_vec(&cache.state().recurrent_state(slot, 1))
    }

    #[test]
    fn a_committed_prefix_is_reused_by_the_next_request() {
        let cache = DeviceCache::new(&params(), 32, 4, KvDType::F32, auto_device());
        let mut a = cache.begin();
        assert_eq!(
            cache.adopt_prefix(&mut a, &[1, 2, 3]),
            0,
            "the empty tree is cold"
        );
        cache.reserve(&mut a, 4).unwrap();
        let a_pages = a.filled().to_vec();
        stamp(cache.state(), a.state_slot(), 5.0);
        cache.commit(&mut a, &[1, 2, 3, 4], true);
        cache.end(&mut a);

        let mut b = cache.begin();
        assert_eq!(
            cache.adopt_prefix(&mut b, &[1, 2, 3, 4]),
            4,
            "the whole committed prefix matches"
        );
        assert_eq!(
            b.filled(),
            &a_pages[..],
            "b reads the tree's canonical pages, not its own"
        );
        assert!(
            stamp_of(&cache, b.state_slot()).iter().all(|v| *v == 5.0),
            "b resumes from a's donated GDN state"
        );
        cache.end(&mut b);
    }

    #[test]
    fn a_prefix_shorter_than_a_snapshot_boundary_reuses_nothing() {
        let cache = DeviceCache::new(&params(), 32, 4, KvDType::F32, auto_device());
        let mut a = cache.begin();
        cache.reserve(&mut a, 4).unwrap();
        stamp(cache.state(), a.state_slot(), 7.0);
        cache.commit(&mut a, &[1, 2, 3, 4], true);
        cache.end(&mut a);

        let mut b = cache.begin();
        assert_eq!(
            cache.adopt_prefix(&mut b, &[1, 2, 3]),
            0,
            "a strict prefix of a snapshot node has no resumable state"
        );
        assert!(
            stamp_of(&cache, b.state_slot()).iter().all(|v| *v == 0.0),
            "a cold start keeps the zeroed slot"
        );
        cache.end(&mut b);
    }

    #[test]
    fn a_prompt_end_commit_lets_a_follower_reuse_the_prompt() {
        let cache = DeviceCache::new(&params(), 64, 4, KvDType::F32, auto_device());
        let mut a = cache.begin();
        cache.reserve(&mut a, 4).unwrap();
        stamp(cache.state(), a.state_slot(), 3.0);
        cache.commit(&mut a, &[1, 2, 3, 4], false);
        assert_eq!(a.cached(), 4, "a keeps decoding over its own pages");
        assert_eq!(a.reused(), 4, "the tree now owns a's prompt");
        cache.reserve(&mut a, 1).unwrap();
        cache.commit(&mut a, &[1, 2, 3, 4, 5], true);
        cache.end(&mut a);

        let mut b = cache.begin();
        assert_eq!(
            cache.adopt_prefix(&mut b, &[1, 2, 3, 4]),
            4,
            "the prompt boundary is reusable"
        );
        assert!(
            stamp_of(&cache, b.state_slot()).iter().all(|v| *v == 3.0),
            "b resumes from the prompt-end snapshot"
        );
        cache.end(&mut b);
    }

    #[test]
    fn pages_and_slots_are_conserved_across_reuse() {
        let cache = DeviceCache::new(&params(), 32, 4, KvDType::F32, auto_device());
        let total_pages = cache.num_pages();
        let mut a = cache.begin();
        cache.reserve(&mut a, 6).unwrap();
        stamp(cache.state(), a.state_slot(), 1.0);
        cache.commit(&mut a, &[1, 2, 3, 4, 5, 6], true);
        cache.end(&mut a);

        // A second request reuses the prefix, adds a tail, and commits it.
        let mut b = cache.begin();
        assert_eq!(cache.adopt_prefix(&mut b, &[1, 2, 3, 4, 5, 6]), 6);
        cache.reserve(&mut b, 2).unwrap();
        cache.commit(&mut b, &[1, 2, 3, 4, 5, 6, 7, 8], true);
        cache.end(&mut b);

        // The tree owns eight pages and one snapshot; everything else is free.
        assert_eq!(cache.available(), total_pages - 8);
    }

    #[test]
    fn adopting_the_same_prefix_twice_shares_one_tree_snapshot() {
        let cache = DeviceCache::new(&params(), 32, 4, KvDType::F32, auto_device());
        let mut a = cache.begin();
        cache.reserve(&mut a, 4).unwrap();
        stamp(cache.state(), a.state_slot(), 9.0);
        cache.commit(&mut a, &[1, 2, 3, 4], true);
        cache.end(&mut a);

        let mut b = cache.begin();
        assert_eq!(cache.adopt_prefix(&mut b, &[1, 2, 3, 4]), 4);
        let mut c = cache.begin();
        assert_eq!(
            cache.adopt_prefix(&mut c, &[1, 2, 3, 4]),
            4,
            "two live requests share the same snapshot"
        );
        assert_ne!(
            b.state_slot(),
            c.state_slot(),
            "each restores into its own slot"
        );
        assert!(stamp_of(&cache, b.state_slot()).iter().all(|v| *v == 9.0));
        assert!(stamp_of(&cache, c.state_slot()).iter().all(|v| *v == 9.0));
        cache.end(&mut b);
        cache.end(&mut c);
    }

    #[test]
    fn a_prompt_end_snapshot_shares_pages_with_the_still_decoding_request() {
        let cache = DeviceCache::new(&params(), 32, 4, KvDType::F32, auto_device());
        let mut a = cache.begin();
        cache.reserve(&mut a, 4).unwrap();
        stamp(cache.state(), a.state_slot(), 2.0);
        cache.commit(&mut a, &[1, 2, 3, 4], false);
        let a_pages = a.filled().to_vec();

        // A second request adopts the same prefix while a is still decoding.
        let mut b = cache.begin();
        assert_eq!(cache.adopt_prefix(&mut b, &[1, 2, 3, 4]), 4);
        assert_eq!(b.filled(), &a_pages[..], "both point at the tree's pages");

        // a keeps decoding and finishes over a longer chain.
        cache.reserve(&mut a, 1).unwrap();
        cache.commit(&mut a, &[1, 2, 3, 4, 5], true);
        cache.end(&mut a);

        // b finishes over a different tail without losing its adopted pages.
        cache.reserve(&mut b, 1).unwrap();
        cache.commit(&mut b, &[1, 2, 3, 4, 6], true);
        cache.end(&mut b);

        assert_eq!(
            cache.available(),
            32 - 6,
            "the shared prefix and both tails are the tree's six pages"
        );
    }
}
