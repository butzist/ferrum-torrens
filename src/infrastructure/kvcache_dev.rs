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

use crate::domain::kvcache::{
    EvictResult, HybridRadixIndex, KvCache, KvcacheError, LinearStatePool, PagePool, SequenceCache,
};
use crate::domain::model::{LayerType, ModelParams};
use crate::infrastructure::cudarc::{Runtime, Tensor, TensorDType};
use crate::infrastructure::torch::Device;

/// ALGORITHM.md 3.1: one token per page.
pub const PAGE_SIZE: usize = 1;

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
    fn tensor_dtype(self) -> TensorDType {
        match self {
            KvDType::F32 => TensorDType::F32,
            KvDType::BFloat16 => TensorDType::Bf16,
        }
    }
}

fn runtime(device: Device) -> Runtime {
    let ordinal = match device {
        Device::Cuda(ordinal) => ordinal,
        Device::Cpu => 0,
    };
    Runtime::open(ordinal).expect("open the cudarc compute runtime")
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
    kv_heads: usize,
    head_dim: usize,
    dtype: KvDType,
    rt: Runtime,
}

impl MhaPool {
    pub fn new(params: &ModelParams, num_pages: usize, dtype: KvDType, device: Device) -> Self {
        Self::with_runtime(params, num_pages, dtype, runtime(device))
    }

    fn with_runtime(params: &ModelParams, num_pages: usize, dtype: KvDType, rt: Runtime) -> Self {
        let dense_of = dense_map(params, LayerType::FullAttention);
        let layers = dense_of.iter().filter(|d| d.is_some()).count();
        let (kv_heads, head_dim) = (params.num_kv_heads, params.head_dim);
        let kv = rt
            .zeros(
                &[2, layers, num_pages, PAGE_SIZE, kv_heads, head_dim],
                dtype.tensor_dtype(),
            )
            .expect("allocate the kv pool");
        Self {
            kv,
            dense_of,
            kv_heads,
            head_dim,
            dtype,
            rt,
        }
    }

    /// The slab index for a global layer id, or `None` for a linear-attention
    /// layer, which has no KV slab.
    pub fn dense_of(&self, layer: usize) -> Option<usize> {
        self.dense_of.get(layer).copied().flatten()
    }

    pub fn num_pages(&self) -> usize {
        self.kv.shape()[2]
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
        let k = self
            .kv
            .select(0, 0)
            .expect("kv half")
            .select(0, slab)
            .expect("kv slab");
        let v = self
            .kv
            .select(0, 1)
            .expect("kv half")
            .select(0, slab)
            .expect("kv slab");
        (k.device_ptr(), v.device_ptr())
    }

    /// Bytes the pool holds, for the config's memory arithmetic.
    pub fn bytes(&self) -> usize {
        self.kv.numel() * self.dtype.tensor_dtype().element_size()
    }

    /// Write this step's `k`/`v` (`[t, kv_heads, head_dim]`) at `locs`.
    pub fn store(&self, layer: usize, locs: &[u32], k: &Tensor, v: &Tensor) {
        let idx = self.index(locs);
        self.store_indexed(layer, &idx, locs.len(), k, v);
    }

    /// `store`, but the locations are already a device `[t]` i32 index tensor.
    ///
    /// A captured decode step bakes the address of the index tensor, so the
    /// locations live on the device rather than being rebuilt host-side each
    /// step (decision 51).
    pub fn store_at(&self, layer: usize, locs: &Tensor, k: &Tensor, v: &Tensor) {
        self.store_indexed(layer, locs, locs.numel(), k, v);
    }

    fn store_indexed(&self, layer: usize, idx: &Tensor, t: usize, k: &Tensor, v: &Tensor) {
        let slab = self.slab(layer);
        debug_assert_eq!(k.shape(), &[t, self.kv_heads, self.head_dim]);
        debug_assert_eq!(v.shape(), &[t, self.kv_heads, self.head_dim]);
        let cols = self.kv_heads * self.head_dim;
        for (half, tt) in [(0usize, k), (1, v)] {
            let dst = self
                .kv
                .select(0, half)
                .expect("kv half")
                .select(0, slab)
                .expect("kv slab")
                .reshape(&[self.num_pages(), cols])
                .expect("kv slab view");
            let src = tt
                .contiguous()
                .expect("kv rows")
                .cast(self.dtype.tensor_dtype())
                .expect("kv dtype")
                .reshape(&[t, cols])
                .expect("kv rows flat");
            dst.scatter_rows(idx, &src).expect("store kv rows");
        }
    }

    /// Read the K/V at `locs` back, in the order asked, widened to f32.
    pub fn gather(&self, layer: usize, locs: &[u32]) -> (Tensor, Tensor) {
        let idx = self.index(locs);
        self.gather_indexed(layer, &idx)
    }

    /// `gather` in the pool's own dtype, so a bf16 pool feeds the bf16 SDPA
    /// without a round trip through f32.
    pub fn gather_native(&self, layer: usize, locs: &[u32]) -> (Tensor, Tensor) {
        let idx = self.index(locs);
        self.gather_indexed_native(layer, &idx)
    }

    /// Read one fixed-length window per row: `locs` is `[rows, window]` and the
    /// result is `[rows, window, kv_heads, head_dim]`, so a batched decode
    /// gathers every request's window in one op (decision 51).
    pub fn gather_rows(&self, layer: usize, locs: &Tensor) -> (Tensor, Tensor) {
        let (rows, window) = (locs.shape()[0], locs.shape()[1]);
        let idx = locs
            .contiguous()
            .expect("window locations")
            .reshape(&[rows * window])
            .expect("flat window");
        let (k, v) = self.gather_indexed(layer, &idx);
        let shape = [rows, window, self.kv_heads, self.head_dim];
        (
            k.reshape(&shape).expect("k window"),
            v.reshape(&shape).expect("v window"),
        )
    }

    fn gather_indexed(&self, layer: usize, idx: &Tensor) -> (Tensor, Tensor) {
        // Gathers here are always f32: the attention math and the paged-kernel
        // fallback read f32, and a narrower pool stores bf16 only to save bytes.
        // Upcasting bf16 -> f32 is exact.
        let (k, v) = self.gather_indexed_native(layer, idx);
        (
            k.cast(TensorDType::F32).expect("kv f32"),
            v.cast(TensorDType::F32).expect("kv f32"),
        )
    }

    fn gather_indexed_native(&self, layer: usize, idx: &Tensor) -> (Tensor, Tensor) {
        let slab = self.slab(layer);
        let cols = self.kv_heads * self.head_dim;
        let n = idx.shape()[0];
        let half = |h: usize| -> Tensor {
            let rows = self
                .kv
                .select(0, h)
                .expect("kv half")
                .select(0, slab)
                .expect("kv slab")
                .reshape(&[self.num_pages(), cols])
                .expect("kv slab view");
            rows.gather_rows(idx)
                .expect("gather kv rows")
                .reshape(&[n, self.kv_heads, self.head_dim])
                .expect("kv rows")
        };
        (half(0), half(1))
    }

    fn index(&self, locs: &[u32]) -> Tensor {
        let data: Vec<i32> = locs.iter().map(|&l| l as i32).collect();
        self.rt
            .from_slice_i32(&data, &[data.len()])
            .expect("stage kv index")
    }

    fn slab(&self, layer: usize) -> usize {
        self.dense_of[layer].expect("layer has no kv slab")
    }
}

/// GDN state slots the pool must hold: one live slot per running request, one
/// locked prompt/chunk snapshot per running request (a non-finished commit COWs
/// the live state into a snapshot while the request still holds its previous
/// lock), plus one slot of progress headroom, plus the captured graph's padding
/// row. Sizing at exactly `max_running_requests` panicked as soon as a request
/// committed a prompt-end snapshot (TODO.md item 10).
pub fn state_slots(max_running_requests: usize, reserve_padding: bool) -> usize {
    2 * max_running_requests + 1 + usize::from(reserve_padding)
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
    slots: usize,
    conv_dim: usize,
    width: usize,
    v_heads: usize,
    k_dim: usize,
    v_dim: usize,
    rt: Runtime,
}

impl LinearStateSlabs {
    pub fn new(params: &ModelParams, num_slots: usize, device: Device) -> Self {
        Self::with_runtime(params, num_slots, runtime(device))
    }

    fn with_runtime(params: &ModelParams, num_slots: usize, rt: Runtime) -> Self {
        let dense_of = dense_map(params, LayerType::LinearAttention);
        let layers = dense_of.iter().filter(|d| d.is_some()).count();
        let (k_heads, k_dim) = (params.linear_num_key_heads, params.linear_key_head_dim);
        let (v_heads, v_dim) = (params.linear_num_value_heads, params.linear_value_head_dim);
        let width = params.linear_conv_kernel_dim.saturating_sub(1);
        let conv_dim = 2 * k_heads * k_dim + v_heads * v_dim;
        let conv = rt
            .zeros(&[layers, num_slots, conv_dim, width], TensorDType::F32)
            .expect("allocate the conv state");
        let recurrent = rt
            .zeros(
                &[layers, num_slots, v_heads, k_dim, v_dim],
                TensorDType::F32,
            )
            .expect("allocate the recurrent state");
        Self {
            conv,
            recurrent,
            dense_of,
            slots: num_slots,
            conv_dim,
            width,
            v_heads,
            k_dim,
            v_dim,
            rt,
        }
    }

    pub fn dense_of(&self, layer: usize) -> Option<usize> {
        self.dense_of.get(layer).copied().flatten()
    }

    pub fn num_slots(&self) -> usize {
        self.slots
    }

    /// Bytes the state slabs hold, for the config's memory arithmetic.
    pub fn bytes(&self) -> usize {
        (self.conv.numel() + self.recurrent.numel()) * TensorDType::F32.element_size()
    }

    pub fn conv_size(&self) -> Vec<usize> {
        vec![self.conv_dim, self.width]
    }

    pub fn recurrent_size(&self) -> Vec<usize> {
        vec![self.v_heads, self.k_dim, self.v_dim]
    }

    fn conv_view(&self, slot: u32, layer: usize) -> Tensor {
        self.conv
            .select(0, self.slab(layer))
            .expect("conv layer")
            .select(0, slot as usize)
            .expect("conv slot")
    }

    fn recurrent_view(&self, slot: u32, layer: usize) -> Tensor {
        self.recurrent
            .select(0, self.slab(layer))
            .expect("recurrent layer")
            .select(0, slot as usize)
            .expect("recurrent slot")
    }

    /// The slot's conv state for this layer, zero before anything is written.
    pub fn conv_state(&self, slot: u32, layer: usize) -> Tensor {
        self.conv_view(slot, layer)
    }

    pub fn set_conv_state(&self, slot: u32, layer: usize, state: Tensor) {
        let dst = self.conv_view(slot, layer);
        let src = state
            .contiguous()
            .expect("conv state")
            .reshape(&self.conv_size())
            .expect("conv state view");
        src.copy_to(&dst).expect("write conv state");
    }

    /// The slot's recurrent state for this layer, zero before anything is
    /// written.
    pub fn recurrent_state(&self, slot: u32, layer: usize) -> Tensor {
        self.recurrent_view(slot, layer)
    }

    pub fn set_recurrent_state(&self, slot: u32, layer: usize, state: Tensor) {
        let dst = self.recurrent_view(slot, layer);
        let src = state
            .contiguous()
            .expect("recurrent state")
            .reshape(&self.recurrent_size())
            .expect("recurrent state view");
        src.copy_to(&dst).expect("write recurrent state");
    }

    /// The rows' conv state, `[rows, conv_dim, width]`, addressed by a device
    /// `[rows]` i32 slot index: a batched decode step reads the slots it staged
    /// rather than host numbers (decision 51).
    pub fn conv_state_rows(&self, slots: &Tensor, layer: usize) -> Tensor {
        let rows = self
            .conv
            .select(0, self.slab(layer))
            .expect("conv layer")
            .reshape(&[self.slots, self.conv_dim * self.width])
            .expect("conv rows");
        let idx = slots.contiguous().expect("slot index");
        rows.gather_rows(&idx)
            .expect("gather conv rows")
            .reshape(&[idx.shape()[0], self.conv_dim, self.width])
            .expect("conv rows shape")
    }

    /// `set_conv_state` for a device `[rows]` i32 slot index; `state` is
    /// `[rows, conv_dim, width]`.
    pub fn set_conv_state_rows(&self, slots: &Tensor, layer: usize, state: &Tensor) {
        let dst = self
            .conv
            .select(0, self.slab(layer))
            .expect("conv layer")
            .reshape(&[self.slots, self.conv_dim * self.width])
            .expect("conv rows");
        let src = state
            .contiguous()
            .expect("conv rows")
            .cast(TensorDType::F32)
            .expect("conv dtype")
            .reshape(&[state.shape()[0], self.conv_dim * self.width])
            .expect("conv rows flat");
        dst.scatter_rows(slots, &src).expect("write conv rows");
    }

    /// The rows' recurrent state, `[rows, v_heads, k_dim, v_dim]`, addressed by a
    /// device `[rows]` i32 slot index.
    pub fn recurrent_state_rows(&self, slots: &Tensor, layer: usize) -> Tensor {
        let rows = self
            .recurrent
            .select(0, self.slab(layer))
            .expect("recurrent layer")
            .reshape(&[self.slots, self.v_heads * self.k_dim * self.v_dim])
            .expect("recurrent rows");
        let idx = slots.contiguous().expect("slot index");
        rows.gather_rows(&idx)
            .expect("gather recurrent rows")
            .reshape(&[idx.shape()[0], self.v_heads, self.k_dim, self.v_dim])
            .expect("recurrent rows shape")
    }

    /// `set_recurrent_state` for a device `[rows]` i32 slot index; `state` is
    /// `[rows, v_heads, k_dim, v_dim]`.
    pub fn set_recurrent_state_rows(&self, slots: &Tensor, layer: usize, state: &Tensor) {
        let dst = self
            .recurrent
            .select(0, self.slab(layer))
            .expect("recurrent layer")
            .reshape(&[self.slots, self.v_heads * self.k_dim * self.v_dim])
            .expect("recurrent rows");
        let src = state
            .contiguous()
            .expect("recurrent rows")
            .cast(TensorDType::F32)
            .expect("recurrent dtype")
            .reshape(&[state.shape()[0], self.v_heads * self.k_dim * self.v_dim])
            .expect("recurrent rows flat");
        dst.scatter_rows(slots, &src).expect("write recurrent rows");
    }

    /// Zero a whole slot, so a recycled slot cannot leak the previous request's
    /// state into its first prefill chunk.
    pub fn reset_slot(&self, slot: u32) {
        let conv = self
            .rt
            .zeros(&self.conv_size(), TensorDType::F32)
            .expect("zero conv state");
        let recurrent = self
            .rt
            .zeros(&self.recurrent_size(), TensorDType::F32)
            .expect("zero recurrent state");
        for layer in 0..self.dense_of.len() {
            if self.dense_of[layer].is_some() {
                self.set_conv_state(slot, layer, conv.clone());
                self.set_recurrent_state(slot, layer, recurrent.clone());
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
                let conv = self.conv_view(dst, layer);
                self.conv_view(src, layer)
                    .copy_to(&conv)
                    .expect("copy conv slot");
                let recurrent = self.recurrent_view(dst, layer);
                self.recurrent_view(src, layer)
                    .copy_to(&recurrent)
                    .expect("copy recurrent slot");
            }
        }
    }

    fn slab(&self, layer: usize) -> usize {
        self.dense_of[layer].expect("layer has no linear state")
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
    _rt: Runtime,
}

struct SharedSlabs {
    mha: MhaPool,
    state: LinearStateSlabs,
}

impl DeviceCache {
    pub fn new(
        params: &ModelParams,
        num_pages: usize,
        num_slots: usize,
        dtype: KvDType,
        device: Device,
    ) -> Self {
        Self::with_runtime(params, num_pages, num_slots, dtype, runtime(device))
    }

    fn with_runtime(
        params: &ModelParams,
        num_pages: usize,
        num_slots: usize,
        dtype: KvDType,
        rt: Runtime,
    ) -> Self {
        Self {
            pages: Mutex::new(PagePool::new(PAGE_SIZE, num_pages)),
            slots: Mutex::new(LinearStatePool::new(num_slots)),
            tree: Mutex::new(HybridRadixIndex::new(PAGE_SIZE)),
            slabs: SharedSlabs {
                mha: MhaPool::with_runtime(params, num_pages, dtype, rt.clone()),
                state: LinearStateSlabs::with_runtime(params, num_slots, rt.clone()),
            },
            _rt: rt,
        }
    }

    pub fn mha(&self) -> &MhaPool {
        &self.slabs.mha
    }

    pub fn state(&self) -> &LinearStateSlabs {
        &self.slabs.state
    }

    /// The runtime that owns the pool's stream and buffers. A model reading or
    /// writing these slabs has to run on the same stream or the two are
    /// unordered.
    pub fn runtime(&self) -> &Runtime {
        &self._rt
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

    fn rollback(&self, cache: &mut SequenceCache, n: usize) -> Result<(), KvcacheError> {
        if n > cache.cached() - cache.reused() {
            return Err(KvcacheError::MisalignedCache {
                why: format!(
                    "rollback of {n} pages over the {} this request owns",
                    cache.cached() - cache.reused()
                ),
            });
        }
        cache.truncate(n, &mut self.pages.lock().expect("kvcache lock"));
        Ok(())
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
    use half::bf16;

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
            mrope: None,
            expert_slot_bytes: 1,
        }
    }

    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 9) as f32 / (1 << 23) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn f32_tensor(like: &Tensor, values: &[f32], shape: &[usize]) -> Tensor {
        like.runtime()
            .from_slice_f32(values, shape)
            .expect("stage f32")
    }

    fn bf16_tensor(like: &Tensor, values: &[f32], shape: &[usize]) -> Tensor {
        let bits: Vec<bf16> = values.iter().map(|v| bf16::from_f32(*v)).collect();
        like.runtime()
            .from_slice_bf16(&bits, shape)
            .expect("stage bf16")
    }

    fn dev_index(like: &Tensor, values: &[i32]) -> Tensor {
        like.runtime()
            .from_slice_i32(values, &[values.len()])
            .expect("stage index")
    }

    fn to_f32(t: &Tensor) -> Vec<f32> {
        t.to_host_f32().expect("read device")
    }

    /// Values already rounded to bf16, so a store/gather round trip is exact
    /// rather than close: the pool promises it hands back what it was given.
    fn kv_pair(like: &Tensor, t: usize, kvh: usize, hd: usize, seed: u32) -> (Tensor, Tensor) {
        let n = t * kvh * hd;
        let shape = [t, kvh, hd];
        (
            bf16_tensor(like, &data(n, seed), &shape),
            bf16_tensor(like, &data(n, seed + 1), &shape),
        )
    }

    fn pool(num_pages: usize, dtype: KvDType) -> MhaPool {
        MhaPool::with_runtime(&params(), num_pages, dtype, runtime(Device::Cuda(0)))
    }

    fn state(num_slots: usize) -> LinearStateSlabs {
        LinearStateSlabs::with_runtime(&params(), num_slots, runtime(Device::Cuda(0)))
    }

    fn cache(num_pages: usize, num_slots: usize) -> DeviceCache {
        DeviceCache::with_runtime(
            &params(),
            num_pages,
            num_slots,
            KvDType::F32,
            runtime(Device::Cuda(0)),
        )
    }

    #[test]
    fn only_full_attention_layers_get_a_slab() {
        let pool = pool(16, KvDType::F32);
        assert_eq!(pool.dense_of(0), Some(0));
        assert_eq!(pool.dense_of(1), None, "a linear layer has no kv");
        assert_eq!(pool.dense_of(3), Some(1));
        assert_eq!(pool.num_pages(), 16);
    }

    #[test]
    fn a_stored_location_gathers_back_exactly() {
        let pool = pool(16, KvDType::F32);
        let (k, v) = kv_pair(&pool.kv, 3, 2, 4, 61);
        pool.store(0, &[5, 2, 9], &k, &v);
        let (gk, gv) = pool.gather(0, &[5, 2, 9]);
        assert_eq!(to_f32(&gk), to_f32(&k));
        assert_eq!(to_f32(&gv), to_f32(&v));
    }

    #[test]
    fn gather_reads_locations_in_the_order_asked() {
        let pool = pool(16, KvDType::F32);
        let (k, v) = kv_pair(&pool.kv, 3, 2, 4, 63);
        pool.store(0, &[5, 2, 9], &k, &v);
        let (gk, _) = pool.gather(0, &[9, 5, 2]);
        let want: Vec<f32> = [2usize, 0, 1]
            .iter()
            .flat_map(|&r| to_f32(&k.select(0, r).unwrap()))
            .collect();
        assert_eq!(to_f32(&gk), want);
    }

    #[test]
    fn a_later_store_replaces_the_earlier_one_at_the_same_location() {
        let pool = pool(16, KvDType::F32);
        let (k, v) = kv_pair(&pool.kv, 2, 2, 4, 65);
        pool.store(0, &[1, 2], &k, &v);
        let (k2, v2) = kv_pair(&pool.kv, 1, 2, 4, 67);
        pool.store(0, &[2], &k2, &v2);
        let (gk, gv) = pool.gather(0, &[2]);
        assert_eq!(to_f32(&gk), to_f32(&k2));
        assert_eq!(to_f32(&gv), to_f32(&v2));
    }

    #[test]
    fn one_layers_slab_does_not_disturb_another() {
        let pool = pool(16, KvDType::F32);
        let (k, v) = kv_pair(&pool.kv, 2, 2, 4, 69);
        pool.store(0, &[3, 4], &k, &v);
        let (k2, v2) = kv_pair(&pool.kv, 2, 2, 4, 71);
        pool.store(3, &[3, 4], &k2, &v2);
        assert_eq!(to_f32(&pool.gather(0, &[3, 4]).0), to_f32(&k));
        assert_eq!(to_f32(&pool.gather(3, &[3, 4]).0), to_f32(&k2));
    }

    #[test]
    fn an_unwritten_location_reads_as_zero() {
        let pool = pool(8, KvDType::F32);
        let (gk, gv) = pool.gather(0, &[7]);
        assert_eq!(to_f32(&gk), vec![0.0; 8]);
        assert_eq!(to_f32(&gv), vec![0.0; 8]);
    }

    #[test]
    fn a_bf16_pool_hands_back_the_value_rounded_to_bf16() {
        let pool = pool(8, KvDType::BFloat16);
        let n = 2 * 2 * 4;
        let want = data(n, 83);
        let k = f32_tensor(&pool.kv, &want, &[2, 2, 4]);
        pool.store(0, &[4, 6], &k, &k);
        let (gk, _) = pool.gather(0, &[4, 6]);
        assert_eq!(
            gk.dtype(),
            TensorDType::F32,
            "a bf16 pool must gather back as f32 for the attention math"
        );
        assert_eq!(
            to_f32(&gk),
            to_f32(&k.cast(TensorDType::Bf16).unwrap()),
            "a lossy pool must round, not truncate"
        );
        assert_ne!(to_f32(&gk), want);
    }

    #[test]
    fn the_kv_slab_pointers_step_by_one_layer_of_tokens() {
        let p = params();
        let pool = pool(7, KvDType::F32);
        let (k0, v0) = pool.kv_slab_ptrs(0);
        let (k1, _) = pool.kv_slab_ptrs(3);
        // Two dense slabs in the tower, K then V, so V starts two layers in.
        let per = 7 * PAGE_SIZE * p.num_kv_heads * p.head_dim * 4;
        assert_eq!(v0 - k0, 2 * per as u64, "V follows both K slabs");
        assert_eq!(k1 - k0, per as u64, "slab 1 is one layer after slab 0");
        assert_eq!(pool.dtype(), KvDType::F32);
    }

    #[test]
    fn the_state_pool_leaves_room_for_snapshots_and_padding() {
        // One live slot and one locked snapshot per running request, one slot of
        // headroom, and the capture padding row.
        assert_eq!(state_slots(1, false), 3);
        assert_eq!(state_slots(4, false), 9);
        assert_eq!(state_slots(4, true), 10);
        assert!(
            state_slots(1, false) > 1,
            "item 10: not just the concurrency"
        );
    }

    #[test]
    fn a_device_cache_hands_out_pages_and_state_slots() {
        let cache = cache(8, 2);
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
        let cache = cache(4, 1);
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
        let cache = cache(8, 1);
        let a_slot = {
            let a = cache.begin();
            let state = &cache.state().recurrent;
            cache.state().set_recurrent_state(
                a.state_slot(),
                1,
                f32_tensor(state, &[1.0; 32], &[2, 4, 4]),
            );
            a.state_slot()
        };
        let mut a = SequenceCache::new(a_slot);
        cache.end(&mut a);
        let b = cache.begin();
        assert_eq!(b.state_slot(), a.state_slot());
        assert!(
            to_f32(&cache.state().recurrent_state(b.state_slot(), 1))
                .iter()
                .all(|v| *v == 0.0),
            "a recycled slot must not leak the previous request's state"
        );
    }

    #[test]
    fn only_linear_layers_hold_linear_state() {
        let s = state(4);
        assert_eq!(s.dense_of(0), None);
        assert_eq!(s.dense_of(1), Some(0));
        assert_eq!(s.dense_of(2), Some(1));
        assert_eq!(s.dense_of(3), None);
        assert_eq!(s.num_slots(), 4);
    }

    #[test]
    fn a_linear_state_slot_starts_at_zero_and_round_trips() {
        let s = state(2);
        let conv = s.conv_state(0, 1);
        assert_eq!(conv.shape(), &[2 * 2 * 4 + 2 * 4, 3]);
        assert!(to_f32(&conv).iter().all(|v| *v == 0.0));
        let rec = s.recurrent_state(0, 1);
        assert_eq!(rec.shape(), &[2, 4, 4]);

        let want_conv = data(conv.numel(), 73);
        s.set_conv_state(0, 1, f32_tensor(&s.conv, &want_conv, conv.shape()));
        assert_eq!(to_f32(&s.conv_state(0, 1)), want_conv);

        let want_rec = data(rec.numel(), 75);
        s.set_recurrent_state(0, 1, f32_tensor(&s.recurrent, &want_rec, rec.shape()));
        assert_eq!(to_f32(&s.recurrent_state(0, 1)), want_rec);
    }

    #[test]
    fn linear_state_slots_are_independent_and_resettable() {
        let s = state(2);
        let one = vec![1.0f32; 32];
        s.set_recurrent_state(0, 1, f32_tensor(&s.recurrent, &one, &[2, 4, 4]));
        assert_eq!(to_f32(&s.recurrent_state(1, 1)), vec![0.0; 32]);
        assert_eq!(to_f32(&s.recurrent_state(0, 1)), one);
        s.reset_slot(0);
        assert!(
            to_f32(&s.recurrent_state(0, 1)).iter().all(|v| *v == 0.0),
            "a recycled slot must not leak the previous request's state"
        );
    }

    #[test]
    fn a_linear_state_slot_copies_conv_and_recurrent_to_another() {
        let s = state(2);
        let cv = s.conv_size();
        let rv = s.recurrent_size();
        // Both linear layers carry distinct state on slot 0.
        for layer in [1usize, 2] {
            s.set_conv_state(
                0,
                layer,
                f32_tensor(&s.conv, &data(cv[0] * cv[1], 81 + layer as u32), &cv[..]),
            );
            s.set_recurrent_state(
                0,
                layer,
                f32_tensor(
                    &s.recurrent,
                    &data(rv[0] * rv[1] * rv[2], 91 + layer as u32),
                    &rv[..],
                ),
            );
        }
        s.copy_slot(0, 1);
        for layer in [1usize, 2] {
            assert_eq!(
                to_f32(&s.conv_state(1, layer)),
                to_f32(&s.conv_state(0, layer)),
                "the copy matches the source conv"
            );
            assert_eq!(
                to_f32(&s.recurrent_state(1, layer)),
                to_f32(&s.recurrent_state(0, layer)),
                "the copy matches the source recurrent"
            );
        }
        // Copy-on-write: writing the live slot afterwards leaves the snapshot.
        let before = to_f32(&s.recurrent_state(1, 1));
        s.set_recurrent_state(
            0,
            1,
            f32_tensor(&s.recurrent, &vec![7.0f32; before.len()], &rv[..]),
        );
        assert_eq!(
            to_f32(&s.recurrent_state(1, 1)),
            before,
            "a snapshot slot is not aliased by a later live write"
        );
    }

    #[test]
    fn a_device_index_store_and_gather_match_the_host_path() {
        let pool = pool(16, KvDType::F32);
        let (k, v) = kv_pair(&pool.kv, 2, 2, 4, 91);
        let locs = dev_index(&pool.kv, &[5, 2]);
        pool.store_at(0, &locs, &k, &v);
        let (gk, gv) = pool.gather_rows(0, &locs.reshape(&[1, 2]).unwrap());
        assert_eq!(to_f32(&gk), to_f32(&k));
        assert_eq!(to_f32(&gv), to_f32(&v));
        // The host accessor reads back the very values the device path wrote.
        assert_eq!(to_f32(&pool.gather(0, &[5, 2]).0), to_f32(&k));
    }

    #[test]
    fn a_batched_gather_reads_each_rows_window_in_order() {
        let pool = pool(16, KvDType::F32);
        let (k, v) = kv_pair(&pool.kv, 4, 2, 4, 97);
        // Token `r` lands at pool location `r`, so the window can name rows back.
        pool.store_at(0, &dev_index(&pool.kv, &[0, 1, 2, 3]), &k, &v);
        let windows = dev_index(&pool.kv, &[2, 0, 3, 1]).reshape(&[2, 2]).unwrap();
        let (gk, _) = pool.gather_rows(0, &windows);
        assert_eq!(gk.shape(), &[2, 2, 2, 4]);
        let flat = to_f32(&gk);
        let want: Vec<f32> = [2usize, 0, 3, 1]
            .iter()
            .flat_map(|&r| to_f32(&k.select(0, r).unwrap()))
            .collect();
        assert_eq!(flat, want, "row 0 then row 1, each window in order");
    }

    #[test]
    fn a_device_slot_reads_and_writes_the_same_state_as_the_host_slot() {
        let s = state(3);
        let slot = dev_index(&s.conv, &[1]);
        let cv = s.conv_size();
        let conv = f32_tensor(&s.conv, &data(cv[0] * cv[1], 93), &cv[..]);
        s.set_conv_state_rows(&slot, 1, &conv.unsqueeze(0).unwrap());
        assert_eq!(
            to_f32(&s.conv_state_rows(&slot, 1).select(0, 0).unwrap()),
            to_f32(&conv)
        );
        assert_eq!(
            to_f32(&s.conv_state(1, 1)),
            to_f32(&conv),
            "the host accessor sees the device write"
        );

        let rv = s.recurrent_size();
        let rec = f32_tensor(&s.recurrent, &data(rv[0] * rv[1] * rv[2], 95), &rv[..]);
        s.set_recurrent_state_rows(&slot, 1, &rec.unsqueeze(0).unwrap());
        assert_eq!(
            to_f32(&s.recurrent_state_rows(&slot, 1).select(0, 0).unwrap()),
            to_f32(&rec)
        );
        assert_eq!(to_f32(&s.recurrent_state(1, 1)), to_f32(&rec));
        assert!(
            to_f32(&s.recurrent_state_rows(&dev_index(&s.conv, &[0]), 1))
                .iter()
                .all(|v| *v == 0.0),
            "the device accessor never aliases another slot"
        );
    }

    /// Mark a slot's first linear layer so a restore can be told apart from a
    /// zeroed fresh slot.
    fn stamp(s: &LinearStateSlabs, slot: u32, value: f32) {
        let n = s.recurrent_size().iter().product::<usize>();
        s.set_recurrent_state(
            slot,
            1,
            f32_tensor(&s.recurrent, &vec![value; n], &s.recurrent_size()[..]),
        );
    }

    fn stamp_of(cache: &DeviceCache, slot: u32) -> Vec<f32> {
        to_f32(&cache.state().recurrent_state(slot, 1))
    }

    #[test]
    fn a_committed_prefix_is_reused_by_the_next_request() {
        let cache = cache(32, 4);
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
        let cache = cache(32, 4);
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
        let cache = cache(64, 4);
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

    /// A chunked prefill commits at every chunk boundary with `finished=false`.
    /// Each commit must advance the reused span by the new chunk without freeing
    /// a page (the request shares the pages the tree adopted), and leave the
    /// boundary resumable for a later mid-prompt request.
    #[test]
    fn a_chunk_boundary_commit_is_reusable_mid_prompt() {
        let cache = cache(64, 8);
        let total_pages = cache.num_pages();
        let mut a = cache.begin();
        cache.reserve(&mut a, 2).unwrap();
        let first_pages = a.filled().to_vec();
        stamp(cache.state(), a.state_slot(), 4.0);

        cache.commit(&mut a, &[1, 2], false);
        assert_eq!(a.reused(), 2, "the first boundary is tree-owned");
        cache.reserve(&mut a, 2).unwrap();
        cache.commit(&mut a, &[1, 2, 3, 4], false);
        assert_eq!(a.reused(), 4, "the second advances the reused span");
        cache.reserve(&mut a, 2).unwrap();
        cache.commit(&mut a, &[1, 2, 3, 4, 5, 6], false);
        assert_eq!(a.reused(), 6);
        assert_eq!(
            &a.filled()[..2],
            &first_pages[..],
            "the leading pages stay the ones the tree adopted"
        );

        // A follower whose prompt ends on the first chunk boundary resumes
        // there -- the mid-prompt snapshot a chunk boundary buys.
        let mut b = cache.begin();
        assert_eq!(
            cache.adopt_prefix(&mut b, &[1, 2, 9]),
            2,
            "the first chunk boundary is reusable mid-prompt"
        );
        assert!(
            stamp_of(&cache, b.state_slot()).iter().all(|v| *v == 4.0),
            "b resumes from the chunk-boundary GDN snapshot"
        );
        cache.end(&mut b);

        cache.reserve(&mut a, 1).unwrap();
        cache.commit(&mut a, &[1, 2, 3, 4, 5, 6, 7], true);
        cache.end(&mut a);
        assert_eq!(
            cache.available(),
            total_pages - 7,
            "the tree owns one page per committed token"
        );
    }

    #[test]
    fn pages_and_slots_are_conserved_across_reuse() {
        let cache = cache(32, 4);
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
        let cache = cache(32, 4);
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
        let cache = cache(32, 4);
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
