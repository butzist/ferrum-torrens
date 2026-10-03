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

use tch::{Kind, Tensor};

use crate::domain::model::{LayerType, ModelParams};
use crate::infrastructure::torch::Device;

/// ALGORITHM.md 3.1: one token per page.
pub const PAGE_SIZE: i64 = 1;

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
}

impl MhaPool {
    pub fn new(params: &ModelParams, num_pages: usize, device: Device) -> Self {
        let dense_of = dense_map(params, LayerType::FullAttention);
        let layers = dense_of.iter().filter(|d| d.is_some()).count() as i64;
        let (kv_heads, head_dim) = (params.num_kv_heads as i64, params.head_dim as i64);
        let kv = Tensor::zeros(
            [2, layers, num_pages as i64, PAGE_SIZE, kv_heads, head_dim],
            (Kind::BFloat16, device.to_tch()),
        );
        Self {
            kv,
            dense_of,
            kv_heads,
            head_dim,
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

    /// Bytes the pool holds, for the config's memory arithmetic.
    pub fn bytes(&self) -> usize {
        self.kv.numel() * Kind::BFloat16.elt_size_in_bytes()
    }

    /// Write this step's `k`/`v` (`[t, kv_heads, head_dim]`) at `locs`.
    pub fn store(&self, layer: usize, locs: &[u32], k: &Tensor, v: &Tensor) {
        let idx = self.index(locs);
        let slab = self.slab(layer);
        debug_assert_eq!(
            k.size(),
            vec![locs.len() as i64, self.kv_heads, self.head_dim]
        );
        debug_assert_eq!(
            v.size(),
            vec![locs.len() as i64, self.kv_heads, self.head_dim]
        );
        for (half, t) in [(0i64, k), (1, v)] {
            // `select` hands back a view, so the in-place write lands in the
            // pool's storage rather than in a copy.
            let mut view = self.kv.select(0, half).select(0, slab).shallow_clone();
            let _ = view.index_copy_(0, &idx, &t.unsqueeze(1).to_kind(Kind::BFloat16));
        }
    }

    /// Read the K/V at `locs` back, in the order asked.
    pub fn gather(&self, layer: usize, locs: &[u32]) -> (Tensor, Tensor) {
        let idx = self.index(locs);
        let slab = |half: i64| -> Tensor {
            self.kv
                .select(0, half)
                .select(0, self.slab(layer))
                .index_select(0, &idx)
                .squeeze_dim(1)
        };
        (slab(0), slab(1))
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

    fn slab(&self, layer: usize) -> i64 {
        self.dense_of[layer].expect("layer has no linear state") as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let pool = MhaPool::new(&params(), 16, auto_device());
        assert_eq!(pool.dense_of(0), Some(0));
        assert_eq!(pool.dense_of(1), None, "a linear layer has no kv");
        assert_eq!(pool.dense_of(3), Some(1));
        assert_eq!(pool.num_pages(), 16);
    }

    #[test]
    fn a_stored_location_gathers_back_exactly() {
        let pool = MhaPool::new(&params(), 16, auto_device());
        let (k, v) = kv_pair(3, 2, 4, 61);
        pool.store(0, &[5, 2, 9], &k, &v);
        let (gk, gv) = pool.gather(0, &[5, 2, 9]);
        assert_eq!(to_f32_vec(&gk), to_f32_vec(&k));
        assert_eq!(to_f32_vec(&gv), to_f32_vec(&v));
    }

    #[test]
    fn gather_reads_locations_in_the_order_asked() {
        let pool = MhaPool::new(&params(), 16, auto_device());
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
        let pool = MhaPool::new(&params(), 16, auto_device());
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
        let pool = MhaPool::new(&params(), 16, auto_device());
        let (k, v) = kv_pair(2, 2, 4, 69);
        pool.store(0, &[3, 4], &k, &v);
        let (k2, v2) = kv_pair(2, 2, 4, 71);
        pool.store(3, &[3, 4], &k2, &v2);
        assert_eq!(to_f32_vec(&pool.gather(0, &[3, 4]).0), to_f32_vec(&k));
        assert_eq!(to_f32_vec(&pool.gather(3, &[3, 4]).0), to_f32_vec(&k2));
    }

    #[test]
    fn an_unwritten_location_reads_as_zero() {
        let pool = MhaPool::new(&params(), 8, auto_device());
        let (gk, gv) = pool.gather(0, &[7]);
        assert_eq!(to_f32_vec(&gk), vec![0.0; 8]);
        assert_eq!(to_f32_vec(&gv), vec![0.0; 8]);
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
}
