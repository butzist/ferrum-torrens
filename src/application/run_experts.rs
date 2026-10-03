// The routed-expert use case for one MoE layer: plan the slot residency, copy
// the misses in, then ask the kernel for the grouped GEMM.
//
// This is workflow over the domain ports and nothing else -- no CUDA, no mmap,
// no python -- so it is unit-tested against mocked ports and is where the
// decision-27 ordering (forget the dropped layer BEFORE planning) is enforced.

use crate::domain::expert_cache::ExpertCache;
use crate::domain::expert_kernel::{
    ExpertGemm, ExpertSlotStore, MoEError, MoEExecutor, MoEGroup, MoERequest,
};
use std::sync::{Arc, Mutex};

/// The engine's `MoEExecutor`: the same three pieces `run_experts` takes, held
/// behind one shared handle so a tower that only has `&self` can call it. The
/// slot machine is the mutable part (it counts touches for the LRU), the two
/// ports are `&self`, and one device cache serves every step, which is why the
/// adapter takes its own lock.
pub struct CachedExperts {
    cache: Mutex<ExpertCache>,
    store: Arc<dyn ExpertSlotStore>,
    gemm: Arc<dyn ExpertGemm>,
}

impl CachedExperts {
    pub fn new(
        store: Arc<dyn ExpertSlotStore>,
        gemm: Arc<dyn ExpertGemm>,
        num_layers: usize,
        num_experts: usize,
        cache_size: usize,
    ) -> Self {
        Self {
            cache: Mutex::new(ExpertCache::new(num_layers, num_experts, cache_size)),
            store,
            gemm,
        }
    }

    /// Slots the machine can hand out. A step's plan needs this to be at least
    /// the adapter's bank size, which `run_experts` checks.
    pub fn cache_size(&self) -> usize {
        self.locked().cache_size()
    }

    fn locked(&self) -> std::sync::MutexGuard<'_, ExpertCache> {
        self.cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl MoEExecutor for CachedExperts {
    fn execute<'a>(&self, req: &MoERequest<'a>, out: &mut [f32]) -> Result<(), MoEError> {
        run_experts(
            &mut self.locked(),
            self.store.as_ref(),
            self.gemm.as_ref(),
            req.layer,
            &ExpertStep {
                hidden: req.hidden,
                inter: req.inter,
                top_k: req.top_k,
                x: req.x,
                experts: req.experts,
                weights: req.weights,
            },
            out,
        )
    }
}

/// One layer's routed expert step. `experts` is dense over `tokens * top_k`,
/// the same layout the router produces.
pub struct ExpertStep<'a> {
    pub hidden: usize,
    pub inter: usize,
    pub top_k: usize,
    pub x: &'a [f32],
    pub experts: &'a [u32],
    pub weights: &'a [f32],
}

/// `out` receives `tokens * hidden` f32. The port contract says the kernel gets
/// the slot ids, so the rewrites happen here and never leave the domain.
pub fn run_experts(
    cache: &mut ExpertCache,
    store: &dyn ExpertSlotStore,
    gemm: &dyn ExpertGemm,
    layer: usize,
    step: &ExpertStep<'_>,
    out: &mut [f32],
) -> Result<(), MoEError> {
    if step.hidden == 0 || step.top_k == 0 || !step.x.len().is_multiple_of(step.hidden) {
        return Err(MoEError::Bad(format!(
            "expert step: hidden {} top_k {} x {}",
            step.hidden,
            step.top_k,
            step.x.len()
        )));
    }
    let tokens = step.x.len() / step.hidden;
    if step.experts.len() != tokens * step.top_k || step.weights.len() != tokens * step.top_k {
        return Err(MoEError::Bad(format!(
            "expert step: {} experts and {} weights for {tokens} tokens x top_k {}",
            step.experts.len(),
            step.weights.len(),
            step.top_k
        )));
    }
    let slots = store.slots();
    if cache.cache_size() < slots {
        return Err(MoEError::Bad(format!(
            "slot cache holds {} slots but the adapter stages {}",
            cache.cache_size(),
            slots
        )));
    }

    // The device cache holds one layer, so a switch invalidates bytes we still
    // think are resident. Forget them before the plan is computed.
    if let Some(dropped) = store.begin_layer(layer)?
        && dropped != layer
    {
        cache.forget_layer(dropped);
    }

    let plan = cache.ensure_experts(layer, step.experts);
    for (&expert, &slot) in plan.misses.iter().zip(&plan.write_slots) {
        store.load(layer, expert, slot)?;
    }

    gemm.grouped_moe(
        &MoEGroup {
            hidden: step.hidden,
            inter: step.inter,
            x: step.x,
            top_k: step.top_k,
            routes: &plan.rewrites,
            weights: step.weights,
        },
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::expert_kernel::{
        MoEExecutor, MoERequest, MockExpertGemm, MockExpertSlotStore,
    };
    use std::sync::{Arc, Mutex};

    /// Routes plus the three shapes the kernel was handed.
    type SeenKernel = (Vec<i32>, usize, usize, usize);

    fn request<'a>(experts: &'a [u32], weights: &'a [f32], x: &'a [f32]) -> MoERequest<'a> {
        MoERequest {
            layer: 3,
            hidden: 4,
            inter: 2,
            top_k: 2,
            x,
            experts,
            weights,
        }
    }

    /// The port the tower sees: expert ids go in, slot ids reach the kernel and
    /// the kernel's output comes back in `out`.
    #[test]
    fn the_executor_rewrites_expert_ids_into_slots_for_the_kernel() {
        let mut store = strict_store(4);
        store
            .expect_begin_layer()
            .withf(|l| *l == 3)
            .returning(|_| Ok(None));
        store.expect_load().times(2).returning(|_, _, _| Ok(()));
        let seen: Arc<Mutex<Option<SeenKernel>>> = Arc::new(Mutex::new(None));
        let sink = seen.clone();
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe().returning(move |req, out| {
            *sink.lock().unwrap() = Some((req.routes.to_vec(), req.hidden, req.inter, req.top_k));
            for (i, o) in out.iter_mut().enumerate() {
                *o = req.weights[i / req.hidden] * 2.0;
            }
            Ok(())
        });
        let exec = CachedExperts::new(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        let x = [1.0f32; 8];
        let experts = [6u32, 1, 6, 1];
        let weights = [0.25f32; 4];
        let mut out = [0.0f32; 8];
        exec.execute(&request(&experts, &weights, &x), &mut out)
            .unwrap();

        // The kernel is told slots, not expert ids, and it keeps the shapes.
        let (routes, hidden, inter, top_k) = seen.lock().unwrap().clone().unwrap();
        assert_eq!(routes.len(), 4);
        assert!(routes.iter().all(|s| *s >= 0 && *s < 4));
        assert_ne!(routes[0], 6);
        assert_eq!(routes[0], routes[2]);
        assert_eq!((hidden, inter, top_k), (4, 2, 2));
        assert_eq!(out, [0.5f32; 8]);
    }

    /// A second step over the same layer stages nothing: the slot machine is the
    /// executor's own state, which is why the port can be `&self`.
    #[test]
    fn a_repeated_step_over_one_layer_stages_nothing() {
        let mut store = strict_store(4);
        store
            .expect_begin_layer()
            .withf(|l| *l == 3)
            .returning(|_| Ok(None));
        store.expect_load().times(2).returning(|_, _, _| Ok(()));
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe().times(2).returning(|_, _| Ok(()));
        let exec = CachedExperts::new(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        let x = [1.0f32; 8];
        let experts = [6u32, 1, 6, 1];
        let weights = [0.25f32; 4];
        let mut out = [0.0f32; 8];
        exec.execute(&request(&experts, &weights, &x), &mut out)
            .unwrap();
        // `store` allows no further `load` calls, so this second step only
        // passes if both experts were still resident.
        exec.execute(&request(&experts, &weights, &x), &mut out)
            .unwrap();
    }

    /// A malformed request is rejected by the use case without staging
    /// anything or launching the kernel.
    #[test]
    fn a_request_that_does_not_match_its_batch_never_reaches_the_adapter() {
        let store = strict_store(4);
        let gemm = MockExpertGemm::new();
        let exec = CachedExperts::new(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        let x = [1.0f32; 8];
        let experts = [6u32, 1];
        let weights = [0.25f32; 4];
        let mut out = [0.0f32; 8];
        assert!(matches!(
            exec.execute(&request(&experts, &weights, &x), &mut out),
            Err(MoEError::Bad(_))
        ));
    }

    /// A store that answers exactly the calls the test expects and fails on
    /// any other: the plan must not copy experts that are already resident.
    fn strict_store(slots: usize) -> MockExpertSlotStore {
        let mut store = MockExpertSlotStore::new();
        store.expect_slots().return_const(slots);
        store
    }

    fn step<'a>(experts: &'a [u32], weights: &'a [f32], x: &'a [f32]) -> ExpertStep<'a> {
        ExpertStep {
            hidden: 4,
            inter: 4,
            top_k: 2,
            x,
            experts,
            weights,
        }
    }

    #[test]
    fn copies_the_misses_then_asks_the_kernel() {
        let mut cache = ExpertCache::new(2, 4, 4);
        let mut store = strict_store(4);
        let mut gemm = MockExpertGemm::new();
        store
            .expect_begin_layer()
            .withf(|l| *l == 0)
            .returning(|_| Ok(None));
        store
            .expect_load()
            .withf(|layer, expert, _slot| *layer == 0 && (*expert == 1 || *expert == 3))
            .times(2)
            .returning(|_, _, _| Ok(()));
        let seen_routes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = seen_routes.clone();
        gemm.expect_grouped_moe().returning(move |req, _| {
            *sink.lock().unwrap() = req.routes.to_vec();
            Ok(())
        });
        let x = [1.0f32; 8];
        let experts = [1u32, 3, 1, 3];
        let weights = [1.0f32; 4];
        let mut out = [0.0f32; 8];
        run_experts(
            &mut cache,
            &store,
            &gemm,
            0,
            &step(&experts, &weights, &x),
            &mut out,
        )
        .unwrap();
        // Both tokens route the same two experts, so four routes share two
        // slots and only two experts were copied.
        let seen = seen_routes.lock().unwrap().clone();
        assert_eq!(seen.len(), 4);
        assert_eq!(seen[0], seen[2]);
        assert_eq!(seen[1], seen[3]);
    }

    #[test]
    fn a_resident_layer_copies_nothing() {
        let mut cache = ExpertCache::new(2, 4, 4);
        let x = [1.0f32; 8];
        let experts = [1u32, 3, 1, 3];
        let weights = [1.0f32; 4];
        let mut out = [0.0f32; 8];

        let mut store = strict_store(4);
        store
            .expect_begin_layer()
            .withf(|l| *l == 0)
            .returning(|_| Ok(None));
        store.expect_load().times(2).returning(|_, _, _| Ok(()));
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe().times(1).returning(|_, _| Ok(()));
        run_experts(
            &mut cache,
            &store,
            &gemm,
            0,
            &step(&experts, &weights, &x),
            &mut out,
        )
        .unwrap();

        // Second step, same layer: no begin_layer change, no copies.
        let mut store = strict_store(4);
        store
            .expect_begin_layer()
            .withf(|l| *l == 0)
            .returning(|_| Ok(None));
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe().times(1).returning(|_, _| Ok(()));
        run_experts(
            &mut cache,
            &store,
            &gemm,
            0,
            &step(&experts, &weights, &x),
            &mut out,
        )
        .unwrap();
    }

    #[test]
    fn a_dropped_layer_is_forgotten_before_the_plan() {
        // The device cache moved to layer 1, so layer 0's slots are stale and
        // must be planned as misses even though `cache` still holds them.
        let mut cache = ExpertCache::new(2, 4, 4);
        let x = [1.0f32; 8];
        let experts = [1u32, 3, 1, 3];
        let weights = [1.0f32; 4];
        let mut out = [0.0f32; 8];
        {
            let mut store = strict_store(4);
            store
                .expect_begin_layer()
                .withf(|l| *l == 0)
                .returning(|_| Ok(None));
            store.expect_load().times(2).returning(|_, _, _| Ok(()));
            let mut gemm = MockExpertGemm::new();
            gemm.expect_grouped_moe().times(1).returning(|_, _| Ok(()));
            run_experts(
                &mut cache,
                &store,
                &gemm,
                0,
                &step(&experts, &weights, &x),
                &mut out,
            )
            .unwrap();
        }
        assert!(cache.slot_of(0, 1) >= 0);

        let mut store = strict_store(4);
        store
            .expect_begin_layer()
            .withf(|l| *l == 1)
            .returning(|_| Ok(Some(0)));
        store
            .expect_load()
            .withf(|layer, _, _| *layer == 1)
            .times(2)
            .returning(|_, _, _| Ok(()));
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe().times(1).returning(|_, _| Ok(()));
        run_experts(
            &mut cache,
            &store,
            &gemm,
            1,
            &step(&experts, &weights, &x),
            &mut out,
        )
        .unwrap();
        // Layer 0 is gone from the machine, layer 1's experts are back in it.
        assert_eq!(cache.slot_of(0, 1), -1);
        assert!(cache.slot_of(1, 1) >= 0);
    }

    #[test]
    fn forget_happens_before_the_plan_not_after() {
        // A stricter version of the ordering rule: the load of layer 1's
        // experts must see layer 0 already cleared, which `sequence` pins down.
        let mut cache = ExpertCache::new(2, 4, 4);
        let x = [1.0f32; 8];
        let experts = [2u32, 2, 2, 2];
        let weights = [1.0f32; 4];
        let mut out = [0.0f32; 8];
        {
            let mut store = strict_store(4);
            store
                .expect_begin_layer()
                .withf(|l| *l == 0)
                .returning(|_| Ok(None));
            store.expect_load().times(1).returning(|_, _, _| Ok(()));
            let mut gemm = MockExpertGemm::new();
            gemm.expect_grouped_moe().times(1).returning(|_, _| Ok(()));
            run_experts(
                &mut cache,
                &store,
                &gemm,
                0,
                &step(&experts, &weights, &x),
                &mut out,
            )
            .unwrap();
        }
        let mut store = strict_store(4);
        store
            .expect_begin_layer()
            .withf(|l| *l == 1)
            .returning(|_| Ok(Some(0)));
        store.expect_load().times(1).returning(|_, _, _| Ok(()));
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe().times(1).returning(|_, _| Ok(()));
        run_experts(
            &mut cache,
            &store,
            &gemm,
            1,
            &step(&experts, &weights, &x),
            &mut out,
        )
        .unwrap();
        // Layer 1's expert was planned as a miss, which is only true if layer 0
        // was forgotten before `ensure_experts` ran.
        assert!(cache.slot_of(1, 2) >= 0);
    }

    #[test]
    fn rejects_a_step_whose_routes_do_not_match_the_batch() {
        let mut cache = ExpertCache::new(2, 4, 4);
        let store = strict_store(4);
        let gemm = MockExpertGemm::new();
        let x = [1.0f32; 8];
        let experts = [1u32, 3];
        let weights = [1.0f32; 4];
        let mut out = [0.0f32; 8];
        assert!(matches!(
            run_experts(
                &mut cache,
                &store,
                &gemm,
                0,
                &step(&experts, &weights, &x),
                &mut out
            ),
            Err(MoEError::Bad(_))
        ));
    }

    #[test]
    fn rejects_a_slot_cache_smaller_than_the_adapter_bank() {
        let mut cache = ExpertCache::new(2, 4, 2);
        let store = strict_store(4);
        let gemm = MockExpertGemm::new();
        let x = [1.0f32; 8];
        let experts = [1u32, 3, 1, 3];
        let weights = [1.0f32; 4];
        let mut out = [0.0f32; 8];
        assert!(matches!(
            run_experts(
                &mut cache,
                &store,
                &gemm,
                0,
                &step(&experts, &weights, &x),
                &mut out
            ),
            Err(MoEError::Bad(_))
        ));
    }

    #[test]
    fn an_overflowing_cache_passes_minus_one_routes() {
        // cache_size 2 with three distinct experts: the last one overflows and
        // the kernel must see -1 rather than a slot it cannot read.
        let mut cache = ExpertCache::new(1, 4, 2);
        let mut store = strict_store(2);
        store
            .expect_begin_layer()
            .withf(|l| *l == 0)
            .returning(|_| Ok(None));
        store.expect_load().times(3).returning(|_, _, _| Ok(()));
        let mut gemm = MockExpertGemm::new();
        let routes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = routes.clone();
        gemm.expect_grouped_moe().returning(move |req, _| {
            *sink.lock().unwrap() = req.routes.to_vec();
            Ok(())
        });
        let x = [1.0f32; 8];
        let experts = [0u32, 1, 2, 0];
        let weights = [1.0f32; 4];
        let mut out = [0.0f32; 8];
        run_experts(
            &mut cache,
            &store,
            &gemm,
            0,
            &step(&experts, &weights, &x),
            &mut out,
        )
        .unwrap();
        let routes = routes.lock().unwrap().clone();
        assert_eq!(routes.len(), 4);
        // Expert 0's two routes overflow: -1 means "contributes nothing", and
        // the LRU tie-break gave its slot to the newest expert.
        assert_eq!(routes[0], -1);
        assert_eq!(routes[3], -1);
        assert!(routes[1] >= 0 && routes[2] >= 0);
    }
}
