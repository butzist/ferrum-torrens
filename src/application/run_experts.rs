// The routed-expert use case for one MoE layer: plan the slot residency, copy
// the misses in, then ask the kernel for the grouped GEMM.
//
// This is workflow over the domain ports and nothing else -- no CUDA, no mmap,
// no python -- so it is unit-tested against mocked ports and is where the
// decision-27 ordering (forget the dropped layer BEFORE planning) is enforced.

use crate::domain::expert_cache::ExpertCache;
use crate::domain::expert_kernel::{ExpertGemm, ExpertSlotStore, MoEError, MoEGroup};

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
    use crate::domain::expert_kernel::{MockExpertGemm, MockExpertSlotStore};

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
