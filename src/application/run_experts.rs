// The routed-expert use case for one MoE layer: plan the slot residency, copy
// the misses in, then ask the kernel for the grouped GEMM.
//
// This is workflow over the domain ports and nothing else -- no CUDA, no mmap,
// no python -- so it is unit-tested against mocked ports. The slot machine's ids
// are flat (`layer * num_experts + expert`) and the device banks hold every
// layer at once (decision 40 prerequisite 2), so a layer switch needs no
// invalidation: the plan alone decides what is resident.

use crate::domain::expert_cache::ExpertCache;
use crate::domain::expert_kernel::{
    ExpertGemm, ExpertSlotStore, MoEDeviceGroup, MoEDeviceRequest, MoEError, MoEExecutor, MoEGroup,
    MoETensor, PendingMoE,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

/// The engine's `MoEExecutor`: the same three pieces `run_experts` takes, held
/// behind one shared handle so a tower that only has `&self` can call it. The
/// slot machine is the mutable part (it counts touches for the LRU), the two
/// ports are `&self`, and one device cache serves every step, which is why the
/// adapter takes its own lock.
pub struct CachedExperts {
    cache: Mutex<ExpertCache>,
    store: Arc<dyn ExpertSlotStore>,
    gemm: Arc<dyn ExpertGemm>,
    steps: AtomicU64,
    /// Whether the adapter has device staging. When it does, every step goes
    /// through `grouped_moe_staged` -- the adapter chunks a large route list
    /// internally, so prefill uses the device LRU and the pinned banks too, not
    /// just decode. Off falls back to the host plan.
    staged: bool,
    cache_size: usize,
    num_layers: usize,
    num_experts: usize,
}

impl CachedExperts {
    pub fn new(
        store: Arc<dyn ExpertSlotStore>,
        gemm: Arc<dyn ExpertGemm>,
        num_layers: usize,
        num_experts: usize,
        cache_size: usize,
    ) -> Self {
        Self::build(store, gemm, num_layers, num_experts, cache_size, false)
    }

    /// The host-plan executor unless the adapter reports device staging. The
    /// registry uses this once the pinned banks are loaded, so a step whose
    /// routes fit `staged_route_cap` never reads the router's ids on the host.
    pub fn with_device_staging(
        store: Arc<dyn ExpertSlotStore>,
        gemm: Arc<dyn ExpertGemm>,
        num_layers: usize,
        num_experts: usize,
        cache_size: usize,
    ) -> Self {
        Self::build(store, gemm, num_layers, num_experts, cache_size, true)
    }

    fn build(
        store: Arc<dyn ExpertSlotStore>,
        gemm: Arc<dyn ExpertGemm>,
        num_layers: usize,
        num_experts: usize,
        cache_size: usize,
        staged: bool,
    ) -> Self {
        Self {
            cache: Mutex::new(ExpertCache::new(num_layers, num_experts, cache_size)),
            store,
            gemm,
            steps: AtomicU64::new(0),
            staged,
            cache_size,
            num_layers,
            num_experts,
        }
    }

    /// Slots the machine can hand out. A step's plan needs this to be at least
    /// the adapter's bank size, which `run_experts` checks.
    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    /// Steps this executor has run, counted at the port so a caller can prove
    /// the kernel was in the loop.
    pub fn steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }

    fn locked(&self) -> std::sync::MutexGuard<'_, ExpertCache> {
        self.cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl MoEExecutor for CachedExperts {
    fn steps(&self) -> u64 {
        self.steps()
    }

    /// Free the device banks and forget the slot plan. The store keeps the
    /// pinned host source, so the next step restages exactly the experts it
    /// routes to; the slot machine starts empty because every bank slot is
    /// uninitialized memory again.
    fn release_cache(&self) -> Result<(), MoEError> {
        self.store.release_banks()?;
        *self.locked() = ExpertCache::new(self.num_layers, self.num_experts, self.cache_size);
        Ok(())
    }

    fn rehydrate_cache(&self) -> Result<(), MoEError> {
        self.store.rehydrate_banks()
    }

    fn cache_resident(&self) -> bool {
        self.store.banks_resident()
    }

    fn execute_device<'a>(
        &self,
        req: &MoEDeviceRequest<'a>,
        out: &mut dyn MoETensor,
    ) -> Result<(), MoEError> {
        // Counted before the work, so a step the kernel refuses still shows up:
        // the count is evidence that the port was reached, not that it succeeded.
        self.steps.fetch_add(1, Ordering::Relaxed);
        let tokens = check_device_step(req)?;
        if out.numel() < tokens * req.hidden {
            return Err(MoEError::Bad(format!(
                "device expert step: {} outputs for {tokens} tokens x hidden {}",
                out.numel(),
                req.hidden
            )));
        }
        let routes = tokens * req.top_k;
        if routes == 0 {
            return Err(MoEError::Bad("device expert step: no routes".into()));
        }

        // The staged path owns the plan on the device, so a staged executor uses
        // it for every step. A route list past the committed `lru_ensure` block
        // is chunked inside the adapter (prefill), not sent to the host plan, so
        // the router's ids and the misses never leave the device once staging is
        // on. The host plan stays the whole path when there is no staging.
        if self.staged {
            return self.gemm.grouped_moe_staged(req, out);
        }
        run_experts_device(
            &mut self.locked(),
            self.store.as_ref(),
            self.gemm.as_ref(),
            req,
            out,
        )
    }

    fn begin_device<'a>(
        &self,
        req: &MoEDeviceRequest<'a>,
    ) -> Result<Option<Box<dyn PendingMoE>>, MoEError> {
        if !self.staged {
            return Ok(None);
        }
        if std::env::var("FERRUM_MOE_OVERLAP").ok().as_deref() == Some("0") {
            return Ok(None);
        }
        let tokens = check_device_step(req)?;
        if req.x.numel() < tokens * req.hidden {
            return Err(MoEError::Bad(format!(
                "device expert step: {} outputs for {tokens} tokens x hidden {}",
                req.x.numel(),
                req.hidden
            )));
        }
        match self.gemm.begin_staged(req)? {
            Some(pending) => {
                self.steps.fetch_add(1, Ordering::Relaxed);
                Ok(Some(pending))
            }
            None => Ok(None),
        }
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
    let routes = stage_and_route(cache, store, layer, step.experts)?;
    gemm.grouped_moe(
        &MoEGroup {
            hidden: step.hidden,
            inter: step.inter,
            x: step.x,
            top_k: step.top_k,
            routes: &routes,
            weights: step.weights,
        },
        out,
    )
}

/// Reject a device step whose buffers do not divide into a dense batch and
/// report the token count. Element counts only: a rejected step reads nothing,
/// which is what keeps the validation identical on both hand-offs.
fn check_device_step(step: &MoEDeviceRequest<'_>) -> Result<usize, MoEError> {
    let (hidden, top_k) = (step.hidden, step.top_k);
    let (x_len, expert_count, weight_count) =
        (step.x.numel(), step.experts.numel(), step.weights.numel());
    if hidden == 0 || top_k == 0 || x_len == 0 || !x_len.is_multiple_of(hidden) {
        return Err(MoEError::Bad(format!(
            "device expert step: hidden {hidden} top_k {top_k} x {x_len}"
        )));
    }
    let tokens = x_len / hidden;
    if expert_count != tokens * top_k || weight_count != tokens * top_k {
        return Err(MoEError::Bad(format!(
            "device expert step: {expert_count} experts and {weight_count} weights \
             for {tokens} tokens x top_k {top_k}"
        )));
    }
    Ok(tokens)
}

/// One routed layer step with the activations, the weights and the output
/// still in their device buffers (decision 43). Only the routes leave the
/// device, because the plan is computed here -- prerequisite 4 moves that too.
///
/// `out` receives `tokens * hidden` f32, and the slot rewrites happen exactly
/// as they do on the host path: nothing about the residency changes with the
/// hand-off.
pub fn run_experts_device(
    cache: &mut ExpertCache,
    store: &dyn ExpertSlotStore,
    gemm: &dyn ExpertGemm,
    step: &MoEDeviceRequest<'_>,
    out: &mut dyn MoETensor,
) -> Result<(), MoEError> {
    let hidden = step.hidden;
    let tokens = check_device_step(step)?;
    if out.numel() < tokens * hidden {
        return Err(MoEError::Bad(format!(
            "device expert step: {} outputs for {tokens} tokens x hidden {hidden}",
            out.numel()
        )));
    }

    // The one crossing this hand-off still makes: the router's dense ids, read
    // once so the residency plan can be computed on the host.
    let experts: Vec<u32> = step
        .experts
        .to_host_f32()?
        .into_iter()
        .map(|e| e as u32)
        .collect();
    let routes = stage_and_route(cache, store, step.layer, &experts)?;
    gemm.grouped_moe_dev(
        &MoEDeviceGroup {
            hidden: step.hidden,
            inter: step.inter,
            top_k: step.top_k,
            x: step.x,
            routes: &routes,
            weights: step.weights,
        },
        out,
    )
}

/// The part of a step that is identical whichever way the data is handed over:
/// reject a cache smaller than the adapter's banks, plan the residency, stage
/// the misses and hand back the slot routes the kernel reads (`-1` for an
/// expert the cache could not hold).
fn stage_and_route(
    cache: &mut ExpertCache,
    store: &dyn ExpertSlotStore,
    layer: usize,
    experts: &[u32],
) -> Result<Vec<i32>, MoEError> {
    let slots = store.slots();
    if cache.cache_size() < slots {
        return Err(MoEError::Bad(format!(
            "slot cache holds {} slots but the adapter stages {}",
            cache.cache_size(),
            slots
        )));
    }
    let plan = cache.ensure_experts(layer, experts);
    for (&expert, &slot) in plan.misses.iter().zip(&plan.write_slots) {
        store.load(layer, expert, slot)?;
    }
    Ok(plan.rewrites)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::expert_kernel::{
        MoEDeviceRequest, MoETensor, MockExpertGemm, MockExpertSlotStore, MockPendingMoE,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Routes plus the three shapes the kernel was handed.
    type SeenKernel = (Vec<i32>, usize, usize, usize);

    /// A host buffer behind the domain's opaque device port. These tests run
    /// with nothing on a device at all, and the read counter is how they prove
    /// which buffers the step really touched: the plan reads the routes once and
    /// never the activations or the weights.
    struct Buf {
        vals: Vec<f32>,
        reads: AtomicUsize,
    }

    impl Buf {
        fn new(vals: Vec<f32>) -> Self {
            Self {
                vals,
                reads: AtomicUsize::new(0),
            }
        }

        fn zeros(len: usize) -> Self {
            Self::new(vec![0.0; len])
        }

        fn reads(&self) -> usize {
            self.reads.load(Ordering::Relaxed)
        }

        /// The address a port sees, so a test can tell the buffer itself from a
        /// copy of it without borrowing the local into a `'static` closure.
        fn addr(&self) -> usize {
            let fat: *const dyn MoETensor = self;
            fat as *const () as usize
        }
    }

    impl MoETensor for Buf {
        fn numel(&self) -> usize {
            self.vals.len()
        }

        fn to_host_f32(&self) -> Result<Vec<f32>, MoEError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            Ok(self.vals.clone())
        }

        fn set_host_f32(&mut self, values: &[f32]) -> Result<(), MoEError> {
            if values.len() != self.vals.len() {
                return Err(MoEError::Bad(format!(
                    "{} values for a {}-element buffer",
                    values.len(),
                    self.vals.len()
                )));
            }
            self.vals.copy_from_slice(values);
            Ok(())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    /// The shapes every request in this module uses: two tokens, hidden 4,
    /// inter 4, top_k 2.
    fn device_request<'a>(
        layer: usize,
        x: &'a Buf,
        experts: &'a Buf,
        weights: &'a Buf,
    ) -> MoEDeviceRequest<'a> {
        MoEDeviceRequest {
            layer,
            hidden: 4,
            inter: 4,
            top_k: 2,
            x,
            experts,
            weights,
        }
    }

    fn step_buffers() -> (Buf, Buf, Buf, Buf) {
        (
            Buf::new(vec![1.0f32; 8]),
            Buf::new(vec![6.0, 1.0, 6.0, 1.0]),
            Buf::new(vec![0.25f32; 4]),
            Buf::zeros(8),
        )
    }

    /// The port the tower sees: expert ids go in, slot ids reach the kernel and
    /// the kernel's output comes back in `out`. The activations are handed over
    /// as the very buffer the tower passed in -- no host view of them anywhere,
    /// which is decision 43's whole point.
    #[test]
    fn the_executor_rewrites_expert_ids_into_slots_for_the_kernel() {
        let mut store = strict_store(4);
        store.expect_load().times(2).returning(|_, _, _| Ok(()));
        let seen: Arc<Mutex<Option<SeenKernel>>> = Arc::new(Mutex::new(None));
        let sink = seen.clone();
        let (x, experts, weights, mut out) = step_buffers();
        let x_addr = x.addr();
        let w_addr = weights.addr();
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe_dev().returning(move |req, out| {
            assert_eq!(
                req.x as *const dyn MoETensor as *const () as usize, x_addr,
                "the activations crossed as the tower's own buffer, not a copy"
            );
            assert_eq!(
                req.weights as *const dyn MoETensor as *const () as usize, w_addr,
                "the weights crossed as the tower's own buffer too"
            );
            *sink.lock().unwrap() = Some((req.routes.to_vec(), req.hidden, req.inter, req.top_k));
            out.set_host_f32(&vec![0.5f32; out.numel()])
        });
        let exec = CachedExperts::new(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        exec.execute_device(&device_request(3, &x, &experts, &weights), &mut out)
            .unwrap();

        // The kernel is told slots, not expert ids, and it keeps the shapes.
        let (routes, hidden, inter, top_k) = seen.lock().unwrap().clone().unwrap();
        assert_eq!(routes.len(), 4);
        assert!(routes.iter().all(|s| *s >= 0 && *s < 4));
        assert_ne!(routes[0], 6);
        assert_eq!(routes[0], routes[2]);
        assert_eq!((hidden, inter, top_k), (4, 4, 2));
        assert_eq!(out.vals, vec![0.5f32; 8]);
        // The plan read the routes once; the activations and the weights were
        // never asked for a host view.
        assert_eq!(experts.reads(), 1, "the routes are read exactly once");
        assert_eq!(x.reads(), 0, "the activations stayed on the device");
        assert_eq!(weights.reads(), 0, "the weights stayed on the device");
    }

    /// The executor counts the steps it ran, which is the only way a caller can
    /// tell the kernel path from the inline host one: both answer with logits,
    /// so a gate that only checks the token cannot see a silently dropped port.
    #[test]
    fn the_executor_counts_the_steps_it_ran() {
        let mut store = strict_store(4);
        store.expect_load().times(2).returning(|_, _, _| Ok(()));
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe_dev()
            .times(2)
            .returning(|_, _| Ok(()));
        let exec = CachedExperts::new(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        assert_eq!(exec.steps(), 0, "nothing has run yet");

        let (x, experts, weights, mut out) = step_buffers();
        exec.execute_device(&device_request(3, &x, &experts, &weights), &mut out)
            .unwrap();
        exec.execute_device(&device_request(3, &x, &experts, &weights), &mut out)
            .unwrap();
        assert_eq!(exec.steps(), 2);
    }

    /// A second step over the same layer stages nothing: the slot machine is the
    /// executor's own state, which is why the port can be `&self`.
    #[test]
    fn a_repeated_step_over_one_layer_stages_nothing() {
        let mut store = strict_store(4);
        store.expect_load().times(2).returning(|_, _, _| Ok(()));
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe_dev()
            .times(2)
            .returning(|_, _| Ok(()));
        let exec = CachedExperts::new(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        let (x, experts, weights, mut out) = step_buffers();
        exec.execute_device(&device_request(3, &x, &experts, &weights), &mut out)
            .unwrap();
        // `store` allows no further `load` calls, so this second step only
        // passes if both experts were still resident.
        exec.execute_device(&device_request(3, &x, &experts, &weights), &mut out)
            .unwrap();
    }

    /// With device staging on, a step that fits the cap never reads the router's
    /// ids on the host: the adapter's `lru_ensure` rewrites them on the device.
    /// The store is never asked to stage either, because the kernel copied the
    /// misses itself.
    #[test]
    fn a_staged_step_keeps_the_router_ids_on_the_device() {
        let store = strict_store(4);
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe_staged()
            .times(1)
            .returning(|_, out| out.set_host_f32(&vec![0.5f32; out.numel()]));
        let exec = CachedExperts::with_device_staging(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        let (x, experts, weights, mut out) = step_buffers();
        exec.execute_device(&device_request(3, &x, &experts, &weights), &mut out)
            .unwrap();
        assert_eq!(experts.reads(), 0, "the ids stayed on the device");
        assert_eq!(x.reads(), 0);
        assert_eq!(weights.reads(), 0);
        assert_eq!(out.vals, vec![0.5f32; 8]);
    }

    /// A step larger than the cubin's route cap stays on the device too: the
    /// adapter chunks it internally (`grouped_moe_staged`), so prefill never
    /// falls back to the host plan and never reads the router's ids on the host.
    /// The cap is the adapter's business, so this test does not set one.
    #[test]
    fn a_staged_step_past_the_cap_still_stages_on_the_device() {
        let store = strict_store(4);
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe_staged()
            .times(1)
            .returning(|_, _| Ok(()));
        let exec = CachedExperts::with_device_staging(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        let (x, experts, weights, mut out) = step_buffers();
        exec.execute_device(&device_request(3, &x, &experts, &weights), &mut out)
            .unwrap();
        assert_eq!(experts.reads(), 0, "the ids never reached the host");
    }

    #[test]
    fn a_layer_switch_keeps_the_other_layer_resident() {
        // Decision 40 prerequisite 2: the device banks hold every layer at once
        // and are never dropped, so switching layers must not invalidate the
        // residency the slot machine knows about. Four slots, four experts over
        // two layers: the return to layer 0 is a hit, and the store's budget of
        // exactly four `load` calls turns any re-stage into a panic.
        let mut cache = ExpertCache::new(2, 4, 4);
        let x = [1.0f32; 8];
        let experts = [1u32, 3, 1, 3];
        let weights = [0.25f32; 4];
        let mut out = [0.0f32; 8];

        let mut store = strict_store(4);
        store.expect_load().times(4).returning(|_, _, _| Ok(()));
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe().times(3).returning(|_, _| Ok(()));

        run_experts(
            &mut cache,
            &store,
            &gemm,
            0,
            &step(&experts, &weights, &x),
            &mut out,
        )
        .unwrap();
        run_experts(
            &mut cache,
            &store,
            &gemm,
            1,
            &step(&experts, &weights, &x),
            &mut out,
        )
        .unwrap();
        assert!(cache.slot_of(0, 1) >= 0, "layer 0 stayed resident");
        assert!(cache.slot_of(1, 1) >= 0, "layer 1 is resident too");
        // Back to layer 0: still resident, so the fourth and last `load` was
        // spent on layer 1 above and this call stages nothing.
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

    /// A malformed request is rejected by the use case without staging
    /// anything or launching the kernel.
    #[test]
    fn a_request_that_does_not_match_its_batch_never_reaches_the_adapter() {
        let store = strict_store(4);
        let gemm = MockExpertGemm::new();
        let exec = CachedExperts::new(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        let x = Buf::new(vec![1.0f32; 8]);
        let experts = Buf::new(vec![6.0, 1.0]);
        let weights = Buf::new(vec![0.25f32; 4]);
        let mut out = Buf::zeros(8);
        assert!(matches!(
            exec.execute_device(&device_request(3, &x, &experts, &weights), &mut out),
            Err(MoEError::Bad(_))
        ));
    }

    /// The device hand-off rejects the batch the host one rejects, and it does
    /// so on element counts rather than on a host view of the buffers.
    #[test]
    fn a_device_step_whose_activations_do_not_divide_never_reaches_the_adapter() {
        let store = strict_store(4);
        let gemm = MockExpertGemm::new();
        let exec = CachedExperts::new(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        let x = Buf::new(vec![1.0f32; 7]);
        let experts = Buf::new(vec![6.0, 1.0, 6.0, 1.0]);
        let weights = Buf::new(vec![0.25f32; 4]);
        let mut out = Buf::zeros(8);
        assert!(matches!(
            exec.execute_device(&device_request(3, &x, &experts, &weights), &mut out),
            Err(MoEError::Bad(_))
        ));
        assert_eq!(x.reads(), 0, "a rejected step reads nothing at all");
    }

    /// `begin_device` counts the step once when it actually starts an
    /// overlapped step, and leaves the count alone when it declines (so the
    /// later `execute_device` fallback is the one that counts).
    #[test]
    fn begin_device_counts_a_started_step_once() {
        let store = strict_store(4);
        let mut gemm = MockExpertGemm::new();
        gemm.expect_begin_staged().returning(|_| {
            let mut pending = MockPendingMoE::new();
            pending.expect_finish().returning(|_| Ok(()));
            Ok(Some(Box::new(pending)))
        });
        let exec = CachedExperts::with_device_staging(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        let (x, experts, weights, mut out) = step_buffers();
        assert_eq!(exec.steps(), 0);
        let handle = exec
            .begin_device(&device_request(3, &x, &experts, &weights))
            .unwrap();
        assert!(handle.is_some());
        assert_eq!(exec.steps(), 1);
        handle.unwrap().finish(&mut out).unwrap();
        assert_eq!(exec.steps(), 1);
    }

    #[test]
    fn begin_device_declining_does_not_count_and_fallback_does() {
        let store = strict_store(4);
        let mut gemm = MockExpertGemm::new();
        gemm.expect_begin_staged().returning(|_| Ok(None));
        gemm.expect_grouped_moe_staged()
            .times(1)
            .returning(|_, out| out.set_host_f32(&vec![0.5f32; out.numel()]));
        let exec = CachedExperts::with_device_staging(Arc::new(store), Arc::new(gemm), 4, 8, 4);
        let (x, experts, weights, mut out) = step_buffers();
        assert_eq!(exec.steps(), 0);
        let handle = exec
            .begin_device(&device_request(3, &x, &experts, &weights))
            .unwrap();
        assert!(handle.is_none());
        assert_eq!(exec.steps(), 0);
        exec.execute_device(&device_request(3, &x, &experts, &weights), &mut out)
            .unwrap();
        assert_eq!(exec.steps(), 1);
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

        // Second step, same layer: nothing changed in the store, no copies.
        let store = strict_store(4);
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

    /// Freeing the banks must also forget the slot plan: every expert the next
    /// step routes to has to be staged again, because the bank memory is fresh.
    #[test]
    fn releasing_the_expert_cache_restages_every_expert_next_step() {
        let mut store = strict_store(4);
        store.expect_banks_resident().return_const(true);
        store.expect_load().times(4).returning(|_, _, _| Ok(()));
        store.expect_release_banks().times(1).returning(|| Ok(()));
        store.expect_rehydrate_banks().times(1).returning(|| Ok(()));
        let mut gemm = MockExpertGemm::new();
        gemm.expect_grouped_moe_dev()
            .times(2)
            .returning(|_, _| Ok(()));
        let executor = CachedExperts::new(Arc::new(store), Arc::new(gemm), 2, 4, 4);
        let (x, _defaults, weights, mut out) = step_buffers();
        let experts = Buf::new(vec![1.0, 3.0, 1.0, 3.0]);
        let req = device_request(0, &x, &experts, &weights);
        executor.execute_device(&req, &mut out).unwrap();
        assert!(executor.cache_resident());

        executor.release_cache().unwrap();
        executor.rehydrate_cache().unwrap();
        executor.execute_device(&req, &mut out).unwrap();
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
