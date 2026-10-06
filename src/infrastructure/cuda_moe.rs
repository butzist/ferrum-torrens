// The NVFP4 expert adapter on the CUDA driver API: decision 36 slice 4, and the
// replacement for the pyo3 `triton.rs` adapter.
//
// What changed with Python gone: the slot cache's banks are libtorch tensors on
// the CUDA device, and the four kernels are launched through `cuda_driver` from
// the cubins committed under `kernel/`. Decision 25's hazard -- two libtorch
// builds having to agree on C++ ABI so tensors could be shared by pointer -- is
// gone with the interpreter, because the banks now *are* libtorch tensors and the
// launches run on libtorch's own stream.
//
// The banks are byte banks on purpose. The GEMV wants `*fp8e4nv` scales and
// `*fp16` globals, and libtorch 2.9 cannot build those views from a host slice
// through tch (`from_blob` is absent, and a dtype-reinterpreting `view` is a
// newer C++ API than the bindings expose). A `uint8` tensor over the same bytes
// is the same device memory with the same alignment, and the kernel reinterprets
// the address itself, so bytes are all that has to cross. The one dtype this
// adapter does use is bf16, for the activations and the intermediates: that is
// what the cubins were specialized for and what the checkpoint itself stores.

use std::sync::{Arc, Mutex};

use tch::Tensor;

use crate::domain::expert_kernel::{
    ExpertGemm, ExpertShapes, ExpertSlabSource, ExpertSlotStore, MoEError, MoEGroup, Part, Slab,
    route_lists,
};
use crate::infrastructure::cuda_driver::{CUstream, CudaError, CudaKernels, Driver};
use crate::infrastructure::kernel_abi::{
    Abi, GEMV_DOWN, GEMV_GATE_UP, MoEArgs, SILU_MUL, SUM_REDUCE, gemv_down_args, gemv_gate_up_args,
    grid, silu_mul_args, sum_reduce_args,
};
use crate::infrastructure::torch::{Device, bf16_tensor, f32_tensor, to_f32_vec};

/// e2m1 magnitudes by nibble, the decode table the GEMV indexes. The low nibble
/// is the magnitude and the high bit the sign, which is why the negatives follow.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn driver(e: CudaError) -> MoEError {
    MoEError::Bad(format!("driver: {e}"))
}

fn bad_abi(e: MoEError) -> MoEError {
    MoEError::Bad(format!("abi: {e}"))
}

/// One part's device banks: `[slots, rows, cols / 2]` packed, `[slots, rows,
/// cols / 16]` scales and `[slots, rows * 2]` globals, all uint8 bytes.
///
/// Each slot is a contiguous run in all three, so `narrow` gives a view of exactly
/// one expert's bytes. `copy_` does not flatten, so the staged host tensor's
/// length has to match that view's element count or libtorch raises a shape error.
struct Banks {
    packed: Tensor,
    scale: Tensor,
    global: Tensor,
}

impl Banks {
    fn new(slots: usize, rows: usize, cols: usize, device: Device) -> Self {
        let bytes = |n: usize| Tensor::zeros([n as i64], (tch::Kind::Uint8, device.to_tch()));
        Self {
            packed: bytes(slots * rows * cols / 2),
            scale: bytes(slots * rows * cols / 16),
            global: bytes(slots * rows * 2),
        }
    }

    /// Copy one slab into `slot`: the H2D half of a cache miss.
    fn stage(
        &mut self,
        slot: usize,
        slab: &Slab,
        rows: usize,
        cols: usize,
    ) -> Result<(), MoEError> {
        stage_bytes(&mut self.packed, slot, &slab.packed, rows * cols / 2)?;
        stage_bytes(&mut self.scale, slot, &slab.scale, rows * cols / 16)?;
        stage_bytes(&mut self.global, slot, &slab.global, rows * 2)
    }
}

fn stage_bytes(bank: &mut Tensor, slot: usize, bytes: &[u8], len: usize) -> Result<(), MoEError> {
    if bytes.len() != len {
        return Err(MoEError::Bad(format!(
            "slab is {} bytes for a {len}-byte slot row",
            bytes.len()
        )));
    }
    bank.narrow(0, slot as i64 * len as i64, len as i64)
        .copy_(&Tensor::from_slice(bytes));
    Ok(())
}

/// The device slot cache for one layer, plus the launch path.
///
/// The ports are implemented for `Mutex<CudaMoE>`: staging a miss is `&mut self`
/// (it rewrites banks) while one device cache serves every step, so the engine
/// holds the adapter behind one lock and hands out the same handle as both ports.
pub struct CudaMoE {
    kernels: CudaKernels,
    abi: Abi,
    source: Arc<dyn ExpertSlabSource>,
    shapes: ExpertShapes,
    slots: usize,
    device: Device,
    /// libtorch's default stream, whose handle is 0 (the legacy stream). Torch's
    /// own copies and our launches are therefore ordered by CUDA rather than by
    /// an explicit sync. It is a field because CUDA-graph capture will need a
    /// side stream, and torch's C API exports no `getCurrentCUDAStream` to ask.
    stream: CUstream,
    layer: Option<usize>,
    gate_up: Option<Banks>,
    down: Option<Banks>,
}

// SAFETY: the fields are libtorch tensors, Arcs, plain data, and driver handles
// that are immutable after construction. Two concurrent `grouped_moe` calls
// would read the same banks (read-only) and write their own activations, and
// both launches land on the same stream in order anyway. The ports hand this out
// behind a Mutex regardless.
unsafe impl Send for CudaMoE {}
unsafe impl Sync for CudaMoE {}

/// One step's device buffers, kept alive across calls so a CUDA graph can be
/// captured over their addresses (decision 39): the launches are the graph,
/// and everything that writes these tensors happens before it. A group must
/// outlive every graph captured on top of it -- dropping it frees the memory
/// replay would read -- and the batch shape is pinned with it, because a
/// captured grid cannot reshape. `stage_into` is the only door into a replay.
pub struct StagedGroup {
    tokens: usize,
    top_k: usize,
    routes: usize,
    x: Tensor,
    slot_t: Tensor,
    token_t: Tensor,
    weight_t: Tensor,
    lut: Tensor,
    ic1: Tensor,
    ic2: Tensor,
    ic3: Tensor,
    result: Tensor,
}

impl CudaMoE {
    /// Open the driver, load the committed cubins and check they were built for
    /// `shapes`. Nothing is allocated yet: the banks come with the first layer.
    pub fn new(
        device: Device,
        source: Arc<dyn ExpertSlabSource>,
        shapes: ExpertShapes,
        slots: usize,
    ) -> Result<Self, MoEError> {
        Self::new_with_stream(device, source, shapes, slots, std::ptr::null_mut())
    }

    pub fn new_with_stream(
        device: Device,
        source: Arc<dyn ExpertSlabSource>,
        shapes: ExpertShapes,
        slots: usize,
        stream: CUstream,
    ) -> Result<Self, MoEError> {
        let Device::Cuda(ordinal) = device else {
            return Err(MoEError::Bad(
                "the driver-API MoE adapter needs a CUDA device".into(),
            ));
        };
        if slots == 0 {
            return Err(MoEError::Bad(
                "the slot cache needs at least one slot".into(),
            ));
        }
        // The banks are libtorch tensors, so ATen needs its CUDA kernels
        // registered before the first allocation (see `load_cuda_backend`).
        crate::infrastructure::torch::load_cuda_backend()
            .map_err(|e| MoEError::Bad(format!("libtorch cuda: {e}")))?;
        let lib = Driver::open().map_err(driver)?;
        lib.init().map_err(driver)?;
        lib.attach_primary_context(ordinal as i32).map_err(driver)?;
        let abi = Abi::committed().map_err(bad_abi)?;
        check_shapes(&shapes, &abi)?;
        let kernels = CudaKernels::load(Arc::new(lib), abi.clone()).map_err(driver)?;
        Ok(Self {
            kernels,
            abi,
            source,
            shapes,
            slots,
            device,
            stream,
            layer: None,
            gate_up: None,
            down: None,
        })
    }

    pub fn stream(&self) -> CUstream {
        self.stream
    }

    pub fn shapes(&self) -> ExpertShapes {
        self.shapes
    }

    /// The layer whose experts the cache currently holds.
    pub fn layer(&self) -> Option<usize> {
        self.layer
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Bytes per expert, so the engine can check the VRAM budget before the first
    /// miss allocates anything.
    pub fn expert_bytes(&self) -> usize {
        per_expert(self.shapes)
    }

    pub fn cache_bytes(&self) -> usize {
        self.slots * self.expert_bytes()
    }

    /// Switch the banks to `layer`, reporting the layer whose bytes were dropped
    /// (decision 27). The caller replays that into `ExpertCache::forget_layer`
    /// before planning, so the slots it still believes in are gone exactly when
    /// the plan is computed.
    pub fn begin_layer(&mut self, layer: usize) -> Result<Option<usize>, MoEError> {
        if self.layer == Some(layer) {
            return Ok(None);
        }
        let dropped = self.layer.take();
        self.gate_up = None;
        self.down = None;
        self.layer = Some(layer);
        Ok(dropped)
    }

    /// Pull both slabs of one expert from the source and stage them.
    pub fn load(&mut self, layer: usize, expert: u32, slot: usize) -> Result<(), MoEError> {
        self.begin_layer(layer)?;
        if slot >= self.slots {
            return Err(MoEError::SlotOutOfRange {
                slot,
                capacity: self.slots,
            });
        }
        for part in [Part::GateUp, Part::Down] {
            let slab = self
                .source
                .slab(layer, expert, part)
                .map_err(|e| MoEError::Bad(format!("slab L{layer} E{expert} {part:?}: {e}")))?;
            let (rows, cols) = self.shapes.slab_shape(part);
            check_slab(&slab, rows, cols)?;
            if self.gate_up.is_none() {
                self.banks_for();
            }
            let banks = match part {
                Part::GateUp => self.gate_up.as_mut().expect("gate_up banks"),
                Part::Down => self.down.as_mut().expect("down banks"),
            };
            banks.stage(slot, &slab, rows, cols)?;
        }
        Ok(())
    }

    fn banks_for(&mut self) {
        if self.gate_up.is_some() {
            return;
        }
        let (gu_rows, gu_cols) = self.shapes.slab_shape(Part::GateUp);
        let (dn_rows, dn_cols) = self.shapes.slab_shape(Part::Down);
        self.gate_up = Some(Banks::new(self.slots, gu_rows, gu_cols, self.device));
        self.down = Some(Banks::new(self.slots, dn_rows, dn_cols, self.device));
    }

    /// Every check the launches need, before anything is allocated: the
    /// cubins' shapes and top_k, the batch's shapes, and banks to read. The
    /// output buffer is checked by `collect`, which knows how many values the
    /// reduce produces.
    fn check(&self, req: &MoEGroup<'_>) -> Result<usize, MoEError> {
        let (hidden, inter, top_k) = (self.shapes.hidden, self.shapes.inter, req.top_k);
        if hidden != self.abi.hidden || inter != self.abi.inter {
            return Err(MoEError::Bad(format!(
                "the committed cubins are for hidden {} inter {}, not {hidden}x{inter}",
                self.abi.hidden, self.abi.inter
            )));
        }
        if top_k != self.abi.top_k {
            return Err(MoEError::Bad(format!(
                "the committed cubins bake top_k {} into the reduce, got {top_k}",
                self.abi.top_k
            )));
        }
        if hidden == 0 || !req.x.len().is_multiple_of(hidden) {
            return Err(MoEError::Bad(format!(
                "grouped_moe: {} activations for hidden {hidden}",
                req.x.len()
            )));
        }
        let tokens = req.x.len() / hidden;
        if tokens == 0 {
            return Err(MoEError::Bad("grouped_moe: a step with no tokens".into()));
        }
        let routes = tokens * top_k;
        if req.routes.len() != routes || req.weights.len() < routes {
            return Err(MoEError::Bad(format!(
                "grouped_moe: {} routes and {} weights for {tokens} tokens x top_k {top_k}",
                req.routes.len(),
                req.weights.len()
            )));
        }
        if self.gate_up.is_none() || self.down.is_none() {
            return Err(MoEError::Bad(
                "grouped_moe before any expert was staged".into(),
            ));
        }
        Ok(tokens)
    }

    /// Validate `req` and write everything the launches read into a fresh set
    /// of device tensors. This is the data-dependent half of a step: the
    /// activations, the routes and the weights arrive from the host here, and
    /// no capture ever sees any of it (decision 39).
    pub fn stage(&self, req: &MoEGroup<'_>) -> Result<StagedGroup, MoEError> {
        let tokens = self.check(req)?;
        let (hidden, inter, top_k) = (self.shapes.hidden, self.shapes.inter, req.top_k);
        let routes = tokens * top_k;
        let (route_token, route_slot) = route_lists(req.routes, tokens, top_k)?;

        // Activations go over as bf16, which is what the cubins were specialized
        // for. The route weights stay f32: the down GEMV applies them as f32.
        let dev = self.device;
        let bf16 = (tch::Kind::BFloat16, dev.to_tch());
        Ok(StagedGroup {
            tokens,
            top_k,
            routes,
            x: bf16_tensor(req.x, &[tokens as i64, hidden as i64], dev),
            slot_t: Tensor::from_slice(&route_slot).to(dev.to_tch()),
            token_t: Tensor::from_slice(&route_token).to(dev.to_tch()),
            weight_t: f32_tensor(&req.weights[..routes], &[routes as i64], dev),
            lut: f32_tensor(&E2M1, &[E2M1.len() as i64], dev),
            ic1: Tensor::empty([routes as i64, (2 * inter) as i64], bf16),
            ic2: Tensor::empty([routes as i64, inter as i64], bf16),
            ic3: Tensor::empty([routes as i64, hidden as i64], bf16),
            result: Tensor::empty([tokens as i64, hidden as i64], bf16),
        })
    }

    /// Rewrite an existing group for the next step of the same shape, keeping
    /// every tensor address: the launches a captured graph holds still point
    /// at these buffers, so this is what makes a replay read new data. A
    /// different batch shape is an error, because the captured grid cannot
    /// follow it.
    pub fn stage_into(&self, req: &MoEGroup<'_>, g: &mut StagedGroup) -> Result<(), MoEError> {
        let tokens = self.check(req)?;
        if tokens != g.tokens || req.top_k != g.top_k {
            return Err(MoEError::Bad(format!(
                "stage_into: the captured group holds {} tokens x top_k {},                  the step is {tokens} x top_k {}",
                g.tokens, g.top_k, req.top_k
            )));
        }
        let (route_token, route_slot) = route_lists(req.routes, tokens, req.top_k)?;
        let dev = self.device;
        g.x.copy_(&bf16_tensor(
            req.x,
            &[tokens as i64, self.shapes.hidden as i64],
            dev,
        ));
        g.slot_t.copy_(&Tensor::from_slice(&route_slot));
        g.token_t.copy_(&Tensor::from_slice(&route_token));
        g.weight_t.copy_(&f32_tensor(
            &req.weights[..g.routes],
            &[g.routes as i64],
            dev,
        ));
        Ok(())
    }

    /// The four launches, in order, on `self.stream` and the banks this owns:
    /// no allocation, no host readback, no synchronization -- only what a
    /// CUDA-graph capture can record (decision 39).
    ///
    /// Routes arrive as slot ids with `-1` for "not resident", and the GEMV
    /// handles that branch-free (it reads slot 0 and zeroes the accumulator), so
    /// nothing is compacted on the host and the reduce stays dense over `top_k`.
    pub fn launch(&self, g: &StagedGroup) -> Result<(), MoEError> {
        let gu = self
            .gate_up
            .as_ref()
            .ok_or_else(|| MoEError::Bad("launch before any expert was staged".into()))?;
        let dn = self
            .down
            .as_ref()
            .ok_or_else(|| MoEError::Bad("launch before any expert was staged".into()))?;
        let (tokens, routes) = (g.tokens, g.routes);

        let ptr = |t: &Tensor| t.data_ptr() as u64;
        let args = MoEArgs {
            tokens,
            routes,
            x: ptr(&g.x),
            gu_packed: ptr(&gu.packed),
            gu_scale: ptr(&gu.scale),
            gu_global: ptr(&gu.global),
            dn_packed: ptr(&dn.packed),
            dn_scale: ptr(&dn.scale),
            dn_global: ptr(&dn.global),
            route_slot: ptr(&g.slot_t),
            route_token: ptr(&g.token_t),
            route_weight: ptr(&g.weight_t),
            lut: ptr(&g.lut),
            ic1: ptr(&g.ic1),
            ic2: ptr(&g.ic2),
            ic3: ptr(&g.ic3),
            out: ptr(&g.result),
        };
        let stream = self.stream;
        for (name, rows, pack) in [
            (
                GEMV_GATE_UP,
                routes,
                gemv_gate_up_args(&self.abi, &args).map_err(bad_abi)?,
            ),
            (
                SILU_MUL,
                routes,
                silu_mul_args(&self.abi, &args).map_err(bad_abi)?,
            ),
            (
                GEMV_DOWN,
                routes,
                gemv_down_args(&self.abi, &args).map_err(bad_abi)?,
            ),
            (
                SUM_REDUCE,
                tokens,
                sum_reduce_args(&self.abi, &args).map_err(bad_abi)?,
            ),
        ] {
            let kernel = self.abi.kernel(name).map_err(bad_abi)?;
            let mut params = pack.kernel_params().map_err(bad_abi)?;
            // SAFETY: every pointer in `args` is a live libtorch allocation in
            // the context made current at construction, `pack` was built from
            // this kernel's manifest entry, and `stream` is the stream this
            // adapter was built for (the capture stream when one is used).
            unsafe {
                self.kernels
                    .launch(kernel, grid(rows, kernel), &mut params, stream)
                    .map_err(driver)?;
            }
        }
        Ok(())
    }

    /// The reduce's output back to the host. The readback is a copy on torch's
    /// stream, and the side stream this adapter can hold keeps the legacy
    /// stream's ordering, so it waits behind whatever `launch` queued.
    pub fn collect(&self, g: &StagedGroup, out: &mut [f32]) -> Result<(), MoEError> {
        let (tokens, hidden) = (g.tokens, self.shapes.hidden);
        if out.len() < tokens * hidden {
            return Err(MoEError::Bad(format!(
                "collect: {tokens} tokens want {} outputs, got {}",
                tokens * hidden,
                out.len()
            )));
        }
        // The port's output is host memory, so the reduce has to be done.
        let values = to_f32_vec(&g.result);
        if values.len() < tokens * hidden {
            return Err(MoEError::Bad(format!(
                "the reduce produced {} values for {tokens} tokens",
                values.len()
            )));
        }
        out[..tokens * hidden].copy_from_slice(&values[..tokens * hidden]);
        Ok(())
    }

    /// The port's whole step over fresh buffers: `stage` + `launch` +
    /// `collect`. The three stay separate so a capture can hold one
    /// `StagedGroup` alive and replay only the launches (decision 39).
    pub fn grouped_moe(&self, req: &MoEGroup<'_>, out: &mut [f32]) -> Result<(), MoEError> {
        let g = self.stage(req)?;
        self.launch(&g)?;
        self.collect(&g, out)
    }
}

/// Packed plus scales plus globals for one expert's two slabs.
fn per_expert(shapes: ExpertShapes) -> usize {
    let per = |(rows, cols): (usize, usize)| rows * cols / 2 + rows * cols / 16 + rows * 2;
    per(shapes.slab_shape(Part::GateUp)) + per(shapes.slab_shape(Part::Down))
}

/// The committed cubins are constexpr-specialized, so a model they were not built
/// for cannot be served by them at all. Checking at load turns a silently wrong
/// answer into an error.
fn check_shapes(shapes: &ExpertShapes, abi: &Abi) -> Result<(), MoEError> {
    if shapes.hidden != abi.hidden || shapes.inter != abi.inter {
        return Err(MoEError::Bad(format!(
            "the committed cubins are for hidden {} inter {} (bf16, top_k {}), not hidden {} inter {}",
            abi.hidden, abi.inter, abi.top_k, shapes.hidden, shapes.inter
        )));
    }
    Ok(())
}

fn check_slab(slab: &Slab, rows: usize, cols: usize) -> Result<(), MoEError> {
    if slab.packed.len() != rows * cols / 2
        || slab.scale.len() != rows * cols / 16
        || slab.global.len() != rows * 2
    {
        return Err(MoEError::Bad(format!(
            "slab bytes packed {} scale {} global {} for {rows}x{cols}",
            slab.packed.len(),
            slab.scale.len(),
            slab.global.len()
        )));
    }
    Ok(())
}

/// One `Mutex<CudaMoE>` is both ports, so the plan and the kernel call see the
/// same banks without the caller juggling borrows.
impl ExpertSlotStore for Mutex<CudaMoE> {
    fn begin_layer(&self, layer: usize) -> Result<Option<usize>, MoEError> {
        self.lock()
            .map_err(|_| MoEError::Bad("cuda slot cache poisoned".into()))?
            .begin_layer(layer)
    }

    fn load(&self, layer: usize, expert: u32, slot: usize) -> Result<(), MoEError> {
        self.lock()
            .map_err(|_| MoEError::Bad("cuda slot cache poisoned".into()))?
            .load(layer, expert, slot)
    }

    fn slots(&self) -> usize {
        self.lock().map(|m| m.slots).unwrap_or(0)
    }
}

impl ExpertGemm for Mutex<CudaMoE> {
    fn grouped_moe<'a>(&self, req: &MoEGroup<'a>, out: &mut [f32]) -> Result<(), MoEError> {
        self.lock()
            .map_err(|_| MoEError::Bad("cuda slot cache poisoned".into()))?
            .grouped_moe(req, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The slab source is only reached after a successful load, which no CPU-only
    /// test gets to, so these refusals never call it.
    struct Unused;

    impl ExpertSlabSource for Unused {
        fn slab(&self, _: usize, _: u32, _: Part) -> Result<Slab, MoEError> {
            unreachable!("the CPU paths must refuse before staging")
        }
    }

    /// Staging is byte indexing, so it is checkable on the CPU with the same code
    /// the device path runs: a slot's bytes land at its own offset and nowhere
    /// else.
    #[test]
    fn a_slab_lands_in_its_own_slot_row() {
        let (rows, cols) = (32usize, 64usize);
        let mut banks = Banks::new(4, rows, cols, Device::Cpu);
        let n = |i: usize| (i % 251) as u8;
        let slab = Slab {
            packed: (0..rows * cols / 2).map(n).collect(),
            scale: (0..rows * cols / 16).map(|i| n(i + 7)).collect(),
            global: (0..rows * 2).map(|i| n(i + 13)).collect(),
        };
        banks.stage(2, &slab, rows, cols).unwrap();
        let read = |t: &Tensor| {
            let flat = t.flatten(0, -1).contiguous();
            let mut out = vec![0u8; flat.numel()];
            let len = out.len();
            flat.copy_data_u8(&mut out, len);
            out
        };
        let len = rows * cols / 2;
        assert_eq!(
            read(&banks.packed)[2 * len..3 * len],
            slab.packed,
            "packed bytes must land in slot 2"
        );
        assert!(
            read(&banks.packed)[..2 * len].iter().all(|b| *b == 0),
            "slot 1 must still be zeros"
        );
        let scale_len = rows * cols / 16;
        assert_eq!(read(&banks.scale)[2 * scale_len..3 * scale_len], slab.scale);
        let global_len = rows * 2;
        assert_eq!(
            read(&banks.global)[2 * global_len..3 * global_len],
            slab.global
        );
    }

    #[test]
    fn a_slab_of_the_wrong_size_is_an_error() {
        let (rows, cols) = (8usize, 16usize);
        let slab = Slab {
            packed: vec![0; 10],
            scale: vec![0; rows * cols / 16],
            global: vec![0; rows * 2],
        };
        let err = check_slab(&slab, rows, cols).unwrap_err().to_string();
        assert!(err.contains("slab bytes"), "{err}");
        let mut banks = Banks::new(2, rows, cols, Device::Cpu);
        assert!(banks.stage(0, &slab, rows, cols).is_err());
    }

    /// The shapes come from the checkpoint and the constexprs from the cubins, so
    /// a mismatch is an error rather than a wrong answer.
    #[test]
    fn cubins_built_for_other_shapes_are_rejected() {
        let abi = Abi::committed().unwrap();
        check_shapes(&ExpertShapes::new(abi.hidden, abi.inter).unwrap(), &abi).unwrap();
        let other = ExpertShapes::new(abi.hidden * 2, abi.inter).unwrap();
        let err = check_shapes(&other, &abi).unwrap_err().to_string();
        assert!(err.contains("the committed cubins are for hidden"), "{err}");
    }

    /// `CudaMoE` holds driver handles, so it is not `Debug` and `unwrap_err`
    /// would rather not exist. The refusal is the whole point of these two.
    fn refusal(device: Device, slots: usize) -> String {
        let built = CudaMoE::new(
            device,
            Arc::new(Unused),
            ExpertShapes::new(2048, 512).unwrap(),
            slots,
        );
        match built {
            Ok(_) => panic!("{device:?} with {slots} slots must be refused"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn a_cpu_device_cannot_run_the_committed_cubins() {
        let err = refusal(Device::Cpu, 8);
        assert!(err.contains("needs a CUDA device"), "{err}");
    }

    #[test]
    fn zero_slots_is_refused() {
        let err = refusal(Device::Cuda(0), 0);
        // On a GPU host this is the slot refusal; on a CPU-only host the device
        // check comes first. Either way it is an error, not a zero-capacity cache.
        assert!(
            err.contains("at least one slot") || err.contains("needs a CUDA device"),
            "{err}"
        );
    }

    /// 1.69 MiB per expert at this checkpoint's shapes, which is what the
    /// registry's VRAM budget divides by the slot count.
    #[test]
    fn the_cache_budget_is_169_mib_per_expert() {
        let shapes = ExpertShapes::new(2048, 512).unwrap();
        assert_eq!(shapes.slab_shape(Part::GateUp), (1024, 2048));
        assert_eq!(shapes.slab_shape(Part::Down), (2048, 512));
        // gate_up is 1 MiB packed + 128 KiB scales + 2 KiB globals, down is
        // 512 KiB + 64 KiB + 4 KiB.
        assert_eq!(
            per_expert(shapes),
            1024 * 1024 + 1024 * 128 + 2048 + 512 * 1024 + 2048 * 32 + 4096
        );
        assert_eq!(per_expert(shapes), 1_775_616);
    }

    /// The route layout is the kernel's contract: token-major, `-1` for a route
    /// that is not resident.
    #[test]
    fn routes_reach_the_kernel_token_major() {
        let (token, slot) = route_lists(&[3, -1, 1, 1, 0, 2], 3, 2).unwrap();
        assert_eq!(token, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(slot, vec![3, -1, 1, 1, 0, 2]);
    }
}
