// The NVFP4 expert adapter on the CUDA driver API: decision 36 slice 4, and the
// replacement for the pyo3 `triton.rs` adapter.
//
// The slot cache's banks are cudarc tensors on the CUDA device, and the four
// kernels are launched through `cuda_driver` from the cubins committed under
// `kernel/`. The banks are byte banks on purpose: the GEMV wants `*fp8e4nv`
// scales and `*fp16` globals, and the kernel reinterprets the address itself, so
// bytes are all that has to cross. The one dtype this adapter does use is bf16,
// for the activations and the intermediates: that is what the cubins were
// specialized for and what the checkpoint itself stores.

use std::sync::{Arc, Mutex};

use half::bf16;

use crate::domain::expert_kernel::{
    BANKS, ExpertGemm, ExpertShapes, ExpertSlabSource, ExpertSlotStore, MoEDeviceGroup,
    MoEDeviceRequest, MoEError, MoEGroup, MoETensor, Part, PinnedBank, PinnedBankSource, Slab,
    route_lists,
};
use crate::infrastructure::cuda_driver::{
    CUstream, CudaError, CudaKernels, Driver, OwnedEvent, OwnedStream, STREAM_NON_BLOCKING,
    StagingKernels,
};
use crate::infrastructure::cudarc::{Runtime, Tensor, TensorDType};
use crate::infrastructure::kernel_abi::{
    Abi, CopyArgs, GEMV_DOWN, GEMV_GATE_UP, GROUPED_DOWN, GROUPED_GATE_UP, INDEX_COPY, LRU_ENSURE,
    LruArgs, MoEArgs, SILU_MUL, SUM_REDUCE, StagingAbi, gemv_down_args, gemv_gate_up_args, grid,
    grouped_down_args, grouped_gate_up_args, index_copy_args, lru_ensure_args, silu_mul_args,
    sum_reduce_args,
};
use crate::infrastructure::torch::Device;

/// The concrete buffer behind a port handle. The adapter is the one place
/// allowed to know what a `MoETensor` is made of, so every copy it makes goes
/// through here.
fn device_tensor<'a>(t: &'a dyn MoETensor, what: &str) -> Result<&'a Tensor, MoEError> {
    t.as_any().downcast_ref::<Tensor>().ok_or_else(|| {
        MoEError::Bad(format!(
            "the device MoE hand-off was handed {what} that is not a device tensor"
        ))
    })
}

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

fn bf16_tensor(rt: &Runtime, values: &[f32], shape: &[usize]) -> Result<Tensor, MoEError> {
    let bits: Vec<bf16> = values.iter().map(|v| bf16::from_f32(*v)).collect();
    rt.from_slice_bf16(&bits, shape)
}

fn host_i32(t: &Tensor) -> Result<Vec<i32>, MoEError> {
    match t.dtype() {
        TensorDType::I32 => t.to_host_i32(),
        TensorDType::I64 => Ok(t.to_host_i64()?.into_iter().map(|v| v as i32).collect()),
        TensorDType::F32 => Ok(t.to_host_f32()?.into_iter().map(|v| v as i32).collect()),
        other => Err(MoEError::Bad(format!(
            "expert ids cannot be {}",
            other.name()
        ))),
    }
}

fn as_i32_ids(t: &Tensor, n: usize, rt: &Runtime) -> Result<Tensor, MoEError> {
    let flat = t.contiguous()?.reshape(&[n])?;
    if flat.dtype() == TensorDType::I32 {
        return Ok(flat);
    }
    let host = host_i32(&flat)?;
    rt.from_slice_i32(&host, &[n])
}

/// One part's device banks: `[slots, rows, cols / 2]` packed, `[slots, rows,
/// cols / 16]` scales and `[slots, rows * 2]` globals, all uint8 bytes.
///
/// Each slot is a contiguous run in all three, so `narrow` gives a view of exactly
/// one expert's bytes. `copy_to` writes the destination linearly, so the staged
/// host tensor's length has to match that view's element count.
struct Banks {
    packed: Tensor,
    scale: Tensor,
    global: Tensor,
}

impl Banks {
    fn new(rt: &Runtime, slots: usize, rows: usize, cols: usize) -> Result<Self, MoEError> {
        let bytes = |n: usize| rt.zeros(&[n], TensorDType::U8);
        Ok(Self {
            packed: bytes(slots * rows * cols / 2)?,
            scale: bytes(slots * rows * cols / 16)?,
            global: bytes(slots * rows * 2)?,
        })
    }

    /// Copy one slab into `slot`: the H2D half of a cache miss.
    fn stage(
        &mut self,
        slot: usize,
        slab: &Slab,
        rows: usize,
        cols: usize,
    ) -> Result<(), MoEError> {
        stage_bytes(&self.packed, slot, &slab.packed, rows * cols / 2)?;
        stage_bytes(&self.scale, slot, &slab.scale, rows * cols / 16)?;
        stage_bytes(&self.global, slot, &slab.global, rows * 2)
    }
}

fn stage_bytes(bank: &Tensor, slot: usize, bytes: &[u8], len: usize) -> Result<(), MoEError> {
    if bytes.len() != len {
        return Err(MoEError::Bad(format!(
            "slab is {} bytes for a {len}-byte slot row",
            bytes.len()
        )));
    }
    let src = bank.runtime().from_slice_u8(bytes, &[len])?;
    src.copy_to(&bank.narrow(0, slot * len, len)?)
}

/// The device slot cache, plus the launch path.
///
/// The banks are allocated once, on the first staging, and never dropped: a
/// layer switch overwrites bytes at the same addresses (decision 40
/// prerequisite 2), which is both what lets one bank serve every layer and what
/// makes the addresses a captured graph bakes in stay valid.
///
/// The ports are implemented for `Mutex<CudaMoE>`: staging a miss is `&mut self`
/// (it rewrites banks) while one device cache serves every step, so the engine
/// holds the adapter behind one lock and hands out the same handle as both ports.
pub struct CudaMoE {
    rt: Runtime,
    kernels: CudaKernels,
    abi: Abi,
    staging_abi: StagingAbi,
    staging: StagingKernels,
    source: Arc<dyn ExpertSlabSource>,
    shapes: ExpertShapes,
    slots: usize,
    /// The E2M1 decode table the GEMV reads. Resident and never moved, so a
    /// captured graph bakes one stable address instead of uploading it per
    /// step (decision 40 prerequisite 4).
    lut: Tensor,
    /// The launch stream. It is the runtime's side stream unless a caller hands
    /// in its own, and every copy this adapter makes runs on the same runtime,
    /// so a launch is ordered behind the staging that fed it.
    stream: CUstream,
    gate_up: Option<Banks>,
    down: Option<Banks>,
    /// The device slot tables and pinned-bank plan, present once
    /// `enable_device_staging` has run.
    device_staging: Option<DeviceStaging>,
    /// The pinned host banks the copy kernel reads. Owned here because a
    /// captured graph bakes their device aliases in and they must outlive every
    /// replay (decision 40 prerequisite 4).
    pinned: Option<Box<dyn PinnedBankSource>>,
    /// The whole-layer prefill harness, present once `enable_prefill_overlap`
    /// has run (item 8 slice 2). Retained across prefill/decode transitions, so
    /// nothing is reallocated per prefill; `None` when overlap is off.
    prefill: Option<PrefillHarness>,
    /// The live residency policy over the shared slot banks.
    residency: Residency,
}

// SAFETY: the driver handles are immutable after construction, and the tensor
// buffers are cudarc `Send + Sync`. Two concurrent `grouped_moe` calls would
// read the same banks (read-only) and write their own activations, and both
// launches land on the same stream in order anyway. The ports hand this out
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

/// The device-resident half of the offload cache: the LRU tables `lru_ensure`
/// reads and writes, the miss plan `index_copy` follows, and the six device
/// pointer arrays the copy needs. Allocated once when device staging is enabled
/// and, like the banks, never moved -- a captured graph bakes every one of these
/// addresses (decision 40 prerequisite 4).
///
/// `src_ptrs` is one `[6]` array per layer, because the copy kernel reads
/// layer-local `src_idx` rows (the reference's per-layer host banks). The pinned
/// source is flat across layers (decision 47), so layer `l`'s window starts at
/// `addr + l * experts * row_bytes`; the offset is a constant of the capture and
/// needs no per-step update.
struct DeviceStaging {
    layers: usize,
    experts: usize,
    plan_cap: usize,
    slot_for_id: Tensor,
    id_of_slot: Tensor,
    usage: Tensor,
    step: Tensor,
    misses: Tensor,
    slots_out: Tensor,
    num: Tensor,
    dst_ptrs: Tensor,
    feat_bytes: Tensor,
    src_ptrs: Vec<Tensor>,
}

/// Which residency mapping over the shared slot banks is live. The banks are
/// allocated once and never move (decision 40 prerequisite 2); this state only
/// records *how* their slots are read, which is the whole difference between the
/// decode LRU and the prefill double buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Residency {
    /// Flat cross-layer LRU: the device tables are authoritative. Decode, and
    /// any step too small to be worth a whole-layer stream.
    Decode,
    /// Whole-layer double buffer over slots `[0, 2*experts)`: a prefill sequence
    /// is in flight, the harness is copying layers into the halves, and the LRU
    /// tables are stale (reset on the way back to `Decode`).
    Prefill,
}

/// The prefill's ordering machinery and static copy plan (item 8 slice 2). The
/// two halves borrow `2 * experts` of the cache's own slots, so this owns no
/// expert memory -- only the non-blocking copy stream, five events and the row-index
/// tensors, all retained across prefill/decode transitions.
///
/// A prefetch copies a whole layer from the pinned host banks into its half on
/// the copy stream; the compute stream waits `ready[half]` before the GEMV reads
/// that half and records `release[half]` when it is done, so the next reuse of
/// the half waits the release. `begin` fences the copy stream behind everything
/// already enqueued on the compute stream when a prefill starts.
///
/// The default copy is one `cuMemcpyBatchAsync` of six runs through the copy
/// engine (R10.6b): the buffer is a contiguous slot window, so each bank's
/// whole layer is one contiguous source and one contiguous destination run, and
/// the copy occupies no SMs. `FERRUM_PF_DMA=0` restores the committed
/// `index_copy` cubin, whose grid takes one `BLOCK_K`-row plan and therefore
/// copies a whole layer in `BLOCK_K`-row groups.
struct PrefillHarness {
    copy_stream: OwnedStream,
    begin: OwnedEvent,
    ready: [OwnedEvent; 2],
    release: [OwnedEvent; 2],
    ready_armed: [bool; 2],
    release_armed: [bool; 2],
    /// Layer-local source rows per copy group, shared by both halves.
    src_idx: Vec<Tensor>,
    /// Slot rows per half and group (`half * experts + layer-local row`).
    dst_idx: [Vec<Tensor>; 2],
    /// Group size (rows), one int64 scalar each.
    num: Vec<Tensor>,
    /// The `2 * experts` cache slots this prefill's buffer owns, chosen as the
    /// coldest contiguous window so the hottest decode experts stay resident.
    victims_host: Vec<i32>,
    /// The layer each half currently holds, `-1` when empty.
    half_layer: [i64; 2],
    /// Host copy plan: the pinned banks' device aliases and the device banks'
    /// base pointers. Both are stable for the adapter's life, so a whole-layer
    /// run is pure host arithmetic at prefetch time.
    banks: [PinnedBank; BANKS.len()],
    dst_base: [u64; BANKS.len()],
    /// First slot of the contiguous `2 * experts` buffer window.
    window_start: usize,
    /// How the prefetch copies a layer (`FERRUM_PF_DMA=0` forces the
    /// `index_copy` kernel).
    dma: PrefillDma,
}

/// The whole-layer prefetch engine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PrefillDma {
    /// The committed `index_copy` cubin, for the A/B and as a fallback.
    Off,
    /// `cuMemcpyBatchAsync` through the copy engine.
    On,
}

impl CudaMoE {
    /// Open the driver, load the committed cubins and check they were built for
    /// `shapes`. Nothing is allocated yet: the banks come with the first miss,
    /// and stay for the life of the adapter.
    pub fn new(
        device: Device,
        source: Arc<dyn ExpertSlabSource>,
        shapes: ExpertShapes,
        slots: usize,
    ) -> Result<Self, MoEError> {
        Self::new_with_stream(device, source, shapes, slots, std::ptr::null_mut())
    }

    // The stream is an opaque driver handle, not a pointer this code derefs.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
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
        let lib = Driver::open().map_err(driver)?;
        lib.init().map_err(driver)?;
        lib.attach_primary_context(ordinal as i32).map_err(driver)?;
        let abi = Abi::committed().map_err(bad_abi)?;
        check_shapes(&shapes, &abi)?;
        let staging_abi = StagingAbi::committed().map_err(bad_abi)?;
        let lib = Arc::new(lib);
        let rt = if stream.is_null() {
            Runtime::open_with(Arc::clone(&lib), ordinal)?
        } else {
            // SAFETY: the caller owns `stream` for the adapter's lifetime.
            unsafe { Runtime::open_with_stream(Arc::clone(&lib), ordinal, stream)? }
        };
        let stream = if stream.is_null() {
            rt.raw_stream()
        } else {
            stream
        };
        let kernels = CudaKernels::load(Arc::clone(&lib), abi.clone()).map_err(driver)?;
        let staging = StagingKernels::load(lib, staging_abi.clone()).map_err(driver)?;
        let lut = rt.from_slice_f32(&E2M1, &[E2M1.len()])?;
        Ok(Self {
            rt,
            kernels,
            abi,
            staging_abi,
            staging,
            source,
            shapes,
            slots,
            lut,
            stream,
            gate_up: None,
            down: None,
            device_staging: None,
            pinned: None,
            prefill: None,
            residency: Residency::Decode,
        })
    }

    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// The runtime that owns this adapter's stream and buffers, so a caller
    /// building device hand-off tensors shares the stream they are copied on.
    pub fn runtime(&self) -> &Runtime {
        &self.rt
    }

    /// The process-wide driver handle, so the composition root can build the
    /// pinned banks on the same context the adapter launches in.
    pub fn driver(&self) -> Arc<Driver> {
        self.kernels.driver()
    }

    pub fn shapes(&self) -> ExpertShapes {
        self.shapes
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

    /// The six device pointers `launch` feeds the kernels. A captured graph
    /// bakes these into its nodes, so they have to stay where they are for the
    /// life of the capture -- the banks are allocated once and never dropped
    /// (decision 40 prerequisite 2). Errs before the first staging, when there
    /// are no banks to point at yet.
    pub fn bank_ptrs(&self) -> Result<[u64; 6], MoEError> {
        let gu = self
            .gate_up
            .as_ref()
            .ok_or_else(|| MoEError::Bad("bank_ptrs before any expert was staged".into()))?;
        let dn = self
            .down
            .as_ref()
            .ok_or_else(|| MoEError::Bad("bank_ptrs before any expert was staged".into()))?;
        let ptr = |t: &Tensor| t.device_ptr();
        Ok([
            ptr(&gu.packed),
            ptr(&gu.scale),
            ptr(&gu.global),
            ptr(&dn.packed),
            ptr(&dn.scale),
            ptr(&dn.global),
        ])
    }

    /// Pull both slabs of one expert from the source and stage them.
    pub fn load(&mut self, layer: usize, expert: u32, slot: usize) -> Result<(), MoEError> {
        if slot >= self.slots {
            return Err(MoEError::SlotOutOfRange {
                slot,
                capacity: self.slots,
            });
        }
        if self.gate_up.is_none() {
            self.banks_for()?;
        }
        for part in [Part::GateUp, Part::Down] {
            let slab = self
                .source
                .slab(layer, expert, part)
                .map_err(|e| MoEError::Bad(format!("slab L{layer} E{expert} {part:?}: {e}")))?;
            let (rows, cols) = self.shapes.slab_shape(part);
            check_slab(&slab, rows, cols)?;
            let banks = match part {
                Part::GateUp => self.gate_up.as_mut().expect("gate_up banks"),
                Part::Down => self.down.as_mut().expect("down banks"),
            };
            banks.stage(slot, &slab, rows, cols)?;
        }
        Ok(())
    }

    /// Allocate both banks once. Shapes are fixed by the cubins, so a bank built
    /// here serves every layer for the life of the adapter.
    fn banks_for(&mut self) -> Result<(), MoEError> {
        if self.gate_up.is_some() {
            return Ok(());
        }
        let (gu_rows, gu_cols) = self.shapes.slab_shape(Part::GateUp);
        let (dn_rows, dn_cols) = self.shapes.slab_shape(Part::Down);
        self.gate_up = Some(Banks::new(&self.rt, self.slots, gu_rows, gu_cols)?);
        self.down = Some(Banks::new(&self.rt, self.slots, dn_rows, dn_cols)?);
        Ok(())
    }

    /// Allocate the device slot tables and the per-layer source-pointer arrays,
    /// then the pinned host banks are the source the copy kernel reads.
    ///
    /// The banks are allocated here too, so `dst_ptrs` already points where a
    /// capture will bake it. The constexprs are checked against the shapes: a
    /// plan that cannot fit `BLOCK_K`, or a slot count past `BLOCK_C`, would
    /// silently produce a wrong plan, so both are refused.
    pub fn enable_device_staging(
        &mut self,
        pinned: Box<dyn PinnedBankSource>,
        layers: usize,
        experts: usize,
    ) -> Result<(), MoEError> {
        if layers == 0 || experts == 0 {
            return Err(MoEError::Bad(
                "device staging needs at least one layer and one expert".into(),
            ));
        }
        let lru = self.staging_abi.kernel(LRU_ENSURE).map_err(bad_abi)?;
        let block_c = lru.constexpr("BLOCK_C").map_err(bad_abi)? as usize;
        if self.slots > block_c {
            return Err(MoEError::Bad(format!(
                "device staging bakes BLOCK_C {block_c}, but the cache holds {} slots",
                self.slots
            )));
        }
        self.banks_for()?;
        let banks = pinned.banks()?;
        if banks.len() != BANKS.len() {
            return Err(MoEError::Bad("pinned source must report six banks".into()));
        }
        let rows = layers * experts;
        let feat: Vec<i64> = BANKS
            .iter()
            .map(|b| self.shapes.bank_row_bytes(*b) as i64)
            .collect();
        for (i, bank) in BANKS.iter().enumerate() {
            let want = self.shapes.bank_row_bytes(*bank);
            if banks[i].row_bytes != want || banks[i].rows != rows {
                return Err(MoEError::Bad(format!(
                    "pinned bank {i} is {} rows of {} bytes, want {rows} of {want}",
                    banks[i].rows, banks[i].row_bytes
                )));
            }
        }
        let rt = self.rt.clone();
        let src_ptrs = (0..layers)
            .map(|layer| {
                let ptrs: Vec<i64> = BANKS
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        (banks[i].addr + (layer * experts * banks[i].row_bytes) as u64) as i64
                    })
                    .collect();
                rt.from_slice_i64(&ptrs, &[ptrs.len()])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dst: Vec<i64> = self.bank_ptrs()?.iter().map(|p| *p as i64).collect();

        let table = |n: usize, v: i32| rt.from_slice_i32(&vec![v; n], &[n]);
        self.device_staging = Some(DeviceStaging {
            layers,
            experts,
            plan_cap: self.slots,
            slot_for_id: table(rows, -1)?,
            id_of_slot: table(self.slots, -1)?,
            usage: rt.zeros(&[self.slots], TensorDType::I64)?,
            step: rt.zeros(&[1], TensorDType::I64)?,
            misses: rt.zeros(&[self.slots], TensorDType::I32)?,
            slots_out: rt.zeros(&[self.slots], TensorDType::I32)?,
            num: rt.zeros(&[1], TensorDType::I64)?,
            dst_ptrs: rt.from_slice_i64(&dst, &[dst.len()])?,
            feat_bytes: rt.from_slice_i64(&feat, &[feat.len()])?,
            src_ptrs,
        });
        self.pinned = Some(pinned);
        Ok(())
    }

    /// Build the whole-layer prefill harness (item 8 slice 2): a non-blocking
    /// copy stream, its ordering events and the static row-index tensors. The
    /// two buffers borrow `2 * experts` of the cache's slots, so this allocates
    /// no expert memory and only changes how those slots are read.
    ///
    /// Refused unless the cache holds `2 * experts` slots, which is exactly the
    /// condition the config gates `prefill_overlap` on.
    pub fn enable_prefill_overlap(&mut self) -> Result<(), MoEError> {
        let experts = self
            .device_staging
            .as_ref()
            .ok_or_else(|| MoEError::Bad("enable_prefill_overlap before device staging".into()))?
            .experts;
        if self.slots < 2 * experts {
            return Err(MoEError::Bad(format!(
                "prefill overlap needs 2*{experts} slots, the cache holds {}",
                self.slots
            )));
        }
        if self.prefill.is_some() {
            return Ok(());
        }
        // The committed copy cubin covers one `BLOCK_K`-row plan, so a whole
        // layer is copied in that many groups; each group is a static index list.
        let block_k = self.lru_block_k()?;
        let groups = experts.div_ceil(block_k);
        let lib = self.kernels.driver();
        let copy_stream = lib.owned_stream(STREAM_NON_BLOCKING).map_err(driver)?;
        let begin = lib.owned_event().map_err(driver)?;
        let ready = [
            lib.owned_event().map_err(driver)?,
            lib.owned_event().map_err(driver)?,
        ];
        let release = [
            lib.owned_event().map_err(driver)?,
            lib.owned_event().map_err(driver)?,
        ];
        let rt = self.rt.clone();
        let mut src_idx = Vec::with_capacity(groups);
        let mut dst_idx: [Vec<Tensor>; 2] =
            [Vec::with_capacity(groups), Vec::with_capacity(groups)];
        let mut num = Vec::with_capacity(groups);
        for g in 0..groups {
            let start = g * block_k;
            let n = (experts - start).min(block_k);
            // The row indices are `i32` (the copy kernel's index type) and the
            // count is `i64`.
            let src: Vec<i32> = (start..start + n).map(|r| r as i32).collect();
            src_idx.push(rt.from_slice_i32(&src, &[src.len()])?);
            num.push(rt.from_slice_i64(&[n as i64], &[1])?);
            for (half, dsts) in dst_idx.iter_mut().enumerate() {
                let dst: Vec<i32> = (0..n)
                    .map(|r| (half * experts + start + r) as i32)
                    .collect();
                dsts.push(rt.from_slice_i32(&dst, &[dst.len()])?);
            }
        }
        let victims_host: Vec<i32> = (0..2 * experts).map(|s| s as i32).collect();
        let banks = self
            .pinned
            .as_ref()
            .ok_or_else(|| MoEError::Bad("enable_prefill_overlap before the pinned banks".into()))?
            .banks()?;
        let dst_base = self.bank_ptrs()?;
        // Decide the copy engine once, here: the batch symbol is optional, so a
        // pre-CUDA-13 driver keeps the `index_copy` path. Say so once, not per
        // layer.
        let requested = std::env::var("FERRUM_PF_DMA").ok();
        let available = lib.batch_copy_available();
        let dma = prefill_dma_from(requested.as_deref(), available);
        if !available && dma == PrefillDma::Off && requested.as_deref() != Some("0") {
            eprintln!(
                "[moe] prefill DMA copy unavailable (cuMemcpyBatchAsync_v2); using index_copy"
            );
        }
        self.prefill = Some(PrefillHarness {
            copy_stream,
            begin,
            ready,
            release,
            ready_armed: [false; 2],
            release_armed: [false; 2],
            src_idx,
            dst_idx,
            num,
            victims_host,
            half_layer: [-1, -1],
            banks,
            dst_base,
            window_start: 0,
            dma,
        });
        Ok(())
    }

    /// Whether the whole-layer prefill harness is available.
    pub fn prefill_overlap(&self) -> bool {
        self.prefill.is_some()
    }

    /// How many slots currently map to a resident expert. A test hook for the
    /// prefill-retention gate (decision 94); it syncs, so the serving path never
    /// calls it.
    pub fn resident_count(&self) -> Result<usize, MoEError> {
        let Some(st) = self.device_staging.as_ref() else {
            return Ok(0);
        };
        let ids = st.id_of_slot.to_host_i32()?;
        Ok(ids.iter().filter(|&&id| id >= 0).count())
    }

    /// Whether a step is prefill-shaped: more routes than one `lru_ensure`
    /// launch handles. Below that it stays on the flat LRU, where a whole-layer
    /// stream would over-copy for a token count that hides nothing.
    fn is_prefill_step(&self, routes: usize) -> Result<bool, MoEError> {
        Ok(self.prefill.is_some() && routes > self.lru_block_k()?)
    }

    /// The `lru_ensure` route cap, or zero when device staging is off. The
    /// cubin bakes it, so this is the executor's answer for whether a step fits
    /// the staged path.
    pub fn staged_route_cap(&self) -> usize {
        if self.device_staging.is_none() {
            return 0;
        }
        self.staging_abi
            .kernel(LRU_ENSURE)
            .ok()
            .and_then(|k| k.constexpr("BLOCK_K").ok())
            .map(|v| v as usize)
            .unwrap_or(0)
    }

    /// Empty the device LRU: every id maps to `-1`, no slot is resident and the
    /// clock resets. The host plan writes the same banks at slots of its own
    /// choosing, so this is what keeps a stale table from reading them.
    pub fn reset_staging(&mut self) -> Result<(), MoEError> {
        let Some(st) = self.device_staging.as_mut() else {
            return Ok(());
        };
        let rt = self.rt.clone();
        let n = st.slot_for_id.numel();
        rt.from_slice_i32(&vec![-1; n], &[n])?
            .copy_to(&st.slot_for_id)?;
        let n = st.id_of_slot.numel();
        rt.from_slice_i32(&vec![-1; n], &[n])?
            .copy_to(&st.id_of_slot)?;
        let n = st.usage.numel();
        rt.from_slice_i64(&vec![0; n], &[n])?.copy_to(&st.usage)?;
        let n = st.step.numel();
        rt.from_slice_i64(&vec![0; n], &[n])?.copy_to(&st.step)?;
        let n = st.num.numel();
        rt.from_slice_i64(&vec![0; n], &[n])?.copy_to(&st.num)?;
        Ok(())
    }

    /// `lru_ensure`: rewrite `experts` in place from layer-local ids to slots and
    /// emit this step's `(miss, slot)` plan. `experts` must be an int32 device
    /// tensor; the kernel owns the rewrite so the host never sees the ids.
    pub fn ensure_experts(&mut self, layer: usize, experts: &Tensor) -> Result<(), MoEError> {
        let routes = experts.numel();
        let lru = self.staging_abi.kernel(LRU_ENSURE).map_err(bad_abi)?;
        let block_k = lru.constexpr("BLOCK_K").map_err(bad_abi)? as usize;
        if routes == 0 || routes > block_k {
            return Err(MoEError::Bad(format!(
                "lru_ensure bakes BLOCK_K {block_k}, but this step routes {routes}"
            )));
        }
        let st = self
            .device_staging
            .as_ref()
            .ok_or_else(|| MoEError::Bad("device staging is not enabled".into()))?;
        if layer >= st.layers {
            return Err(MoEError::Bad(format!(
                "layer {layer} past the {} the staging holds",
                st.layers
            )));
        }
        let ptr = |t: &Tensor| t.device_ptr();
        let args = LruArgs {
            experts: experts.device_ptr(),
            slot_for_id: ptr(&st.slot_for_id),
            id_of_slot: ptr(&st.id_of_slot),
            usage: ptr(&st.usage),
            step: ptr(&st.step),
            misses: ptr(&st.misses),
            slots: ptr(&st.slots_out),
            num: ptr(&st.num),
            routes: routes as u32,
            num_cached: st.plan_cap as u32,
            id_base: (layer * st.experts) as u32,
        };
        let pack = lru_ensure_args(&self.staging_abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        // SAFETY: every pointer in `args` is a live libtorch allocation in the
        // context made current at construction, `params` was built from this
        // kernel's manifest entry, and `stream` is this adapter's stream.
        unsafe { self.staging.launch(lru, &mut params, self.stream) }.map_err(driver)
    }

    /// `index_copy`: copy the plan's missed rows from the layer's pinned host
    /// banks into the device slot banks. Reads the plan `ensure_experts` wrote on
    /// the device, so it is data-independent.
    pub fn copy_missing(&mut self, layer: usize) -> Result<(), MoEError> {
        let copy = self.staging_abi.kernel(INDEX_COPY).map_err(bad_abi)?;
        let st = self
            .device_staging
            .as_ref()
            .ok_or_else(|| MoEError::Bad("device staging is not enabled".into()))?;
        let src = st
            .src_ptrs
            .get(layer)
            .ok_or_else(|| MoEError::Bad(format!("layer {layer} past the pinned source plan")))?;
        let ptr = |t: &Tensor| t.device_ptr();
        let args = CopyArgs {
            dst_ptrs: ptr(&st.dst_ptrs),
            src_ptrs: ptr(src),
            feat_bytes: ptr(&st.feat_bytes),
            dst_idx: ptr(&st.slots_out),
            src_idx: ptr(&st.misses),
            num: ptr(&st.num),
        };
        let pack = index_copy_args(&self.staging_abi, &args).map_err(bad_abi)?;
        let mut params = pack.kernel_params().map_err(bad_abi)?;
        // SAFETY: as `ensure_experts`: every pointer is a live allocation and
        // `params` matches this kernel's manifest entry.
        unsafe { self.staging.launch(copy, &mut params, self.stream) }.map_err(driver)
    }

    /// Every check the launches need, before anything is allocated: the
    /// cubins' shapes and top_k, the batch's shapes, and banks to read. The
    /// output buffer is checked by `collect`, which knows how many values the
    /// reduce produces. Written against element counts so the host and the
    /// device hand-offs run the same checks (decision 43).
    fn check_batch(
        &self,
        top_k: usize,
        x_numel: usize,
        route_count: usize,
        weights_len: usize,
    ) -> Result<usize, MoEError> {
        let (hidden, inter) = (self.shapes.hidden, self.shapes.inter);
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
        if hidden == 0 || x_numel == 0 || !x_numel.is_multiple_of(hidden) {
            return Err(MoEError::Bad(format!(
                "grouped_moe: {x_numel} activations for hidden {hidden}"
            )));
        }
        let tokens = x_numel / hidden;
        let routes = tokens * top_k;
        if route_count != routes || weights_len < routes {
            return Err(MoEError::Bad(format!(
                "grouped_moe: {route_count} routes and {weights_len} weights \
                 for {tokens} tokens x top_k {top_k}"
            )));
        }
        if self.gate_up.is_none() || self.down.is_none() {
            return Err(MoEError::Bad(
                "grouped_moe before any expert was staged".into(),
            ));
        }
        Ok(tokens)
    }

    fn check(&self, req: &MoEGroup<'_>) -> Result<usize, MoEError> {
        self.check_batch(req.top_k, req.x.len(), req.routes.len(), req.weights.len())
    }

    fn check_dev(&self, req: &MoEDeviceGroup<'_>) -> Result<usize, MoEError> {
        self.check_batch(
            req.top_k,
            req.x.numel(),
            req.routes.len(),
            req.weights.numel(),
        )
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
        Ok(StagedGroup {
            tokens,
            top_k,
            routes,
            x: bf16_tensor(&self.rt, req.x, &[tokens, hidden])?,
            slot_t: self.rt.from_slice_i32(&route_slot, &[routes])?,
            token_t: self.rt.from_slice_i32(&route_token, &[routes])?,
            weight_t: self.rt.from_slice_f32(&req.weights[..routes], &[routes])?,
            lut: self.lut.clone(),
            ic1: self.rt.zeros(&[routes, 2 * inter], TensorDType::Bf16)?,
            ic2: self.rt.zeros(&[routes, inter], TensorDType::Bf16)?,
            ic3: self.rt.zeros(&[routes, hidden], TensorDType::Bf16)?,
            result: self.rt.zeros(&[tokens, hidden], TensorDType::Bf16)?,
        })
    }

    /// The same fresh group, with the activations and the weights arriving
    /// from buffers that are already on the device (decision 43). Only the
    /// plan's routes still come from the host: slot ids have to exist in host
    /// memory before they can be uploaded, and prerequisite 4 is what moves
    /// that off the host too.
    pub fn stage_dev(&self, req: &MoEDeviceGroup<'_>) -> Result<StagedGroup, MoEError> {
        let tokens = self.check_dev(req)?;
        let top_k = req.top_k;
        let routes = tokens * top_k;
        let (route_token, route_slot) = route_lists(req.routes, tokens, top_k)?;
        self.group_with_slots(
            req.x,
            req.weights,
            self.rt.from_slice_i32(&route_slot, &[routes])?,
            self.rt.from_slice_i32(&route_token, &[routes])?,
            tokens,
            top_k,
        )
    }

    /// Assemble a step's device buffers from activations and weights that are
    /// already on the device, with the route lists supplied as device tensors:
    /// the host path builds them from `route_lists`, the staged path from the
    /// kernel's own rewrite.
    fn group_with_slots(
        &self,
        x: &dyn MoETensor,
        weights: &dyn MoETensor,
        slot_t: Tensor,
        token_t: Tensor,
        tokens: usize,
        top_k: usize,
    ) -> Result<StagedGroup, MoEError> {
        let (hidden, inter) = (self.shapes.hidden, self.shapes.inter);
        let routes = tokens * top_k;
        let x = device_tensor(x, "the activations")?;
        let w = device_tensor(weights, "the weights")?;
        let g = StagedGroup {
            tokens,
            top_k,
            routes,
            x: self.rt.zeros(&[tokens, hidden], TensorDType::Bf16)?,
            slot_t,
            token_t,
            weight_t: self.rt.zeros(&[routes], TensorDType::F32)?,
            lut: self.lut.clone(),
            ic1: self.rt.zeros(&[routes, 2 * inter], TensorDType::Bf16)?,
            ic2: self.rt.zeros(&[routes, inter], TensorDType::Bf16)?,
            ic3: self.rt.zeros(&[routes, hidden], TensorDType::Bf16)?,
            result: self.rt.zeros(&[tokens, hidden], TensorDType::Bf16)?,
        };
        // Device to device, narrowing to bf16 on the way in -- the conversion
        // the host path makes during its upload, made here instead.
        let x = x.contiguous()?.reshape(&[tokens, hidden])?;
        let x = if x.dtype() == TensorDType::Bf16 {
            x
        } else {
            x.cast(TensorDType::Bf16)?
        };
        x.copy_to(&g.x)?;
        let w = w.contiguous()?.reshape(&[w.numel()])?;
        w.narrow(0, 0, routes)?.copy_to(&g.weight_t)?;
        Ok(g)
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
        let routes = g.routes;
        let (route_token, route_slot) = route_lists(req.routes, tokens, req.top_k)?;
        bf16_tensor(&self.rt, req.x, &[tokens, self.shapes.hidden])?.copy_to(&g.x)?;
        self.rt
            .from_slice_i32(&route_slot, &[routes])?
            .copy_to(&g.slot_t)?;
        self.rt
            .from_slice_i32(&route_token, &[routes])?
            .copy_to(&g.token_t)?;
        self.rt
            .from_slice_f32(&req.weights[..routes], &[routes])?
            .copy_to(&g.weight_t)?;
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

        let ptr = |t: &Tensor| t.device_ptr();
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
            seg: 0,
            seg_ids: 0,
            route_idx: 0,
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

    /// The staged banks' slot capacity: the grouped CSR indexes `seg` by the
    /// slot id, so it sizes to the cache's slot count. `Banks` stores each of
    /// its three arrays flat, so the slot count is not a tensor dimension.
    fn slot_capacity(&self) -> usize {
        self.slots
    }

    /// The prefill grouped GEMM launches (R10.2, OPTIMIZATION.md R10): both
    /// GEMVs become one program per (compacted segment, N tile) over the CSR
    /// `seg`/`route_idx`, so an expert's weight is read once per (segment,
    /// n-tile) however many routes name it and empty slots never launch a CTA.
    /// `seg_ids` lists the slots the grid may touch (ascending, so it aligns
    /// with `route_idx`'s slot-sorted order). Decode keeps `launch`'s per-route
    /// GEMV; the silu and the reduce are the same kernels in the same order,
    /// and `route_idx` keeps `ic1`/`ic3` indexed by the original route. The
    /// caller owns the CSR tensors and must hold them until the stream has
    /// drained, like it holds `g`.
    pub fn launch_grouped(
        &self,
        g: &StagedGroup,
        seg: &Tensor,
        seg_ids: &Tensor,
        route_idx: &Tensor,
    ) -> Result<(), MoEError> {
        let gu = self
            .gate_up
            .as_ref()
            .ok_or_else(|| MoEError::Bad("launch before any expert was staged".into()))?;
        let dn = self
            .down
            .as_ref()
            .ok_or_else(|| MoEError::Bad("launch before any expert was staged".into()))?;
        let (tokens, routes) = (g.tokens, g.routes);
        let segments = seg_ids.numel();

        let ptr = |t: &Tensor| t.device_ptr();
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
            seg: ptr(seg),
            seg_ids: ptr(seg_ids),
            route_idx: ptr(route_idx),
        };
        let stream = self.stream;
        for (name, rows, pack) in [
            (
                GROUPED_GATE_UP,
                segments,
                grouped_gate_up_args(&self.abi, &args).map_err(bad_abi)?,
            ),
            (
                SILU_MUL,
                routes,
                silu_mul_args(&self.abi, &args).map_err(bad_abi)?,
            ),
            (
                GROUPED_DOWN,
                segments,
                grouped_down_args(&self.abi, &args).map_err(bad_abi)?,
            ),
            (
                SUM_REDUCE,
                tokens,
                sum_reduce_args(&self.abi, &args).map_err(bad_abi)?,
            ),
        ] {
            let kernel = self.abi.kernel(name).map_err(bad_abi)?;
            let mut params = pack.kernel_params().map_err(bad_abi)?;
            // SAFETY: every pointer in `args` is a live allocation the caller
            // holds (`g` and the CSR) in the context made current at
            // construction, `pack` came from this kernel's manifest entry, and
            // `stream` is this adapter's.
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
        // The host path's output is host memory, so the reduce has to be done.
        let values = g.result.to_host_f32()?;
        if values.len() < tokens * hidden {
            return Err(MoEError::Bad(format!(
                "the reduce produced {} values for {tokens} tokens",
                values.len()
            )));
        }
        out[..tokens * hidden].copy_from_slice(&values[..tokens * hidden]);
        Ok(())
    }

    /// The reduce's output into the caller's own device buffer, device to
    /// device: the readback the host path makes never happens on the device
    /// hand-off (decision 43). The copy widens the bf16 reduce to f32, the same
    /// widening the host readback does.
    pub fn collect_dev(&self, g: &StagedGroup, out: &mut dyn MoETensor) -> Result<(), MoEError> {
        let (tokens, hidden) = (g.tokens, self.shapes.hidden);
        let t = out.as_any_mut().downcast_mut::<Tensor>().ok_or_else(|| {
            MoEError::Bad("the device MoE hand-off needs a device tensor for the routed sum".into())
        })?;
        if t.shape() != [tokens, hidden] {
            return Err(MoEError::Bad(format!(
                "collect_dev: the routed sum is {:?}, wanted [{tokens}, {hidden}]",
                t.shape()
            )));
        }
        if t.dtype() == g.result.dtype() {
            g.result.copy_to(t)
        } else {
            g.result.cast(t.dtype())?.copy_to(t)
        }
    }

    /// The port's whole step over fresh buffers: `stage` + `launch` +
    /// `collect`. The three stay separate so a capture can hold one
    /// `StagedGroup` alive and replay only the launches (decision 39).
    pub fn grouped_moe(&self, req: &MoEGroup<'_>, out: &mut [f32]) -> Result<(), MoEError> {
        let g = self.stage(req)?;
        self.launch(&g)?;
        self.collect(&g, out)
    }

    /// The device hand-off's whole step: `stage_dev` + `launch` +
    /// `collect_dev`. Same three pieces as the host path, so the capture
    /// contract is the same on both (decision 39), with only the staging's
    /// inputs having moved onto the device (decision 43).
    pub fn grouped_moe_dev(
        &self,
        req: &MoEDeviceGroup<'_>,
        out: &mut dyn MoETensor,
    ) -> Result<(), MoEError> {
        let g = self.stage_dev(req)?;
        self.launch(&g)?;
        self.collect_dev(&g, out)
    }

    /// The whole step with the plan on the device: rewrite the router's ids into
    /// slots (`lru_ensure`), copy the misses (`index_copy`), then the same three
    /// pieces as the other hand-offs (decision 40 prerequisite 4). The ids never
    /// cross to the host, so a capture sees only data-independent launches.
    pub fn grouped_moe_staged(
        &mut self,
        req: &MoEDeviceRequest<'_>,
        out: &mut dyn MoETensor,
    ) -> Result<(), MoEError> {
        let tokens = self.check_batch(
            req.top_k,
            req.x.numel(),
            req.experts.numel(),
            req.weights.numel(),
        )?;
        let top_k = req.top_k;
        let block_k = self.lru_block_k()?;
        let x = device_tensor(req.x, "the activations")?;
        let experts = device_tensor(req.experts, "the expert ids")?;
        let weights = device_tensor(req.weights, "the weights")?;
        let out_t = out.as_any_mut().downcast_mut::<Tensor>().ok_or_else(|| {
            MoEError::Bad("the device MoE hand-off needs a device tensor for the routed sum".into())
        })?;
        if out_t.shape() != [tokens, self.shapes.hidden] {
            return Err(MoEError::Bad(format!(
                "grouped_moe_staged: the routed sum is {:?}, wanted [{tokens}, {}]",
                out_t.shape(),
                self.shapes.hidden
            )));
        }

        // A prefill-shaped step (`>` one `lru_ensure` launch of routes) with the
        // overlap harness takes the whole-layer path: one streamed copy of the
        // layer and one GEMV pass, no per-chunk `lru_ensure`. Anything smaller
        // stays on the flat LRU. Entering the prefill frees only the buffer
        // slots; the rest of the decode LRU survives (TODO.md item 24).
        if self.is_prefill_step(tokens * top_k)? {
            if self.residency != Residency::Prefill {
                self.enter_prefill()?;
            }
            return self.prefill_layer(req.layer, x, experts, weights, out_t, top_k);
        }
        if self.residency == Residency::Prefill {
            self.leave_prefill()?;
        }

        // `lru_ensure` dedups first occurrences inside one CTA, so its committed
        // `BLOCK_K` is a route cap (decisions 44/48). A prefill's route list is
        // far larger, so chunk it by tokens: each chunk fits the cap, the device
        // LRU and the pinned banks carry across the chunks, and each chunk's
        // reduce lands in its own output rows. An expert an earlier chunk made
        // resident is a hit here, so the copy is not repeated.
        if tokens * top_k <= block_k {
            return self.staged_group(req.layer, x, experts, weights, out_t, top_k);
        }
        let chunk = staged_chunk_tokens(block_k, top_k);
        let mut start = 0usize;
        while start < tokens {
            let n = chunk.min(tokens - start);
            let xc = x.narrow(0, start, n)?;
            let ec = experts.narrow(0, start, n)?;
            let wc = weights.narrow(0, start, n)?;
            let mut oc = out_t.narrow(0, start, n)?;
            self.staged_group(req.layer, &xc, &ec, &wc, &mut oc, top_k)?;
            start += n;
        }
        Ok(())
    }

    /// Dev-only: with `FT_MISS_TRACE` set, accumulate this layer's miss count (a
    /// D2H read of the plan's `num`) and print the per-step total at the last
    /// layer. A D2H cannot be captured, so run eager (`FERRUM_SMOKE_NOGRAPH=1`);
    /// it is a measurement probe for the copy investigation, never in serving.
    fn probe_misses(&self, layer: usize) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TOTAL: AtomicU64 = AtomicU64::new(0);
        if std::env::var_os("FT_MISS_TRACE").is_none() {
            return;
        }
        let Some(st) = &self.device_staging else {
            return;
        };
        if layer == 0 {
            TOTAL.store(0, Ordering::Relaxed);
        }
        let n = st
            .num
            .to_host_i64()
            .ok()
            .and_then(|v| v.first().copied())
            .unwrap_or(0) as u64;
        let total = TOTAL.fetch_add(n, Ordering::Relaxed) + n;
        if layer + 1 == st.layers {
            eprintln!("[miss] step_experts={total}");
        }
    }

    /// The committed `lru_ensure` `BLOCK_K`: the route cap one launch handles.
    fn lru_block_k(&self) -> Result<usize, MoEError> {
        self.staging_abi
            .kernel(LRU_ENSURE)
            .map_err(bad_abi)?
            .constexpr("BLOCK_K")
            .map(|v| v as usize)
            .map_err(bad_abi)
    }

    /// Enter the whole-layer prefill: fence the copy stream behind the compute
    /// stream (a preceding decode may still be reading the borrowed slots), free
    /// only the `2 * num_experts` buffer slots, and reset the half bookkeeping.
    ///
    /// Only the buffer range is touched; the decode LRU in the rest of the cache
    /// is retained across the prefill (TODO.md item 24, OPTIMIZATION.md R7.2).
    fn enter_prefill(&mut self) -> Result<(), MoEError> {
        self.select_buffer_slots()?;
        let copy_stream = {
            let pf = self
                .prefill
                .as_mut()
                .ok_or_else(|| MoEError::Bad("enter_prefill without a harness".into()))?;
            pf.half_layer = [-1, -1];
            pf.ready_armed = [false, false];
            pf.release_armed = [false, false];
            // SAFETY: `self.stream` is the adapter's live stream in the context
            // made current at construction.
            unsafe { pf.begin.record(self.stream) }.map_err(driver)?;
            pf.copy_stream.raw()
        };
        let pf = self.prefill.as_ref().expect("harness");
        // SAFETY: the copy stream is this harness's own live stream.
        unsafe { pf.begin.wait_on(copy_stream) }.map_err(driver)?;
        self.residency = Residency::Prefill;
        Ok(())
    }

    /// Leave the whole-layer prefill. The buffer slots were freed on entry and
    /// the LRU tables never referenced them during the prefill, so the next
    /// decode step picks them up as free slots; the retained LRU is intact.
    fn leave_prefill(&mut self) -> Result<(), MoEError> {
        self.residency = Residency::Decode;
        Ok(())
    }

    /// Pick the `2 * num_experts` slots the whole-layer double buffer owns. The
    /// buffer has to be a contiguous slot window so each prefetched bank is one
    /// DMA run (R10.6b), so this picks the window with the lowest total usage:
    /// free slots go first and a warm cache loses as little of its LRU as a
    /// contiguous window can. Those slots are freed from the reverse map and the
    /// per-half copy row lists are rebuilt from them; every other slot's LRU
    /// state is untouched.
    fn select_buffer_slots(&mut self) -> Result<(), MoEError> {
        let buffer = {
            let st = self
                .device_staging
                .as_ref()
                .ok_or_else(|| MoEError::Bad("select_buffer_slots before staging".into()))?;
            2 * st.experts
        };
        let rt = self.rt.clone();
        let start = {
            let st = self.device_staging.as_ref().expect("checked above");
            let usage = st.usage.to_host_i64().map_err(|e| {
                MoEError::Bad(format!("select_buffer_slots cannot read usage: {e}"))
            })?;
            pick_window(&usage, buffer)
        };
        let victims: Vec<i32> = (start..start + buffer).map(|s| s as i32).collect();
        {
            let st = self.device_staging.as_mut().expect("checked above");
            let mut id_of_slot = st.id_of_slot.to_host_i32()?;
            let mut slot_for_id = st.slot_for_id.to_host_i32()?;
            let mut usage = st.usage.to_host_i64()?;
            for &victim in &victims {
                let slot = victim as usize;
                let id = id_of_slot[slot];
                if id >= 0 {
                    slot_for_id[id as usize] = -1;
                }
                id_of_slot[slot] = -1;
                usage[slot] = 0;
            }
            let n = slot_for_id.len();
            rt.from_slice_i32(&slot_for_id, &[n])?
                .copy_to(&st.slot_for_id)?;
            let n = id_of_slot.len();
            rt.from_slice_i32(&id_of_slot, &[n])?
                .copy_to(&st.id_of_slot)?;
            let n = usage.len();
            rt.from_slice_i64(&usage, &[n])?.copy_to(&st.usage)?;
        }
        let experts = self
            .device_staging
            .as_ref()
            .map(|st| st.experts)
            .expect("checked above");
        let block_k = self.lru_block_k()?;
        let pf = self
            .prefill
            .as_mut()
            .ok_or_else(|| MoEError::Bad("select_buffer_slots without a harness".into()))?;
        for (half, dsts) in pf.dst_idx.iter_mut().enumerate() {
            for (g, dst) in dsts.iter_mut().enumerate() {
                let row_start = g * block_k;
                let n = (experts - row_start).min(block_k);
                let off = half * experts + row_start;
                let host = &victims[off..off + n];
                *dst = rt.from_slice_i32(host, &[n])?;
            }
        }
        pf.victims_host = victims;
        pf.window_start = start;
        Ok(())
    }

    /// Copy a whole layer into one half of the double buffer on the copy stream,
    /// waiting for that half's release event first, then record the ready event
    /// the compute stream waits on. The caller owns the `half_layer` bookkeeping.
    fn prefetch_half(&mut self, layer: usize, half: usize) -> Result<(), MoEError> {
        let dma = self
            .prefill
            .as_ref()
            .map(|pf| pf.dma)
            .unwrap_or(PrefillDma::Off);
        if dma != PrefillDma::Off {
            return self.prefetch_half_dma(layer, half);
        }
        let (src, dst_ptrs, feat_bytes) = {
            let st = self
                .device_staging
                .as_ref()
                .ok_or_else(|| MoEError::Bad("prefetch_half before device staging".into()))?;
            let src = st
                .src_ptrs
                .get(layer)
                .ok_or_else(|| MoEError::Bad(format!("layer {layer} past the pinned source plan")))?
                .clone();
            (src, st.dst_ptrs.clone(), st.feat_bytes.clone())
        };
        let copy = self.staging_abi.kernel(INDEX_COPY).map_err(bad_abi)?;
        let ptr = |t: &Tensor| t.device_ptr();
        let pf = self
            .prefill
            .as_mut()
            .ok_or_else(|| MoEError::Bad("prefetch_half without a harness".into()))?;
        let copy_stream = pf.copy_stream.raw();
        if pf.release_armed[half] {
            // SAFETY: the copy stream is this harness's own live stream.
            unsafe { pf.release[half].wait_on(copy_stream) }.map_err(driver)?;
        }
        for g in 0..pf.num.len() {
            let args = CopyArgs {
                dst_ptrs: ptr(&dst_ptrs),
                src_ptrs: ptr(&src),
                feat_bytes: ptr(&feat_bytes),
                dst_idx: ptr(&pf.dst_idx[half][g]),
                src_idx: ptr(&pf.src_idx[g]),
                num: ptr(&pf.num[g]),
            };
            let pack = index_copy_args(&self.staging_abi, &args).map_err(bad_abi)?;
            let mut params = pack.kernel_params().map_err(bad_abi)?;
            // SAFETY: as `copy_missing`: live allocations and params packed from
            // this kernel's manifest entry, launched on the copy stream.
            unsafe { self.staging.launch(copy, &mut params, copy_stream) }.map_err(driver)?;
        }
        // SAFETY: the copy stream is this harness's own live stream.
        unsafe { pf.ready[half].record(copy_stream) }.map_err(driver)?;
        pf.ready_armed[half] = true;
        Ok(())
    }

    /// Whole-layer prefetch through the copy engine (R10.6b). The buffer window
    /// is contiguous and the pinned layer is contiguous, so each bank is exactly
    /// one `(dst, src, bytes)` run: six batch entries per layer instead of
    /// `6 * experts` `index_copy` CTAs that would occupy the SMs and stall the
    /// compute stream on PCIe. Every run is a whole layer (hundreds of KiB and
    /// up), far above the driver's small-entry floor that makes a batch degrade
    /// to a synchronous copy.
    fn prefetch_half_dma(&mut self, layer: usize, half: usize) -> Result<(), MoEError> {
        let experts = self
            .device_staging
            .as_ref()
            .ok_or_else(|| MoEError::Bad("prefetch_half_dma before device staging".into()))?
            .experts;
        let lib = self.kernels.driver();
        let pf = self
            .prefill
            .as_mut()
            .ok_or_else(|| MoEError::Bad("prefetch_half_dma without a harness".into()))?;
        let copy_stream = pf.copy_stream.raw();
        if pf.release_armed[half] {
            // SAFETY: the copy stream is this harness's own live stream.
            unsafe { pf.release[half].wait_on(copy_stream) }.map_err(driver)?;
        }
        let runs = whole_layer_runs(
            &pf.banks,
            &pf.dst_base,
            experts,
            layer,
            pf.window_start,
            half,
        );
        let mut dsts = [0u64; BANKS.len()];
        let mut srcs = [0u64; BANKS.len()];
        let mut sizes = [0usize; BANKS.len()];
        for (i, run) in runs.iter().enumerate() {
            dsts[i] = run.dst;
            srcs[i] = run.src;
            sizes[i] = run.bytes;
        }
        // SAFETY: every run is a live range of the pinned source's layer window
        // and this half's slots, and `copy_stream` is the harness's own stream.
        unsafe { lib.copy_batch_async(&dsts, &srcs, &sizes, copy_stream) }.map_err(driver)?;
        // SAFETY: the copy stream is this harness's own live stream.
        unsafe { pf.ready[half].record(copy_stream) }.map_err(driver)?;
        pf.ready_armed[half] = true;
        Ok(())
    }

    /// One layer of a whole-layer prefill: ensure the layer is in its half, kick
    /// off the next layer's copy, wait for the half, run the GEMV per route chunk
    /// over the materialized layer, then release the half. Position == expert id
    /// (offset by the half), so no `lru_ensure` and no route rewrite happen.
    fn prefill_layer(
        &mut self,
        layer: usize,
        x: &Tensor,
        experts: &Tensor,
        weights: &Tensor,
        out: &mut Tensor,
        top_k: usize,
    ) -> Result<(), MoEError> {
        let hidden = self.shapes.hidden;
        let tokens = x.numel() / hidden;
        let routes = tokens * top_k;
        let (experts_per_layer, layers) = {
            let st = self.device_staging.as_ref().expect("staging");
            (st.experts, st.layers)
        };
        let half = layer % 2;
        let victims = self.prefill.as_ref().expect("harness").victims_host.clone();
        // The grid runs over this half's buffer slots only: every route of the
        // chunk names one of them (whole-layer staging), and `victims_host` is
        // ascending, so its ranges line up with the CSR's slot-sorted
        // `route_idx`. Launching over all cache slots made ~6x of the grid
        // retire on an empty segment.
        let seg_host: Vec<i32> = if prefill_all_segments() {
            (0..self.slot_capacity()).map(|s| s as i32).collect()
        } else {
            let off = half * experts_per_layer;
            victims[off..off + experts_per_layer].to_vec()
        };
        let seg_ids = self.rt.from_slice_i32(&seg_host, &[seg_host.len()])?;

        // (1) This layer's half. Layer 0 has no lookahead, so it starts here; for
        // every later layer the previous step already prefetched it.
        if self.half_missing(half, layer) {
            let sp = crate::infrastructure::profile::begin("pf_prefetch");
            self.prefetch_half(layer, half)?;
            crate::infrastructure::profile::end("pf_prefetch", sp);
            self.prefill.as_mut().expect("harness").half_layer[half] = layer as i64;
        }
        // (2) Next layer's copy, which overlaps everything below.
        if layer + 1 < layers {
            let nh = (layer + 1) % 2;
            if self.half_missing(nh, layer + 1) {
                let sp = crate::infrastructure::profile::begin("pf_prefetch");
                self.prefetch_half(layer + 1, nh)?;
                crate::infrastructure::profile::end("pf_prefetch", sp);
                self.prefill.as_mut().expect("harness").half_layer[nh] = (layer + 1) as i64;
            }
        }
        // (3) Wait for this half (the copy stream's ready event), then run the
        // GEMV over the materialized layer, chunked only to bound the scratch.
        {
            let sp = crate::infrastructure::profile::begin("pf_wait");
            let pf = self.prefill.as_ref().expect("harness");
            if pf.ready_armed[half] {
                // SAFETY: `self.stream` is the adapter's live compute stream.
                unsafe { pf.ready[half].wait_on(self.stream) }.map_err(driver)?;
            }
            crate::infrastructure::profile::end("pf_wait", sp);
        }
        let sp = crate::infrastructure::profile::begin("pf_gemm");
        let offset = half * experts_per_layer;
        // The buffer's slot for `(half, expert_id)` is the selected victim, not
        // the fixed `half * E + id` range (decision 94 follow-up).
        let experts_host = host_i32(experts)?;
        let slots_host: Vec<i32> = experts_host
            .iter()
            .map(|&expert| {
                victims.get(offset + expert as usize).copied().ok_or_else(|| {
                    MoEError::Bad(format!(
                        "prefill expert id {expert} is past the {experts_per_layer}-expert layer"
                    ))
                })
            })
            .collect::<Result<_, _>>()?;
        let slots = self.rt.from_slice_i32(&slots_host, &[routes])?;
        // The whole-layer path never runs `lru_ensure`, so `block_k` does not
        // bound it; the chunk only bounds the scratch and keeps the grouped
        // GEMV's route list small. A large chunk is a large win: with 16-token
        // chunks almost every route in a chunk names a distinct expert, so the
        // per-route GEMV re-reads each expert from HBM, while a 256-token chunk
        // gives each expert enough routes that its weight is reused from L2.
        let chunk = prefill_chunk_tokens().min(tokens);
        let mut start = 0usize;
        while start < tokens {
            let n = chunk.min(tokens - start);
            let xc = x.narrow(0, start, n)?;
            let wc = weights.narrow(0, start, n)?;
            let sc = slots.narrow(0, start * top_k, n * top_k)?;
            let token_host: Vec<i32> = (0..n)
                .flat_map(|t| std::iter::repeat_n(t as i32, top_k))
                .collect();
            let token_t = self.rt.from_slice_i32(&token_host, &[n * top_k])?;
            let mut oc = out.narrow(0, start, n)?;
            let g = self.group_with_slots(&xc, &wc, sc, token_t, n, top_k)?;
            if prefill_use_grouped() {
                // The CSR lives as long as `g`: the queued launches read it
                // asynchronously, the same lifetime the staged buffers have.
                let chunk_slots = &slots_host[start * top_k..(start + n) * top_k];
                let (seg, route_idx) =
                    build_grouped_csr_host(chunk_slots, self.slot_capacity(), &self.rt)?;
                self.launch_grouped(&g, &seg, &seg_ids, &route_idx)?;
                self.collect_dev(&g, &mut oc)?;
            } else {
                self.launch(&g)?;
                self.collect_dev(&g, &mut oc)?;
            }
            start += n;
        }
        crate::infrastructure::profile::end("pf_gemm", sp);
        // (4) Release the half so a future layer's prefetch may reuse it.
        let pf = self.prefill.as_mut().expect("harness");
        // SAFETY: `self.stream` is the adapter's live compute stream.
        unsafe { pf.release[half].record(self.stream) }.map_err(driver)?;
        pf.release_armed[half] = true;
        Ok(())
    }

    /// Whether `half` does not already hold `layer`. A cheap read of the harness
    /// bookkeeping, split out so the two call sites do not fight the borrow
    /// checker over `self`.
    fn half_missing(&self, half: usize, layer: usize) -> bool {
        self.prefill
            .as_ref()
            .map(|p| p.half_layer[half] != layer as i64)
            .unwrap_or(false)
    }

    /// One staged chunk: rewrite the chunk's layer-local ids into slots on the
    /// device (`lru_ensure`), copy the misses (`index_copy`), then the same three
    /// pieces as the other hand-offs. The ids never cross to the host.
    fn staged_group(
        &mut self,
        layer: usize,
        x: &Tensor,
        experts: &Tensor,
        weights: &Tensor,
        out: &mut Tensor,
        top_k: usize,
    ) -> Result<(), MoEError> {
        let hidden = self.shapes.hidden;
        let routes = experts.numel();
        let tokens = x.numel() / hidden;
        if top_k == 0 || routes != tokens * top_k {
            return Err(MoEError::Bad(format!(
                "staged chunk: {routes} routes for {tokens} tokens x top_k {top_k}"
            )));
        }
        // The kernel rewrites this copy in place; a fresh buffer keeps the
        // router's own `[tokens, top_k]` ids untouched.
        let slots = as_i32_ids(experts, routes, &self.rt)?;
        let sp = crate::infrastructure::profile::begin("moe_lru");
        self.ensure_experts(layer, &slots)?;
        crate::infrastructure::profile::end("moe_lru", sp);
        let sp = crate::infrastructure::profile::begin("moe_copy");
        self.copy_missing(layer)?;
        crate::infrastructure::profile::end("moe_copy", sp);
        self.probe_misses(layer);

        let token_host: Vec<i32> = (0..tokens)
            .flat_map(|t| std::iter::repeat_n(t as i32, top_k))
            .collect();
        let token_t = self.rt.from_slice_i32(&token_host, &[routes])?;
        let sp = crate::infrastructure::profile::begin("moe_gemm");
        let g = self.group_with_slots(x, weights, slots, token_t, tokens, top_k)?;
        self.launch(&g)?;
        let out = self.collect_dev(&g, out);
        crate::infrastructure::profile::end("moe_gemm", sp);
        out
    }
}

/// Tokens one staged chunk carries so `tokens * top_k <= block_k`. Floored at
/// one so a `block_k` smaller than `top_k` still makes progress; `ensure_experts`
/// then refuses the overflow instead of the loop spinning forever.
fn staged_chunk_tokens(block_k: usize, top_k: usize) -> usize {
    (block_k / top_k).max(1)
}

/// Tokens one grouped GEMM launch of the whole-layer prefill path covers.
///
/// That path never runs `lru_ensure`, so the committed `BLOCK_K` (a route cap
/// for the device LRU) does not apply; only the scratch and the route list are
/// bounded. The chunk is a launch granularity, not a numerics change: the CSR
/// buckets a chunk's routes by slot, so every chunk re-reads each routed
/// expert's weight from HBM. R10.7 measured a 1991-token prefill with the DMA
/// copy hidden: 512 -> 1024 -> one chunk moved pp 583.2 -> 633.6 -> 664.6
/// tok/s (+8.6%, +4.9%, +14.0% total) and dropped `pf_gemm` 1464.9 -> 1037.8
/// ms (-29.2%) for one chunk. The win is the per-chunk expert stream, so the
/// plateau is one chunk for the measured prompt; 2048 captures it with ~130
/// MiB of transient scratch at 2048 tokens x top_k 8. A longer
/// `max_prefill_length` splits into 2048-token chunks, which keeps that bound.
/// `FERRUM_PF_CHUNK` overrides for measurement (a diagnostic knob, never
/// needed in serving).
const PREFILL_CHUNK_TOKENS: usize = 2048;

/// The live prefill chunk ceiling: `FERRUM_PF_CHUNK` when it parses to a
/// non-zero token count, else the default.
fn prefill_chunk_tokens() -> usize {
    prefill_chunk_from(std::env::var("FERRUM_PF_CHUNK").ok().as_deref())
}

fn prefill_chunk_from(value: Option<&str>) -> usize {
    value
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(PREFILL_CHUNK_TOKENS)
}

/// Whether the whole-layer prefill runs the grouped GEMM (R10.2, the default)
/// or the per-route GEMV it replaced. `FERRUM_PF_GEMV=1` forces the GEMV, for
/// the A/B measurement and as a fallback.
fn prefill_use_grouped() -> bool {
    prefill_grouped_from(std::env::var("FERRUM_PF_GEMV").ok().as_deref())
}

fn prefill_grouped_from(value: Option<&str>) -> bool {
    value != Some("1")
}

/// The grouped GEMM grid runs over the half's `2E` buffer slots by default
/// (R10.10); `FERRUM_PF_SEGMENTS=all` launches over every cache slot instead,
/// reproducing the pre-R10.10 grid for an A/B.
fn prefill_all_segments() -> bool {
    prefill_all_segments_from(std::env::var("FERRUM_PF_SEGMENTS").ok().as_deref())
}

fn prefill_all_segments_from(value: Option<&str>) -> bool {
    value == Some("all")
}

/// Resolve the whole-layer prefetch engine once: `FERRUM_PF_DMA=0` forces the
/// committed `index_copy` kernel (the A/B and the fallback), and a driver
/// without `cuMemcpyBatchAsync_v2` takes it too because DMA is an optimization.
fn prefill_dma_from(value: Option<&str>, batch_available: bool) -> PrefillDma {
    if batch_available && value != Some("0") {
        PrefillDma::On
    } else {
        PrefillDma::Off
    }
}

/// One DMA run of a whole-layer prefetch, in raw device addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CopyRun {
    dst: u64,
    src: u64,
    bytes: usize,
}

/// The whole-layer copy plan: one run per bank, because the source (every
/// expert of `layer` in id order) and the destination (the half's contiguous
/// slot window) are both contiguous. Absolute addresses are stable for the
/// adapter's life, so this is pure host arithmetic per prefetch.
fn whole_layer_runs(
    banks: &[PinnedBank; BANKS.len()],
    dst_base: &[u64; BANKS.len()],
    experts: usize,
    layer: usize,
    window_start: usize,
    half: usize,
) -> [CopyRun; BANKS.len()] {
    std::array::from_fn(|i| {
        let row = banks[i].row_bytes;
        CopyRun {
            dst: dst_base[i] + ((window_start + half * experts) * row) as u64,
            src: banks[i].addr + (layer * experts * row) as u64,
            bytes: experts * row,
        }
    })
}

/// Start of the contiguous `buffer`-slot window with the lowest total usage.
/// Free slots (usage 0) naturally win, so the buffer only takes slots the decode
/// LRU has not touched recently. Contiguity is what makes one DMA run per bank
/// cover a whole layer.
fn pick_window(usage: &[i64], buffer: usize) -> usize {
    assert!(buffer > 0 && usage.len() >= buffer);
    let mut sum: i64 = usage[..buffer].iter().sum();
    let mut best = (sum, 0usize);
    for i in buffer..usage.len() {
        sum += usage[i] - usage[i - buffer];
        if sum < best.0 {
            best = (sum, i - buffer + 1);
        }
    }
    best.1
}

/// Bucket the prefill's routes per slot into a CSR the grouped launches read:
/// `seg[s]..seg[s + 1]` indexes `route_idx`, which holds the original route
/// ids in stable per-slot order, so the segment index is the slot id. Prefill
/// routes are always resident (whole-layer staging); the decode-only shapes a
/// non-resident or out-of-bank slot produces are refused here rather than
/// misread by a segment program.
fn build_grouped_csr_host(
    route_slot: &[i32],
    slots: usize,
    rt: &Runtime,
) -> Result<(Tensor, Tensor), MoEError> {
    let mut counts = vec![0i32; slots];
    for &slot in route_slot {
        if slot < 0 {
            return Err(MoEError::Bad(
                "prefill grouped CSR: a route is not resident".into(),
            ));
        }
        let Some(count) = counts.get_mut(slot as usize) else {
            return Err(MoEError::Bad(format!(
                "prefill grouped CSR: a route names a slot past the {slots}-slot bank"
            )));
        };
        *count += 1;
    }
    let mut seg = vec![0i32; slots + 1];
    for s in 0..slots {
        seg[s + 1] = seg[s] + counts[s];
    }
    let mut cursor = seg[..slots].to_vec();
    let mut route_idx = vec![0i32; route_slot.len()];
    for (route, &slot) in route_slot.iter().enumerate() {
        let at = &mut cursor[slot as usize];
        route_idx[*at as usize] = route as i32;
        *at += 1;
    }
    Ok((
        rt.from_slice_i32(&seg, &[seg.len()])?,
        rt.from_slice_i32(&route_idx, &[route_idx.len()])?,
    ))
}

/// Packed plus scales plus globals for one expert's two slabs.
fn per_expert(shapes: ExpertShapes) -> usize {
    let per = |(rows, cols): (usize, usize)| rows * cols / 2 + rows * cols / 16 + rows * 2;
    per(shapes.slab_shape(Part::GateUp)) + per(shapes.slab_shape(Part::Down))
}

/// Bytes one expert slot occupies in the device cache. The auto cache budget
/// (item 12) divides the free device memory by this.
pub fn expert_slot_bytes(shapes: ExpertShapes) -> usize {
    per_expert(shapes)
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

    fn grouped_moe_dev<'a>(
        &self,
        req: &MoEDeviceGroup<'a>,
        out: &mut dyn MoETensor,
    ) -> Result<(), MoEError> {
        self.lock()
            .map_err(|_| MoEError::Bad("cuda slot cache poisoned".into()))?
            .grouped_moe_dev(req, out)
    }

    fn grouped_moe_staged<'a>(
        &self,
        req: &MoEDeviceRequest<'a>,
        out: &mut dyn MoETensor,
    ) -> Result<(), MoEError> {
        self.lock()
            .map_err(|_| MoEError::Bad("cuda slot cache poisoned".into()))?
            .grouped_moe_staged(req, out)
    }

    fn staged_route_cap(&self) -> usize {
        self.lock().map(|m| m.staged_route_cap()).unwrap_or(0)
    }

    fn reset_staging(&self) -> Result<(), MoEError> {
        self.lock()
            .map_err(|_| MoEError::Bad("cuda slot cache poisoned".into()))?
            .reset_staging()
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

    /// The prefill chunk ceiling: `FERRUM_PF_CHUNK` overrides the default, and
    /// anything unparseable (or zero, which would loop forever) falls back.
    #[test]
    fn a_prefill_chunk_env_override_parses() {
        assert_eq!(prefill_chunk_from(None), PREFILL_CHUNK_TOKENS);
        assert_eq!(prefill_chunk_from(Some("512")), 512);
        assert_eq!(prefill_chunk_from(Some("garbage")), PREFILL_CHUNK_TOKENS);
        assert_eq!(prefill_chunk_from(Some("0")), PREFILL_CHUNK_TOKENS);
    }

    /// The grouped prefill is the default (R10.2); `FERRUM_PF_GEMV=1` forces
    /// the per-route GEMV path it replaced, for the A/B and as a fallback.
    #[test]
    fn a_prefill_grouped_env_override_parses() {
        assert!(prefill_grouped_from(None));
        assert!(prefill_grouped_from(Some("0")));
        assert!(prefill_grouped_from(Some("garbage")));
        assert!(!prefill_grouped_from(Some("1")));
    }

    /// The whole-layer prefetch uses the copy engine by default (R10.6b);
    /// `FERRUM_PF_DMA=0` restores the `index_copy` kernel for the A/B.
    /// The compacted grid is the default (R10.10); `FERRUM_PF_SEGMENTS=all`
    /// restores the all-slots grid for the A/B.
    #[test]
    fn a_prefill_segment_env_override_parses() {
        assert!(!prefill_all_segments_from(None));
        assert!(!prefill_all_segments_from(Some("0")));
        assert!(!prefill_all_segments_from(Some("garbage")));
        assert!(prefill_all_segments_from(Some("all")));
    }

    #[test]
    fn a_prefill_dma_env_override_parses() {
        assert_eq!(prefill_dma_from(None, true), PrefillDma::On);
        assert_eq!(prefill_dma_from(Some("1"), true), PrefillDma::On);
        assert_eq!(prefill_dma_from(Some("garbage"), true), PrefillDma::On);
        assert_eq!(prefill_dma_from(Some("0"), true), PrefillDma::Off);
    }

    /// The batch copy is an optimization: a driver without the symbol keeps the
    /// `index_copy` kernel whatever the env asks for, so the engine still opens
    /// and serves decode plus prefill on a pre-CUDA-13 driver.
    #[test]
    fn a_driver_without_the_batch_copy_falls_back_to_the_kernel() {
        assert_eq!(prefill_dma_from(None, false), PrefillDma::Off);
        assert_eq!(prefill_dma_from(Some("1"), false), PrefillDma::Off);
        assert_eq!(prefill_dma_from(Some("0"), false), PrefillDma::Off);
    }

    /// The buffer is the coldest *contiguous* window, because contiguity is what
    /// makes one DMA run per bank cover a whole layer. A warm cache must lose as
    /// little as a window can avoid, and ties keep the lowest start.
    #[test]
    fn the_buffer_window_is_the_coldest_contiguous_run() {
        assert_eq!(pick_window(&[9, 9, 0, 0, 0, 0, 9, 9], 4), 2);
        assert_eq!(pick_window(&[9, 9, 1, 1, 0, 0], 2), 4);
        assert_eq!(pick_window(&[1, 1, 1, 1], 2), 0);
        assert_eq!(pick_window(&[7, 3, 5], 3), 0);
    }

    /// A whole-layer DMA plan is exactly six contiguous runs: each bank's layer
    /// window (every expert in id order) into the half's contiguous slot window.
    /// The two halves abut, so the buffer is one contiguous slot range.
    #[test]
    fn a_whole_layer_dma_plan_is_one_run_per_bank() {
        let banks: [PinnedBank; BANKS.len()] = std::array::from_fn(|i| PinnedBank {
            addr: 0x1_0000 + (i as u64) * 0x10_0000,
            row_bytes: 1024 << i,
            rows: 8 * 4,
        });
        let dst_base: [u64; BANKS.len()] =
            std::array::from_fn(|i| 0x1_0000_0000 + (i as u64) * 0x100_0000);
        let (experts, layer, window, half) = (8usize, 3usize, 100usize, 1usize);

        let runs = whole_layer_runs(&banks, &dst_base, experts, layer, window, half);
        for (i, run) in runs.iter().enumerate() {
            let row = banks[i].row_bytes;
            assert_eq!(run.src, banks[i].addr + (layer * experts * row) as u64);
            assert_eq!(
                run.dst,
                dst_base[i] + ((window + half * experts) * row) as u64
            );
            assert_eq!(run.bytes, experts * row);
        }

        let first = whole_layer_runs(&banks, &dst_base, experts, layer, window, 0);
        for i in 0..BANKS.len() {
            assert_eq!(
                first[i].dst + (experts * banks[i].row_bytes) as u64,
                runs[i].dst,
                "the halves must abut in bank {i}"
            );
        }
    }

    /// The adapter is device-only, so the tensor tests need a live runtime.
    /// On a host with no device they are skipped rather than failed.
    fn test_rt() -> Option<Runtime> {
        Runtime::open(0).ok()
    }

    /// The CSR the grouped launches read: bucketed per slot in stable route
    /// order, a complete permutation of the routes, and it refuses the shapes
    /// only the decode GEMV handles (a non-resident slot, a slot past the bank).
    #[test]
    fn a_grouped_csr_buckets_and_refuses_bad_slots() {
        let Some(rt) = test_rt() else {
            return;
        };
        let ints = |t: &Tensor| t.to_host_i32().unwrap();

        let (seg, route_idx) = build_grouped_csr_host(&[2, 0, 2, 0, 0, 1], 3, &rt).unwrap();
        assert_eq!(ints(&seg), vec![0, 3, 4, 6]);
        assert_eq!(ints(&route_idx), vec![1, 3, 4, 5, 0, 2]);

        let err = build_grouped_csr_host(&[-1, 0], 3, &rt)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(err.contains("not resident"), "{err}");

        let err = build_grouped_csr_host(&[3], 3, &rt)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(err.contains("past the"), "{err}");
    }

    /// The prefill chunker keeps every chunk inside the committed route cap, and
    /// never returns zero (which would loop forever).
    #[test]
    fn a_staged_chunk_fits_the_route_cap() {
        // top_k 8, cap 128 -> 16 tokens per chunk.
        assert_eq!(staged_chunk_tokens(128, 8), 16);
        // A cap that is an exact multiple still leaves the last chunk full.
        assert_eq!(staged_chunk_tokens(64, 8), 8);
        // A cap smaller than top_k cannot fit one route; the chunk is one token
        // and `ensure_experts` is what refuses the overflow.
        assert_eq!(staged_chunk_tokens(2, 8), 1);
    }

    /// Staging is byte indexing, so it is checkable on the CPU with the same code
    /// the device path runs: a slot's bytes land at its own offset and nowhere
    /// else.
    #[test]
    fn a_slab_lands_in_its_own_slot_row() {
        let Some(rt) = test_rt() else {
            return;
        };
        let (rows, cols) = (32usize, 64usize);
        let mut banks = Banks::new(&rt, 4, rows, cols).unwrap();
        let n = |i: usize| (i % 251) as u8;
        let slab = Slab {
            packed: (0..rows * cols / 2).map(n).collect(),
            scale: (0..rows * cols / 16).map(|i| n(i + 7)).collect(),
            global: (0..rows * 2).map(|i| n(i + 13)).collect(),
        };
        banks.stage(2, &slab, rows, cols).unwrap();
        let read = |t: &Tensor| t.to_host_u8().unwrap();
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
        let Some(rt) = test_rt() else {
            return;
        };
        let mut banks = Banks::new(&rt, 2, rows, cols).unwrap();
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
