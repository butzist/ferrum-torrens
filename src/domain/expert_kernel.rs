// MoE expert-compute ports: slot-resident NVFP4 experts plus the grouped GEMM
// that consumes them. Pure data and traits, no CUDA and no IO. `grouped_moe_reference`
// is the arithmetic oracle and mirrors the scalar tower's `moe_experts` exactly;
// adapters implement the two ports with host buffers (the CPU path) or the
// Triton kernel plus a device slot cache (the serving path).

use crate::domain::numerics::{e4m3fn_to_f32, f16_to_f32, fp4_nibble};
use thiserror::Error;

/// Whether an error text describes an allocation that ran out of device memory.
///
/// Two spellings reach this engine: cudarc renders the driver result as
/// `DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`, while our own
/// driver path carries the `cuGetErrorString` text, which is `out of memory`.
/// Both are stable across CUDA 12/13 and mean the same retryable condition, so
/// the classification lives next to `MoEError` instead of in each adapter.
pub fn is_out_of_device_memory(text: &str) -> bool {
    text.contains("CUDA_ERROR_OUT_OF_MEMORY") || text.contains("out of memory")
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MoEError {
    #[error("moe: {0}")]
    Bad(String),
    #[error("moe: slot {slot} out of range (capacity {capacity})")]
    SlotOutOfRange { slot: usize, capacity: usize },
    /// The checkpoint has no such expert-bank entry.
    #[error("moe: no expert bank entry {entry}")]
    Missing { entry: String },
    /// A device allocation for the expert cache failed because the GPU has no
    /// free memory. Distinct from `Bad` because the server answers it with a
    /// retryable 503 instead of treating it as a fault.
    #[error("moe: out of device memory: {0}")]
    OutOfMemory(String),
    /// The bytes behind an entry could not be read; `why` is the reader port's
    /// own message, kept verbatim so the shard/file detail survives.
    #[error("moe: read {entry}: {why}")]
    Read { entry: String, why: String },
}

/// Which half of an expert the two GEMMs read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    GateUp,
    Down,
}

/// The three NVFP4 fields of one expert half, in the order the copy kernel and
/// the FTW index spell them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Packed,
    Scale,
    Global,
}

/// The six expert-bank families of the offload cache. This order is a
/// cross-adapter contract: `index_copy`'s `src_ptrs`/`dst_ptrs` and
/// `CudaMoE::bank_ptrs` both index it, and the copy cubin bakes `NUM_BANKS = 6`
/// at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bank {
    GateUpPacked,
    GateUpScale,
    GateUpGlobal,
    DownPacked,
    DownScale,
    DownGlobal,
}

/// Every bank, in the order the copy kernel and the device cache index them:
/// gate_up then down, packed/scale/global within each.
pub const BANKS: [Bank; 6] = [
    Bank::GateUpPacked,
    Bank::GateUpScale,
    Bank::GateUpGlobal,
    Bank::DownPacked,
    Bank::DownScale,
    Bank::DownGlobal,
];

impl Bank {
    pub fn part(self) -> Part {
        match self {
            Bank::GateUpPacked | Bank::GateUpScale | Bank::GateUpGlobal => Part::GateUp,
            Bank::DownPacked | Bank::DownScale | Bank::DownGlobal => Part::Down,
        }
    }

    pub fn field(self) -> Field {
        match self {
            Bank::GateUpPacked | Bank::DownPacked => Field::Packed,
            Bank::GateUpScale | Bank::DownScale => Field::Scale,
            Bank::GateUpGlobal | Bank::DownGlobal => Field::Global,
        }
    }
}

/// One page-locked host bank, as the copy kernel sees it: `addr` is the
/// device-visible base (the allocation is mapped into device space), `row_bytes`
/// the pitch between experts, and `rows` every expert of every layer the source
/// holds. `src_idx` is therefore a flat `layer * experts + expert`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedBank {
    pub addr: u64,
    pub row_bytes: usize,
    pub rows: usize,
}

/// The pinned host side of the offload cache: the six device-visible bank
/// regions the copy kernel reads a miss from. Filled once from the FTW
/// checkpoint and kept for the life of the engine, so a captured graph's copy
/// node always reads the same addresses (decision 40 prerequisite 4).
pub trait PinnedBankSource: Send + Sync {
    fn banks(&self) -> Result<[PinnedBank; BANKS.len()], MoEError>;
}

/// One expert slab's bytes, already sliced to that expert's rows (decision 23
/// layouts): `packed [rows, cols/2]` u8 with the low nibble holding the even
/// k, `scale [rows, cols/16]` e4m3fn, `global [rows]` f16.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Slab {
    pub packed: Vec<u8>,
    pub scale: Vec<u8>,
    pub global: Vec<u8>,
}

/// Host bytes for every expert of one layer; mmap-backed in production.
pub trait ExpertSlabSource: Send + Sync {
    fn slab(&self, layer: usize, expert: u32, part: Part) -> Result<Slab, MoEError>;
}

/// The slot cache's write side: stage one expert's bytes into `slot`. `&self`,
/// not `&mut self`: one device cache serves every step, so the adapter takes the
/// lock itself and callers hold a shared handle. The banks hold every layer at
/// once with stable addresses (decision 40 prerequisite 2), so a layer switch is
/// just another copy into a slot the plan picked -- there is nothing to declare
/// and nothing to invalidate.
#[cfg_attr(test, mockall::automock)]
pub trait ExpertSlotStore: Send + Sync {
    fn load(&self, layer: usize, expert: u32, slot: usize) -> Result<(), MoEError>;
    fn slots(&self) -> usize;

    /// Free the device expert banks, if this store owns any. The slot tables
    /// and the pinned host source stay put, so `rehydrate_banks` can rebuild the
    /// banks at the same slot count. A host-only store has nothing to free.
    fn release_banks(&self) -> Result<(), MoEError> {
        Ok(())
    }

    /// Reallocate the device expert banks after `release_banks`. Every slot is
    /// empty again, which is why the caller also resets its slot machine.
    fn rehydrate_banks(&self) -> Result<(), MoEError> {
        Ok(())
    }

    /// Whether the device banks are currently allocated.
    fn banks_resident(&self) -> bool {
        true
    }
}

/// One layer's grouped MoE GEMM input: every `(token, top_k)` route is one
/// expert GEMV, `silu(gate) * up` sits between the two, and the routes are
/// summed weighted. `routes` holds slot ids (`-1` = not resident, contributes
/// nothing). The output stays a separate `&mut`, so nothing can alias it.
pub struct MoEGroup<'a> {
    pub hidden: usize,
    pub inter: usize,
    pub x: &'a [f32],
    pub top_k: usize,
    pub routes: &'a [i32],
    pub weights: &'a [f32],
}

/// Shapes of one expert's two slabs, the numbers the kernels index with:
/// `gate_up [2 * inter, hidden]`, `down [hidden, inter]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertShapes {
    pub hidden: usize,
    pub inter: usize,
}

impl ExpertShapes {
    pub fn new(hidden: usize, inter: usize) -> Result<Self, MoEError> {
        if hidden == 0 || inter == 0 || !hidden.is_multiple_of(16) || !inter.is_multiple_of(16) {
            return Err(MoEError::Bad(format!(
                "expert shapes hidden {hidden} inter {inter} must be nonzero multiples of 16"
            )));
        }
        Ok(Self { hidden, inter })
    }

    pub fn gate_up_rows(self) -> usize {
        2 * self.inter
    }

    /// `(rows, cols)` of one part, the shape both the device bank and the
    /// per-expert slab use.
    pub fn slab_shape(self, part: Part) -> (usize, usize) {
        match part {
            Part::GateUp => (self.gate_up_rows(), self.hidden),
            Part::Down => (self.hidden, self.inter),
        }
    }

    /// Bytes of one expert's row in `bank`, the copy kernel's `feat_bytes` and
    /// the pitch the device banks are written at. The three NVFP4 fields pack
    /// differently: `packed` holds two k per byte, `scale` one e4m3 per 16, and
    /// `global` one f16 per row.
    pub fn bank_row_bytes(self, bank: Bank) -> usize {
        let (rows, cols) = self.slab_shape(bank.part());
        match bank.field() {
            Field::Packed => rows * cols / 2,
            Field::Scale => rows * cols / 16,
            Field::Global => rows * 2,
        }
    }
}

/// The route lists the kernel wants, flattened out of the domain's dense
/// `[tokens, top_k]` slot ids. Pure, so it is tested without python.
pub fn route_lists(
    routes: &[i32],
    tokens: usize,
    top_k: usize,
) -> Result<(Vec<i32>, Vec<i32>), MoEError> {
    if tokens == 0 || top_k == 0 || routes.len() != tokens * top_k {
        return Err(MoEError::Bad(format!(
            "routes {} for {tokens} tokens x top_k {top_k}",
            routes.len()
        )));
    }
    let mut route_token = Vec::with_capacity(routes.len());
    let mut route_slot = Vec::with_capacity(routes.len());
    for (t, chunk) in routes.chunks(top_k).enumerate() {
        for &slot in chunk {
            route_token.push(t as i32);
            route_slot.push(slot);
        }
    }
    Ok((route_token, route_slot))
}

#[cfg_attr(test, mockall::automock)]
pub trait ExpertGemm: Send + Sync {
    fn grouped_moe<'a>(&self, req: &MoEGroup<'a>, out: &mut [f32]) -> Result<(), MoEError>;

    /// The same grouped MoE with the activations, the weights and the output
    /// kept on the device (decision 43). Only `routes` is a host slice: the
    /// plan is still computed here, and prerequisite 4 moves it.
    fn grouped_moe_dev<'a>(
        &self,
        req: &MoEDeviceGroup<'a>,
        out: &mut dyn MoETensor,
    ) -> Result<(), MoEError>;

    /// The device-resident hand-off of decision 40 prerequisite 4: the adapter
    /// rewrites the router's dense ids to slot ids with `lru_ensure` and copies
    /// the misses in with `index_copy`, both on the device, so nothing about the
    /// residency plan crosses to the host. `req.experts` is the router's dense
    /// `[tokens, top_k]` ids; the adapter works on its own device copy.
    fn grouped_moe_staged<'a>(
        &self,
        req: &MoEDeviceRequest<'a>,
        out: &mut dyn MoETensor,
    ) -> Result<(), MoEError>;

    /// Start a routed step that may finish later, so the shared expert can
    /// compute on the main stream while the missed experts are copied. The
    /// default returns `None`, meaning "no overlapped path: call
    /// `grouped_moe_staged` as today". An adapter that can split the copy onto
    /// a second stream returns `Some(handle)`; the caller computes the shared
    /// expert and then calls `handle.finish(out)`.
    fn begin_staged<'a>(
        &self,
        req: &MoEDeviceRequest<'a>,
    ) -> Result<Option<Box<dyn PendingMoE>>, MoEError> {
        let _ = req;
        Ok(None)
    }

    /// The largest `tokens * top_k` one `grouped_moe_staged` step can take, or
    /// zero when this adapter has no device staging. The committed `lru_ensure`
    /// bakes this bound, so the executor reports it and falls back to the host
    /// plan for a step past it rather than launching a kernel that cannot fit.
    fn staged_route_cap(&self) -> usize;

    /// Drop the device LRU's residency. The host plan writes the same device
    /// banks at slots the device table would still map, so entering the host
    /// path after the staged one has to invalidate the table first. A no-op when
    /// device staging is off.
    fn reset_staging(&self) -> Result<(), MoEError>;
}

/// An opaque buffer of device memory handed across the MoE ports: the tower's
/// activations, the router's ids and weights, the routed sum. The domain counts
/// elements and -- while the residency plan is still computed on the host --
/// asks for a host view of the routes; it never learns the representation, and
/// a tower on the device never converts anything itself to cross this boundary
/// (decision 43).
///
/// No automock: a mock of `as_any` would have to hand back a reference it owns,
/// and mockall's `Box<dyn Any>` expectation is not `Sync`. The tests use small
/// host-backed doubles instead.
pub trait MoETensor: Send + Sync {
    /// Elements the buffer holds, the way a host slice reports its length.
    fn numel(&self) -> usize;

    /// A host view, element-major. Only the routes are read this way, for the
    /// plan; activations and weights are not, which is what keeps them on the
    /// device.
    fn to_host_f32(&self) -> Result<Vec<f32>, MoEError>;

    /// Write `values` into the buffer from the host. The device adapter copies
    /// device to device instead and never calls this; it is how a host-side
    /// stand-in for the kernel answers.
    fn set_host_f32(&mut self, values: &[f32]) -> Result<(), MoEError>;

    /// The adapter downcasts to its own tensor type to copy out of or into a
    /// buffer, so the domain never has to describe a device tensor beyond
    /// counting its elements.
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// One routed-expert layer step, device-resident: the router's dense expert ids,
/// the top-k weights and the post-norm activations, every one of them a buffer
/// the adapter owns. Slot ids still deliberately do not appear here -- the
/// executor rewrites them behind this request. The output is a separate `&mut`,
/// as in `ExpertGemm`, so nothing can alias it.
pub struct MoEDeviceRequest<'a> {
    pub layer: usize,
    pub hidden: usize,
    pub inter: usize,
    pub top_k: usize,
    pub x: &'a dyn MoETensor,
    pub experts: &'a dyn MoETensor,
    pub weights: &'a dyn MoETensor,
}

/// The device half of [`MoEGroup`]: activations and weights stay in their
/// buffers, while `routes` is the plan's own host output (`-1` for a slot that
/// is not resident).
pub struct MoEDeviceGroup<'a> {
    pub hidden: usize,
    pub inter: usize,
    pub top_k: usize,
    pub x: &'a dyn MoETensor,
    pub routes: &'a [i32],
    pub weights: &'a dyn MoETensor,
}

/// A routed step that has been started but not finished: the missed experts
/// have been copied (possibly on a second stream) but the grouped GEMV has not
/// run yet. The caller finishes it after the independent shared-expert work.
/// Pure domain: no CUDA, no streams live in this type.
#[cfg_attr(test, mockall::automock)]
pub trait PendingMoE: Send {
    fn finish(self: Box<Self>, out: &mut dyn MoETensor) -> Result<(), MoEError>;
}

/// The engine's MoE port: one call per routed layer per step. Arch-agnostic on
/// purpose. The offload cache and the grouped GEMM are core engine obligations
/// shared by every supported model, so an arch forward only hands over its
/// routing decision and never widens an expert slab itself.
///
/// The hand-off is device-resident (decision 43): a tower whose activations
/// already live on the device hands the port its buffers and gets the routed
/// sum back as one, so nothing crosses to the host at the layer boundary.
#[cfg_attr(test, mockall::automock)]
pub trait MoEExecutor: Send + Sync {
    fn execute_device<'a>(
        &self,
        req: &MoEDeviceRequest<'a>,
        out: &mut dyn MoETensor,
    ) -> Result<(), MoEError>;

    /// Start a routed step that may overlap its miss copy with independent
    /// work. The default returns `None`, so callers fall back to
    /// `execute_device` exactly as before. Pure domain: no streams here.
    fn begin_device<'a>(
        &self,
        req: &MoEDeviceRequest<'a>,
    ) -> Result<Option<Box<dyn PendingMoE>>, MoEError> {
        let _ = req;
        Ok(None)
    }

    /// How many steps this executor has run since it was built. A forward that
    /// widens routed experts on the host answers with the same logits as one
    /// that goes through this port, so this is the only way a caller (or a gate)
    /// can tell which one actually happened. No default: an adapter that cannot
    /// count must say so rather than report zero.
    fn steps(&self) -> u64;

    /// Free the device expert cache and forget which expert sits in which slot.
    /// The next step stages its misses again. Used by the residency controller
    /// before a vision pass borrows the device; a host-only executor is a no-op.
    fn release_cache(&self) -> Result<(), MoEError> {
        Ok(())
    }

    /// Reallocate the device expert cache at its configured slot count after
    /// `release_cache`.
    fn rehydrate_cache(&self) -> Result<(), MoEError> {
        Ok(())
    }

    /// Whether the device expert cache is currently allocated.
    fn cache_resident(&self) -> bool {
        true
    }

    /// Drop the device LRU's residency tables (slot_for_id, id_of_slot, usage,
    /// step counter) without freeing the device banks. Used after a CUDA-graph
    /// capture, whose padding row ran `lru_ensure` and polluted the tables; the
    /// first real replay must start from a cold cache so its slot assignment
    /// matches the eager path (decision 164). A no-op when device staging is
    /// off or the executor has no device LRU.
    fn reset_staging(&self) -> Result<(), MoEError> {
        Ok(())
    }
}

/// Slot-major banks for the reference path: `packed [slots, rows, cols/2]`, the
/// scales and globals the same way. This is exactly what a device slot cache
/// holds, so the Triton kernel sees the same byte layout.
pub struct HostBanks<'a> {
    pub gu_packed: &'a [u8],
    pub gu_scale: &'a [u8],
    pub gu_global: &'a [u8],
    pub dn_packed: &'a [u8],
    pub dn_scale: &'a [u8],
    pub dn_global: &'a [u8],
    pub slots: usize,
    pub gu_rows: usize,
    pub gu_cols: usize,
    pub dn_rows: usize,
    pub dn_cols: usize,
}

/// `out[t] = sum_j w[t][j] * down^T(silu_mul(gate_up^T x[t]))` over the resident
/// slots, in f32 and in the same order the scalar tower accumulates it.
pub fn grouped_moe_reference(
    banks: &HostBanks<'_>,
    req: &MoEGroup<'_>,
    out: &mut [f32],
) -> Result<(), MoEError> {
    grouped_moe_impl(banks, req, out, None)
}

/// The same sum, also reporting `abs_sum[t] = sum_j |w[t][j]| * |ic3_j(t)|`.
/// A route's terms nearly cancel, so `out` alone understates their size by
/// orders of magnitude and is the wrong yardstick for a kernel's error.
pub fn grouped_moe_reference_abs_sum(
    banks: &HostBanks<'_>,
    req: &MoEGroup<'_>,
    out: &mut [f32],
    abs_sum: &mut [f32],
) -> Result<(), MoEError> {
    grouped_moe_impl(banks, req, out, Some(abs_sum))
}

fn grouped_moe_impl(
    banks: &HostBanks<'_>,
    req: &MoEGroup<'_>,
    out: &mut [f32],
    mut abs_sum: Option<&mut [f32]>,
) -> Result<(), MoEError> {
    let h = req.hidden;
    let i = req.inter;
    if banks.gu_rows != 2 * i
        || banks.gu_cols != h
        || banks.dn_rows != h
        || banks.dn_cols != i
        || h == 0
        || i == 0
        || !h.is_multiple_of(16)
        || !i.is_multiple_of(16)
    {
        return Err(MoEError::Bad(format!(
            "banks gu {}x{} dn {}x{} vs hidden {h} inter {i}",
            banks.gu_rows, banks.gu_cols, banks.dn_rows, banks.dn_cols
        )));
    }
    let slots = banks.slots;
    let gu_pb = banks.gu_rows * h / 2;
    let gu_sb = banks.gu_rows * h / 16;
    let gu_gb = banks.gu_rows * 2;
    let dn_pb = h * i / 2;
    let dn_sb = h * i / 16;
    let dn_gb = h * 2;
    if banks.gu_packed.len() != slots * gu_pb
        || banks.gu_scale.len() != slots * gu_sb
        || banks.gu_global.len() != slots * gu_gb
        || banks.dn_packed.len() != slots * dn_pb
        || banks.dn_scale.len() != slots * dn_sb
        || banks.dn_global.len() != slots * dn_gb
    {
        return Err(MoEError::Bad("slot bank byte lengths".into()));
    }

    let tokens = req.x.len() / h;
    let routes = tokens * req.top_k;
    if req.x.len() != tokens * h
        || req.routes.len() < routes
        || req.weights.len() < routes
        || out.len() < tokens * h
    {
        return Err(MoEError::Bad(format!(
            "request buffer lengths: x {} routes {} weights {} out {}",
            req.x.len(),
            req.routes.len(),
            req.weights.len(),
            out.len()
        )));
    }

    let out = &mut out[..tokens * h];
    out.fill(0.0);
    let mut abs_sum = match abs_sum.as_mut() {
        Some(buf) if buf.len() >= tokens * h => {
            let buf = &mut buf[..tokens * h];
            buf.fill(0.0);
            Some(&mut buf[..])
        }
        Some(buf) => {
            return Err(MoEError::Bad(format!(
                "abs_sum needs {tokens} * {h}, got {}",
                buf.len()
            )));
        }
        None => None,
    };
    let mut ic1 = vec![0.0f32; 2 * i];
    let mut ic2 = vec![0.0f32; i];
    let mut ic3 = vec![0.0f32; h];

    for t in 0..tokens {
        let xrow = &req.x[t * h..(t + 1) * h];
        for j in 0..req.top_k {
            let slot = req.routes[t * req.top_k + j];
            if slot < 0 {
                continue;
            }
            let slot = slot as usize;
            if slot >= slots {
                return Err(MoEError::SlotOutOfRange {
                    slot,
                    capacity: slots,
                });
            }
            let w = req.weights[t * req.top_k + j];

            gemv_rows(
                xrow,
                &banks.gu_packed[slot * gu_pb..(slot + 1) * gu_pb],
                &banks.gu_scale[slot * gu_sb..(slot + 1) * gu_sb],
                &banks.gu_global[slot * gu_gb..(slot + 1) * gu_gb],
                2 * i,
                h,
                &mut ic1,
            );
            for k in 0..i {
                let g = ic1[k] / (1.0 + (-ic1[k]).exp());
                ic2[k] = g * ic1[i + k];
            }
            gemv_rows(
                &ic2,
                &banks.dn_packed[slot * dn_pb..(slot + 1) * dn_pb],
                &banks.dn_scale[slot * dn_sb..(slot + 1) * dn_sb],
                &banks.dn_global[slot * dn_gb..(slot + 1) * dn_gb],
                h,
                i,
                &mut ic3,
            );
            for d in 0..h {
                out[t * h + d] += w * ic3[d];
                if let Some(abs) = abs_sum.as_deref_mut() {
                    abs[t * h + d] += (w * ic3[d]).abs();
                }
            }
        }
    }
    Ok(())
}

/// `y[r] = (sum_k x[k] * fp4(packed, r, k) * e4m3(scale, r, k / 16)) *
/// f16(global, r)`, the same order the scalar tower's `nvfp4_gemv` accumulates.
fn gemv_rows(
    x: &[f32],
    packed: &[u8],
    scale: &[u8],
    global: &[u8],
    rows: usize,
    cols: usize,
    y: &mut [f32],
) {
    let cols2 = cols / 2;
    let cols16 = cols / 16;
    for r in 0..rows {
        let g = if global.len() >= 2 * rows {
            f16_to_f32(u16::from_le_bytes([global[2 * r], global[2 * r + 1]]))
        } else {
            1.0
        };
        let mut acc = 0.0f32;
        for b in 0..cols16 {
            let s = e4m3fn_to_f32(scale[r * cols16 + b]);
            for j in 0..16 {
                let k = b * 16 + j;
                acc += x[k] * fp4_nibble(packed[r * cols2 + k / 2], k % 2 == 1) * s;
            }
        }
        y[r] = acc * g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two spellings the engine actually sees, plus the near misses that
    /// must stay ordinary faults. The first string is the idle-release OOM this
    /// classifier exists for.
    #[test]
    fn out_of_memory_texts_are_recognized() {
        assert!(is_out_of_device_memory(
            "cudarc: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        ));
        assert!(is_out_of_device_memory("cuda: out of memory"));
        assert!(!is_out_of_device_memory(
            "cudarc: DriverError(CUDA_ERROR_INVALID_VALUE, \"invalid argument\")"
        ));
        assert!(!is_out_of_device_memory(
            "moe: no expert bank entry down#L00003"
        ));
    }

    #[test]
    fn routes_flatten_token_major() {
        let (token, slot) = route_lists(&[3, -1, 1, 1, 0, 2], 3, 2).unwrap();
        assert_eq!(token, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(slot, vec![3, -1, 1, 1, 0, 2]);
    }
    #[test]
    fn routes_reject_wrong_shapes() {
        assert!(route_lists(&[0, 1], 2, 2).is_err());
        assert!(route_lists(&[], 0, 1).is_err());
        assert!(route_lists(&[0], 1, 0).is_err());
    }
    #[test]
    fn shapes_need_multiples_of_sixteen() {
        assert!(ExpertShapes::new(1024, 512).is_ok());
        assert!(ExpertShapes::new(1000, 512).is_err());
        assert!(ExpertShapes::new(1024, 0).is_err());
        let s = ExpertShapes::new(1024, 512).unwrap();
        assert_eq!(s.slab_shape(Part::GateUp), (1024, 1024));
        assert_eq!(s.slab_shape(Part::Down), (1024, 512));
    }

    #[test]
    fn the_bank_order_covers_both_parts_and_all_three_fields() {
        assert_eq!(
            BANKS.map(|b| (b.part(), b.field())),
            [
                (Part::GateUp, Field::Packed),
                (Part::GateUp, Field::Scale),
                (Part::GateUp, Field::Global),
                (Part::Down, Field::Packed),
                (Part::Down, Field::Scale),
                (Part::Down, Field::Global),
            ]
        );
    }

    #[test]
    fn bank_row_bytes_matches_the_nvfp4_field_packing() {
        let s = ExpertShapes::new(2048, 512).unwrap();
        // gate_up is [2*inter=1024, hidden=2048], down is [hidden=2048, inter=512].
        assert_eq!(s.bank_row_bytes(Bank::GateUpPacked), 1024 * 2048 / 2);
        assert_eq!(s.bank_row_bytes(Bank::GateUpScale), 1024 * 2048 / 16);
        assert_eq!(s.bank_row_bytes(Bank::GateUpGlobal), 1024 * 2);
        assert_eq!(s.bank_row_bytes(Bank::DownPacked), 2048 * 512 / 2);
        assert_eq!(s.bank_row_bytes(Bank::DownScale), 2048 * 512 / 16);
        assert_eq!(s.bank_row_bytes(Bank::DownGlobal), 2048 * 2);
        // The FTW `gate_up_packed#L*` bank is [experts, 1024, 1024]: one expert
        // row is 1 MiB, the number the copy's `splits` is derived from.
        assert_eq!(s.bank_row_bytes(Bank::GateUpPacked), 1024 * 1024);
    }

    const H: usize = 16;
    const I: usize = 16;

    const E4M3_ONE: u8 = 0x38;
    const E4M3_TWO: u8 = 0x40;

    /// Only the round values the tests use.
    fn f16_bits(v: f32) -> [u8; 2] {
        let bits: u16 = match v {
            0.5 => 0x3800,
            1.0 => 0x3c00,
            -1.0 => 0xbc00,
            4.0 => 0x4400,
            _ => panic!("unsupported f16 literal"),
        };
        bits.to_le_bytes()
    }

    /// One slab whose every packed byte is `packed` and every scale is `scale`.
    fn slab(rows: usize, cols: usize, packed: u8, scale: u8, global: f32) -> Slab {
        Slab {
            packed: vec![packed; rows * cols / 2],
            scale: vec![scale; rows * cols / 16],
            global: vec![f16_bits(global); rows].concat(),
        }
    }

    /// Slot-major bytes, exactly what a device slot cache holds.
    fn flat(slabs: &[Slab]) -> Vec<u8> {
        slabs
            .iter()
            .flat_map(|s| s.packed.iter().copied())
            .collect()
    }
    fn flat_scale(slabs: &[Slab]) -> Vec<u8> {
        slabs.iter().flat_map(|s| s.scale.iter().copied()).collect()
    }
    fn flat_global(slabs: &[Slab]) -> Vec<u8> {
        slabs
            .iter()
            .flat_map(|s| s.global.iter().copied())
            .collect()
    }

    fn close(a: f32, b: f32) {
        assert!(
            (a - b).abs() <= 1e-4 * b.abs().max(1.0),
            "got {a}, want {b}"
        );
    }

    #[test]
    fn hand_computed_case_matches() {
        // Byte 0x10 is lo-nibble code 1 (0.5) and hi-nibble code 0 (0.0), so
        // every weight row is 0.5 on even k and 0.0 on odd k. With x all ones
        // and unit scales: ic1[r] = 8 * 0.5 = 4, gate and up are both 4,
        // ic2 = silu(4) * 4 everywhere, ic3[d] = 8 * 0.5 * ic2.
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let want = 0.5 * (8.0 * 0.5 * silu(4.0) * 4.0);

        let gu = flat(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let gu_s = flat_scale(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let gu_g = flat_global(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let dn = flat(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let dn_s = flat_scale(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let dn_g = flat_global(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 1,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let mut out = vec![0.0f32; H];
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 1,
                routes: &[0],
                weights: &[0.5],
            },
            &mut out,
        )
        .unwrap();
        for v in out {
            close(v, want);
        }
    }

    #[test]
    fn the_abs_sum_reports_the_terms_a_cancelling_sum_hides() {
        // Slot 1 is slot 0 with its down global negated, so two unit-weighted
        // routes cancel exactly (silu is not odd, so the gate_up global cannot
        // do it). `out` is then uninformative about the size of the terms that
        // produced it; `abs_sum` is what an error must be judged by.
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let term = 4.0 * silu(4.0) * 4.0;

        let gu_slabs = [
            slab(2 * I, H, 0x10, E4M3_ONE, 1.0),
            slab(2 * I, H, 0x10, E4M3_ONE, 1.0),
        ];
        let dn_slabs = [
            slab(H, I, 0x10, E4M3_ONE, 1.0),
            slab(H, I, 0x10, E4M3_ONE, -1.0),
        ];
        let gu = flat(&gu_slabs);
        let gu_s = flat_scale(&gu_slabs);
        let gu_g = flat_global(&gu_slabs);
        let dn = flat(&dn_slabs);
        let dn_s = flat_scale(&dn_slabs);
        let dn_g = flat_global(&dn_slabs);
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 2,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let mut out = vec![0.0f32; H];
        let mut abs_sum = vec![0.0f32; H];
        grouped_moe_reference_abs_sum(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 2,
                routes: &[0, 1],
                weights: &[1.0, 1.0],
            },
            &mut out,
            &mut abs_sum,
        )
        .unwrap();
        for (o, a) in out.iter().zip(&abs_sum) {
            close(*o, 0.0);
            close(*a, 2.0 * term);
        }
    }

    #[test]
    fn non_resident_route_contributes_nothing() {
        let gu = flat(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let gu_s = flat_scale(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let gu_g = flat_global(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let dn = flat(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let dn_s = flat_scale(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let dn_g = flat_global(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 1,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let mut out = vec![7.0f32; H];
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 1,
                routes: &[-1],
                weights: &[1.0],
            },
            &mut out,
        )
        .unwrap();
        assert_eq!(out, vec![0.0; H]);
    }

    #[test]
    fn slot_selects_its_own_expert() {
        // Slot 0 is 0x01 (0.5 on even k), slot 1 the 0x10 one (0.5 on odd k),
        // so alternating x tells the two slots apart.
        let gus = [
            slab(2 * I, H, 0x01, E4M3_ONE, 1.0),
            slab(2 * I, H, 0x10, E4M3_ONE, 1.0),
        ];
        let dns = [
            slab(H, I, 0x01, E4M3_ONE, 1.0),
            slab(H, I, 0x10, E4M3_ONE, 1.0),
        ];
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));

        let run = |banks: &HostBanks<'_>, slot: i32| {
            let x: Vec<f32> = (0..H).map(|k| if k % 2 == 0 { 1.0 } else { 2.0 }).collect();
            let mut out = vec![0.0f32; H];
            grouped_moe_reference(
                banks,
                &MoEGroup {
                    hidden: H,
                    inter: I,
                    x: &x,
                    top_k: 1,
                    routes: &[slot],
                    weights: &[1.0],
                },
                &mut out,
            )
            .unwrap();
            out
        };

        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 2,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let silu = |v: f32| v / (1.0 + (-v).exp());
        // 8 nonzero weights per row: ic1 = 8 * 0.5 * 1 = 4 on slot 0 and
        // 8 * 0.5 * 2 = 8 on slot 1, then ic2 = silu(ic1) * ic1.
        let want0 = 8.0 * 0.5 * silu(4.0) * 4.0;
        let want1 = 8.0 * 0.5 * silu(8.0) * 8.0;
        for v in run(&banks, 0) {
            close(v, want0);
        }
        for v in run(&banks, 1) {
            close(v, want1);
        }

        // Zeroing slot 1's bytes must zero the slot-1 route only.
        let mut gus = gus.clone();
        let mut dns = dns.clone();
        gus[1] = Slab {
            packed: vec![0; 2 * I * H / 2],
            scale: vec![0; 2 * I * H / 16],
            global: vec![0; 4 * I],
        };
        dns[1] = Slab {
            packed: vec![0; H * I / 2],
            scale: vec![0; H * I / 16],
            global: vec![0; 2 * H],
        };
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 2,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        assert_eq!(run(&banks, 1), vec![0.0; H]);
        assert!(run(&banks, 0).iter().all(|&v| v > 0.0));
    }

    #[test]
    fn global_scale_applies_once_after_the_k_reduction() {
        // gate_up global 4x and down global 0.5x: the down half scales
        // linearly, the gate_up half goes through silu(4 * ic1).
        let gus = [slab(2 * I, H, 0x10, E4M3_ONE, 4.0)];
        let dns = [slab(H, I, 0x10, E4M3_ONE, 0.5)];
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 1,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let mut out = vec![0.0f32; H];
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 1,
                routes: &[0],
                weights: &[1.0],
            },
            &mut out,
        )
        .unwrap();
        // ic1 = 8 * 0.5 * 4 = 16, so ic2 = silu(16) * 16, ic3 = 8 * 0.5 * ic2 * 0.5.
        let ic2 = (16.0 / (1.0 + (-16.0f32).exp())) * 16.0;
        let want = 8.0 * 0.5 * ic2 * 0.5;
        for v in out {
            close(v, want);
        }
    }

    #[test]
    fn block_scale_is_per_sixteen_k() {
        // cols = 32: two e4m3 blocks per row, the second one 2x.
        let s0 = slab(2 * I, 32, 0x10, E4M3_ONE, 1.0);
        let mut s1 = slab(2 * I, 32, 0x10, E4M3_TWO, 1.0);
        s1.scale = (0..2 * I * 32 / 16)
            .map(|b| if b % 2 == 1 { E4M3_TWO } else { E4M3_ONE })
            .collect();
        let gus = [s0, s1];
        let dns = [
            slab(32, I, 0x10, E4M3_ONE, 1.0),
            slab(32, I, 0x10, E4M3_ONE, 1.0),
        ];
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 2,
            gu_rows: 2 * I,
            gu_cols: 32,
            dn_rows: 32,
            dn_cols: I,
        };
        let x = vec![1.0f32; 32];
        let mut out = vec![0.0f32; 32];
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: 32,
                inter: I,
                x: &x,
                top_k: 1,
                routes: &[1],
                weights: &[1.0],
            },
            &mut out,
        )
        .unwrap();
        // 16 k per row: block 0 carries the even k at scale 1 (8 * 0.5) and
        // block 1 the odd k at scale 2 (8 * 0.5 * 2), so ic1 = 12.
        let ic2 = (12.0 / (1.0 + (-12.0f32).exp())) * 12.0;
        let want = 8.0 * 0.5 * ic2;
        for v in out {
            close(v, want);
        }
    }

    #[test]
    fn routes_sum_weighted_over_top_k() {
        let gus = [
            slab(2 * I, H, 0x10, E4M3_ONE, 1.0),
            slab(2 * I, H, 0x11, E4M3_ONE, 1.0),
        ];
        let dns = [
            slab(H, I, 0x10, E4M3_ONE, 1.0),
            slab(H, I, 0x11, E4M3_ONE, 1.0),
        ];
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 2,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let one = |slot: i32, w: f32| {
            let mut out = vec![0.0f32; H];
            grouped_moe_reference(
                &banks,
                &MoEGroup {
                    hidden: H,
                    inter: I,
                    x: &x,
                    top_k: 1,
                    routes: &[slot],
                    weights: &[w],
                },
                &mut out,
            )
            .unwrap();
            out
        };
        let mut both = vec![0.0f32; H];
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 2,
                routes: &[0, 1],
                weights: &[0.25, 0.75],
            },
            &mut both,
        )
        .unwrap();
        let a = one(0, 0.25);
        let b = one(1, 0.75);
        for i in 0..H {
            close(both[i], a[i] + b[i]);
        }
    }

    #[test]
    fn shape_and_slot_mismatches_are_errors() {
        let gus = [slab(2 * I, H, 0x10, E4M3_ONE, 1.0)];
        let dns = [slab(H, I, 0x10, E4M3_ONE, 1.0)];
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 1,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let mut out = vec![0.0f32; H];
        let wide = MoEGroup {
            hidden: H + 16,
            inter: I,
            x: &x,
            top_k: 1,
            routes: &[0],
            weights: &[1.0],
        };
        assert!(grouped_moe_reference(&banks, &wide, &mut out).is_err());

        let ok = MoEGroup { hidden: H, ..wide };
        let overflow = MoEGroup { routes: &[5], ..ok };
        assert!(matches!(
            grouped_moe_reference(&banks, &overflow, &mut out),
            Err(MoEError::SlotOutOfRange {
                slot: 5,
                capacity: 1
            })
        ));
    }

    #[test]
    fn reference_matches_the_scalar_gemv_order() {
        // Same bytes through the scalar tower's `nvfp4_gemv` + silu-mul path,
        // which is the oracle the Triton kernel will be compared against.
        let rows = 2 * I;
        let mut packed = vec![0u8; rows * H / 2];
        let mut scale = vec![0u8; rows * H / 16];
        let mut global = vec![0u8; rows * 2];
        for (i, b) in packed.iter_mut().enumerate() {
            *b = (i * 37 + 11) as u8;
        }
        for (i, b) in scale.iter_mut().enumerate() {
            *b = (i * 7 + 3) as u8;
        }
        for r in 0..rows {
            global[2 * r..2 * r + 2].copy_from_slice(&f16_bits(1.0));
        }
        let dn = slab(H, I, 0x21, E4M3_ONE, 1.0);
        let dns = flat(std::slice::from_ref(&dn));
        let dns_s = flat_scale(std::slice::from_ref(&dn));
        let dns_g = flat_global(std::slice::from_ref(&dn));

        let x: Vec<f32> = (0..H).map(|k| ((k * 13 % 9) as f32) * 0.25 - 1.0).collect();
        let mut out = vec![0.0f32; H];
        let banks = HostBanks {
            gu_packed: &packed,
            gu_scale: &scale,
            gu_global: &global,
            dn_packed: &dns,
            dn_scale: &dns_s,
            dn_global: &dns_g,
            slots: 1,
            gu_rows: rows,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 1,
                routes: &[0],
                weights: &[1.0],
            },
            &mut out,
        )
        .unwrap();

        let ic1 =
            crate::models::qwen3_6::forward::nvfp4_gemv(&x, &packed, &scale, &global, rows, H);
        let mut ic2 = vec![0.0f32; I];
        for k in 0..I {
            let g = ic1[k] / (1.0 + (-ic1[k]).exp());
            ic2[k] = g * ic1[I + k];
        }
        let ic3 = crate::models::qwen3_6::forward::nvfp4_gemv(&ic2, &dns, &dns_s, &dns_g, H, I);
        for i in 0..H {
            close(out[i], ic3[i]);
        }
    }

    #[test]
    fn dequant_helpers_agree_with_the_reference_scales() {
        // Pins the two scale decoders the kernel has to mirror bit for bit.
        assert_eq!(e4m3fn_to_f32(E4M3_ONE), 1.0);
        assert_eq!(e4m3fn_to_f32(E4M3_TWO), 2.0);
        assert_eq!(f16_to_f32(u16::from_le_bytes(f16_bits(4.0))), 4.0);
        // Low nibble is the even k (decision 23).
        assert_eq!(fp4_nibble(0x10, false), 0.0);
        assert_eq!(fp4_nibble(0x10, true), 0.5);
        assert_eq!(fp4_nibble(0x01, false), 0.5);
        assert_eq!(fp4_nibble(0x01, true), 0.0);
    }
}
