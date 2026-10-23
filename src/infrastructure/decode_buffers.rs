// Fixed-address device buffers for one decode step: the pinned batch-metadata
// staging of TODO.md's P1 and the `GraphCaptureBuffer` of decision 40's slice 2.
//
// A captured graph bakes the addresses of everything its kernels read, so the
// per-step metadata a replay overwrites -- the input tokens, the rope positions,
// each request's GDN state slot and its KV pool location, and the logits the
// step produces -- has to sit at fixed addresses for the life of the engine.
// FreeToken's `engine/graph.py` keeps exactly this set on device tensors and
// stages the next step into them between replays; this owns the same set here.
//
// Same shape of boundary as decision 47's pinned expert banks: allocated once,
// never moved, filled in place. Unlike the expert banks (read by the driver-API
// copy kernel out of pinned host memory), these are ordinary device tensors,
// because the staging is a host H2D copy outside any capture -- the graph only
// ever reads them.

use crate::infrastructure::cudarc::{Runtime, Tensor, TensorDType};
use crate::infrastructure::torch::Device;

/// Re-export the row type so callers that stage buffers get it from one place.
pub use crate::domain::decode::DecodeRow;

/// The out-of-range pool location a padded (dummy) request carries. Real
/// locations are never negative, so the sentinel can never alias one; every
/// location that reads back as this was padding rather than a live request.
pub const SENTINEL_LOC: i32 = -1;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DecodeBufferError {
    #[error("decode step has {rows} rows, over the {max} the buffers hold")]
    TooManyRows { rows: usize, max: usize },
    #[error("cannot pad {rows} rows to {padded}, under the staged count")]
    PaddedUnderRows { rows: usize, padded: usize },
    #[error("padding to {padded} is over the {max} the buffers hold")]
    PaddedOverCapacity { padded: usize, max: usize },
    #[error("kv window of {locs} locations is over the {window} the buffer holds")]
    WindowOverCapacity { locs: usize, window: usize },
}

/// The device tensors one captured decode step reads and writes, allocated once
/// at `[max_bs]` (plus `[max_bs, vocab]` for the logits and `[max_bs, window]`
/// for the KV window) and staged in place.
pub struct DecodeBuffers {
    input_ids: Tensor,
    positions: Tensor,
    state_slots: Tensor,
    out_locs: Tensor,
    kv_locs: Tensor,
    logits: Tensor,
    max_bs: usize,
    vocab: usize,
    window: usize,
    device: Device,
}

impl DecodeBuffers {
    /// Allocate the whole set at the captured batch size. The logits are the one
    /// large buffer: `max_bs * vocab * 4` bytes, which is what a decode graph
    /// writes and the sampler reads back. `window` is the fixed number of pool
    /// locations a step gathers over; it must cover the largest context the
    /// captured step will ever serve, because a graph pins the gather length.
    pub fn new(max_bs: usize, vocab: usize, window: usize, device: Device) -> Self {
        assert!(max_bs > 0 && vocab > 0 && window > 0);
        let ordinal = match device {
            Device::Cuda(ordinal) => ordinal,
            Device::Cpu => 0,
        };
        let rt = Runtime::open(ordinal).expect("open the cudarc compute runtime");
        Self {
            input_ids: rt
                .zeros(&[max_bs], TensorDType::I32)
                .expect("allocate decode input ids"),
            positions: rt
                .zeros(&[max_bs], TensorDType::I32)
                .expect("allocate decode positions"),
            state_slots: rt
                .zeros(&[max_bs], TensorDType::I32)
                .expect("allocate decode state slots"),
            out_locs: rt
                .zeros(&[max_bs], TensorDType::I32)
                .expect("allocate decode out locs"),
            // Every window entry starts at pool location 0. The padding value is
            // irrelevant to the arithmetic -- the device mask drops every entry
            // past the step's position -- but it must be a *valid* pool index,
            // because the gather reads it as a pool location.
            kv_locs: rt
                .zeros(&[max_bs, window], TensorDType::I32)
                .expect("allocate decode kv locs"),
            logits: rt
                .zeros(&[max_bs, vocab], TensorDType::F32)
                .expect("allocate decode logits"),
            max_bs,
            vocab,
            window,
            device,
        }
    }

    pub fn max_bs(&self) -> usize {
        self.max_bs
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// The fixed attention window length: how many pool locations a gathered
    /// step reads, padding sentinel included.
    pub fn window(&self) -> usize {
        self.window
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The staged input tokens, `[max_bs]` i32: the rows the embedding gathers.
    pub fn input_ids(&self) -> &Tensor {
        &self.input_ids
    }

    /// The staged absolute rope positions, `[max_bs]` i32.
    pub fn positions(&self) -> &Tensor {
        &self.positions
    }

    /// The staged GDN state slot per row, `[max_bs]` i32.
    pub fn state_slots(&self) -> &Tensor {
        &self.state_slots
    }

    /// The staged KV pool location per row, `[max_bs]` i32.
    pub fn out_locs(&self) -> &Tensor {
        &self.out_locs
    }

    /// The staged attention windows, `[max_bs, window]` i32. Each row is the pool
    /// locations attention gathers over, head-aligned and padded to the captured
    /// window length, so a graph reads a fixed address and shape.
    pub fn kv_locs(&self) -> &Tensor {
        &self.kv_locs
    }

    /// One row's window, `[window]`, the tensor a decode gather reads.
    pub fn kv_locs_row(&self, row: usize) -> Tensor {
        assert!(row < self.max_bs, "kv_locs_row past the captured batch");
        self.kv_locs.select(0, row).expect("kv locs row")
    }

    /// The logits `[max_bs, vocab]` f32 the step writes into, and the sampler
    /// reads back. A capture writes the first `rows` of them.
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }

    /// The first `n` logit rows, a view for a caller that has fewer live rows
    /// than the captured batch.
    pub fn logits_rows(&self, n: usize) -> Tensor {
        assert!(n <= self.max_bs, "logits_rows past the captured batch");
        self.logits.narrow(0, 0, n).expect("logit rows")
    }

    /// Stage `rows` at the head of every buffer. The tail keeps whatever the
    /// previous step left there, so a replay over a shorter batch reads stale
    /// valid rows rather than an undefined buffer; `stage_padded` marks them.
    pub fn stage(&self, rows: &[DecodeRow]) -> Result<(), DecodeBufferError> {
        if rows.len() > self.max_bs {
            return Err(DecodeBufferError::TooManyRows {
                rows: rows.len(),
                max: self.max_bs,
            });
        }
        self.copy_i32(
            &self.input_ids,
            &rows.iter().map(|r| r.input_id).collect::<Vec<_>>(),
        );
        self.copy_i32(
            &self.positions,
            &rows.iter().map(|r| r.position).collect::<Vec<_>>(),
        );
        self.copy_i32(
            &self.state_slots,
            &rows.iter().map(|r| r.state_slot).collect::<Vec<_>>(),
        );
        self.copy_i32(
            &self.out_locs,
            &rows.iter().map(|r| r.out_loc).collect::<Vec<_>>(),
        );
        Ok(())
    }

    /// Stage `rows` and pad every row up to `padded_bs` with sentinel rows, so a
    /// replay captured at `padded_bs` sees the whole batch defined. This is the
    /// dummy-request padding step 3 of decision 40's capture plan needs.
    pub fn stage_padded(
        &self,
        rows: &[DecodeRow],
        padded_bs: usize,
    ) -> Result<(), DecodeBufferError> {
        if padded_bs < rows.len() {
            return Err(DecodeBufferError::PaddedUnderRows {
                rows: rows.len(),
                padded: padded_bs,
            });
        }
        if padded_bs > self.max_bs {
            return Err(DecodeBufferError::PaddedOverCapacity {
                padded: padded_bs,
                max: self.max_bs,
            });
        }
        let mut full = rows.to_vec();
        full.resize(padded_bs, DecodeRow::sentinel());
        self.stage(&full)
    }

    /// Stage one row's KV window: `locs` are the request's pool locations in
    /// token order, written head-aligned and padded to the captured window length
    /// with location 0. The whole row is rewritten every step, so a shorter
    /// window than the last step cannot leave a stale location in the tail.
    ///
    /// The padding value does not matter to the arithmetic -- the device mask
    /// drops every position past the step's own -- but it must be a *valid* pool
    /// index, because the gather reads it as a pool location. Location 0 always
    /// exists.
    pub fn stage_window(&self, row: usize, locs: &[u32]) -> Result<(), DecodeBufferError> {
        assert!(row < self.max_bs, "stage_window past the captured batch");
        if locs.len() > self.window {
            return Err(DecodeBufferError::WindowOverCapacity {
                locs: locs.len(),
                window: self.window,
            });
        }
        let mut host: Vec<i32> = locs
            .iter()
            .map(|v| {
                debug_assert!(
                    *v != u32::MAX,
                    "a token location is never the sentinel page"
                );
                if *v == u32::MAX { 0 } else { *v as i32 }
            })
            .collect();
        host.resize(self.window, 0);
        let dst = self.kv_locs.select(0, row).expect("kv locs row");
        let src = dst
            .runtime()
            .from_slice_i32(&host, &[self.window])
            .expect("stage kv window");
        src.copy_to(&dst).expect("write kv window");
        Ok(())
    }

    /// Copy `values` into the head of an i32 buffer, as a device H2D. The values
    /// are already the row's `u32` narrowed to the kernel's `i32`; the sentinel
    /// `u32::MAX` is the only one that does not round-trip, by design.
    fn copy_i32(&self, dst: &Tensor, values: &[u32]) {
        if values.is_empty() {
            return;
        }
        let host: Vec<i32> = values
            .iter()
            .map(|v| {
                if *v == u32::MAX {
                    SENTINEL_LOC
                } else {
                    *v as i32
                }
            })
            .collect();
        let dst = dst.narrow(0, 0, values.len()).expect("decode buffer head");
        let src = dst
            .runtime()
            .from_slice_i32(&host, &[values.len()])
            .expect("stage decode metadata");
        src.copy_to(&dst).expect("write decode metadata");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::Cuda(0)
    }

    fn i32_vec(t: &Tensor) -> Vec<i32> {
        t.to_host_i32().expect("read i32 buffer")
    }

    fn row(seed: u32) -> DecodeRow {
        DecodeRow {
            input_id: 100 + seed,
            position: 7 + seed,
            state_slot: seed,
            out_loc: 4096 + seed,
        }
    }

    #[test]
    fn buffers_start_zeroed_with_the_asked_shape() {
        let b = DecodeBuffers::new(4, 8, 16, dev());
        assert_eq!(b.max_bs(), 4);
        assert_eq!(b.vocab(), 8);
        assert_eq!(b.input_ids().shape(), vec![4]);
        assert_eq!(b.logits().shape(), vec![4, 8]);
        assert_eq!(i32_vec(b.out_locs()), vec![0; 4]);
    }

    #[test]
    fn staging_a_step_writes_every_row() {
        let b = DecodeBuffers::new(4, 8, 16, dev());
        b.stage(&[row(0), row(1)]).unwrap();
        assert_eq!(i32_vec(b.input_ids())[..2], [100, 101]);
        assert_eq!(i32_vec(b.positions())[..2], [7, 8]);
        assert_eq!(i32_vec(b.state_slots())[..2], [0, 1]);
        assert_eq!(i32_vec(b.out_locs())[..2], [4096, 4097]);
    }

    #[test]
    fn staging_keeps_the_buffer_addresses_fixed() {
        let b = DecodeBuffers::new(2, 8, 4, dev());
        let ptrs = [
            b.input_ids().device_ptr(),
            b.positions().device_ptr(),
            b.state_slots().device_ptr(),
            b.out_locs().device_ptr(),
            b.kv_locs().device_ptr(),
            b.logits().device_ptr(),
        ];
        b.stage(&[row(0), row(9)]).unwrap();
        b.stage_window(0, &[7, 8]).unwrap();
        let after = [
            b.input_ids().device_ptr(),
            b.positions().device_ptr(),
            b.state_slots().device_ptr(),
            b.out_locs().device_ptr(),
            b.kv_locs().device_ptr(),
            b.logits().device_ptr(),
        ];
        assert_eq!(ptrs, after, "a capture bakes these addresses");
    }

    #[test]
    fn staging_more_rows_than_capacity_is_refused() {
        let b = DecodeBuffers::new(2, 8, 16, dev());
        assert_eq!(
            b.stage(&[row(0), row(1), row(2)]),
            Err(DecodeBufferError::TooManyRows { rows: 3, max: 2 })
        );
    }

    #[test]
    fn padding_fills_the_tail_with_sentinel_rows() {
        let b = DecodeBuffers::new(4, 8, 16, dev());
        b.stage_padded(&[row(0), row(1)], 4).unwrap();
        assert_eq!(i32_vec(b.input_ids())[..2], [100, 101]);
        assert_eq!(i32_vec(b.out_locs())[..2], [4096, 4097]);
        assert_eq!(
            i32_vec(b.out_locs())[2..],
            [SENTINEL_LOC, SENTINEL_LOC],
            "padded rows read as the sentinel, not a live location"
        );
        assert_eq!(i32_vec(b.input_ids())[2..], [0, 0]);
    }

    #[test]
    fn padding_smaller_than_the_staged_rows_is_refused() {
        let b = DecodeBuffers::new(4, 8, 16, dev());
        assert_eq!(
            b.stage_padded(&[row(0), row(1)], 1),
            Err(DecodeBufferError::PaddedUnderRows { rows: 2, padded: 1 })
        );
        assert_eq!(
            b.stage_padded(&[row(0)], 5),
            Err(DecodeBufferError::PaddedOverCapacity { padded: 5, max: 4 })
        );
    }

    #[test]
    fn a_second_step_overwrites_the_first_in_place() {
        let b = DecodeBuffers::new(2, 8, 16, dev());
        b.stage(&[row(0), row(1)]).unwrap();
        let first = i32_vec(b.input_ids());
        b.stage(&[row(9)]).unwrap();
        assert_eq!(
            i32_vec(b.input_ids())[0],
            109,
            "the head row is the new step"
        );
        assert_ne!(i32_vec(b.input_ids()), first);
        assert_eq!(
            i32_vec(b.input_ids())[1],
            first[1],
            "an unstaged tail row keeps the previous step's value"
        );
    }

    #[test]
    fn the_kv_window_starts_zeroed() {
        let b = DecodeBuffers::new(1, 8, 4, dev());
        assert_eq!(b.window(), 4);
        assert_eq!(b.kv_locs().shape(), vec![1, 4]);
        assert_eq!(
            i32_vec(&b.kv_locs_row(0)),
            vec![0; 4],
            "an unstaged window reads as valid location 0, which the mask drops"
        );
    }

    #[test]
    fn stage_window_aligns_the_locations_and_pads_the_tail() {
        let b = DecodeBuffers::new(2, 8, 4, dev());
        b.stage_window(0, &[7, 8]).unwrap();
        assert_eq!(
            i32_vec(&b.kv_locs_row(0)),
            vec![7, 8, 0, 0],
            "the window is head-aligned and padded to the captured length"
        );
        // A later, shorter step must not leave the previous tail live.
        b.stage_window(0, &[9]).unwrap();
        assert_eq!(i32_vec(&b.kv_locs_row(0)), vec![9, 0, 0, 0]);
        // The other row is untouched.
        assert_eq!(i32_vec(&b.kv_locs_row(1)), vec![0; 4]);
    }

    #[test]
    fn a_window_longer_than_the_buffer_is_refused() {
        let b = DecodeBuffers::new(1, 8, 2, dev());
        assert_eq!(
            b.stage_window(0, &[1, 2, 3]),
            Err(DecodeBufferError::WindowOverCapacity { locs: 3, window: 2 })
        );
    }
}
