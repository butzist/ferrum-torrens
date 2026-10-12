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

use tch::{Kind, Tensor};

use crate::infrastructure::torch::Device;

/// The out-of-range pool location a padded (dummy) request carries. Real
/// locations are never negative, so the sentinel can never alias one; every
/// location that reads back as this was padding rather than a live request.
pub const SENTINEL_LOC: i32 = -1;

/// One request's row in a decode step: the token it is advancing, where in the
/// sequence it sits, the GDN state slot it owns, and the KV pool location it
/// writes this step's K/V at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRow {
    pub input_id: u32,
    pub position: u32,
    pub state_slot: u32,
    pub out_loc: u32,
}

impl DecodeRow {
    /// The row a padded request carries: no real token, no real position, and
    /// sentinel locations that read as padding. Padding to the captured batch
    /// size must leave the graph reading well-defined buffers, so a dummy row is
    /// a real row with a marker value rather than an unwritten one.
    pub const fn sentinel() -> Self {
        Self {
            input_id: 0,
            position: 0,
            state_slot: 0,
            out_loc: u32::MAX,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DecodeBufferError {
    #[error("decode step has {rows} rows, over the {max} the buffers hold")]
    TooManyRows { rows: usize, max: usize },
    #[error("cannot pad {rows} rows to {padded}, under the staged count")]
    PaddedUnderRows { rows: usize, padded: usize },
    #[error("padding to {padded} is over the {max} the buffers hold")]
    PaddedOverCapacity { padded: usize, max: usize },
}

/// The device tensors one captured decode step reads and writes, allocated once
/// at `[max_bs]` (plus `[max_bs, vocab]` for the logits) and staged in place.
pub struct DecodeBuffers {
    input_ids: Tensor,
    positions: Tensor,
    state_slots: Tensor,
    out_locs: Tensor,
    logits: Tensor,
    max_bs: usize,
    vocab: usize,
    device: Device,
}

impl DecodeBuffers {
    /// Allocate the whole set at the captured batch size. The logits are the one
    /// large buffer: `max_bs * vocab * 4` bytes, which is what a decode graph
    /// writes and the sampler reads back.
    pub fn new(max_bs: usize, vocab: usize, device: Device) -> Self {
        assert!(max_bs > 0 && vocab > 0);
        let i32_dev = (Kind::Int, device.to_tch());
        let f32_dev = (Kind::Float, device.to_tch());
        Self {
            input_ids: Tensor::zeros([max_bs as i64], i32_dev),
            positions: Tensor::zeros([max_bs as i64], i32_dev),
            state_slots: Tensor::zeros([max_bs as i64], i32_dev),
            out_locs: Tensor::zeros([max_bs as i64], i32_dev),
            logits: Tensor::zeros([max_bs as i64, vocab as i64], f32_dev),
            max_bs,
            vocab,
            device,
        }
    }

    pub fn max_bs(&self) -> usize {
        self.max_bs
    }

    pub fn vocab(&self) -> usize {
        self.vocab
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

    /// The logits `[max_bs, vocab]` f32 the step writes into, and the sampler
    /// reads back. A capture writes the first `rows` of them.
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }

    /// The first `n` logit rows, a view for a caller that has fewer live rows
    /// than the captured batch.
    pub fn logits_rows(&self, n: usize) -> Tensor {
        assert!(n <= self.max_bs, "logits_rows past the captured batch");
        self.logits.narrow(0, 0, n as i64)
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
        let src = Tensor::from_slice(&host).to_device(dst.device());
        dst.narrow(0, 0, values.len() as i64).copy_(&src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::torch::auto_device;

    fn i32_vec(t: &Tensor) -> Vec<i32> {
        let flat = t.contiguous().flatten(0, -1);
        let mut out = vec![0i32; flat.numel()];
        flat.copy_data(&mut out, flat.numel());
        out
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
        let b = DecodeBuffers::new(4, 8, auto_device());
        assert_eq!(b.max_bs(), 4);
        assert_eq!(b.vocab(), 8);
        assert_eq!(b.input_ids().size(), vec![4]);
        assert_eq!(b.logits().size(), vec![4, 8]);
        assert_eq!(i32_vec(b.out_locs()), vec![0; 4]);
    }

    #[test]
    fn staging_a_step_writes_every_row() {
        let b = DecodeBuffers::new(4, 8, auto_device());
        b.stage(&[row(0), row(1)]).unwrap();
        assert_eq!(i32_vec(b.input_ids())[..2], [100, 101]);
        assert_eq!(i32_vec(b.positions())[..2], [7, 8]);
        assert_eq!(i32_vec(b.state_slots())[..2], [0, 1]);
        assert_eq!(i32_vec(b.out_locs())[..2], [4096, 4097]);
    }

    #[test]
    fn staging_more_rows_than_capacity_is_refused() {
        let b = DecodeBuffers::new(2, 8, auto_device());
        assert_eq!(
            b.stage(&[row(0), row(1), row(2)]),
            Err(DecodeBufferError::TooManyRows { rows: 3, max: 2 })
        );
    }

    #[test]
    fn padding_fills_the_tail_with_sentinel_rows() {
        let b = DecodeBuffers::new(4, 8, auto_device());
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
        let b = DecodeBuffers::new(4, 8, auto_device());
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
        let b = DecodeBuffers::new(2, 8, auto_device());
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
}
