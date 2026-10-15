// The captured decode port: one batched, replayable decode step over fixed
// address buffers. The engine owns the service (it sizes the buffers from
// `max_running_requests` and the context window); the compute adapter records
// the CUDA graphs and replays them. Nothing here touches device tensors.
//
// `DecodeRow` is the per-request staging of one decode step, the same shape
// FreeToken's `GraphCaptureBuffer` keeps on device (DONE.md decision 49): the
// token the request advances, its absolute rope position, the GDN state slot it
// owns and the KV pool location it writes this step's K/V at. One row per live
// request; a captured batch is padded to its fixed size with a reserved dummy
// row.

/// One request's row in a decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRow {
    pub input_id: u32,
    pub position: u32,
    pub state_slot: u32,
    pub out_loc: u32,
}

impl DecodeRow {
    /// The row a padded request carries: no real token, no real position, and an
    /// out-of-range location that reads as padding. Padding to the captured batch
    /// size must leave the graph reading well-defined buffers, so a dummy row is
    /// a real row with a marker value rather than an unwritten one. The real
    /// engine pads with a reserved location and state slot instead, because a
    /// dummy row runs the whole forward and its writes are real side effects.
    pub const fn sentinel() -> Self {
        Self {
            input_id: 0,
            position: 0,
            state_slot: 0,
            out_loc: u32::MAX,
        }
    }
}

/// Why a captured decode step could not run. The adapter maps its own graph and
/// driver errors onto these, so the engine never sees an infra type.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DecodeError {
    #[error("a {rows}-row decode step has no captured graph at or above it")]
    NoBatchFits { rows: usize },
    #[error("a {rows}-row decode step was given {windows} windows")]
    RowWindowMismatch { rows: usize, windows: usize },
    #[error("the decode step over-ran the captured window: {why}")]
    Window { why: String },
    #[error("the captured decode step failed: {0}")]
    Step(String),
}

/// One batched, replayable decode step: stage the rows and their KV windows,
/// replay the graph captured at the smallest size that fits, and read back one
/// logit vector per live row.
///
/// The engine holds this as `Option<Arc<dyn CapturedDecode>>`: `None` on a
/// build or device without capture, and then the cached decode path stays the
/// per-request host one.
pub trait CapturedDecode: Send + Sync {
    /// The largest captured batch size, i.e. the most rows one step may carry.
    fn max_batch(&self) -> usize;

    /// The fixed number of pool locations each row's window is staged into.
    fn window(&self) -> usize;

    /// Run one step: `rows` live requests, each with a `windows[i]` of pool
    /// locations in token order. Returns one `[vocab]` logit vector per live row
    /// (not per padded row). Errors if `rows` is over every captured size or a
    /// window is longer than the captured `window`.
    fn step(&self, rows: &[DecodeRow], windows: &[&[u32]]) -> Result<Vec<Vec<f32>>, DecodeError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sentinel_row_is_a_defined_marker() {
        let s = DecodeRow::sentinel();
        assert_eq!(s.input_id, 0);
        assert_eq!(s.position, 0);
        assert_eq!(s.state_slot, 0);
        assert_eq!(s.out_loc, u32::MAX, "padding reads as an out-of-range loc");
    }

    #[test]
    fn the_errors_render_the_counts_they_carry() {
        assert_eq!(
            DecodeError::NoBatchFits { rows: 3 }.to_string(),
            "a 3-row decode step has no captured graph at or above it"
        );
        assert_eq!(
            DecodeError::RowWindowMismatch {
                rows: 2,
                windows: 1
            }
            .to_string(),
            "a 2-row decode step was given 1 windows"
        );
    }
}
