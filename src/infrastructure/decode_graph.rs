// Per-batch-size CUDA-graph executables for the decode step (TODO.md P1 item 3).
//
// A captured decode graph bakes the shapes and the addresses of every op it
// recorded, so one executable serves exactly one batch size. This owns that set:
// a `DecodeGraph` per captured size -- its own private graph pool and its own
// fixed-address `DecodeBuffers` -- and a `DecodeGraphSet` that picks the smallest
// captured size that fits a step and pads the tail with dummy rows, which is the
// "dummy-req padding to the captured size" of decision 40's capture plan.
//
// The capture itself is the same shape as `tests/decode_graph_capture.rs`: warm
// the kernels, synchronize, open the pool, record one forward, instantiate. The
// pool must outlive every replay, so it lives inside the `DecodeGraph`.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::domain::decode::{CapturedDecode, DecodeError};
use crate::domain::kvcache::SequenceCache;
use crate::infrastructure::cuda_driver::{
    CAPTURE_MODE_GLOBAL, CUgraphExec, CUstream, CudaError, Driver,
};
use crate::infrastructure::cuda_graph::GraphPool;
use crate::infrastructure::decode_buffers::{DecodeBufferError, DecodeBuffers, DecodeRow};
use crate::infrastructure::torch::{Device, to_f32_vec};

/// A `step` failed while a capture was being recorded.
#[derive(Debug)]
pub enum CaptureError<E> {
    Step(E),
    Driver(CudaError),
}

impl<E: std::fmt::Display> std::fmt::Display for CaptureError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::Step(e) => write!(f, "the captured decode step failed: {e}"),
            CaptureError::Driver(e) => write!(f, "the capture driver call failed: {e}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CaptureError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CaptureError::Step(e) => Some(e),
            CaptureError::Driver(e) => Some(e),
        }
    }
}

/// A batch-decode step captured at one fixed batch size.
///
/// The private pool and the `DecodeBuffers` are the graph's baked addresses, so
/// both must outlive every replay; `Drop` destroys the instance before the pool
/// closes. `buffers()` is where a caller stages the next step between replays.
pub struct DecodeGraph {
    lib: Arc<Driver>,
    stream: CUstream,
    bufs: DecodeBuffers,
    /// Kept alive so the recorded region's allocations outlive the graph; it
    /// closes when this drops, after the instance is destroyed.
    _pool: GraphPool,
    exec: CUgraphExec,
}

impl DecodeGraph {
    /// Record `step` over `bufs` into an instantiated graph at `bufs`' batch
    /// size.
    ///
    /// `step` runs once to warm the kernels, the stream synchronizes, and then
    /// `step` is recorded with the pool open. `step` must run the model's decode
    /// forward over the buffers and nothing else: a capture cannot record a
    /// pageable host copy, an allocation outside the pool, or a synchronization.
    ///
    /// # Safety
    /// `stream` must be a non-legacy stream on the primary context of the
    /// buffers' device, torch's current stream must already be `stream`
    /// (`cuda_graph::set_current_stream`), and `step` must be replayable
    /// identically from the same buffers.
    pub unsafe fn capture<F, E>(
        lib: Arc<Driver>,
        stream: CUstream,
        bufs: DecodeBuffers,
        mut step: F,
    ) -> Result<Self, CaptureError<E>>
    where
        F: FnMut(&DecodeBuffers) -> Result<(), E>,
    {
        let index = match bufs.device() {
            Device::Cuda(i) => i,
            other => panic!("a decode graph needs a CUDA device, got {other:?}"),
        };
        // Warm every kernel outside the capture, so the recording only sees the
        // launches and their allocations, not a lazy JIT or a one-time setup.
        step(&bufs).map_err(CaptureError::Step)?;
        // SAFETY: the caller owns the stream and its capture preconditions.
        unsafe { lib.synchronize(stream) }.map_err(CaptureError::Driver)?;

        // SAFETY: the pool must outlive every allocation the recorded region
        // makes, which it does: it moves into the returned `DecodeGraph`.
        let pool = unsafe { GraphPool::begin(stream, index) };
        // SAFETY: the stream is non-legacy and idle, per the caller.
        unsafe { lib.begin_capture(stream, CAPTURE_MODE_GLOBAL) }.map_err(CaptureError::Driver)?;
        if let Err(e) = step(&bufs) {
            // Never leave the stream capturing; a graph still comes back, so
            // drop it before the pool closes.
            if let Ok(g) = unsafe { lib.end_capture(stream) } {
                unsafe { lib.graph_destroy(g) }.ok();
            }
            return Err(CaptureError::Step(e));
        }
        let graph = unsafe { lib.end_capture(stream) }.map_err(CaptureError::Driver)?;
        let exec = unsafe { lib.graph_instantiate(graph) }.map_err(CaptureError::Driver)?;
        unsafe { lib.graph_destroy(graph) }.map_err(CaptureError::Driver)?;
        Ok(Self {
            lib,
            stream,
            bufs,
            _pool: pool,
            exec,
        })
    }

    /// The batch size this executable was captured at; a replay must stage that
    /// many rows.
    pub fn batch(&self) -> usize {
        self.bufs.max_bs()
    }

    /// The fixed-address buffers the graph reads and writes; stage the next
    /// step's rows and windows into these between replays.
    pub fn buffers(&self) -> &DecodeBuffers {
        &self.bufs
    }

    /// The side stream the graph records and replays on.
    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// Launch the recorded step on the capture stream. The buffers must already
    /// be staged; the caller synchronizes the stream before reading them back.
    ///
    /// # Safety
    /// The device state the step reads must be the state its staged metadata
    /// names, and every baked address (`self.bufs`) must still be live.
    pub unsafe fn replay(&self) -> Result<(), CudaError> {
        unsafe { self.lib.graph_launch(self.exec, self.stream) }
    }
}

impl Drop for DecodeGraph {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            // The instance owns its memory; the pool that held the recorded
            // region's allocations drops after this.
            unsafe { self.lib.graph_exec_destroy(self.exec) }.ok();
        }
    }
}

// The recorded handles are raw CUDA objects, but the driver and every buffer the
// graph reads are `Send + Sync` (same call as `CudaMoE`): the engine holds the
// service behind an `Arc` and runs one step at a time.
unsafe impl Send for DecodeGraph {}
unsafe impl Sync for DecodeGraph {}

/// What a [`DecodeGraphSet`] needs from one batch-size slot: its size, the
/// buffers a step is staged into, and a launch. `DecodeGraph` is the real one;
/// the unit tests stand in a fake so the selection and padding are exercised
/// without a device.
pub trait DecodeExec {
    fn batch(&self) -> usize;
    fn buffers(&self) -> &DecodeBuffers;
    fn launch(&self) -> Result<(), CudaError>;
}

impl DecodeExec for DecodeGraph {
    fn batch(&self) -> usize {
        DecodeGraph::batch(self)
    }

    fn buffers(&self) -> &DecodeBuffers {
        DecodeGraph::buffers(self)
    }

    fn launch(&self) -> Result<(), CudaError> {
        unsafe { self.replay() }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeGraphError {
    #[error("a {rows}-row decode step has no captured graph at or above it")]
    NoBatchFits { rows: usize },
    #[error("a {rows}-row step was given {windows} windows")]
    RowWindowMismatch { rows: usize, windows: usize },
    #[error(transparent)]
    Layout(#[from] DecodeBufferError),
    #[error(transparent)]
    Driver(#[from] CudaError),
}

/// The captured executables an engine holds, one per batch size.
///
/// A step of `n` rows replays on the smallest captured size `>= n`, padded with
/// `padding` dummy rows so every baked row is defined. The padding row should
/// name a reserved state slot and a reserved pool location: the dummy rows run
/// the whole forward, so their K/V and GDN writes are real side effects.
pub struct DecodeGraphSet<E: DecodeExec> {
    entries: BTreeMap<usize, E>,
    padding: DecodeRow,
}

impl<E: DecodeExec> DecodeGraphSet<E> {
    pub fn new(padding: DecodeRow) -> Self {
        Self {
            entries: BTreeMap::new(),
            padding,
        }
    }

    /// Add a captured executable, keyed by its batch size. Replaces one already
    /// captured at the same size.
    pub fn insert(&mut self, exec: E) {
        self.entries.insert(exec.batch(), exec);
    }

    /// The captured batch sizes, ascending.
    pub fn batches(&self) -> impl Iterator<Item = usize> + '_ {
        self.entries.keys().copied()
    }

    /// The executable captured at exactly `bs`, if any.
    pub fn get(&self, bs: usize) -> Option<&E> {
        self.entries.get(&bs)
    }

    /// The smallest captured size that fits `rows`, or `None` if every captured
    /// size is smaller.
    pub fn plan(&self, rows: usize) -> Option<usize> {
        self.entries.range(rows..).next().map(|(bs, _)| *bs)
    }

    /// Stage `rows` (each with its own attention window) into the chosen
    /// executable's buffers, pad the tail to its batch size, and launch it.
    /// Returns the batch size that ran.
    pub fn stage_and_launch(
        &self,
        rows: &[DecodeRow],
        windows: &[&[u32]],
    ) -> Result<usize, DecodeGraphError> {
        if rows.len() != windows.len() {
            return Err(DecodeGraphError::RowWindowMismatch {
                rows: rows.len(),
                windows: windows.len(),
            });
        }
        let bs = self
            .plan(rows.len())
            .ok_or(DecodeGraphError::NoBatchFits { rows: rows.len() })?;
        let exec = &self.entries[&bs];
        let bufs = exec.buffers();

        let mut staged = rows.to_vec();
        staged.resize(bs, self.padding);
        bufs.stage(&staged)?;
        for (i, w) in windows.iter().enumerate() {
            bufs.stage_window(i, w)?;
        }
        // A padded row keeps no window from an earlier, longer step: its mask
        // drops every key but the padding must still be a defined buffer.
        for i in rows.len()..bs {
            bufs.stage_window(i, &[])?;
        }
        exec.launch()?;
        Ok(bs)
    }
}

/// What a replay must do before the logits are valid: the driver synchronize on
/// the capture stream. The unit tests use `None` so the staging and readback run
/// without a device; the real capture carries the driver and stream.
enum ReplaySync {
    None,
    Driver(Arc<Driver>, usize, Device),
}

impl ReplaySync {
    /// Point torch at the capture stream before staging, so the H2D copies of
    /// the row/window tensors land on the same stream the graph launches on.
    fn bind(&self) {
        if let ReplaySync::Driver(_, stream, Device::Cuda(ordinal)) = self {
            // SAFETY: `stream` is the side stream the graph was captured on,
            // live for the adapter's life.
            unsafe {
                crate::infrastructure::cuda_graph::set_current_stream(
                    *stream as CUstream,
                    *ordinal,
                );
            }
        }
    }

    fn run(&self) -> Result<(), CudaError> {
        match self {
            ReplaySync::None => Ok(()),
            // SAFETY: the stream is the one the graph was captured on, still
            // owned by the driver the adapter holds.
            ReplaySync::Driver(lib, stream, _) => unsafe { lib.synchronize(*stream as CUstream) },
        }
    }
}

/// The engine-facing captured decode service: a [`DecodeGraphSet`] plus the
/// stream synchronize a replay needs, behind the domain [`CapturedDecode`] port.
///
/// The composition root builds one and hands it to the engine, which replays the
/// smallest captured size that fits its step. `_padding` keeps the dummy row's
/// state slot and pool page reserved for the service's whole life: the padding
/// row runs the real forward, so its writes must land outside every live
/// request.
pub struct EngineDecodeGraphs<E: DecodeExec> {
    set: DecodeGraphSet<E>,
    window: usize,
    sync: ReplaySync,
    _padding: Option<SequenceCache>,
}

impl EngineDecodeGraphs<DecodeGraph> {
    /// Record one graph per size in `sizes` and wrap them in the engine's
    /// service. `padding` must name a reserved state slot and pool location;
    /// `padding_cache` is the reservation itself, kept alive by the service.
    ///
    /// `step` runs the model's decode forward over the buffers (see
    /// [`DecodeGraph::capture`] for the capture preconditions). Its error is
    /// stringified: a capture failure surfaces at load, not at every step.
    ///
    /// # Safety
    /// `stream` must be a non-legacy side stream on `device`, and `step` must be
    /// the strictly device-only decode forward: it cannot copy from pageable
    /// host memory, allocate outside the graph pool, or synchronize.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn capture(
        lib: Arc<Driver>,
        stream: CUstream,
        device: Device,
        vocab: usize,
        window: usize,
        sizes: &[usize],
        padding: DecodeRow,
        padding_cache: SequenceCache,
        mut step: impl FnMut(&DecodeBuffers) -> Result<(), String>,
    ) -> Result<Self, CaptureError<String>> {
        let mut set = DecodeGraphSet::new(padding);
        for &bs in sizes {
            let bufs = DecodeBuffers::new(bs, vocab, window, device);
            // SAFETY: the caller guarantees the stream and the step.
            let graph = unsafe { DecodeGraph::capture(lib.clone(), stream, bufs, |b| step(b)) }?;
            set.insert(graph);
        }
        Ok(Self {
            set,
            window,
            sync: ReplaySync::Driver(lib, stream as usize, device),
            _padding: Some(padding_cache),
        })
    }
}

impl<E: DecodeExec> EngineDecodeGraphs<E> {
    /// Wrap an already-built set; the unit tests stand in a fake so the staging,
    /// the size selection and the logit readback run without a device.
    pub fn from_set(set: DecodeGraphSet<E>, window: usize) -> Self {
        Self {
            set,
            window,
            sync: ReplaySync::None,
            _padding: None,
        }
    }

    fn run_step(
        &self,
        rows: &[DecodeRow],
        windows: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>, DecodeError> {
        if rows.len() != windows.len() {
            return Err(DecodeError::RowWindowMismatch {
                rows: rows.len(),
                windows: windows.len(),
            });
        }
        self.sync.bind();
        let bs = self
            .set
            .stage_and_launch(rows, windows)
            .map_err(graph_error)?;
        self.sync
            .run()
            .map_err(|e| DecodeError::Step(e.to_string()))?;
        let bufs = self
            .set
            .get(bs)
            .expect("stage_and_launch picked a size")
            .buffers();
        let mut out = Vec::with_capacity(rows.len());
        for i in 0..rows.len() {
            out.push(to_f32_vec(&bufs.logits().select(0, i as i64)));
        }
        Ok(out)
    }
}

/// The graph set's own error, mapped onto the port so the engine never names an
/// infra type. A layout error is a window that over-ran the captured length.
fn graph_error(e: DecodeGraphError) -> DecodeError {
    match e {
        DecodeGraphError::NoBatchFits { rows } => DecodeError::NoBatchFits { rows },
        DecodeGraphError::RowWindowMismatch { rows, windows } => {
            DecodeError::RowWindowMismatch { rows, windows }
        }
        DecodeGraphError::Layout(e) => DecodeError::Window { why: e.to_string() },
        DecodeGraphError::Driver(e) => DecodeError::Step(e.to_string()),
    }
}

impl CapturedDecode for EngineDecodeGraphs<DecodeGraph> {
    fn max_batch(&self) -> usize {
        self.set.batches().max().unwrap_or(0)
    }

    fn window(&self) -> usize {
        self.window
    }

    fn step(&self, rows: &[DecodeRow], windows: &[&[u32]]) -> Result<Vec<Vec<f32>>, DecodeError> {
        self.run_step(rows, windows)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::infrastructure::torch::auto_device;
    use tch::Kind;

    /// A device-free stand-in: real CPU `DecodeBuffers`, a launch that just
    /// counts.
    struct Fake {
        bufs: DecodeBuffers,
        launches: Cell<usize>,
    }

    impl DecodeExec for Fake {
        fn batch(&self) -> usize {
            self.bufs.max_bs()
        }

        fn buffers(&self) -> &DecodeBuffers {
            &self.bufs
        }

        fn launch(&self) -> Result<(), CudaError> {
            self.launches.set(self.launches.get() + 1);
            Ok(())
        }
    }

    fn i32_vec(t: &tch::Tensor) -> Vec<i32> {
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

    fn fake(bs: usize, window: usize) -> DecodeGraphSet<Fake> {
        let mut set = DecodeGraphSet::new(DecodeRow::sentinel());
        set.insert(Fake {
            bufs: DecodeBuffers::new(bs, 8, window, auto_device()),
            launches: Cell::new(0),
        });
        set
    }

    #[test]
    fn the_set_picks_the_smallest_captured_size_that_fits() {
        let mut set = DecodeGraphSet::new(DecodeRow::sentinel());
        for bs in [4, 1, 2] {
            set.insert(Fake {
                bufs: DecodeBuffers::new(bs, 8, 4, auto_device()),
                launches: Cell::new(0),
            });
        }
        assert_eq!(set.batches().collect::<Vec<_>>(), vec![1, 2, 4]);
        assert_eq!(set.plan(0), Some(1));
        assert_eq!(set.plan(1), Some(1));
        assert_eq!(set.plan(2), Some(2));
        assert_eq!(set.plan(3), Some(4), "three rows ride the batch-4 graph");
        assert_eq!(set.plan(4), Some(4));
        assert_eq!(set.plan(5), None, "past every captured size");
    }

    #[test]
    fn a_short_step_is_padded_to_the_captured_size() {
        let pad = DecodeRow {
            input_id: 555,
            position: 9,
            state_slot: 3,
            out_loc: 77,
        };
        let mut set = DecodeGraphSet::new(pad);
        set.insert(Fake {
            bufs: DecodeBuffers::new(4, 8, 3, auto_device()),
            launches: Cell::new(0),
        });

        let ran = set.stage_and_launch(&[row(0)], &[&[7, 8]]).unwrap();
        assert_eq!(ran, 4, "one row runs on the batch-4 graph");
        let bufs = set.get(4).unwrap().buffers();
        assert_eq!(
            i32_vec(bufs.input_ids()),
            vec![100, 555, 555, 555],
            "live row then dummy rows"
        );
        assert_eq!(i32_vec(bufs.out_locs()), vec![4096, 77, 77, 77]);
        assert_eq!(
            i32_vec(&bufs.kv_locs_row(0)),
            vec![7, 8, 0],
            "the live window is head-aligned and padded"
        );
        for i in 1..4 {
            assert_eq!(
                i32_vec(&bufs.kv_locs_row(i)),
                vec![0; 3],
                "a dummy row carries no window"
            );
        }
        assert_eq!(set.get(4).unwrap().launches.get(), 1);
    }

    #[test]
    fn a_step_past_every_captured_size_is_refused() {
        let set = fake(2, 4);
        assert!(matches!(
            set.stage_and_launch(&[row(0), row(1), row(2)], &[&[], &[], &[]]),
            Err(DecodeGraphError::NoBatchFits { rows: 3 })
        ));
    }

    #[test]
    fn a_window_count_that_does_not_match_the_rows_is_refused() {
        let set = fake(4, 4);
        assert!(matches!(
            set.stage_and_launch(&[row(0), row(1)], &[&[]]),
            Err(DecodeGraphError::RowWindowMismatch {
                rows: 2,
                windows: 1
            })
        ));
    }

    /// A device-free stand-in whose launch writes each row's token id into its
    /// logits, so the port's readback can be checked without a device.
    struct WriteFake {
        bufs: DecodeBuffers,
        launches: AtomicUsize,
    }

    impl DecodeExec for WriteFake {
        fn batch(&self) -> usize {
            self.bufs.max_bs()
        }

        fn buffers(&self) -> &DecodeBuffers {
            &self.bufs
        }

        fn launch(&self) -> Result<(), CudaError> {
            let rows = self.bufs.max_bs() as i64;
            let vocab = self.bufs.logits().size()[1];
            let ids = self.bufs.input_ids().to_kind(Kind::Float).unsqueeze(1);
            let spread = ids.expand([rows, vocab], false);
            let mut logits = self.bufs.logits().shallow_clone();
            logits.copy_(&spread);
            self.launches.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn write_graphs(bs: usize, window: usize) -> EngineDecodeGraphs<WriteFake> {
        let mut set = DecodeGraphSet::new(DecodeRow::sentinel());
        set.insert(WriteFake {
            bufs: DecodeBuffers::new(bs, 8, window, auto_device()),
            launches: std::sync::atomic::AtomicUsize::new(0),
        });
        EngineDecodeGraphs::from_set(set, window)
    }

    #[test]
    fn the_port_reads_back_one_logit_vector_per_live_row() {
        let graphs = write_graphs(4, 4);
        let out = graphs.run_step(&[row(0), row(1)], &[&[], &[]]).unwrap();
        assert_eq!(out.len(), 2, "one vector per live row, not per padded row");
        assert_eq!(out[0][0], 100.0, "row 0 sees its own staged token");
        assert_eq!(out[1][0], 101.0);
        assert_eq!(graphs.window, 4);
    }

    #[test]
    fn the_port_refuses_a_step_past_every_captured_size() {
        let graphs = write_graphs(2, 4);
        assert_eq!(
            graphs.run_step(&[row(0), row(1), row(2)], &[&[], &[], &[]]),
            Err(DecodeError::NoBatchFits { rows: 3 })
        );
    }

    #[test]
    fn the_port_refuses_a_window_count_that_does_not_match_the_rows() {
        let graphs = write_graphs(4, 4);
        assert_eq!(
            graphs.run_step(&[row(0), row(1)], &[&[]]),
            Err(DecodeError::RowWindowMismatch {
                rows: 2,
                windows: 1
            })
        );
    }

    #[test]
    fn the_port_refuses_a_window_over_the_captured_length() {
        let graphs = write_graphs(4, 2);
        assert_eq!(
            graphs.run_step(&[row(0)], &[&[1, 2, 3]]),
            Err(DecodeError::Window {
                why: "kv window of 3 locations is over the 2 the buffer holds".into()
            })
        );
    }
}
