// Per-batch-size CUDA-graph executables for the decode step (TODO.md P1 item 3).
//
// A captured decode graph bakes the shapes and the addresses of every op it
// recorded, so one executable serves exactly one batch size. This owns that set:
// a `DecodeGraph` per captured size -- its own `CudaGraph` and its own
// fixed-address `DecodeBuffers` -- and a `DecodeGraphSet` that picks the
// smallest captured size that fits a step and pads the tail with a reserved
// dummy row, which is the "dummy-req padding to the captured size" of decision
// 40's capture plan.
//
// cudarc's graph owns its instantiated executable and frees it on drop; the
// recorded region's async allocations belong to the graph's capture pool, so
// there is no separate pool object to outlive the replay.

use std::collections::BTreeMap;
use std::sync::Mutex;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph, DriverError};

use crate::domain::decode::{CapturedDecode, DecodeError};
use crate::domain::kvcache::SequenceCache;
use crate::infrastructure::cudarc::Runtime;
use crate::infrastructure::decode_buffers::{DecodeBufferError, DecodeBuffers, DecodeRow};
use crate::infrastructure::torch::Device;

/// A `step` failed while a capture was being recorded.
#[derive(Debug)]
pub enum CaptureError<E> {
    Step(E),
    Driver(DriverError),
    Layout(DecodeBufferError),
    NoGraph,
}

impl<E> From<DecodeBufferError> for CaptureError<E> {
    fn from(e: DecodeBufferError) -> Self {
        CaptureError::Layout(e)
    }
}

impl<E: std::fmt::Display> std::fmt::Display for CaptureError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::Step(e) => write!(f, "the captured decode step failed: {e}"),
            CaptureError::Driver(e) => write!(f, "the capture driver call failed: {e}"),
            CaptureError::Layout(e) => write!(f, "the capture buffers could not be staged: {e}"),
            CaptureError::NoGraph => write!(f, "the stream capture ended without a graph"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CaptureError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CaptureError::Step(e) => Some(e),
            CaptureError::Driver(e) => Some(e),
            CaptureError::Layout(e) => Some(e),
            CaptureError::NoGraph => None,
        }
    }
}

/// A batch-decode step captured at one fixed batch size.
///
/// The `DecodeBuffers` are the graph's baked addresses, so they must outlive
/// every replay; `buffers()` is where a caller stages the next step between
/// replays.
pub struct DecodeGraph {
    graph: CudaGraph,
    bufs: DecodeBuffers,
    rt: Runtime,
}

impl DecodeGraph {
    /// Record `step` over `bufs` into an instantiated graph at `bufs`' batch
    /// size.
    ///
    /// `step` runs once to warm the kernels, the stream synchronizes, and then
    /// `step` is recorded. `step` must run the model's decode forward over the
    /// buffers and nothing else: a capture cannot record a pageable host copy,
    /// an allocation outside the stream's capture pool, or a synchronization.
    ///
    /// # Safety
    /// `rt`'s stream must be a non-legacy stream on the buffers' device, and
    /// `step` must be replayable identically from the same buffers.
    pub unsafe fn capture<F, E>(
        rt: &Runtime,
        bufs: DecodeBuffers,
        mut step: F,
    ) -> Result<Self, CaptureError<E>>
    where
        F: FnMut(&DecodeBuffers) -> Result<(), E>,
    {
        step(&bufs).map_err(CaptureError::Step)?;
        rt.stream().synchronize().map_err(CaptureError::Driver)?;

        rt.stream()
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(CaptureError::Driver)?;
        if let Err(e) = step(&bufs) {
            // Never leave the stream capturing; the discarded graph, if any,
            // drops at the end of this expression.
            let _ = rt.stream().end_capture(CUgraphInstantiate_flags(0));
            return Err(CaptureError::Step(e));
        }
        let graph = rt
            .stream()
            .end_capture(CUgraphInstantiate_flags(0))
            .map_err(CaptureError::Driver)?
            .ok_or(CaptureError::NoGraph)?;
        graph.upload().map_err(CaptureError::Driver)?;
        Ok(Self {
            graph,
            bufs,
            rt: rt.clone(),
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

    /// The runtime whose stream the graph records and replays on.
    pub fn runtime(&self) -> &Runtime {
        &self.rt
    }

    /// Launch the recorded step on the capture stream. The buffers must already
    /// be staged; the caller synchronizes the stream before reading them back.
    pub fn replay(&self) -> Result<(), DriverError> {
        self.graph.launch()
    }
}

// The raw graph handle is not `Send`, but every replay is serialized by the
// service's mutex and the runtime/buffers it reads are `Send`. `Sync` stays off
// so no caller can launch the same executable concurrently.
unsafe impl Send for DecodeGraph {}

/// What a [`DecodeGraphSet`] needs from one batch-size slot: its size, the
/// buffers a step is staged into, and a launch. `DecodeGraph` is the real one;
/// the unit tests stand in a fake so the selection and padding are exercised
/// without a device.
pub trait DecodeExec {
    fn batch(&self) -> usize;
    fn buffers(&self) -> &DecodeBuffers;
    fn launch(&self) -> Result<(), DriverError>;
}

impl DecodeExec for DecodeGraph {
    fn batch(&self) -> usize {
        DecodeGraph::batch(self)
    }

    fn buffers(&self) -> &DecodeBuffers {
        DecodeGraph::buffers(self)
    }

    fn launch(&self) -> Result<(), DriverError> {
        self.replay()
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
    Driver(#[from] DriverError),
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

        {
            let _stage = crate::infrastructure::host_profile::span("stage");
            bufs.stage_padded_with(rows, bs, self.padding)?;
            for (i, w) in windows.iter().enumerate() {
                bufs.stage_window(i, w)?;
            }
            // A padded row keeps no window from an earlier, longer step: its mask
            // drops every key but the padding must still be a defined buffer.
            for i in rows.len()..bs {
                bufs.stage_window(i, &[])?;
            }
        }
        {
            let _launch = crate::infrastructure::host_profile::span("launch");
            exec.launch()?;
        }
        Ok(bs)
    }
}

struct GraphService<E: DecodeExec> {
    set: DecodeGraphSet<E>,
    rt: Option<Runtime>,
    _padding: Option<SequenceCache>,
}

/// The engine-facing captured decode service: a [`DecodeGraphSet`] behind the
/// domain [`CapturedDecode`] port.
///
/// The composition root builds one and hands it to the engine, which replays
/// the smallest captured size that fits its step. `_padding` keeps the dummy
/// row's state slot and pool page reserved for the service's whole life: the
/// padding row runs the real forward, so its writes must land outside every live
/// request.
///
/// `CudaGraph` is not thread safe, so the mutex serializes the whole
/// stage/launch/read sequence; the engine only ever runs one step at a time.
pub struct EngineDecodeGraphs<E: DecodeExec> {
    inner: Mutex<GraphService<E>>,
    window: usize,
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
    /// `rt`'s stream must be a non-legacy stream on the model's device, and
    /// `step` must be the strictly device-only decode forward: it cannot copy
    /// from pageable host memory, allocate outside the capture pool, or
    /// synchronize.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn capture(
        rt: &Runtime,
        vocab: usize,
        window: usize,
        sizes: &[usize],
        padding: DecodeRow,
        padding_cache: SequenceCache,
        mut step: impl FnMut(&DecodeBuffers) -> Result<(), String>,
    ) -> Result<Self, CaptureError<String>> {
        let device = Device::Cuda(rt.context().ordinal());
        let mut set = DecodeGraphSet::new(padding);
        for &bs in sizes {
            let bufs = DecodeBuffers::with_runtime(rt, bs, vocab, window, device);
            // Warm the recorded shapes over the reserved padding row, never a
            // live request: the warmup and the recording both run the forward.
            bufs.stage_padded_with(&[], bs, padding)?;
            for row in 0..bs {
                bufs.stage_window(row, &[])?;
            }
            // SAFETY: the caller guarantees the stream and the step.
            let graph = unsafe { DecodeGraph::capture(rt, bufs, |b| step(b)) }?;
            set.insert(graph);
        }
        Ok(Self {
            inner: Mutex::new(GraphService {
                set,
                rt: Some(rt.clone()),
                _padding: Some(padding_cache),
            }),
            window,
        })
    }
}

impl<E: DecodeExec> EngineDecodeGraphs<E> {
    /// Wrap an already-built set; the unit tests stand in a fake so the staging,
    /// the size selection and the logit readback run without a device.
    pub fn from_set(set: DecodeGraphSet<E>, window: usize) -> Self {
        Self {
            inner: Mutex::new(GraphService {
                set,
                rt: None,
                _padding: None,
            }),
            window,
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
        let service = {
            let _lock = crate::infrastructure::host_profile::span("lock");
            self.inner
                .lock()
                .map_err(|e| DecodeError::Step(e.to_string()))?
        };
        let bs = service
            .set
            .stage_and_launch(rows, windows)
            .map_err(graph_error)?;
        {
            let _sync = crate::infrastructure::host_profile::span("sync");
            if let Some(rt) = &service.rt {
                rt.synchronize()
                    .map_err(|e| DecodeError::Step(e.to_string()))?;
            }
        }
        let _readback = crate::infrastructure::host_profile::span("readback");
        let bufs = service
            .set
            .get(bs)
            .expect("stage_and_launch picked a size")
            .buffers();
        let logits = bufs.logits_rows(rows.len());
        let mut out = Vec::with_capacity(rows.len());
        for i in 0..rows.len() {
            let row = logits
                .select(0, i)
                .map_err(|e| DecodeError::Step(e.to_string()))?;
            out.push(
                row.to_host_f32()
                    .map_err(|e| DecodeError::Step(e.to_string()))?,
            );
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
        self.inner
            .lock()
            .map(|s| s.set.batches().max().unwrap_or(0))
            .unwrap_or(0)
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

    /// A device-free stand-in: real `DecodeBuffers`, a launch that just counts.
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

        fn launch(&self) -> Result<(), DriverError> {
            self.launches.set(self.launches.get() + 1);
            Ok(())
        }
    }

    fn i32_vec(t: &crate::infrastructure::cudarc::Tensor) -> Vec<i32> {
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

        fn launch(&self) -> Result<(), DriverError> {
            let rows = self.bufs.max_bs();
            let vocab = self.bufs.vocab();
            let ids = self.bufs.input_ids().to_host_i32().expect("read ids");
            let mut logits = Vec::with_capacity(rows * vocab);
            for id in ids.iter().take(rows) {
                logits.extend(std::iter::repeat_n(*id as f32, vocab));
            }
            let src = self
                .bufs
                .logits()
                .runtime()
                .from_slice_f32(&logits, &[rows, vocab])
                .expect("write fake logits");
            src.copy_to(self.bufs.logits()).expect("write logits");
            self.launches.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn write_graphs(bs: usize, window: usize) -> EngineDecodeGraphs<WriteFake> {
        let mut set = DecodeGraphSet::new(DecodeRow::sentinel());
        set.insert(WriteFake {
            bufs: DecodeBuffers::new(bs, 8, window, auto_device()),
            launches: AtomicUsize::new(0),
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
