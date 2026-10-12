// The torch side of CUDA-graph capture: the two facilities `tch` 0.22 does not
// expose, behind the C++ bridge in `shim/c10_graph_shim.cpp`.
//
// Decision 39 established that capture itself goes through the driver API (no
// `tch` capture call exists). That is only half the story: the ops recorded
// into the graph are torch ops, so during a capture they must (a) launch on the
// side stream being captured and (b) allocate out of a private mempool, because
// a `cudaMalloc` between begin and end capture is illegal. Neither is reachable
// from `tch`, which is why the shim exists. Outside a capture this module is not
// used: the engine keeps running torch on the default stream.
//
// Safety: every entry point here mutates process-global torch/thread-local CUDA
// state, so all of them are `unsafe` and expect a valid, non-legacy stream on
// the same primary context torch uses.

use crate::infrastructure::cuda_driver::CUstream;

#[repr(C)]
struct FtGraphPool {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    fn ft_set_current_stream(stream: CUstream, device: i32);
    fn ft_pool_begin(stream: CUstream, device: i32) -> *mut FtGraphPool;
    fn ft_pool_end(pool: *mut FtGraphPool);
}

/// Make `stream` the stream torch launches its ops on for `device`.
///
/// # Safety
/// `stream` must be a live stream on the primary context of `device`, and not
/// the legacy default stream.
pub unsafe fn set_current_stream(stream: CUstream, device: usize) {
    unsafe { ft_set_current_stream(stream, device as i32) }
}

/// A private torch mempool that a capture's allocations route into, so the
/// recorded graph's temporaries live at stable addresses across replays.
///
/// The pool must be open for the whole of `begin_capture..end_capture` and
/// dropped only after `end_capture`, once no more of the graph's ops allocate.
#[must_use = "the pool closes when dropped; keep it alive across the capture"]
pub struct GraphPool {
    raw: *mut FtGraphPool,
    device: usize,
}

impl GraphPool {
    /// Open a pool and start routing `stream`'s allocations into it.
    ///
    /// # Safety
    /// `stream` must be the same non-legacy stream the capture will record, on
    /// the primary context of `device`; the returned pool must outlive every
    /// allocation the captured region makes.
    pub unsafe fn begin(stream: CUstream, device: usize) -> Self {
        let raw = unsafe { ft_pool_begin(stream, device as i32) };
        debug_assert!(!raw.is_null(), "graph_pool_handle never returns null");
        Self { raw, device }
    }

    pub fn device(&self) -> usize {
        self.device
    }

    /// Stop routing allocations into the pool. Called by `drop`; exposed so a
    /// caller can close the pool eagerly and keep the handle only for `Drop`'s
    /// bookkeeping. Idempotent through `raw` being nulled.
    fn close(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `raw` came from `ft_pool_begin` and is closed once.
            unsafe { ft_pool_end(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

impl Drop for GraphPool {
    fn drop(&mut self) {
        self.close();
    }
}

// The pool handle is a heap allocation the shim owns and only this type touches;
// the driver context it belongs to is process-wide, as the rest of the CUDA
// adapters already assume (`Driver` is shared across threads).
unsafe impl Send for GraphPool {}
unsafe impl Sync for GraphPool {}
