//! Does a captured graph own the device blocks it was recorded over?
//!
//! A CUDA-graph capture bakes the device address of every tensor the step
//! allocates, and cudarc frees each of those slices when the step's Rust values
//! drop at the end of the capture (`CudaSlice::drop` -> `cuMemFreeAsync`). The
//! block returns to the driver's memory pool, so a later `cuMemAsyncAlloc` may
//! legitimately hand that address to a different tensor -- and then a replay
//! writes into that tensor instead of into its own buffer. That is the seam TODO
//! item 7 names: the captured path failed with `CUDA_ERROR_ILLEGAL_ADDRESS` at
//! ~370-token prompts and the trigger was a device-layout shift (two more cubins
//! loaded), which is exactly what changes a pool's block layout.
//!
//! This module measures the claim instead of arguing it. It records the base
//! pointer of every allocation made while a capture is being recorded, then
//! (when armed) counts post-capture allocations that return one of those
//! addresses. Zero reuse over a deliberate allocator churn is evidence that the
//! graph owns its blocks; any reuse reproduces the hazard and is a gate failure.
//!
//! Off, the hooks are one relaxed atomic load per allocation, so the serving
//! path pays nothing once a capture has finished and the tracker is disarmed.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

const OFF: u8 = 0;
const RECORD: u8 = 1;
const WATCH: u8 = 2;

static MODE: AtomicU8 = AtomicU8::new(OFF);

#[derive(Default)]
struct State {
    /// Base addresses of the tensors the recorded step allocated.
    recorded: HashSet<u64>,
    /// The same allocations in order, by byte size: a churn that replays this
    /// sequence is the pool's best case for handing one of the blocks back.
    sizes: Vec<usize>,
    bytes: usize,
    /// Post-capture allocations that returned a recorded address.
    reused: Vec<u64>,
    watched: usize,
}

static P: OnceLock<Mutex<State>> = OnceLock::new();

fn state() -> &'static Mutex<State> {
    P.get_or_init(|| Mutex::new(State::default()))
}

/// What one recorded step allocated.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Summary {
    pub tensors: usize,
    pub bytes: usize,
}

/// Record every allocation from now on as part of the capture being recorded.
pub fn begin_record() {
    let mut s = state().lock().unwrap_or_else(|e| e.into_inner());
    *s = State::default();
    MODE.store(RECORD, Ordering::Relaxed);
}

/// Stop recording and return what the recorded step allocated.
pub fn end_record() -> Summary {
    MODE.store(OFF, Ordering::Relaxed);
    let s = state().lock().unwrap_or_else(|e| e.into_inner());
    Summary {
        tensors: s.sizes.len(),
        bytes: s.bytes,
    }
}

/// Count post-capture allocations that land on a recorded address.
pub fn watch() {
    MODE.store(WATCH, Ordering::Relaxed);
}

/// Stop watching (the serving path never watches; only a test does).
pub fn unwatch() {
    MODE.store(OFF, Ordering::Relaxed);
}

/// Every allocation the recorded step made, by byte size in allocation order.
pub fn recorded_sizes() -> Vec<usize> {
    state()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .sizes
        .clone()
}

/// How many watched allocations returned a recorded address, and the last one.
pub fn reused() -> (usize, Option<u64>) {
    let s = state().lock().unwrap_or_else(|e| e.into_inner());
    (s.reused.len(), s.reused.last().copied())
}

/// How many allocations were inspected while watching.
pub fn watched() -> usize {
    state().lock().unwrap_or_else(|e| e.into_inner()).watched
}

/// One fresh device allocation: `ptr` is its base address, `bytes` its size.
pub fn alloc(ptr: u64, bytes: usize) {
    match MODE.load(Ordering::Relaxed) {
        RECORD => {
            let mut s = state().lock().unwrap_or_else(|e| e.into_inner());
            s.recorded.insert(ptr);
            s.bytes += bytes;
            s.sizes.push(bytes);
        }
        WATCH => {
            let mut s = state().lock().unwrap_or_else(|e| e.into_inner());
            s.watched += 1;
            if s.recorded.contains(&ptr) {
                s.reused.push(ptr);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

    fn reset() {
        MODE.store(OFF, Ordering::Relaxed);
        *state().lock().unwrap() = State::default();
    }

    #[test]
    #[serial]
    fn a_recorded_allocation_is_reported_and_a_later_hit_is_counted() {
        reset();
        begin_record();
        alloc(0x1000, 64);
        alloc(0x2000, 128);
        let summary = end_record();
        assert_eq!(
            summary,
            Summary {
                tensors: 2,
                bytes: 192
            }
        );
        assert_eq!(recorded_sizes(), vec![64, 128]);

        watch();
        alloc(0x3000, 128);
        alloc(0x1000, 64);
        let (hits, last) = reused();
        assert_eq!(hits, 1, "only a recorded address counts");
        assert_eq!(last, Some(0x1000));
        assert_eq!(watched(), 2);

        unwatch();
        alloc(0x2000, 128);
        assert_eq!(reused().0, 1, "nothing is inspected while disarmed");
    }

    #[test]
    #[serial]
    fn nothing_is_recorded_before_a_capture_or_after_it_ends() {
        reset();
        alloc(0x1000, 64);
        assert_eq!(recorded_sizes(), Vec::<usize>::new());
        begin_record();
        alloc(0x2000, 64);
        end_record();
        alloc(0x3000, 64);
        assert_eq!(recorded_sizes(), vec![64], "only the recorded step counts");
    }
}
