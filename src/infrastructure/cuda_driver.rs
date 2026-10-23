//! The CUDA driver API, loaded by hand.
//!
//! Decision 36: the serving path must not import Python, and torch 2.11+ ships no
//! kernel-launch binding, so the driver is the only launcher left. `libcuda` is
//! dlopen'd and its entry points resolved by name rather than linked, for two
//! reasons: the crate then builds and tests with no CUDA at all, and the exact
//! version of the driver that ends up loaded is the one already on
//! `LD_LIBRARY_PATH` next to the CUDA runtime this engine is using.
//!
//! Only what the MoE launch path needs is bound. Anything the driver reports as
//! a `CUresult` is an error: a launch that silently failed would show up as
//! garbage tokens much later, so every fallible call returns `Result` and the
//! driver string is carried in the error.

use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::Arc;

use thiserror::Error;

use crate::infrastructure::kernel_abi::{
    Abi, AttentionAbi, AttentionKernel, Kernel, LinearAbi, LinearKernel, StagingAbi, StagingKernel,
};

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("cuda: cannot load {lib} ({why})")]
    Load { lib: String, why: String },
    #[error("cuda: {lib} has no {symbol}")]
    Symbol { lib: String, symbol: &'static str },
    #[error("cuda: {0}")]
    Driver(String),
    #[error("cuda: no device")]
    NoDevice,
}

pub type CUresult = c_int;
pub type CUdevice = c_int;
/// An opaque driver handle. The driver API's own opaque types are pointers on
/// every platform this engine targets.
pub type CUdeviceptr = u64;
#[allow(clippy::type_complexity)]
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUevent = *mut c_void;

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

/// `CU_STREAM_CAPTURE_MODE_GLOBAL`: capture records every launch on the stream
/// and forbids the cross-stream synchronization that would make the graph's
/// boundaries ambiguous.
pub const CAPTURE_MODE_GLOBAL: u32 = 0;

/// `CU_MEMHOSTALLOC_DEVICEMAP`: map the page-locked host allocation into the
/// device address space, which is what gives a pinned buffer a `CUdeviceptr`.
pub const MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;

/// `CU_STREAM_NON_BLOCKING`: the stream does not implicitly synchronize with the
/// legacy stream. The prefill overlap needs this so the expert copy can run
/// concurrently with the compute stream, ordered only by explicit events.
pub const STREAM_NON_BLOCKING: u32 = 0x01;

/// `CU_MEMCPY_SRC_ACCESS_ORDER_STREAM`: the batch copy reads its sources in
/// stream order. A resident pinned bank is rewritten only between prefills, so
/// the driver may overlap the copy with the compute stream's later work.
pub const MEMCPY_SRC_ACCESS_ORDER_STREAM: i32 = 0x01;

/// `CUmemcpyAttributes` (the CUDA 13 `_v1` layout): a source access order, two
/// location hints and a flag word, all 32-bit fields. A zeroed hint means "no
/// location", which is what a device pointer and a device-mapped host bank want.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MemcpyAttributes {
    pub src_access_order: i32,
    pub src_type: i32,
    pub src_id: i32,
    pub dst_type: i32,
    pub dst_id: i32,
    pub flags: u32,
}

/// `libcuda.so.1` is the SONAME every NVIDIA driver ships under, and the stub
/// `libcuda.so` only exists with the toolkit installed. The devshell puts the
/// driver's own mount first on `LD_LIBRARY_PATH`.
const LIB: &str = "libcuda.so.1";

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

/// The bound entry points, one alias per CUDA signature. Spelling the types out
/// here is the whole point: a wrong signature is a memory-corrupting bug, and
/// `dlsym` cannot check it for us.
type InitFn = unsafe extern "C" fn(u32) -> CUresult;
type DriverVersionFn = unsafe extern "C" fn(*mut c_int) -> CUresult;
type ErrorStringFn = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type DeviceCountFn = unsafe extern "C" fn(*mut c_int) -> CUresult;
type PrimaryCtxRetainFn = unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
type CtxSetCurrentFn = unsafe extern "C" fn(CUcontext) -> CUresult;
type ModuleLoadFn = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type ModuleGetFn = unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
#[allow(clippy::too_many_arguments)]
type LaunchFn = unsafe extern "C" fn(
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type StreamSyncFn = unsafe extern "C" fn(CUstream) -> CUresult;
type MemAllocFn = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type MemFreeFn = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type MemcpyHtoDFn = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type MemcpyDtoHFn = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
/// CUDA 13's 8-argument batch signature: 12.8/12.9 carried an extra `failIdx`
/// pointer before the stream, so the versioned `_v2` name is the only safe one
/// to resolve.
type MemcpyBatchAsyncFn = unsafe extern "C" fn(
    *mut CUdeviceptr,
    *mut CUdeviceptr,
    *mut usize,
    usize,
    *mut MemcpyAttributes,
    *mut usize,
    usize,
    CUstream,
) -> CUresult;
type MemHostAllocFn = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> CUresult;
type MemGetInfoFn = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;
type MemHostGetDevicePointerFn =
    unsafe extern "C" fn(*mut CUdeviceptr, *mut c_void, u32) -> CUresult;
type MemFreeHostFn = unsafe extern "C" fn(*mut c_void) -> CUresult;
type StreamCreateFn = unsafe extern "C" fn(*mut CUstream, u32) -> CUresult;
type StreamDestroyFn = unsafe extern "C" fn(CUstream) -> CUresult;
type StreamBeginCaptureFn = unsafe extern "C" fn(CUstream, u32) -> CUresult;
type StreamEndCaptureFn = unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult;
type GraphDestroyFn = unsafe extern "C" fn(CUgraph) -> CUresult;
type GraphInstantiateFn = unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult;
type GraphLaunchFn = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
type GraphExecDestroyFn = unsafe extern "C" fn(CUgraphExec) -> CUresult;
type EventCreateFn = unsafe extern "C" fn(*mut CUevent, u32) -> CUresult;
type EventDestroyFn = unsafe extern "C" fn(CUevent) -> CUresult;
type EventRecordFn = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
type EventElapsedTimeFn = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;
type StreamWaitEventFn = unsafe extern "C" fn(CUstream, CUevent, u32) -> CUresult;

struct Fns {
    cu_init: InitFn,
    cu_driver_get_version: DriverVersionFn,
    cu_get_error_string: ErrorStringFn,
    cu_device_get_count: DeviceCountFn,
    cu_device_primary_ctx_retain: PrimaryCtxRetainFn,
    cu_ctx_set_current: CtxSetCurrentFn,
    cu_module_load_data: ModuleLoadFn,
    cu_module_get_function: ModuleGetFn,
    cu_launch_kernel: LaunchFn,
    cu_stream_synchronize: StreamSyncFn,
    cu_mem_alloc: MemAllocFn,
    cu_mem_free: MemFreeFn,
    cu_memcpy_htod: MemcpyHtoDFn,
    cu_memcpy_dtoh: MemcpyDtoHFn,
    /// Optional: the engine falls back to the `index_copy` kernel when a
    /// pre-CUDA-13 driver does not export the batch copy.
    cu_memcpy_batch_async: Option<MemcpyBatchAsyncFn>,
    cu_mem_host_alloc: MemHostAllocFn,
    cu_mem_host_get_device_pointer: MemHostGetDevicePointerFn,
    cu_mem_free_host: MemFreeHostFn,
    cu_mem_get_info: MemGetInfoFn,
    cu_stream_create: StreamCreateFn,
    cu_stream_destroy: StreamDestroyFn,
    cu_stream_begin_capture: StreamBeginCaptureFn,
    cu_stream_end_capture: StreamEndCaptureFn,
    cu_graph_destroy: GraphDestroyFn,
    cu_graph_instantiate: GraphInstantiateFn,
    cu_graph_launch: GraphLaunchFn,
    cu_graph_exec_destroy: GraphExecDestroyFn,
    cu_event_create: EventCreateFn,
    cu_event_destroy: EventDestroyFn,
    cu_event_record: EventRecordFn,
    cu_event_elapsed_time: EventElapsedTimeFn,
    cu_stream_wait_event: StreamWaitEventFn,
}

/// A loaded `libcuda`, plus whatever context the caller made current.
pub struct Driver {
    /// Kept only so the library outlives every module loaded from it; we never
    /// `dlclose`, because a module holds the driver's allocations.
    _lib: *mut c_void,
    fns: Fns,
}

// SAFETY: the driver API is thread-safe, and this table is immutable after
// `open`. The engine launches from its own worker threads, not just the one
// that loaded the library.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

impl Driver {
    /// dlopen `libcuda` and resolve the symbols below. The batch copy is the
    /// one optional export (CUDA 13's `cuMemcpyBatchAsync_v2`); everything else
    /// must resolve. This is the only fallible part of the boundary that is not
    /// a `CUresult`: every later call goes through `check`.
    pub fn open() -> Result<Self, CudaError> {
        let lib_name = CString::new(LIB).expect("no NUL in the library name");
        // SAFETY: a NUL-terminated name. The handle is kept for the life of the
        // Driver and never closed.
        let lib = unsafe { dlopen(lib_name.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if lib.is_null() {
            let why = unsafe {
                let e = dlerror();
                if e.is_null() {
                    "dlopen failed".to_string()
                } else {
                    std::ffi::CStr::from_ptr(e).to_string_lossy().into_owned()
                }
            };
            return Err(CudaError::Load {
                lib: LIB.to_string(),
                why,
            });
        }

        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let c = CString::new($name).expect("no NUL in a symbol name");
                // SAFETY: the handle came from dlopen above and outlives this
                // struct; the driver's exports are plain C functions.
                let p = unsafe { dlsym(lib, c.as_ptr()) };
                if p.is_null() {
                    return Err(CudaError::Symbol {
                        lib: LIB.to_string(),
                        symbol: $name,
                    });
                }
                // SAFETY: `p` points at an `extern "C"` export whose signature is
                // the CUDA header's for `$name`, named in the call site below.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
            }};
        }

        // An optimization, not a requirement: a driver that predates CUDA 13
        // keeps serving through the `index_copy` kernel.
        macro_rules! sym_opt {
            ($name:literal, $ty:ty) => {{
                let c = CString::new($name).expect("no NUL in a symbol name");
                // SAFETY: as `sym!`; a null result is the absent-symbol case.
                let p = unsafe { dlsym(lib, c.as_ptr()) };
                if p.is_null() {
                    None
                } else {
                    // SAFETY: `p` points at an `extern "C"` export whose
                    // signature is the CUDA header's for `$name`.
                    Some(unsafe { std::mem::transmute::<*mut c_void, $ty>(p) })
                }
            }};
        }

        let fns = Fns {
            cu_init: sym!("cuInit", InitFn),
            cu_driver_get_version: sym!("cuDriverGetVersion", DriverVersionFn),
            cu_get_error_string: sym!("cuGetErrorString", ErrorStringFn),
            cu_device_get_count: sym!("cuDeviceGetCount", DeviceCountFn),
            cu_device_primary_ctx_retain: sym!("cuDevicePrimaryCtxRetain", PrimaryCtxRetainFn),
            cu_ctx_set_current: sym!("cuCtxSetCurrent", CtxSetCurrentFn),
            cu_module_load_data: sym!("cuModuleLoadData", ModuleLoadFn),
            cu_module_get_function: sym!("cuModuleGetFunction", ModuleGetFn),
            cu_launch_kernel: sym!("cuLaunchKernel", LaunchFn),
            cu_stream_synchronize: sym!("cuStreamSynchronize", StreamSyncFn),
            // The `_v2` names are the 64-bit device-pointer ABI; the unsuffixed
            // ones are the 32-bit-pointer era and a modern driver does not export
            // them.
            cu_mem_alloc: sym!("cuMemAlloc_v2", MemAllocFn),
            cu_mem_free: sym!("cuMemFree_v2", MemFreeFn),
            cu_memcpy_htod: sym!("cuMemcpyHtoD_v2", MemcpyHtoDFn),
            cu_memcpy_dtoh: sym!("cuMemcpyDtoH_v2", MemcpyDtoHFn),
            // The whole-layer prefill prefetches through the copy engine rather
            // than an SM `index_copy`, which is what lets it overlap compute.
            cu_memcpy_batch_async: sym_opt!("cuMemcpyBatchAsync_v2", MemcpyBatchAsyncFn),
            // The host allocation is mapped into the device address space
            // (`CU_MEMHOSTALLOC_DEVICEMAP`), so the copy kernel can read the
            // pinned expert banks directly.
            cu_mem_host_alloc: sym!("cuMemHostAlloc", MemHostAllocFn),
            cu_mem_host_get_device_pointer: sym!(
                "cuMemHostGetDevicePointer_v2",
                MemHostGetDevicePointerFn
            ),
            cu_mem_free_host: sym!("cuMemFreeHost", MemFreeHostFn),
            // Free/total device memory for the auto cache-budget planner (item 12).
            cu_mem_get_info: sym!("cuMemGetInfo_v2", MemGetInfoFn),
            cu_stream_create: sym!("cuStreamCreate", StreamCreateFn),
            cu_stream_destroy: sym!("cuStreamDestroy_v2", StreamDestroyFn),
            cu_stream_begin_capture: sym!("cuStreamBeginCapture_v2", StreamBeginCaptureFn),
            cu_stream_end_capture: sym!("cuStreamEndCapture", StreamEndCaptureFn),
            cu_graph_destroy: sym!("cuGraphDestroy", GraphDestroyFn),
            // `cuGraphInstantiateWithFlags` is the unambiguous 3-argument
            // spelling: the plain `cuGraphInstantiate` name is versioned by
            // header era and its current alias is a different signature.
            cu_graph_instantiate: sym!("cuGraphInstantiateWithFlags", GraphInstantiateFn),
            cu_graph_launch: sym!("cuGraphLaunch", GraphLaunchFn),
            cu_graph_exec_destroy: sym!("cuGraphExecDestroy", GraphExecDestroyFn),
            // Events order the prefill copy stream against the compute stream
            // (item 8 slice 2). `cuEventDestroy_v2` is the driver's export.
            cu_event_create: sym!("cuEventCreate", EventCreateFn),
            cu_event_destroy: sym!("cuEventDestroy_v2", EventDestroyFn),
            cu_event_record: sym!("cuEventRecord", EventRecordFn),
            // Durations for the `FT_PROFILE` stage timer (timing-enabled events).
            cu_event_elapsed_time: sym!("cuEventElapsedTime", EventElapsedTimeFn),
            cu_stream_wait_event: sym!("cuStreamWaitEvent", StreamWaitEventFn),
        };
        Ok(Self { _lib: lib, fns })
    }

    fn check(&self, r: CUresult) -> Result<(), CudaError> {
        if r == 0 {
            return Ok(());
        }
        // SAFETY: `cuGetErrorString` writes to the pointer we own, or leaves it
        // null, which we handle rather than deref.
        let mut msg: *const c_char = std::ptr::null();
        // SAFETY: the resolved signature takes (CUresult, *const c_char **).
        unsafe { (self.fns.cu_get_error_string)(r, &raw mut msg) };
        let text = if msg.is_null() {
            format!("CUresult {r}")
        } else {
            // SAFETY: non-null string the driver just wrote; the lifetime is the
            // driver's static table.
            unsafe { std::ffi::CStr::from_ptr(msg) }
                .to_string_lossy()
                .into_owned()
        };
        if r == 100 {
            return Err(CudaError::NoDevice);
        }
        Err(CudaError::Driver(text))
    }

    /// `cuInit(0)`. Must succeed before anything else, and the sandbox has no
    /// device, so this is where a CPU-only box stops.
    pub fn init(&self) -> Result<(), CudaError> {
        // SAFETY: no arguments, no output.
        self.check(unsafe { (self.fns.cu_init)(0) })
    }

    /// Whether the driver exports `cuMemcpyBatchAsync_v2` (CUDA 13). When it
    /// does not, the whole-layer prefill keeps the `index_copy` kernel.
    pub fn batch_copy_available(&self) -> bool {
        self.fns.cu_memcpy_batch_async.is_some()
    }

    pub fn driver_version(&self) -> Result<i32, CudaError> {
        let mut v = 0;
        // SAFETY: `v` is a live i32 for the call's duration.
        self.check(unsafe { (self.fns.cu_driver_get_version)(&raw mut v) })?;
        Ok(v)
    }

    pub fn device_count(&self) -> Result<i32, CudaError> {
        let mut n = 0;
        // SAFETY: `n` is a live i32 for the call's duration.
        self.check(unsafe { (self.fns.cu_device_get_count)(&raw mut n) })?;
        Ok(n)
    }

    /// `(free, total)` device memory in bytes for the current context, the input
    /// to the auto expert-cache budget (item 12, `registry::resolve_cache_slots`).
    pub fn mem_info(&self) -> Result<(usize, usize), CudaError> {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both are live `usize` slots for the call's duration.
        self.check(unsafe { (self.fns.cu_mem_get_info)(&raw mut free, &raw mut total) })?;
        Ok((free, total))
    }

    /// Retain the primary context of `device` and make it current on this
    /// thread. Retaining (not creating) keeps a launch here in the same context
    /// as the cudarc compute runtime.
    pub fn attach_primary_context(&self, device: CUdevice) -> Result<CUcontext, CudaError> {
        let mut ctx: CUcontext = std::ptr::null_mut();
        // SAFETY: `ctx` is a live pointer-sized slot for the duration of the call.
        self.check(unsafe { (self.fns.cu_device_primary_ctx_retain)(&raw mut ctx, device) })?;
        // SAFETY: a retained primary context is a valid handle to set current.
        self.check(unsafe { (self.fns.cu_ctx_set_current)(ctx) })?;
        Ok(ctx)
    }

    /// Load a cubin image (`kernel_abi::cubin`) as a module.
    pub fn load_module(&self, image: &[u8]) -> Result<CUmodule, CudaError> {
        let mut module: CUmodule = std::ptr::null_mut();
        // SAFETY: `image` outlives the call (the driver copies what it needs at
        // load time), and `module` is a live slot.
        self.check(unsafe {
            (self.fns.cu_module_load_data)(&raw mut module, image.as_ptr().cast())
        })?;
        Ok(module)
    }

    /// # Safety
    /// `module` must be a live module from `load_module` on this driver, in the
    /// context that is current on this thread.
    pub unsafe fn function(&self, module: CUmodule, entry: &str) -> Result<CUfunction, CudaError> {
        let c = CString::new(entry).expect("no NUL in an entry name");
        let mut f: CUfunction = std::ptr::null_mut();
        // SAFETY: a live module, a NUL-terminated name, a live output slot.
        let r = unsafe { (self.fns.cu_module_get_function)(&raw mut f, module, c.as_ptr()) };
        self.check(r)
            .map_err(|why| CudaError::Driver(format!("{entry}: {why}")))?;
        Ok(f)
    }

    /// Launch `f` with a grid and block of 1D extents.
    ///
    /// `params` is `kernel_abi`'s `kernelParams` array: one pointer per entry
    /// parameter, and it must stay alive until the launch returns, which it does
    /// because it is borrowed here. The driver copies the values out before
    /// returning, so the launches can be async on a non-default stream.
    ///
    /// # Safety
    /// `f` must be a live function of a module loaded on this driver, `params`
    /// must hold one pointer per entry parameter with the right width, and
    /// `stream` must be a stream of the current context (or null).
    pub unsafe fn launch(
        &self,
        f: CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        stream: CUstream,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        // SAFETY: `f` came from `cuModuleGetFunction`, the grid and block are the
        // kernel's, and `params` holds one pointer per parameter of a live
        // ArgPack. The driver reads the argument values during the call.
        // SAFETY: the caller guarantees a live function and a well-formed
        // `kernelParams` array.
        let r = unsafe {
            (self.fns.cu_launch_kernel)(
                f,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.check(r)
    }

    /// # Safety
    /// `stream` must be a stream of the current context, or null for the legacy
    /// default stream.
    pub unsafe fn synchronize(&self, stream: CUstream) -> Result<(), CudaError> {
        // SAFETY: the caller guarantees `stream` belongs to the current context.
        self.check(unsafe { (self.fns.cu_stream_synchronize)(stream) })
    }

    /// Create a non-legacy CUDA stream in the current context.
    ///
    /// Flags are 0 (`CU_STREAM_DEFAULT`), so the stream keeps the legacy
    /// stream's implicit ordering: work enqueued here waits for stream-0 work
    /// (the staging path) and stream-0 work waits for this stream's, which is
    /// what lets a captured replay read buffers written elsewhere without an
    /// explicit event.
    ///
    /// # Safety
    /// The caller must have a CUDA context current; the returned stream
    /// belongs to that context and must be destroyed with `destroy_stream`.
    pub unsafe fn create_stream(&self) -> Result<CUstream, CudaError> {
        // SAFETY: the caller guarantees a current context.
        unsafe { self.create_stream_with_flags(0) }
    }

    /// # Safety
    /// A CUDA context must be current; the returned stream belongs to it.
    pub unsafe fn create_stream_with_flags(&self, flags: u32) -> Result<CUstream, CudaError> {
        let mut stream: CUstream = std::ptr::null_mut();
        // SAFETY: `stream` is a valid out pointer; `flags` is a stream flag bit.
        self.check(unsafe { (self.fns.cu_stream_create)(&raw mut stream, flags) })?;
        Ok(stream)
    }

    /// An owned stream (destroyed on drop), for a stream that outlives the call
    /// that made it. `flags` is a `CUstream_flags` bit set; pass
    /// `STREAM_NON_BLOCKING` for a copy stream that must not serialize with the
    /// legacy stream.
    pub fn owned_stream(self: &Arc<Self>, flags: u32) -> Result<OwnedStream, CudaError> {
        // SAFETY: the adapter that calls this runs with its context current,
        // which is the invariant `create_stream_with_flags` documents.
        let raw = unsafe { self.create_stream_with_flags(flags) }?;
        Ok(OwnedStream {
            driver: Arc::clone(self),
            raw,
        })
    }

    /// An owned event (destroyed on drop), with timing disabled.
    pub fn owned_event(self: &Arc<Self>) -> Result<OwnedEvent, CudaError> {
        // SAFETY: as `owned_stream`, the caller's context is current.
        let raw = unsafe { self.create_event() }?;
        Ok(OwnedEvent {
            driver: Arc::clone(self),
            raw,
        })
    }

    /// An owned event with timing enabled, for `OwnedEvent::elapsed`.
    pub fn owned_timing_event(self: &Arc<Self>) -> Result<OwnedEvent, CudaError> {
        // SAFETY: as `owned_event`, the caller's context is current.
        let raw = unsafe { self.create_timing_event() }?;
        Ok(OwnedEvent {
            driver: Arc::clone(self),
            raw,
        })
    }

    /// Create a timing-enabled event (`flags = 0`). `FT_PROFILE` uses these to
    /// bracket stages; a duration event costs a device timestamp per record, so
    /// the ordering-only `create_event` stays the default.
    ///
    /// # Safety
    /// The caller must have a CUDA context current; the returned event belongs
    /// to it.
    pub unsafe fn create_timing_event(&self) -> Result<CUevent, CudaError> {
        let mut event: CUevent = std::ptr::null_mut();
        // SAFETY: `event` is a live out slot; flags 0 enables timing.
        self.check(unsafe { (self.fns.cu_event_create)(&raw mut event, 0) })?;
        Ok(event)
    }

    /// Destroy a stream created by `create_stream`.
    ///
    /// # Safety
    /// `stream` must be a valid stream in the current context and not currently
    /// being used by ongoing work. Passing null is allowed by CUDA but we treat
    /// it as an error to avoid silent no-ops.
    pub unsafe fn destroy_stream(&self, stream: CUstream) -> Result<(), CudaError> {
        if stream.is_null() {
            return Err(CudaError::Driver("destroy_stream called with null".into()));
        }
        // SAFETY: `stream` is owned by the caller and belongs to current context.
        self.check(unsafe { (self.fns.cu_stream_destroy)(stream) })
    }

    /// Create an event with timing disabled (`CU_EVENT_DISABLE_TIMING`): the
    /// prefill overlap only needs ordering, not durations, and a timing event
    /// costs a device timestamp per record.
    ///
    /// # Safety
    /// The caller must have a CUDA context current; the returned event belongs
    /// to it and must be destroyed with `destroy_event`.
    pub unsafe fn create_event(&self) -> Result<CUevent, CudaError> {
        let mut event: CUevent = std::ptr::null_mut();
        // SAFETY: `event` is a live out slot; bit 1 is CU_EVENT_DISABLE_TIMING.
        self.check(unsafe { (self.fns.cu_event_create)(&raw mut event, 0x2) })?;
        Ok(event)
    }

    /// Destroy an event created by `create_event`.
    ///
    /// # Safety
    /// `event` must be a valid event of the current context with no record
    /// still being waited on.
    pub unsafe fn destroy_event(&self, event: CUevent) -> Result<(), CudaError> {
        if event.is_null() {
            return Err(CudaError::Driver("destroy_event called with null".into()));
        }
        // SAFETY: the caller owns the event and it belongs to the current context.
        self.check(unsafe { (self.fns.cu_event_destroy)(event) })
    }

    /// Record `event` on `stream`: it completes when every launch queued on
    /// `stream` before this call has completed.
    ///
    /// # Safety
    /// `event` and `stream` must belong to the current context.
    pub unsafe fn record_event(&self, event: CUevent, stream: CUstream) -> Result<(), CudaError> {
        // SAFETY: both handles are the caller's live objects in this context.
        self.check(unsafe { (self.fns.cu_event_record)(event, stream) })
    }

    /// Make every later launch on `stream` wait for `event` to complete.
    ///
    /// # Safety
    /// `event` and `stream` must belong to the current context; `event` must
    /// have been recorded (or be about to be) so the wait can resolve.
    pub unsafe fn stream_wait_event(
        &self,
        stream: CUstream,
        event: CUevent,
    ) -> Result<(), CudaError> {
        // SAFETY: both handles are the caller's live objects in this context.
        self.check(unsafe { (self.fns.cu_stream_wait_event)(stream, event, 0) })
    }

    /// Record everything launched on `stream` into a graph instead of running
    /// it, until `end_capture`. Work already queued elsewhere must be flushed
    /// first (decision 39): a capture stream with legacy-sync semantics that
    /// meets outstanding stream-0 work has a cross-stream dependency capture
    /// cannot record.
    ///
    /// # Safety
    /// `stream` must be a non-legacy stream of the current context and not
    /// already capturing; between this call and `end_capture` nothing may
    /// synchronize `stream` or enqueue work on it from another thread.
    pub unsafe fn begin_capture(&self, stream: CUstream, mode: u32) -> Result<(), CudaError> {
        if stream.is_null() {
            return Err(CudaError::Driver(
                "begin_capture of the legacy stream".into(),
            ));
        }
        // SAFETY: the caller guarantees a live, non-capturing side stream.
        self.check(unsafe { (self.fns.cu_stream_begin_capture)(stream, mode) })
    }

    /// Stop the capture and hand back the recorded graph. The stream returns to
    /// normal, so queued graph work still has to be synchronized before the
    /// staged buffers the graph reads are rewritten.
    ///
    /// # Safety
    /// `stream` must be capturing (paired with `begin_capture` on this thread).
    pub unsafe fn end_capture(&self, stream: CUstream) -> Result<CUgraph, CudaError> {
        let mut graph: CUgraph = std::ptr::null_mut();
        // SAFETY: `graph` is a live out slot; the caller owns the capture.
        self.check(unsafe { (self.fns.cu_stream_end_capture)(stream, &raw mut graph) })?;
        Ok(graph)
    }

    /// Instantiate a captured graph into something launchable. The instance
    /// owns its own memory; the source graph can be destroyed afterwards.
    ///
    /// # Safety
    /// `graph` must be a live graph of the current context.
    pub unsafe fn graph_instantiate(&self, graph: CUgraph) -> Result<CUgraphExec, CudaError> {
        // SAFETY: as `graph_instantiate_with_flags`; flags 0 is the plain
        // instantiation the serving path's fixed-buffer graphs use.
        unsafe { self.graph_instantiate_with_flags(graph, 0) }
    }

    /// `cuGraphInstantiateWithFlags` with explicit flags. The M7.0 spike uses
    /// `CU_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH (1)` for a graph whose
    /// a captured graph allocated memory nodes: without it the allocations are
    /// not returned at the end of a launch, so a second launch of the same
    /// instance fails with `CUDA_ERROR_INVALID_VALUE`.
    ///
    /// # Safety
    /// `graph` must be a live graph of the current context and `flags` must be
    /// the driver's `CUgraphInstantiate_flags` bit set.
    pub unsafe fn graph_instantiate_with_flags(
        &self,
        graph: CUgraph,
        flags: u64,
    ) -> Result<CUgraphExec, CudaError> {
        let mut exec: CUgraphExec = std::ptr::null_mut();
        // SAFETY: `exec` is a live out slot and `graph` is the caller's live
        // graph.
        self.check(unsafe { (self.fns.cu_graph_instantiate)(&raw mut exec, graph, flags) })?;
        Ok(exec)
    }

    /// Launch a captured instance on `stream`.
    ///
    /// # Safety
    /// `exec` must be a live instance and `stream` a stream of the current
    /// context (or null); the device buffers the graph reads must still hold
    /// the addresses and the data its capture saw.
    pub unsafe fn graph_launch(
        &self,
        exec: CUgraphExec,
        stream: CUstream,
    ) -> Result<(), CudaError> {
        // SAFETY: the caller's contract is this method's contract.
        self.check(unsafe { (self.fns.cu_graph_launch)(exec, stream) })
    }

    /// Drop a captured graph (its instance is independent once instantiated).
    ///
    /// # Safety
    /// `graph` must be a live graph with no in-flight capture.
    pub unsafe fn graph_destroy(&self, graph: CUgraph) -> Result<(), CudaError> {
        if graph.is_null() {
            return Err(CudaError::Driver("graph_destroy called with null".into()));
        }
        // SAFETY: the caller guarantees a live graph handle.
        self.check(unsafe { (self.fns.cu_graph_destroy)(graph) })
    }

    /// Drop an instantiated graph, freeing the memory it replayed from.
    ///
    /// # Safety
    /// `exec` must be a live instance with no launch in flight.
    pub unsafe fn graph_exec_destroy(&self, exec: CUgraphExec) -> Result<(), CudaError> {
        if exec.is_null() {
            return Err(CudaError::Driver(
                "graph_exec_destroy called with null".into(),
            ));
        }
        // SAFETY: the caller guarantees a live instance handle.
        self.check(unsafe { (self.fns.cu_graph_exec_destroy)(exec) })
    }

    /// A device allocation, freed on drop. Used by the tests and by the slot
    /// cache's first slice; the steady state is cudarc tensors.
    pub fn alloc<'d>(&'d self, bytes: usize) -> Result<DeviceBuf<'d>, CudaError> {
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: `ptr` is a live slot; the length is the whole allocation.
        self.check(unsafe { (self.fns.cu_mem_alloc)(&raw mut ptr, bytes) })?;
        Ok(DeviceBuf {
            driver: self,
            ptr,
            bytes,
        })
    }

    /// A page-locked host allocation with a device-visible alias, freed on drop.
    /// The MoE offload cache fills these from the FTW checkpoint once and the
    /// copy kernel DMA-reads out of them forever, so the owner keeps the driver
    /// alive for as long as any bank lives.
    pub fn alloc_pinned(self: &Arc<Self>, bytes: usize) -> Result<PinnedBuf, CudaError> {
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host` is a live slot; `DEVICEMAP` is what makes the buffer
        // reachable from the device at all.
        self.check(unsafe {
            (self.fns.cu_mem_host_alloc)(&raw mut host, bytes, MEMHOSTALLOC_DEVICEMAP)
        })?;
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: `host` is the live allocation from the call above and `ptr` is
        // a live slot. The device pointer is only valid while `host` lives,
        // which `PinnedBuf` enforces.
        match self
            .check(unsafe { (self.fns.cu_mem_host_get_device_pointer)(&raw mut ptr, host, 0) })
        {
            Ok(()) => Ok(PinnedBuf {
                driver: Arc::clone(self),
                host,
                ptr,
                bytes,
            }),
            Err(e) => {
                let _ = self.check(unsafe { (self.fns.cu_mem_free_host)(host) });
                Err(e)
            }
        }
    }

    /// # Safety
    /// `dst` must be a live device address with room for `src.len()` bytes.
    pub unsafe fn copy_h2d(&self, dst: CUdeviceptr, src: &[u8]) -> Result<(), CudaError> {
        // SAFETY: the caller guarantees `dst` has room for `src.len()` bytes.
        self.check(unsafe { (self.fns.cu_memcpy_htod)(dst, src.as_ptr().cast(), src.len()) })
    }

    /// # Safety
    /// `src` must be a live device address with at least `dst.len()` bytes.
    pub unsafe fn copy_d2h(&self, dst: &mut [u8], src: CUdeviceptr) -> Result<(), CudaError> {
        // SAFETY: the caller guarantees `src` holds at least `dst.len()` bytes.
        self.check(unsafe { (self.fns.cu_memcpy_dtoh)(dst.as_mut_ptr().cast(), src, dst.len()) })
    }

    /// Enqueue every `(dst, src, bytes)` triple as one `cuMemcpyBatchAsync`.
    /// Entries must be pairwise independent (the driver may run them in any
    /// order), which the whole-layer prefetch's one-run-per-bank plan is.
    /// One attribute covers the batch: sources are read in stream order, which
    /// a resident pinned bank allows.
    ///
    /// # Safety
    /// Every triple must be a live, non-overlapping copy range, and `stream`
    /// must be a stream of the current context (never the legacy null stream,
    /// which the driver would treat as a synchronous call).
    pub unsafe fn copy_batch_async(
        &self,
        dsts: &[CUdeviceptr],
        srcs: &[CUdeviceptr],
        sizes: &[usize],
        stream: CUstream,
    ) -> Result<(), CudaError> {
        if dsts.len() != srcs.len() || dsts.len() != sizes.len() {
            return Err(CudaError::Driver(
                "copy_batch_async: array lengths differ".into(),
            ));
        }
        if dsts.is_empty() {
            return Ok(());
        }
        if stream.is_null() {
            return Err(CudaError::Driver(
                "copy_batch_async on the legacy stream".into(),
            ));
        }
        let Some(copy) = self.fns.cu_memcpy_batch_async else {
            return Err(CudaError::Driver(
                "cuMemcpyBatchAsync_v2 is not available in this driver".into(),
            ));
        };
        let mut attrs = [MemcpyAttributes {
            src_access_order: MEMCPY_SRC_ACCESS_ORDER_STREAM,
            ..MemcpyAttributes::default()
        }];
        let mut attr_idx = [0usize];
        // SAFETY: the arrays are same-length live slices for the call's
        // duration, the driver only reads the pointer/size arrays, and one
        // attribute at index 0 is the documented whole-batch spelling.
        self.check(unsafe {
            copy(
                dsts.as_ptr().cast_mut(),
                srcs.as_ptr().cast_mut(),
                sizes.as_ptr().cast_mut(),
                dsts.len(),
                attrs.as_mut_ptr(),
                attr_idx.as_mut_ptr(),
                attrs.len(),
                stream,
            )
        })
    }
}

impl CudaError {
    pub fn is_no_device(&self) -> bool {
        matches!(self, CudaError::NoDevice)
    }
}

/// Owns a device allocation. Dropping frees it; a leaked cache would pin VRAM
/// across a request, which is exactly the failure the slot cache has to avoid.
pub struct DeviceBuf<'a> {
    driver: &'a Driver,
    pub ptr: CUdeviceptr,
    bytes: usize,
}

impl Drop for DeviceBuf<'_> {
    fn drop(&mut self) {
        // A failure here means the driver is already gone; there is nothing
        // useful to do about it in a destructor.
        let _ = self
            .driver
            .check(unsafe { (self.driver.fns.cu_mem_free)(self.ptr) });
    }
}

impl DeviceBuf<'_> {
    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

/// Owns a page-locked host allocation and its device-visible alias. Dropping
/// frees it; the device pointer is only valid while this lives, so whoever hands
/// it to the device must keep the buffer alive. Unlike `DeviceBuf` this holds an
/// `Arc<Driver>`, because a pinned bank outlives the call that made it.
pub struct PinnedBuf {
    driver: Arc<Driver>,
    host: *mut c_void,
    pub ptr: CUdeviceptr,
    bytes: usize,
}

// SAFETY: the allocation is a plain byte region the driver mapped into device
// space, so touching it from another thread is ordinary host memory access, and
// the `Arc<Driver>` behind it is thread-safe.
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        // A failure here means the driver is already gone; there is nothing
        // useful to do about it in a destructor.
        let _ = self
            .driver
            .check(unsafe { (self.driver.fns.cu_mem_free_host)(self.host) });
    }
}

impl PinnedBuf {
    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// The host view the FTW reader fills.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `host` is this struct's live allocation of `bytes`.
        unsafe { std::slice::from_raw_parts(self.host.cast(), self.bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `host` is this struct's live allocation of `bytes`, and
        // `&mut self` rules out another live slice into it.
        unsafe { std::slice::from_raw_parts_mut(self.host.cast(), self.bytes) }
    }
}

/// Owns a CUDA stream and destroys it on drop. Unlike `DeviceBuf` it holds an
/// `Arc<Driver>`, because the prefill copy stream outlives the call that made
/// it. Dropping is best-effort: a driver already gone needs no cleanup.
pub struct OwnedStream {
    driver: Arc<Driver>,
    raw: CUstream,
}

// SAFETY: a stream handle is a plain driver object reached through the
// thread-safe `Arc<Driver>`, and the driver API itself is thread-safe.
unsafe impl Send for OwnedStream {}
unsafe impl Sync for OwnedStream {}

impl Drop for OwnedStream {
    fn drop(&mut self) {
        let _ = self
            .driver
            .check(unsafe { (self.driver.fns.cu_stream_destroy)(self.raw) });
    }
}

impl OwnedStream {
    pub fn raw(&self) -> CUstream {
        self.raw
    }
}

/// Owns a CUDA event and destroys it on drop. The prefill overlap records and
/// waits on these to order its copy stream against the compute stream.
pub struct OwnedEvent {
    driver: Arc<Driver>,
    raw: CUevent,
}

// SAFETY: as `OwnedStream`: a driver object behind a thread-safe `Arc<Driver>`.
unsafe impl Send for OwnedEvent {}
unsafe impl Sync for OwnedEvent {}

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        let _ = self
            .driver
            .check(unsafe { (self.driver.fns.cu_event_destroy)(self.raw) });
    }
}

impl OwnedEvent {
    pub fn raw(&self) -> CUevent {
        self.raw
    }

    /// Record on `stream`: the event completes once every launch queued on
    /// `stream` before this call has completed.
    ///
    /// # Safety
    /// `stream` must be a live stream of this event's context, which must be
    /// current.
    pub unsafe fn record(&self, stream: CUstream) -> Result<(), CudaError> {
        // SAFETY: the caller guarantees a live stream in the current context.
        self.driver
            .check(unsafe { (self.driver.fns.cu_event_record)(self.raw, stream) })
    }

    /// Make every later launch on `stream` wait for this event to complete.
    ///
    /// # Safety
    /// `stream` must be a live stream of this event's context, which must be
    /// current.
    pub unsafe fn wait_on(&self, stream: CUstream) -> Result<(), CudaError> {
        // SAFETY: the caller guarantees a live stream in the current context.
        self.driver
            .check(unsafe { (self.driver.fns.cu_stream_wait_event)(stream, self.raw, 0) })
    }

    /// Milliseconds between this event and a later `end` event, both recorded
    /// and completed (`cuEventElapsedTime`). Both must be timing-enabled.
    pub fn elapsed(&self, end: &OwnedEvent) -> Result<f32, CudaError> {
        let mut ms: f32 = 0.0;
        // SAFETY: both handles are live timing events in this driver's context.
        self.driver.check(unsafe {
            (self.driver.fns.cu_event_elapsed_time)(&raw mut ms, self.raw, end.raw)
        })?;
        Ok(ms)
    }
}

/// Loaded function handles keyed by manifest kernel name.
///
/// The key is the name and not the Triton entry, because both GEMV cubins are
/// compiled from `_nvfp4_gemv_kernel` with different constexprs. An entry-keyed
/// table resolves the down projection to the gate/up cubin, and that still runs
/// to completion: it reads `inter/2` packed bytes out of a `hidden`-row slab,
/// takes the activation row from `route_token` and drops the route weights, so
/// the result is wrong but plausible.
#[derive(Default)]
struct FunctionTable(Vec<(&'static str, CUfunction)>);

impl FunctionTable {
    fn push(&mut self, name: &'static str, f: CUfunction) {
        self.0.push((name, f));
    }

    fn get(&self, name: &str) -> Option<CUfunction> {
        self.0.iter().find(|(n, _)| *n == name).map(|(_, f)| *f)
    }
}

/// One committed kernel, loaded and ready to launch.
///
/// The module and function handles are owned by the `CudaKernels` that made
/// them; a bare `CUfunction` would be a dangling pointer waiting to happen.
pub struct CudaKernels {
    driver: Arc<Driver>,
    abi: Abi,
    loaded: Vec<(CUmodule, FunctionTable)>,
}

// SAFETY: the module and function handles are immutable once `load` returns and
// the driver behind them is thread-safe (see `Driver`). Launching the same
// function concurrently is what the driver API is for.
unsafe impl Send for CudaKernels {}
unsafe impl Sync for CudaKernels {}

impl CudaKernels {
    /// Load every kernel in `abi` from the committed cubins. The driver is
    /// shared, because the caller launches on the runtime's streams too and there
    /// must be exactly one driver handle in the process.
    pub fn load(driver: Arc<Driver>, abi: Abi) -> Result<Self, CudaError> {
        let mut loaded = Vec::new();
        for name in Abi::kernel_names() {
            let image = crate::infrastructure::kernel_abi::cubin(name)
                .map_err(|e| CudaError::Driver(format!("cubin: {e}")))?;
            let module = driver.load_module(image)?;
            let kernel = abi
                .kernel(name)
                .map_err(|e| CudaError::Driver(format!("abi: {e}")))?;
            // SAFETY: `module` was just loaded by `driver`, whose context is
            // current on this thread.
            let f = unsafe { driver.function(module, &kernel.entry)? };
            let mut table = FunctionTable::default();
            table.push(kernel.name, f);
            loaded.push((module, table));
        }
        Ok(Self {
            driver,
            abi,
            loaded,
        })
    }

    pub fn abi(&self) -> &Abi {
        &self.abi
    }

    /// The shared driver handle, so a sibling allocation (the pinned banks) uses
    /// the same context these kernels launch in.
    pub fn driver(&self) -> Arc<Driver> {
        Arc::clone(&self.driver)
    }

    pub fn function(&self, kernel: &Kernel) -> Result<CUfunction, CudaError> {
        for (_, table) in &self.loaded {
            if let Some(f) = table.get(kernel.name) {
                return Ok(f);
            }
        }
        Err(CudaError::Driver(format!(
            "no loaded function for {}",
            kernel.name
        )))
    }

    /// Launch `kernel` with `grid` and the block its manifest entry recorded.
    ///
    /// # Safety
    /// `stream` must belong to the context that is current on this thread, and
    /// `params` must be the `kernel_abi` pack for this same kernel.
    pub unsafe fn launch(
        &self,
        kernel: &Kernel,
        grid: (u32, u32, u32),
        params: &mut [*mut c_void],
        stream: CUstream,
    ) -> Result<(), CudaError> {
        let f = self.function(kernel)?;
        // SAFETY: `f` came from this type's own `load`, so it is live in the
        // driver's current context, and the grid matches the kernel's constexpr
        // trip counts by construction in `kernel_abi`.
        unsafe {
            self.driver.launch(
                f,
                grid,
                (Abi::block_threads(kernel), 1, 1),
                kernel.shared,
                stream,
                params,
            )
        }
    }

    /// # Safety
    /// As `Driver::synchronize`: `stream` must belong to the current context.
    pub unsafe fn synchronize(&self, stream: CUstream) -> Result<(), CudaError> {
        // SAFETY: the caller's contract is this method's contract.
        unsafe { self.driver.synchronize(stream) }
    }
}

/// The committed paged attention cubin, loaded and ready to launch. Separate
/// from `CudaKernels` and `StagingKernels` because it is a third contract
/// (decision 60): one program per (row, q_head) with the head count baked in,
/// so its launch grid is computed per call rather than fixed in the manifest.
pub struct AttentionKernels {
    driver: Arc<Driver>,
    abi: AttentionAbi,
    loaded: Vec<(CUmodule, FunctionTable)>,
}

// SAFETY: as `CudaKernels`: the handles are immutable once `load` returns and
// the driver behind them is thread-safe.
unsafe impl Send for AttentionKernels {}
unsafe impl Sync for AttentionKernels {}

impl AttentionKernels {
    /// Load every attention kernel in `abi` from the committed cubins.
    pub fn load(driver: Arc<Driver>, abi: AttentionAbi) -> Result<Self, CudaError> {
        let mut loaded = Vec::new();
        for name in AttentionAbi::kernel_names() {
            let image = crate::infrastructure::kernel_abi::attention_cubin(name)
                .map_err(|e| CudaError::Driver(format!("cubin: {e}")))?;
            let module = driver.load_module(image)?;
            let kernel = abi
                .kernel(name)
                .map_err(|e| CudaError::Driver(format!("attention abi: {e}")))?;
            // SAFETY: `module` was just loaded by `driver`, whose context is
            // current on this thread.
            let f = unsafe { driver.function(module, &kernel.entry)? };
            let mut table = FunctionTable::default();
            table.push(kernel.name, f);
            loaded.push((module, table));
        }
        Ok(Self {
            driver,
            abi,
            loaded,
        })
    }

    pub fn abi(&self) -> &AttentionAbi {
        &self.abi
    }

    fn function(&self, kernel: &AttentionKernel) -> Result<CUfunction, CudaError> {
        for (_, table) in &self.loaded {
            if let Some(f) = table.get(kernel.name) {
                return Ok(f);
            }
        }
        Err(CudaError::Driver(format!(
            "no loaded function for {}",
            kernel.name
        )))
    }

    /// Launch an attention kernel on `grid` with the block its manifest entry
    /// recorded.
    ///
    /// # Safety
    /// `stream` must belong to the context that is current on this thread, and
    /// `params` must be the `kernel_abi` pack for this same kernel.
    pub unsafe fn launch(
        &self,
        kernel: &AttentionKernel,
        grid: (u32, u32, u32),
        params: &mut [*mut c_void],
        stream: CUstream,
    ) -> Result<(), CudaError> {
        let f = self.function(kernel)?;
        // SAFETY: `f` came from this type's own `load`, so it is live in the
        // driver's current context, and the grid comes from `kernel_abi`.
        unsafe {
            self.driver.launch(
                f,
                grid,
                (kernel.block_threads(), 1, 1),
                kernel.shared,
                stream,
                params,
            )
        }
    }
}

/// The committed fused GDN decode cubin, loaded and ready to launch. A fourth
/// contract (decision 60 pattern): `num_warps = 1` and a grid computed per call
/// from the baked GDN geometry, so its launch shape cannot come from the cubin
/// alone.
pub struct GdnKernels {
    driver: Arc<Driver>,
    abi: LinearAbi,
    loaded: Vec<(CUmodule, FunctionTable)>,
}

// SAFETY: as `CudaKernels`: the handles are immutable once `load` returns and
// the driver behind them is thread-safe.
unsafe impl Send for GdnKernels {}
unsafe impl Sync for GdnKernels {}

impl GdnKernels {
    /// Load every linear-attention kernel in `abi` from the committed cubins.
    pub fn load(driver: Arc<Driver>, abi: LinearAbi) -> Result<Self, CudaError> {
        let mut loaded = Vec::new();
        for name in LinearAbi::kernel_names() {
            let image = crate::infrastructure::kernel_abi::linear_cubin(name)
                .map_err(|e| CudaError::Driver(format!("cubin: {e}")))?;
            let module = driver.load_module(image)?;
            let kernel = abi
                .kernel(name)
                .map_err(|e| CudaError::Driver(format!("linear abi: {e}")))?;
            // SAFETY: `module` was just loaded by `driver`, whose context is
            // current on this thread.
            let f = unsafe { driver.function(module, &kernel.entry)? };
            let mut table = FunctionTable::default();
            table.push(kernel.name, f);
            loaded.push((module, table));
        }
        Ok(Self {
            driver,
            abi,
            loaded,
        })
    }

    pub fn abi(&self) -> &LinearAbi {
        &self.abi
    }

    fn function(&self, kernel: &LinearKernel) -> Result<CUfunction, CudaError> {
        for (_, table) in &self.loaded {
            if let Some(f) = table.get(kernel.name) {
                return Ok(f);
            }
        }
        Err(CudaError::Driver(format!(
            "no loaded function for {}",
            kernel.name
        )))
    }

    /// Launch a linear-attention kernel on `grid` with the block its manifest
    /// entry recorded.
    ///
    /// # Safety
    /// `stream` must belong to the context that is current on this thread, and
    /// `params` must be the `kernel_abi` pack for this same kernel.
    pub unsafe fn launch(
        &self,
        kernel: &LinearKernel,
        grid: (u32, u32, u32),
        params: &mut [*mut c_void],
        stream: CUstream,
    ) -> Result<(), CudaError> {
        let f = self.function(kernel)?;
        // SAFETY: `f` came from this type's own `load`, so it is live in the
        // driver's current context, and the grid comes from `kernel_abi`.
        unsafe {
            self.driver.launch(
                f,
                grid,
                (kernel.block_threads(), 1, 1),
                kernel.shared,
                stream,
                params,
            )
        }
    }
}

/// The committed staging cubins (`lru_ensure`, `index_copy`), loaded and ready
/// to launch. Separate from `CudaKernels` because it is a separate ABI
/// (decision 46): fixed grids, their own constexprs, no `n`/`block` tile.
pub struct StagingKernels {
    driver: Arc<Driver>,
    abi: StagingAbi,
    loaded: Vec<(CUmodule, FunctionTable)>,
}

impl StagingKernels {
    /// Load every staging kernel from the committed cubins. The driver is
    /// shared with the grouped kernels so there is one handle per process.
    pub fn load(driver: Arc<Driver>, abi: StagingAbi) -> Result<Self, CudaError> {
        let mut loaded = Vec::new();
        for name in StagingAbi::kernel_names() {
            let image = crate::infrastructure::kernel_abi::staging_cubin(name)
                .map_err(|e| CudaError::Driver(format!("cubin: {e}")))?;
            let module = driver.load_module(image)?;
            let kernel = abi
                .kernel(name)
                .map_err(|e| CudaError::Driver(format!("staging abi: {e}")))?;
            // SAFETY: `module` was just loaded by `driver`, whose context is
            // current on this thread.
            let f = unsafe { driver.function(module, &kernel.entry)? };
            let mut table = FunctionTable::default();
            table.push(kernel.name, f);
            loaded.push((module, table));
        }
        Ok(Self {
            driver,
            abi,
            loaded,
        })
    }

    pub fn abi(&self) -> &StagingAbi {
        &self.abi
    }

    fn function(&self, kernel: &StagingKernel) -> Result<CUfunction, CudaError> {
        for (_, table) in &self.loaded {
            if let Some(f) = table.get(kernel.name) {
                return Ok(f);
            }
        }
        Err(CudaError::Driver(format!(
            "no loaded function for {}",
            kernel.name
        )))
    }

    /// Launch a staging kernel whose grid depends on the live step (the CSR
    /// build runs one program per resident slot), keeping the cubin's block
    /// shape and shared memory.
    ///
    /// # Safety
    /// As [`StagingKernels::launch`]: `stream` must be current-context and
    /// `params` the `kernel_abi` pack for this same kernel.
    pub unsafe fn launch_grid(
        &self,
        kernel: &StagingKernel,
        grid: (u32, u32, u32),
        params: &mut [*mut c_void],
        stream: CUstream,
    ) -> Result<(), CudaError> {
        let f = self.function(kernel)?;
        // SAFETY: `f` is live in the driver's current context, the grid is the
        // caller's live slot count, and the block shape comes from the cubin.
        unsafe {
            self.driver.launch(
                f,
                grid,
                (kernel.block_threads(), 1, 1),
                kernel.shared,
                stream,
                params,
            )
        }
    }

    /// Launch a staging kernel with its own fixed grid and block shape.
    ///
    /// # Safety
    /// `stream` must belong to the context that is current on this thread, and
    /// `params` must be the `kernel_abi` pack for this same kernel.
    pub unsafe fn launch(
        &self,
        kernel: &StagingKernel,
        params: &mut [*mut c_void],
        stream: CUstream,
    ) -> Result<(), CudaError> {
        let f = self.function(kernel)?;
        // SAFETY: `f` came from this type's own `load`, so it is live in the
        // driver's current context, and the grid/block come from the cubin's own
        // manifest entry.
        unsafe {
            self.driver.launch(
                f,
                (kernel.grid[0], kernel.grid[1], kernel.grid[2]),
                (kernel.block_threads(), 1, 1),
                kernel.shared,
                stream,
                params,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// The GEMV pair shares one Triton entry, so the table has to key on the
    /// manifest name. Handles here are fabricated integers, never dereferenced:
    /// the table is pure lookup.
    #[test]
    fn the_function_table_keys_on_kernel_name_and_not_on_entry() {
        use crate::infrastructure::kernel_abi::{GEMV_DOWN, GEMV_GATE_UP};

        let abi = Abi::committed().expect("committed manifest");
        let mut table = FunctionTable::default();
        for (i, name) in Abi::kernel_names().iter().enumerate() {
            table.push(name, (i + 1) as CUfunction);
        }

        let gate_up = table.get(GEMV_GATE_UP);
        let down = table.get(GEMV_DOWN);
        assert!(gate_up.is_some() && down.is_some());
        assert_ne!(
            gate_up,
            down,
            "the two GEMV cubins share the entry {} and must not share a handle",
            abi.kernel(GEMV_DOWN).unwrap().entry
        );

        for name in Abi::kernel_names() {
            let handle = table.get(name).expect(name);
            assert!(!handle.is_null(), "{name} resolved to a null handle");
        }

        assert!(table.get("no_such_kernel").is_none());
    }

    /// The riskiest part of the FFI is a symbol name that does not exist, and the
    /// sandbox's driver mount is loadable even though it has no device, so this
    /// test actually resolves all of them.
    #[test]
    fn every_symbol_resolves_and_the_driver_reports_itself() {
        let Ok(driver) = Driver::open() else {
            eprintln!("no libcuda on this box, skipping");
            return;
        };
        assert!(driver.driver_version().is_ok());
        // No device here is fine; that is decision 36's whole premise.
        match driver.init() {
            Ok(()) => assert!(driver.device_count().unwrap_or(0) >= 1),
            Err(e) => {
                assert!(e.is_no_device(), "unexpected init error: {e}");
                assert_eq!(e.to_string(), "cuda: no device");
            }
        }
    }

    #[test]
    fn a_missing_library_is_a_load_error() {
        let err = Driver::open().err();
        if let Some(e) = err {
            assert!(matches!(e, CudaError::Load { .. }), "{e}");
            assert!(e.to_string().contains(LIB), "{e}");
        }
    }

    /// The hand-written batch attribute must match CUDA 13's
    /// `CUmemcpyAttributes_v1` layout, or the driver reads the flags word where
    /// a location hint belongs. This is pure `repr(C)` arithmetic, no device.
    #[test]
    fn the_batch_attribute_layout_matches_cuda_13() {
        use std::mem::{offset_of, size_of};

        assert_eq!(size_of::<MemcpyAttributes>(), 24);
        assert_eq!(offset_of!(MemcpyAttributes, src_access_order), 0);
        assert_eq!(offset_of!(MemcpyAttributes, src_id), 8);
        assert_eq!(offset_of!(MemcpyAttributes, dst_id), 16);
        assert_eq!(offset_of!(MemcpyAttributes, flags), 20);
    }

    /// A `CudaError` from `check` must carry the driver's own string, because
    /// that is what tells us which call failed. CUresult 999 is
    /// `CUDA_ERROR_UNKNOWN`, and its string is driver-provided.
    #[test]
    fn a_driver_error_carries_the_driver_string() {
        let Ok(driver) = Driver::open() else {
            return;
        };
        let err = driver.check(999).unwrap_err();
        assert!(matches!(err, CudaError::Driver(_)), "{err}");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn no_device_is_its_own_variant() {
        let Ok(driver) = Driver::open() else {
            return;
        };
        let r = driver.check(100);
        assert!(matches!(r, Err(CudaError::NoDevice)), "{r:?}");
    }

    /// `cuStreamBeginCapture` refuses the legacy stream (decision 39), and the
    /// refusal has to come from us before the call: the whole stream story
    /// hangs on capture needing a real side stream, and a null stream that
    /// silently reached the driver would blur where the rule is enforced.
    #[test]
    #[serial]
    fn the_legacy_stream_is_refused_as_a_capture_target() {
        let Ok(driver) = Driver::open() else {
            eprintln!("no libcuda on this box, skipping");
            return;
        };
        // SAFETY: null names the legacy stream, which `begin_capture` rejects
        // without touching the driver at all.
        let err = unsafe { driver.begin_capture(std::ptr::null_mut(), CAPTURE_MODE_GLOBAL) }
            .expect_err("the legacy stream cannot be captured");
        assert!(err.to_string().contains("legacy stream"), "{err}");
    }

    /// The full capture round trip on a real side stream: create, begin, end,
    /// instantiate, launch, synchronize, destroy. The graph is empty on
    /// purpose -- the four MoE launches need staged banks, which the gated
    /// harness provides -- so what this proves is the FFI plumbing itself,
    /// every step, on a device. Skips (loudly) without libcuda or a device.
    #[test]
    #[serial]
    fn a_side_stream_round_trips_through_capture() {
        let Ok(driver) = Driver::open() else {
            eprintln!("no libcuda on this box, skipping");
            return;
        };
        if driver.init().is_err() || driver.device_count().unwrap_or(0) == 0 {
            eprintln!("no CUDA device, skipping");
            return;
        }
        driver.attach_primary_context(0).expect("primary context");
        // SAFETY: the context is current on this thread as of the line above
        // and the stream is created here, so it is ours to destroy.
        let stream = unsafe { driver.create_stream() }.expect("side stream");
        assert!(
            !stream.is_null(),
            "a side stream must not be the legacy stream"
        );
        // Nothing is queued on the legacy stream, so capture inherits no
        // cross-stream dependency (decision 39's stream story).
        unsafe { driver.synchronize(std::ptr::null_mut()) }.expect("legacy sync");
        unsafe { driver.begin_capture(stream, CAPTURE_MODE_GLOBAL) }.expect("begin capture");
        let graph = unsafe { driver.end_capture(stream) }.expect("end capture");
        let exec = unsafe { driver.graph_instantiate(graph) }.expect("instantiate");
        unsafe { driver.graph_launch(exec, stream) }.expect("launch");
        unsafe { driver.synchronize(stream) }.expect("sync");
        unsafe { driver.graph_exec_destroy(exec) }.expect("destroy exec");
        unsafe { driver.graph_destroy(graph) }.expect("destroy graph");
        unsafe { driver.destroy_stream(stream) }.expect("destroy stream");
    }

    /// The prefill overlap's ordering primitive: an event recorded on one
    /// stream and waited on by another, then both synchronized and torn down.
    /// Skips (loudly) without libcuda or a device.
    #[test]
    #[serial]
    fn an_event_orders_two_streams_and_round_trips() {
        let Ok(driver) = Driver::open() else {
            eprintln!("no libcuda on this box, skipping");
            return;
        };
        if driver.init().is_err() || driver.device_count().unwrap_or(0) == 0 {
            eprintln!("no CUDA device, skipping");
            return;
        }
        driver.attach_primary_context(0).expect("primary context");
        // SAFETY: the context is current and both streams/events are created
        // here, so they are ours to destroy.
        let a = unsafe { driver.create_stream() }.expect("stream a");
        let b = unsafe { driver.create_stream() }.expect("stream b");
        let event = unsafe { driver.create_event() }.expect("event");
        assert!(!event.is_null());
        unsafe { driver.record_event(event, a) }.expect("record on a");
        unsafe { driver.stream_wait_event(b, event) }.expect("b waits the event");
        unsafe { driver.synchronize(a) }.expect("sync a");
        unsafe { driver.synchronize(b) }.expect("sync b");
        unsafe { driver.destroy_event(event) }.expect("destroy event");
        unsafe { driver.destroy_stream(a) }.expect("destroy a");
        unsafe { driver.destroy_stream(b) }.expect("destroy b");
        // A null handle is an error rather than a silent no-op.
        assert!(unsafe { driver.destroy_event(std::ptr::null_mut()) }.is_err());
    }

    /// The owned wrappers create, order and drop their resources through RAII.
    /// Two copies of each on one driver prove the Drop impls do not disturb the
    /// others (a double-free or a stale handle would show as a later failure).
    /// Skips (loudly) without libcuda or a device.
    #[test]
    #[serial]
    fn owned_streams_and_events_round_trip_and_drop() {
        let Ok(driver) = Driver::open() else {
            eprintln!("no libcuda on this box, skipping");
            return;
        };
        if driver.init().is_err() || driver.device_count().unwrap_or(0) == 0 {
            eprintln!("no CUDA device, skipping");
            return;
        }
        driver.attach_primary_context(0).expect("primary context");
        let driver = Arc::new(driver);
        {
            let stream = driver
                .owned_stream(STREAM_NON_BLOCKING)
                .expect("copy stream");
            assert!(!stream.raw().is_null());
            let ready = driver.owned_event().expect("ready event");
            let begin = driver.owned_event().expect("begin event");
            // SAFETY: `stream` is live in the current context.
            unsafe {
                begin.record(stream.raw()).expect("record begin");
                ready.wait_on(stream.raw()).expect("stream waits ready");
                ready.record(stream.raw()).expect("record ready");
            }
            unsafe { driver.synchronize(stream.raw()) }.expect("sync");
        }
        // A second pair, made after the first dropped, must be equally usable.
        let stream = driver.owned_stream(0).expect("blocking stream");
        let event = driver.owned_event().expect("event");
        // SAFETY: `stream` is live in the current context.
        unsafe {
            event.record(stream.raw()).expect("record");
            event.wait_on(stream.raw()).expect("wait");
        }
        unsafe { driver.synchronize(stream.raw()) }.expect("sync");
    }

    /// The pinned host banks are the source side of the copy kernel, which
    /// reaches them through their device alias, so that alias has to be a real,
    /// readable `CUdeviceptr`. Filling from the host and reading back through
    /// `cuMemcpyDtoH` at the device address only succeeds if `DEVICEMAP` mapped
    /// it. Skips (loudly) without libcuda or a device.
    #[test]
    #[serial]
    fn a_pinned_buffer_has_a_readable_device_alias() {
        let Ok(driver) = Driver::open() else {
            eprintln!("no libcuda on this box, skipping");
            return;
        };
        if driver.init().is_err() || driver.device_count().unwrap_or(0) == 0 {
            eprintln!("no CUDA device, skipping");
            return;
        }
        driver.attach_primary_context(0).expect("primary context");
        let driver = Arc::new(driver);
        let mut buf = driver.alloc_pinned(4096).expect("pinned allocation");
        assert_eq!(buf.len(), 4096);
        assert!(!buf.is_empty());
        for (i, b) in buf.as_mut_slice().iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        assert_ne!(buf.ptr, 0, "the alias must be a real device address");
        let mut out = vec![0u8; 4096];
        // SAFETY: `buf.ptr` aliases the live 4096-byte pinned allocation and
        // `out` has room for exactly that many bytes.
        unsafe { driver.copy_d2h(&mut out, buf.ptr) }.expect("read through the alias");
        assert_eq!(out, buf.as_slice());
    }
}
