//! The CUDA driver API, loaded by hand.
//!
//! Decision 36: the serving path must not import Python, and torch 2.11+ ships no
//! kernel-launch binding, so the driver is the only launcher left. `libcuda` is
//! dlopen'd and its entry points resolved by name rather than linked, for two
//! reasons: the crate then builds and tests with no CUDA at all, and the exact
//! version of the driver that ends up loaded is the one already on
//! `LD_LIBRARY_PATH` next to the libtorch this engine is using.
//!
//! Only what the MoE launch path needs is bound. Anything the driver reports as
//! a `CUresult` is an error: a launch that silently failed would show up as
//! garbage tokens much later, so every fallible call returns `Result` and the
//! driver string is carried in the error.

use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::Arc;

use thiserror::Error;

use crate::infrastructure::kernel_abi::{Abi, Kernel};

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

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

/// `CU_STREAM_CAPTURE_MODE_GLOBAL`: capture records every launch on the stream
/// and forbids the cross-stream synchronization that would make the graph's
/// boundaries ambiguous.
pub const CAPTURE_MODE_GLOBAL: u32 = 0;

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
type StreamCreateFn = unsafe extern "C" fn(*mut CUstream, u32) -> CUresult;
type StreamDestroyFn = unsafe extern "C" fn(CUstream) -> CUresult;
type StreamBeginCaptureFn = unsafe extern "C" fn(CUstream, u32) -> CUresult;
type StreamEndCaptureFn = unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult;
type GraphDestroyFn = unsafe extern "C" fn(CUgraph) -> CUresult;
type GraphInstantiateFn = unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult;
type GraphLaunchFn = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
type GraphExecDestroyFn = unsafe extern "C" fn(CUgraphExec) -> CUresult;

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
    cu_stream_create: StreamCreateFn,
    cu_stream_destroy: StreamDestroyFn,
    cu_stream_begin_capture: StreamBeginCaptureFn,
    cu_stream_end_capture: StreamEndCaptureFn,
    cu_graph_destroy: GraphDestroyFn,
    cu_graph_instantiate: GraphInstantiateFn,
    cu_graph_launch: GraphLaunchFn,
    cu_graph_exec_destroy: GraphExecDestroyFn,
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
    /// dlopen `libcuda` and resolve every symbol below. The only fallible part
    /// of the whole boundary that is not a `CUresult`: everything else goes
    /// through `check`.
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

    /// Retain the primary context of `device` and make it current on this
    /// thread. Retaining (not creating) is what libtorch does too, so a launch
    /// here and a launch from a libtorch op land in the same context.
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
    /// (libtorch's staging) and stream-0 work waits for this stream's, which is
    /// what lets a captured replay read buffers libtorch wrote without an
    /// explicit event.
    ///
    /// # Safety
    /// The caller must have a CUDA context current; the returned stream
    /// belongs to that context and must be destroyed with `destroy_stream`.
    pub unsafe fn create_stream(&self) -> Result<CUstream, CudaError> {
        let mut stream: CUstream = std::ptr::null_mut();
        // SAFETY: `stream` is a valid out pointer; 0 flags create a regular stream.
        self.check(unsafe { (self.fns.cu_stream_create)(&raw mut stream, 0) })?;
        Ok(stream)
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
        let mut exec: CUgraphExec = std::ptr::null_mut();
        // SAFETY: `exec` is a live out slot; flags 0 is the plain instantiation.
        self.check(unsafe { (self.fns.cu_graph_instantiate)(&raw mut exec, graph, 0) })?;
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
    /// cache's first slice; the steady state is libtorch tensors.
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

    fn get(&self, kernel: &Kernel) -> Option<CUfunction> {
        self.0
            .iter()
            .find(|(name, _)| *name == kernel.name)
            .map(|(_, f)| *f)
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
    /// shared, because the caller launches on libtorch's streams too and there
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

    pub fn function(&self, kernel: &Kernel) -> Result<CUfunction, CudaError> {
        for (_, table) in &self.loaded {
            if let Some(f) = table.get(kernel) {
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

#[cfg(test)]
mod tests {
    use super::*;

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

        let gate_up = table.get(abi.kernel(GEMV_GATE_UP).unwrap());
        let down = table.get(abi.kernel(GEMV_DOWN).unwrap());
        assert!(gate_up.is_some() && down.is_some());
        assert_ne!(
            gate_up,
            down,
            "the two GEMV cubins share the entry {} and must not share a handle",
            abi.kernel(GEMV_DOWN).unwrap().entry
        );

        for name in Abi::kernel_names() {
            let kernel = abi.kernel(name).unwrap();
            let handle = table.get(kernel).expect(name);
            assert!(!handle.is_null(), "{name} resolved to a null handle");
        }

        let unknown = Kernel {
            name: "no_such_kernel",
            entry: abi.kernel(GEMV_DOWN).unwrap().entry.clone(),
            args: Vec::new(),
            scratch_params: 0,
            num_warps: 1,
            shared: 0,
            n: 1,
            block: 1,
        };
        assert!(table.get(&unknown).is_none());
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
}
