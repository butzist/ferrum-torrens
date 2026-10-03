// The Triton NVFP4 kernel adapter: an embedded python interpreter (pyo3) that
// owns the device slot cache and calls `ferrum_kernel.nvfp4_moe.grouped_moe`.
//
// Decision 25: the domain port speaks host slices, so this adapter owns every
// CUDA buffer the kernel touches. Nothing is shared with libtorch by pointer --
// two libtorch builds in one process have no business agreeing on C++ ABI -- and
// the only copies are `x` in and `out` out per layer (4 KB/token), while the
// 2.6 MB expert slabs are staged once into the slot cache and then read by the
// kernel straight out of VRAM.
//
// The device slot cache holds ONE layer: 256 experts x ~2.6 MB fits once in
// 8 GB, not forty times, so a layer switch drops the previous layer's slots
// (`take_dropped_layer`, replayed into `ExpertCache::forget_layer`) and
// everything re-misses.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict, PyTuple};

use crate::domain::expert_kernel::{
    ExpertGemm, ExpertSlabSource, ExpertSlotStore, MoEError, MoEGroup, Part, Slab,
};

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

fn err(context: &str, e: impl std::fmt::Display) -> MoEError {
    MoEError::Bad(format!("{context}: {e}"))
}

/// torch is a manylinux wheel whose RPATH names paths that do not exist in a
/// nix store, so the loader cannot find `libstdc++.so.6` unless one is on
/// `LD_LIBRARY_PATH`. That is the overwhelmingly common first failure, and the
/// loader message says nothing about where to put it, so say it here.
fn loader_hint(e: &pyo3::PyErr) -> String {
    let text = e.to_string();
    if text.contains("libstdc++.so.6") || text.contains("cannot open shared object file") {
        " (LD_LIBRARY_PATH needs /run/opengl-driver/lib and the dir holding a \
         matching libstdc++.so.6: the devShell's shellHook exports both; the \
         loader reads the variable when the process starts, so set it before \
         cargo launches the test)"
            .to_string()
    } else {
        String::new()
    }
}

// The python helpers. Every `?` below maps a PyErr into the port error with the
// operation that raised it, because the domain core knows nothing about python.

fn attribute<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    ctx: &str,
) -> Result<Bound<'py, PyAny>, MoEError> {
    obj.getattr(name).map_err(|e| err(ctx, e))
}

fn method<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    args: Bound<'py, PyTuple>,
    ctx: &str,
) -> Result<Bound<'py, PyAny>, MoEError> {
    obj.call_method1(name, args)
        .map(Bound::into_any)
        .map_err(|e| err(ctx, e))
}

fn tuple1<'py>(py: Python<'py>, arg: Bound<'py, PyAny>) -> Result<Bound<'py, PyTuple>, MoEError> {
    PyTuple::new(py, [arg]).map_err(|e| err("argument tuple", e))
}

/// `torch.device(type, ordinal)`. Passing `torch.cuda.device(i)` instead hands
/// over the context-manager class of the same name, which has no `.type`.
fn device_args<'py>(py: Python<'py>, ordinal: i64) -> Result<Bound<'py, PyTuple>, MoEError> {
    let ty = infallible("cuda".into_pyobject(py)).into_any();
    let ordinal = infallible(ordinal.into_pyobject(py)).into_any();
    PyTuple::new(py, [ty, ordinal]).map_err(|e| err("argument tuple", e))
}

fn infallible<T>(r: Result<T, std::convert::Infallible>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => match e {},
    }
}

fn shape<'py>(py: Python<'py>, dims: &[usize]) -> Result<Bound<'py, PyAny>, MoEError> {
    PyTuple::new(py, dims.iter().copied())
        .map(Bound::into_any)
        .map_err(|e| err("shape tuple", e))
}

/// A CPU tensor over a copy of `bytes`, pinned by a python bytearray so
/// `frombuffer` does not warn about a read-only buffer. The copy is the H2D
/// staging: `to(device)` is the transfer.
///
/// `dtype` is keyword-only in torch's signature, so the dict has to travel in
/// pyo3's kwargs slot. Handing `(raw, &kwargs)` to `call_method` built a two
/// element *args* tuple instead, and torch rejected it with "frombuffer() takes
/// from 2 to 1 positional arguments but 2 were given".
fn host_tensor<'py>(
    py: Python<'py>,
    torch: &Bound<'py, PyAny>,
    dtype: &str,
    bytes: &[u8],
) -> Result<Bound<'py, PyAny>, MoEError> {
    let raw = PyByteArray::new(py, bytes);
    let kwargs = PyDict::new(py);
    kwargs
        .set_item("dtype", attribute(torch, dtype, "torch dtype")?)
        .map_err(|e| err("frombuffer dtype", e))?;
    torch
        .call_method("frombuffer", (raw,), Some(&kwargs))
        .map_err(|e| err("torch.frombuffer", e))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn to_device<'py>(t: Bound<'py, PyAny>, device: &str) -> Result<Bound<'py, PyAny>, MoEError> {
    t.call_method1("to", (device,))
        .map_err(|e| err("tensor.to(cuda)", e))
}

/// One field of a slab, ready to be staged: which device bank, the host bytes
/// and the shape the kernel expects inside one slot.
/// One python-side NVFP4 bank set: `packed [slots, rows, cols/2]`,
/// `scale [slots, rows, cols/16]`, `global [slots, rows]`.
type Stage<'a> = (&'a Py<PyAny>, &'a [u8], &'static str, Vec<usize>);

struct DeviceBanks {
    packed: Py<PyAny>,
    scale: Py<PyAny>,
    global: Py<PyAny>,
}

/// `[slots, rows, cols / 2]` uint8, `[slots, rows, cols / 16]` e4m3 and
/// `[slots, rows]` f16, so one expert is a contiguous `banks[slot]` and the
/// kernel indexes the slot directly. `device` is a torch device string; a CPU
/// test drives the same allocation the device path does.
fn new_banks<'py>(
    py: Python<'py>,
    torch: &Bound<'py, PyAny>,
    slots: usize,
    rows: usize,
    cols: usize,
    device: &str,
) -> Result<DeviceBanks, MoEError> {
    let empty = |third: Option<usize>, dtype: &str| -> Result<Py<PyAny>, MoEError> {
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("device", device)
            .map_err(|e| err("device kwarg", e))?;
        kwargs
            .set_item("dtype", attribute(torch, dtype, "torch dtype")?)
            .map_err(|e| err("dtype kwarg", e))?;
        let dims: &[usize] = match third {
            Some(third) => &[slots, rows, third],
            None => &[slots, rows],
        };
        let t = torch
            .call_method("empty", (shape(py, dims)?,), Some(&kwargs))
            .map_err(|e| err("torch.empty", e))?;
        Ok(t.unbind())
    };
    Ok(DeviceBanks {
        packed: empty(Some(cols / 2), "uint8")?,
        scale: empty(Some(cols / 16), "float8_e4m3fn")?,
        global: empty(None, "float16")?,
    })
}

/// Copy one expert slab into `slot`. The bank rows are already
/// `[rows, cols / 2]`, so the staged CPU tensor has to be viewed to the same
/// shape: `copy_` does not flatten, and a flat source is a shape error.
fn stage_into_slot<'py>(
    py: Python<'py>,
    torch: &Bound<'py, PyAny>,
    banks: &DeviceBanks,
    slot: usize,
    slab: &Slab,
    rows: usize,
    cols: usize,
) -> Result<(), MoEError> {
    let parts: [Stage<'_>; 3] = [
        (
            &banks.packed,
            slab.packed.as_slice(),
            "uint8",
            vec![rows, cols / 2],
        ),
        (
            &banks.scale,
            slab.scale.as_slice(),
            "float8_e4m3fn",
            vec![rows, cols / 16],
        ),
        (&banks.global, slab.global.as_slice(), "float16", vec![rows]),
    ];
    for (bank, bytes, dtype, dims) in parts {
        let src = method(
            &host_tensor(py, torch, dtype, bytes)?,
            "view",
            tuple1(py, shape(py, &dims)?)?,
            &format!("{dtype} slab view"),
        )?;
        let slot_view = bank
            .bind(py)
            .get_item(slot)
            .map_err(|e| err("bank[slot]", e))?;
        slot_view
            .call_method1("copy_", (src,))
            .map_err(|e| err("copy_ into the slot cache", e))?;
    }
    Ok(())
}

/// `[tokens, hidden]` on whatever device the kernel wrote, back as a flat
/// `Vec<f32>`. `tolist` is the crossing point: pyo3 cannot extract a sequence
/// straight out of a tensor, and only the list gives it one.
fn to_f32_vec<'py>(t: &Bound<'py, PyAny>) -> Result<Vec<f32>, MoEError> {
    let host = t
        .call_method1("cpu", ())
        .map_err(|e| err("result.cpu()", e))?;
    let flat = host
        .call_method1("reshape", (-1,))
        .map_err(|e| err("result.reshape(-1)", e))?;
    let list = flat
        .call_method0("tolist")
        .map_err(|e| err("result.tolist()", e))?;
    list.extract().map_err(|e| err("result to Vec<f32>", e))
}

/// The interpreter pyo3 linked is not the venv's, so `torch`/`triton` live
/// nowhere on its `sys.path` until their site-packages go on, and this repo's
/// `ferrum_kernel` package is a plain directory that needs the same treatment.
pub fn prepend_kernel_paths<'py>(
    py: Python<'py>,
    site_packages: &[PathBuf],
    python_path: &Path,
) -> Result<(), MoEError> {
    for dir in site_packages {
        if !dir.is_dir() {
            return Err(err(
                "kernel site-packages",
                format!(
                    "{} does not exist: recreate FreeToken/.venv (see AGENTS.md) or point \
                     FERRUM_KERNEL_SITE_PACKAGES at the torch/triton install",
                    dir.display()
                ),
            ));
        }
    }
    let sys = py.import("sys").map_err(|e| err("import sys", e))?;
    let path = attribute(&sys, "path", "sys.path")?;
    for dir in site_packages
        .iter()
        .chain(std::iter::once(&python_path.to_path_buf()))
    {
        let s = dir
            .to_str()
            .ok_or_else(|| err("kernel path", format!("{} is not utf-8", dir.display())))?;
        let there: bool = path
            .call_method1("__contains__", (s,))
            .map_err(|e| err("sys.path lookup", e))?
            .extract()
            .map_err(|e| err("sys.path lookup", e))?;
        if !there {
            path.call_method1("append", (s,))
                .map_err(|e| err("sys.path.append", e))?;
        }
    }
    Ok(())
}

/// The embedded interpreter plus the imported kernel module. Constructing this
/// imports torch, which is expensive, so the engine keeps exactly one.
pub struct TritonRuntime {
    kernel: Py<PyAny>,
    device: String,
}

impl TritonRuntime {
    /// `python_path` holds this repo's `ferrum_kernel` package; `site_packages`
    /// holds the `torch`/`triton` install, because the interpreter pyo3 linked
    /// is not the venv's interpreter.
    pub fn new(python_path: &Path, site_packages: &[PathBuf]) -> Result<Self, MoEError> {
        Python::attach(|py| {
            prepend_kernel_paths(py, site_packages, python_path)?;
            let module = py
                .import("ferrum_kernel.nvfp4_moe")
                .map_err(|e| err("import ferrum_kernel.nvfp4_moe (kernel.python_path?)", e))?;
            let torch = py
                .import("torch")
                .map_err(|e| err(&format!("import torch{}", loader_hint(&e)), e))?;
            let cuda = attribute(&torch, "cuda", "torch.cuda")?;
            let index = method(
                &cuda,
                "current_device",
                PyTuple::empty(py),
                "torch.cuda.current_device",
            )?;
            let ordinal: i64 = index.extract().map_err(|e| err("device index", e))?;
            let device_obj = method(&torch, "device", device_args(py, ordinal)?, "torch.device")?;
            module
                .call_method1("require_native_e4m3", (device_obj,))
                .map_err(|e| err("require_native_e4m3", e))?;
            let ordinal: usize = ordinal.try_into().map_err(|_| {
                err(
                    "device index",
                    format!("cuda ordinal {ordinal} does not fit in usize"),
                )
            })?;
            Ok(TritonRuntime {
                kernel: module.unbind().into(),
                device: format!("cuda:{ordinal}"),
            })
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }
}

/// Device slot cache for one layer plus the two domain ports.
pub struct TritonMoE {
    runtime: Arc<TritonRuntime>,
    source: Arc<dyn ExpertSlabSource>,
    shapes: ExpertShapes,
    slots: usize,
    layer: Option<usize>,
    gate_up: Option<DeviceBanks>,
    down: Option<DeviceBanks>,
    dropped_layer: Option<usize>,
}

impl TritonMoE {
    pub fn new(
        runtime: Arc<TritonRuntime>,
        source: Arc<dyn ExpertSlabSource>,
        shapes: ExpertShapes,
        slots: usize,
    ) -> Self {
        Self {
            runtime,
            source,
            shapes,
            slots,
            layer: None,
            gate_up: None,
            down: None,
            dropped_layer: None,
        }
    }

    pub fn shapes(&self) -> ExpertShapes {
        self.shapes
    }

    /// The layer whose experts the cache currently holds.
    pub fn layer(&self) -> Option<usize> {
        self.layer
    }

    /// Set on the load that switched layers: the domain cache has to forget the
    /// previous layer's slots, since its bytes are gone.
    pub fn take_dropped_layer(&mut self) -> Option<usize> {
        self.dropped_layer.take()
    }

    /// Switch the device banks to `layer`, reporting the layer whose bytes were
    /// dropped (decision 27). The caller replays that into
    /// `ExpertCache::forget_layer` before planning, so the slots it still
    /// believes in are gone exactly when the plan is computed.
    pub fn begin_layer(&mut self, layer: usize) -> Result<Option<usize>, MoEError> {
        if self.layer == Some(layer) {
            return Ok(None);
        }
        let dropped = self.layer.take();
        self.gate_up = None;
        self.down = None;
        self.dropped_layer = dropped;
        self.layer = Some(layer);
        Ok(dropped)
    }

    /// Pull both slabs of one expert from the source and stage them.
    pub fn load(&mut self, layer: usize, expert: u32, slot: usize) -> Result<(), MoEError> {
        // Idempotent: a caller that only stages one expert still gets a
        // consistent cache, and the dropped layer is still reported.
        self.begin_layer(layer)?;
        for part in [Part::GateUp, Part::Down] {
            let slab = self
                .source
                .slab(layer, expert, part)
                .map_err(|e| err(&format!("slab L{layer} E{expert} {part:?}"), e))?;
            self.load_slab(layer, slot, part, &slab)?;
        }
        Ok(())
    }

    /// Bytes per expert: packed + scales + globals, so the engine can check the
    /// 8 GB budget before allocating.
    pub fn expert_bytes(&self) -> usize {
        let per = |(rows, cols): (usize, usize)| rows * cols / 2 + rows * cols / 16 + rows * 2;
        per(self.shapes.slab_shape(Part::GateUp)) + per(self.shapes.slab_shape(Part::Down))
    }

    /// Bytes the whole cache needs on the device.
    pub fn cache_bytes(&self) -> usize {
        self.slots * self.expert_bytes()
    }

    /// Allocate the banks for the layer `begin_layer` just selected.
    fn banks_for(&mut self) -> Result<(), MoEError> {
        if self.gate_up.is_some() {
            return Ok(());
        }
        let shapes = self.shapes;
        let slots = self.slots;
        let device = self.runtime.device.clone();
        Python::attach(|py| {
            let torch = py.import("torch").map_err(|e| err("import torch", e))?;
            let part = |part: Part| -> Result<DeviceBanks, MoEError> {
                let (rows, cols) = shapes.slab_shape(part);
                new_banks(py, &torch, slots, rows, cols, &device)
            };
            self.gate_up = Some(part(Part::GateUp)?);
            self.down = Some(part(Part::Down)?);
            Ok(())
        })
    }

    /// Copy one slab into `slot`, the H2D half of `copy_missing`. Staged one
    /// expert at a time so a step's misses stay ordered and a failure names the
    /// expert it died on.
    pub fn load_slab(
        &mut self,
        _layer: usize,
        slot: usize,
        part: Part,
        slab: &Slab,
    ) -> Result<(), MoEError> {
        if slot >= self.slots {
            return Err(MoEError::SlotOutOfRange {
                slot,
                capacity: self.slots,
            });
        }
        let (rows, cols) = self.shapes.slab_shape(part);
        if slab.packed.len() != rows * cols / 2
            || slab.scale.len() != rows * cols / 16
            || slab.global.len() != rows * 2
        {
            return Err(MoEError::Bad(format!(
                "slab bytes packed {} scale {} global {} for {}x{}",
                slab.packed.len(),
                slab.scale.len(),
                slab.global.len(),
                rows,
                cols
            )));
        }
        self.banks_for()?;
        let banks = match part {
            Part::GateUp => self.gate_up.as_ref().unwrap(),
            Part::Down => self.down.as_ref().unwrap(),
        };
        Python::attach(|py| {
            let torch = py.import("torch").map_err(|e| err("import torch", e))?;
            stage_into_slot(py, &torch, banks, slot, slab, rows, cols)
        })
    }
}

/// The ports over the shared handle: the engine keeps one `Mutex<TritonMoE>` per
/// checkpoint and hands it out as both ports, so the plan and the kernel call
/// see the same banks without the caller juggling borrows.
impl ExpertSlotStore for Mutex<TritonMoE> {
    fn begin_layer(&self, layer: usize) -> Result<Option<usize>, MoEError> {
        self.lock()
            .map_err(|_| MoEError::Bad("triton slot cache poisoned".into()))?
            .begin_layer(layer)
    }

    fn load(&self, layer: usize, expert: u32, slot: usize) -> Result<(), MoEError> {
        self.lock()
            .map_err(|_| MoEError::Bad("triton slot cache poisoned".into()))?
            .load(layer, expert, slot)
    }

    fn slots(&self) -> usize {
        self.lock().map(|m| m.slots).unwrap_or(0)
    }
}

impl ExpertGemm for Mutex<TritonMoE> {
    fn grouped_moe<'a>(&self, req: &MoEGroup<'a>, out: &mut [f32]) -> Result<(), MoEError> {
        let me = self
            .lock()
            .map_err(|_| MoEError::Bad("triton slot cache poisoned".into()))?;
        me.grouped_moe(req, out)
    }
}

/// The kernel call itself, on the adapter. A caller that already holds the lock
/// (the gated test, a single threaded step) implements the port directly; the
/// engine reaches it through the mutex handle.
impl ExpertGemm for TritonMoE {
    fn grouped_moe<'a>(&self, req: &MoEGroup<'a>, out: &mut [f32]) -> Result<(), MoEError> {
        TritonMoE::grouped_moe(self, req, out)
    }
}

impl TritonMoE {
    pub fn grouped_moe<'a>(&self, req: &MoEGroup<'a>, out: &mut [f32]) -> Result<(), MoEError> {
        if self.layer.is_none() {
            return Err(MoEError::Bad(
                "grouped_moe before any expert was staged".into(),
            ));
        }
        let hidden = self.shapes.hidden;
        let tokens = req.x.len() / hidden;
        if tokens == 0 || out.len() < tokens * hidden || req.top_k == 0 {
            return Err(MoEError::Bad(format!(
                "grouped_moe buffers: x {} out {} for top_k {}",
                req.x.len(),
                out.len(),
                req.top_k
            )));
        }
        let (route_token, route_slot) = route_lists(req.routes, tokens, req.top_k)?;
        let device = self.runtime.device.clone();
        let gu = self
            .gate_up
            .as_ref()
            .ok_or_else(|| MoEError::Bad("no slot cache".into()))?;
        let dn = self
            .down
            .as_ref()
            .ok_or_else(|| MoEError::Bad("no slot cache".into()))?;
        let bank_args = (
            &gu.packed, &gu.scale, &gu.global, &dn.packed, &dn.scale, &dn.global,
        );
        let routes = tokens * req.top_k;
        let x = req.x.to_vec();
        let weights = req.weights[..routes].to_vec();
        let slot_list = route_slot;
        let token_list = route_token;
        Python::attach(|py| {
            let kernel = self.runtime.kernel.clone_ref(py);
            let torch = py.import("torch").map_err(|e| err("import torch", e))?;
            let x_t = to_device(
                method(
                    &host_tensor(py, &torch, "float32", &f32_bytes(&x))?,
                    "view",
                    tuple1(py, shape(py, &[tokens, hidden])?)?,
                    "x.view",
                )?,
                &device,
            )?;
            let slot_t = to_device(
                host_tensor(py, &torch, "int32", &i32_bytes(&slot_list))?,
                &device,
            )?;
            let token_t = to_device(
                host_tensor(py, &torch, "int32", &i32_bytes(&token_list))?,
                &device,
            )?;
            let weight_t = to_device(
                method(
                    &host_tensor(py, &torch, "float32", &f32_bytes(&weights))?,
                    "view",
                    tuple1(py, shape(py, &[routes])?)?,
                    "weight.view",
                )?,
                &device,
            )?;
            let result = kernel
                .bind(py)
                .call_method1(
                    "grouped_moe",
                    (
                        x_t,
                        bank_args.0,
                        bank_args.1,
                        bank_args.2,
                        bank_args.3,
                        bank_args.4,
                        bank_args.5,
                        slot_t,
                        token_t,
                        weight_t,
                    ),
                )
                .map_err(|e| err("ferrum_kernel grouped_moe", e))?;
            let values = to_f32_vec(&result)?;
            if values.len() < tokens * hidden {
                return Err(MoEError::Bad(format!(
                    "kernel returned {} values for {tokens} tokens",
                    values.len()
                )));
            }
            out[..tokens * hidden].copy_from_slice(&values[..tokens * hidden]);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The readback: pyo3 extracts a sequence only from a python list, never
    /// straight out of a tensor, so this is where a result comes home.
    #[test]
    fn a_result_tensor_reads_back_as_floats() {
        Python::attach(|py| {
            let sites = std::env::var("FERRUM_KERNEL_SITE_PACKAGES")
                .unwrap_or_else(|_| "FreeToken/.venv/lib/python3.12/site-packages".to_string());
            let dirs: Vec<PathBuf> = sites
                .split(':')
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect();
            if prepend_kernel_paths(py, &dirs, Path::new("python")).is_err() {
                println!("skipping readback test: sys.path setup failed");
                return;
            }
            let Ok(torch) = py.import("torch") else {
                println!("skipping readback test: torch is not importable here");
                return;
            };
            let rows = torch.call_method1("arange", (6,)).unwrap();
            let as_float = rows.call_method1("float", ()).unwrap();
            let shaped = as_float.call_method1("reshape", (2, 3)).unwrap();
            assert_eq!(
                to_f32_vec(&shaped).unwrap(),
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            );
        });
    }

    /// Staging runs on the CPU with the same code path the device uses, so the
    /// copy shapes are checkable without a GPU. `copy_` does not flatten: a
    /// flat staged tensor against a `[rows, cols / 2]` bank row is a shape
    /// error, which is what the device-only gate used to find.
    #[test]
    fn staging_one_slab_lands_in_its_slot_row() {
        Python::attach(|py| {
            let sites = std::env::var("FERRUM_KERNEL_SITE_PACKAGES")
                .unwrap_or_else(|_| "FreeToken/.venv/lib/python3.12/site-packages".to_string());
            let dirs: Vec<PathBuf> = sites
                .split(':')
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect();
            if prepend_kernel_paths(py, &dirs, Path::new("python")).is_err() {
                println!("skipping staging test: sys.path setup failed");
                return;
            }
            let Ok(torch) = py.import("torch") else {
                println!("skipping staging test: torch is not importable here");
                return;
            };
            const SLOTS: usize = 4;
            const ROWS: usize = 32;
            const COLS: usize = 64;
            let banks = new_banks(py, &torch, SLOTS, ROWS, COLS, "cpu").unwrap();
            let n = |i: usize| (i % 251) as u8;
            let slab = Slab {
                packed: (0..ROWS * COLS / 2).map(n).collect(),
                scale: (0..ROWS * COLS / 16).map(|i| n(i + 7)).collect(),
                global: (0..ROWS * 2).map(|i| n(i + 13)).collect(),
            };
            stage_into_slot(py, &torch, &banks, 2, &slab, ROWS, COLS).unwrap();

            let read = |t: &Py<PyAny>, dtype: &str, slot_len: usize| -> Vec<u8> {
                let raw = t.bind(py).call_method0("flatten").unwrap();
                let ints = raw
                    .call_method1("view", (attribute(&torch, dtype, "dtype").unwrap(),))
                    .unwrap();
                let at = (2 * slot_len).min(
                    ints.call_method0("numel")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                );
                (0..slot_len)
                    .map(|i| ints.get_item(at + i).unwrap().extract::<i64>().unwrap() as u8)
                    .collect()
            };
            let want_packed = slab.packed.clone();
            assert_eq!(
                read(&banks.packed, "uint8", ROWS * COLS / 2),
                want_packed,
                "packed bytes must land in slot 2"
            );
            // fp8 and f16 have no byte extraction here; compare the numeric
            // values the kernel would read instead.
            let values = |t: &Py<PyAny>, row: usize| -> Vec<f64> {
                t.bind(py)
                    .get_item(row)
                    .unwrap()
                    .call_method1("to", (attribute(&torch, "float32", "dtype").unwrap(),))
                    .unwrap()
                    .call_method0("flatten")
                    .unwrap()
                    .call_method0("tolist")
                    .unwrap()
                    .extract::<Vec<f64>>()
                    .unwrap()
            };
            let staged_scale = values(&banks.scale, 2);
            assert_eq!(staged_scale.len(), ROWS * COLS / 16);
            assert!(
                staged_scale.iter().any(|v| *v != 0.0),
                "scale row is not all zero"
            );
            let staged_global = values(&banks.global, 2);
            assert_eq!(staged_global.len(), ROWS);
            assert!(staged_global.iter().all(|v| v.is_finite()));
        });
    }

    /// A refreshed `FreeToken/` checkout takes the untracked venv with it, and
    /// the symptom is a bare "No module named 'torch'" from deep inside the
    /// kernel module. Name the missing directory instead.
    #[test]
    fn a_missing_site_packages_dir_is_reported_as_such() {
        let err = Python::attach(|py| {
            prepend_kernel_paths(
                py,
                &[PathBuf::from("/nonexistent/site-packages")],
                Path::new("python"),
            )
            .unwrap_err()
        });
        let msg = err.to_string();
        assert!(msg.contains("/nonexistent/site-packages"), "{msg}");
        assert!(msg.contains("does not exist"), "{msg}");
    }

    /// `torch.frombuffer` takes `dtype` as a keyword. Verified for real only
    /// where torch imports (devshell: the loader path comes from the
    /// shellHook); elsewhere the import is skipped rather than failing the
    /// CPU-only suite.
    #[test]
    fn host_tensor_passes_dtype_as_a_keyword() {
        Python::attach(|py| {
            let sites = std::env::var("FERRUM_KERNEL_SITE_PACKAGES")
                .unwrap_or_else(|_| "FreeToken/.venv/lib/python3.12/site-packages".to_string());
            let dirs: Vec<PathBuf> = sites
                .split(':')
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect();
            if prepend_kernel_paths(py, &dirs, Path::new("python")).is_err() {
                println!("skipping host_tensor test: sys.path setup failed");
                return;
            }
            let Ok(torch) = py.import("torch") else {
                println!("skipping host_tensor test: torch is not importable here");
                return;
            };
            let t = host_tensor(py, &torch, "uint8", &[1, 2, 3, 4]).unwrap();
            assert_eq!(
                t.call_method0("numel").unwrap().extract::<i64>().unwrap(),
                4
            );
            assert_eq!(
                t.getattr("dtype").unwrap().str().unwrap().to_string(),
                "torch.uint8"
            );
        });
    }

    /// The capability probe needs a real `torch.device`. `torch.cuda.device(i)`
    /// is the context-manager class of the same name and has no `.type`, so the
    /// probe blew up with `'device' object has no attribute 'type'`. Pin the
    /// argument shape so it cannot drift back to the context manager.
    #[test]
    fn device_args_are_the_type_and_ordinal() {
        Python::attach(|py| {
            let args = device_args(py, 0).unwrap();
            assert_eq!(args.len(), 2);
            assert_eq!(
                args.get_item(0).unwrap().extract::<String>().unwrap(),
                "cuda"
            );
            assert_eq!(args.get_item(1).unwrap().extract::<i64>().unwrap(), 0);
        });
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
    fn expert_and_cache_sizes_match_the_ftw_layout() {
        let me = TritonMoE::new(
            Arc::new(dummy_runtime()),
            Arc::new(EmptySource),
            ExpertShapes::new(1024, 512).unwrap(),
            4,
        );
        // 1024x1024 packed + scales + f16 globals, twice.
        let per_part = |(rows, cols): (usize, usize)| rows * cols / 2 + rows * cols / 16 + rows * 2;
        let want = per_part((1024, 1024)) + per_part((1024, 512));
        assert_eq!(me.expert_bytes(), want);
        assert_eq!(me.cache_bytes(), 4 * want);
    }

    #[test]
    fn load_rejects_a_slab_of_the_wrong_size() {
        let mut me = TritonMoE::new(
            Arc::new(dummy_runtime()),
            Arc::new(EmptySource),
            ExpertShapes::new(16, 16).unwrap(),
            2,
        );
        assert!(matches!(
            me.load_slab(0, 0, Part::GateUp, &Slab::default()),
            Err(MoEError::Bad(_))
        ));
        assert!(matches!(
            me.load_slab(0, 9, Part::GateUp, &Slab::default()),
            Err(MoEError::SlotOutOfRange {
                slot: 9,
                capacity: 2
            })
        ));
    }

    struct EmptySource;

    impl ExpertSlabSource for EmptySource {
        fn slab(&self, _layer: usize, _expert: u32, _part: Part) -> Result<Slab, MoEError> {
            Err(MoEError::Bad("no source".into()))
        }
    }

    /// A runtime handle without an interpreter: the tests that need no GPU
    /// never touch `kernel` or the device.
    fn dummy_runtime() -> TritonRuntime {
        TritonRuntime {
            kernel: Python::attach(|py| py.None()),
            device: "cuda:0".to_string(),
        }
    }
}
