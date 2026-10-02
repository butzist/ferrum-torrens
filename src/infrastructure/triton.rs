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

fn shape<'py>(py: Python<'py>, dims: &[usize]) -> Result<Bound<'py, PyAny>, MoEError> {
    PyTuple::new(py, dims.iter().copied())
        .map(Bound::into_any)
        .map_err(|e| err("shape tuple", e))
}

/// A CPU tensor over a copy of `bytes`, pinned by a python bytearray so
/// `frombuffer` does not warn about a read-only buffer. The copy is the H2D
/// staging: `to(device)` is the transfer.
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
        .call_method("frombuffer", (raw, &kwargs), None)
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
type Stage<'a> = (&'a Py<PyAny>, &'a [u8], &'static str, Vec<usize>);

/// One python-side NVFP4 bank set: `packed [slots, rows, cols/2]`,
/// `scale [slots, rows, cols/16]`, `global [slots, rows]`.
struct DeviceBanks {
    packed: Py<PyAny>,
    scale: Py<PyAny>,
    global: Py<PyAny>,
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
            let module = py
                .import("ferrum_kernel.nvfp4_moe")
                .map_err(|e| err("import ferrum_kernel.nvfp4_moe (kernel.python_path?)", e))?;
            let torch = py.import("torch").map_err(|e| err("import torch", e))?;
            let cuda = attribute(&torch, "cuda", "torch.cuda")?;
            let index = method(
                &cuda,
                "current_device",
                PyTuple::empty(py),
                "torch.cuda.current_device",
            )?;
            let device_obj = method(
                &cuda,
                "device",
                tuple1(py, index.clone().into_any())?,
                "torch.cuda.device",
            )?;
            module
                .call_method1("require_native_e4m3", (device_obj,))
                .map_err(|e| err("require_native_e4m3", e))?;
            let ordinal: usize = index.extract().map_err(|e| err("device index", e))?;
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
            let empty =
                |rows: usize, third: Option<usize>, dtype: &str| -> Result<Py<PyAny>, MoEError> {
                    let kwargs = PyDict::new(py);
                    kwargs
                        .set_item("device", device.as_str())
                        .map_err(|e| err("device kwarg", e))?;
                    kwargs
                        .set_item("dtype", attribute(&torch, dtype, "torch dtype")?)
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
            let part = |part: Part| -> Result<DeviceBanks, MoEError> {
                let (rows, cols) = shapes.slab_shape(part);
                Ok(DeviceBanks {
                    packed: empty(rows, Some(cols / 2), "uint8")?,
                    scale: empty(rows, Some(cols / 16), "float8_e4m3fn")?,
                    global: empty(rows, None, "float16")?,
                })
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
        let device = self.runtime.device.clone();
        let parts: [Stage<'_>; 3] = [
            (
                &banks.packed,
                slab.packed.as_slice(),
                "uint8",
                vec![rows, cols / 2],
            ),
            (
                &banks.scale,
                &slab.scale,
                "float8_e4m3fn",
                vec![rows, cols / 16],
            ),
            (&banks.global, slab.global.as_slice(), "float16", vec![rows]),
        ];
        Python::attach(|py| {
            let torch = py.import("torch").map_err(|e| err("import torch", e))?;
            for (bank, bytes, dtype, dims) in parts {
                let src = host_tensor(py, &torch, dtype, bytes)?;
                let slot_view = bank
                    .bind(py)
                    .get_item(slot)
                    .map_err(|e| err("bank[slot]", e))?;
                let view = slot_view
                    .call_method1("view", (shape(py, &dims)?,))
                    .map_err(|e| err("slot view", e))?;
                view.call_method1("copy_", (src,))
                    .map_err(|e| err("copy_ into the slot cache", e))?;
            }
            let _ = device;
            Ok(())
        })
    }
}

impl ExpertSlotStore for TritonMoE {
    /// Switch the device banks to `layer`, reporting the layer whose bytes were
    /// dropped (decision 27). The caller replays that into
    /// `ExpertCache::forget_layer` before planning, so the slots it still
    /// believes in are gone exactly when the plan is computed.
    fn begin_layer(&mut self, layer: usize) -> Result<Option<usize>, MoEError> {
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
    fn load(&mut self, layer: usize, expert: u32, slot: usize) -> Result<(), MoEError> {
        // Idempotent: a caller that only stages one expert still gets a
        // consistent cache, and `take_dropped_layer` still reports the switch.
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

    fn slots(&self) -> usize {
        self.slots
    }
}

impl ExpertGemm for Mutex<TritonMoE> {
    fn grouped_moe(&self, req: &MoEGroup<'_>, out: &mut [f32]) -> Result<(), MoEError> {
        let me = self
            .lock()
            .map_err(|_| MoEError::Bad("triton slot cache poisoned".into()))?;
        if me.layer.is_none() {
            return Err(MoEError::Bad(
                "grouped_moe before any expert was staged".into(),
            ));
        }
        let hidden = me.shapes.hidden;
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
        let device = me.runtime.device.clone();
        let gu = me
            .gate_up
            .as_ref()
            .ok_or_else(|| MoEError::Bad("no slot cache".into()))?;
        let dn = me
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
            let kernel = me.runtime.kernel.clone_ref(py);
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
            let host = result
                .call_method1("cpu", ())
                .map_err(|e| err("result.cpu()", e))?;
            let flat = host
                .call_method1("reshape", (-1,))
                .map_err(|e| err("result.reshape(-1)", e))?;
            let values: Vec<f32> = flat.extract().map_err(|e| err("result to Vec<f32>", e))?;
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
