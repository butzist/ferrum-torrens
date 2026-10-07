// Compute-runtime adapter: owns the device choice and the libtorch tensor
// helpers. Infrastructure, so these types never cross into the domain core.
//
// Device resolution is a pure function of (what the config asked for, what the
// host has), kept separate from the probe that inspects the host, so the policy
// is unit-testable on a machine with no GPU at all.

/// What the config asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceRequest {
    /// CUDA when the host has it, CPU otherwise.
    #[default]
    Auto,
    /// CPU even if a GPU is present.
    Cpu,
    /// CUDA, and an error if the host cannot provide it.
    Cuda,
}

/// The device the runtime actually runs on. Resolved once at load, so the
/// forward path never re-probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
        }
    }
}

/// What the host actually has, as seen by the probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CudaProbe {
    pub available: bool,
    pub device_count: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DeviceError {
    #[error("model.device = \"cuda\" but this host has no CUDA device")]
    CudaRequestedUnavailable,
    #[error("model.gpu = {gpu} but this host has {count} CUDA device(s)")]
    CudaIndexOutOfRange { gpu: usize, count: usize },
}

/// Resolve the request against the host. `gpu` is the configured device index.
///
/// The asymmetry is deliberate: `Auto` with *zero* devices is the legitimate
/// no-GPU fallback and lands on CPU, but `Auto` with a device present and a
/// `gpu` index past the end is a config typo and must not be papered over.
pub fn resolve_device(
    request: DeviceRequest,
    gpu: usize,
    probe: CudaProbe,
) -> Result<Device, DeviceError> {
    if matches!(request, DeviceRequest::Cpu) {
        return Ok(Device::Cpu);
    }
    if !probe.available {
        return match request {
            DeviceRequest::Cuda => Err(DeviceError::CudaRequestedUnavailable),
            _ => Ok(Device::Cpu),
        };
    }
    if gpu >= probe.device_count {
        return Err(DeviceError::CudaIndexOutOfRange {
            gpu,
            count: probe.device_count,
        });
    }
    Ok(Device::Cuda(gpu))
}

impl Device {
    /// The libtorch device this maps to.
    #[cfg(feature = "cuda")]
    pub fn to_tch(self) -> tch::Device {
        match self {
            Device::Cpu => tch::Device::Cpu,
            Device::Cuda(i) => tch::Device::Cuda(i),
        }
    }

    /// True for the CUDA device, the cheap guard for graph capture and the NVFP4
    /// kernel (both of which are GPU-only).
    pub fn is_cuda(self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

/// Load libtorch's CUDA backend into the process.
///
/// `libtorch_cuda.so` registers the CUDA kernels with ATen, and a statically
/// dispatched Rust binary never references a symbol from it, so the linker's
/// default `--as-needed` leaves it out of `DT_NEEDED` (only `libtorch_cpu.so` and
/// `libc10.so` make it in). Without it `aten::empty` on a CUDA device raises
/// "only available for these backends: [CPU, ...]" no matter how real the GPU
/// is. Python's `torch` avoids this by importing `_C` with the CUDA extension
/// built in; an embedded runtime has to do what it does by hand.
///
/// The directory comes from where `libtorch_cpu.so` was actually mapped
/// (`/proc/self/maps`) rather than from an env var, so it cannot name a different
/// libtorch than the one already linked in. Idempotent: the loader refcounts, and
/// `RTLD_GLOBAL` makes the registration visible to the already-loaded ATen.
#[cfg(feature = "cuda")]
pub fn load_cuda_backend() -> Result<(), String> {
    use std::ffi::CString;
    unsafe extern "C" {
        fn dlopen(
            filename: *const std::ffi::c_char,
            flag: std::ffi::c_int,
        ) -> *mut std::ffi::c_void;
    }
    const RTLD_NOW: std::ffi::c_int = 2;
    const RTLD_GLOBAL: std::ffi::c_int = 0x100;

    let maps = std::fs::read_to_string("/proc/self/maps").map_err(|e| e.to_string())?;
    let dir = ["libtorch_cpu.so", "libtorch.so"]
        .iter()
        .find_map(|wanted| {
            maps.lines().find_map(|line| {
                let path = line.split_whitespace().last()?;
                path.ends_with(wanted)
                    .then_some(std::path::Path::new(path))?
                    .parent()
            })
        })
        .ok_or_else(|| "no libtorch is mapped, so its lib dir is unknown".to_string())?;
    let entry = dir.join("libtorch_cuda.so");
    let name = CString::new(
        entry
            .to_str()
            .ok_or("libtorch path is not utf-8")?
            .as_bytes(),
    )
    .map_err(|e| e.to_string())?;
    // SAFETY: `name` is a NUL-terminated path that outlives the call, and the
    // loader keeps whatever it mapped; we hold no handle and never unload it.
    let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
    if handle.is_null() {
        return Err(format!("dlopen {} failed", entry.display()));
    }
    Ok(())
}

/// Inspect the host. Returns `available = false` with a zero count rather than
/// erroring when libtorch was built with CUDA but no device is visible, so a
/// CPU-only run needs no config change.
#[cfg(feature = "cuda")]
pub fn probe_cuda() -> CudaProbe {
    // The probe is the first thing in the process to ask libtorch about CUDA, so
    // it is where the backend gets loaded. A failure means no CUDA kernels are
    // registered, which is exactly `available = false`.
    let _ = load_cuda_backend();
    let count = tch::Cuda::device_count().max(0) as usize;
    CudaProbe {
        available: tch::Cuda::is_available() && count > 0,
        device_count: count,
    }
}

/// The device a test or a load should use: whatever `Auto` resolves to on this
/// host. Device-agnostic so one test body runs in the sandbox and in a devshell.
#[cfg(feature = "cuda")]
pub fn auto_device() -> Device {
    resolve_device(DeviceRequest::Auto, 0, probe_cuda()).expect("auto never fails")
}

/// A row-major `data` as an f32 tensor of `shape` on `device`.
///
/// Every dense weight in the checkpoint is stored in a narrow dtype that libtorch
/// cannot matmul, so they are widened once at load and handed over as plain f32
/// here. `shape` is row-major, matching how the FTW index stores shapes.
#[cfg(feature = "cuda")]
pub fn f32_tensor(data: &[f32], shape: &[i64], device: Device) -> tch::Tensor {
    tch::Tensor::from_slice(data)
        .reshape(shape)
        .to(device.to_tch())
}

/// A row-major `data` as a bfloat16 tensor of `shape` on `device`.
///
/// This is the storage dtype for the widened weights: it is what the checkpoint
/// itself uses for most tensors, it halves the resident bytes against the 8 GB
/// target, and libtorch accumulates f32 through a bf16 matmul anyway.
///
/// The round trip goes through an f32 tensor rather than a blob over the raw
/// bytes, so the result owns its data instead of borrowing a slice that the
/// caller has to keep alive.
#[cfg(feature = "cuda")]
pub fn bf16_tensor(data: &[f32], shape: &[i64], device: Device) -> tch::Tensor {
    f32_tensor(data, shape, device).to_kind(tch::Kind::BFloat16)
}

/// Copy a tensor's elements back to the host as f32. The forward's only crossing
/// point back into the domain, which is why the `Model` trait can stay
/// device-agnostic.
#[cfg(feature = "cuda")]
pub fn to_f32_vec(t: &tch::Tensor) -> Vec<f32> {
    let flat = t.to_kind(tch::Kind::Float).contiguous().flatten(0, -1);
    let n = flat.numel();
    let mut out = vec![0.0f32; n];
    flat.copy_data(&mut out, n);
    out
}

/// A libtorch tensor behind the domain's opaque device-buffer port: the only
/// concrete `MoETensor` the serving path has, and the type the MoE adapter
/// downcasts back to in order to copy out of or into one (decision 43).
///
/// `tch::Tensor` is `Send` but not `Sync`. The port borrows one of these for
/// the duration of a single `&self` call on a shared executor, which is the
/// sharing `SharedTensor` documents in the forward: libtorch is thread-safe,
/// and no two threads hold the same tensor while the port does.
#[cfg(feature = "cuda")]
pub struct DevTensor(pub tch::Tensor);

#[cfg(feature = "cuda")]
unsafe impl Sync for DevTensor {}

#[cfg(feature = "cuda")]
impl DevTensor {
    /// Host values as a device buffer: how a caller that still has its numbers
    /// in host memory enters the device hand-off.
    pub fn from_f32(data: &[f32], shape: &[i64], device: Device) -> Self {
        Self(f32_tensor(data, shape, device))
    }

    /// An f32 buffer of `shape`, which is how a tower makes room for the routed
    /// sum it is about to ask for.
    pub fn zeros(shape: &[i64], device: Device) -> Self {
        Self(tch::Tensor::zeros(
            shape,
            (tch::Kind::Float, device.to_tch()),
        ))
    }
}

#[cfg(feature = "cuda")]
impl crate::domain::expert_kernel::MoETensor for DevTensor {
    fn numel(&self) -> usize {
        self.0.numel()
    }

    fn to_host_f32(&self) -> Result<Vec<f32>, crate::domain::expert_kernel::MoEError> {
        Ok(to_f32_vec(&self.0))
    }

    fn set_host_f32(
        &mut self,
        values: &[f32],
    ) -> Result<(), crate::domain::expert_kernel::MoEError> {
        if values.len() != self.0.numel() {
            return Err(crate::domain::expert_kernel::MoEError::Bad(format!(
                "{} values for a {}-element device buffer",
                values.len(),
                self.0.numel()
            )));
        }
        let host = tch::Tensor::from_slice(values).reshape(self.0.size());
        self.0.copy_(&host.to(self.0.device()));
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_CUDA: CudaProbe = CudaProbe {
        available: false,
        device_count: 0,
    };
    const ONE_GPU: CudaProbe = CudaProbe {
        available: true,
        device_count: 1,
    };
    const TWO_GPUS: CudaProbe = CudaProbe {
        available: true,
        device_count: 2,
    };

    #[test]
    fn auto_with_no_cuda_falls_back_to_cpu() {
        assert_eq!(
            resolve_device(DeviceRequest::Auto, 0, NO_CUDA).unwrap(),
            Device::Cpu
        );
    }

    #[test]
    fn auto_with_cuda_uses_the_configured_index() {
        assert_eq!(
            resolve_device(DeviceRequest::Auto, 0, ONE_GPU).unwrap(),
            Device::Cuda(0)
        );
        assert_eq!(
            resolve_device(DeviceRequest::Auto, 1, TWO_GPUS).unwrap(),
            Device::Cuda(1)
        );
    }

    #[test]
    fn auto_with_a_bad_gpu_index_is_an_error_not_a_silent_fallback() {
        // The host has a GPU, so "no device" is not the explanation; gpu = 7 is
        // a typo and must be reported rather than quietly running on CPU.
        assert_eq!(
            resolve_device(DeviceRequest::Auto, 7, TWO_GPUS),
            Err(DeviceError::CudaIndexOutOfRange { gpu: 7, count: 2 })
        );
    }

    #[test]
    fn explicit_cpu_never_uses_cuda() {
        for probe in [NO_CUDA, ONE_GPU, TWO_GPUS] {
            assert_eq!(
                resolve_device(DeviceRequest::Cpu, 3, probe).unwrap(),
                Device::Cpu,
                "device = \"cpu\" must ignore both the GPU and the gpu index"
            );
        }
    }

    #[test]
    fn explicit_cuda_requires_the_device_to_exist() {
        assert_eq!(
            resolve_device(DeviceRequest::Cuda, 0, NO_CUDA),
            Err(DeviceError::CudaRequestedUnavailable)
        );
        assert_eq!(
            resolve_device(DeviceRequest::Cuda, 0, ONE_GPU).unwrap(),
            Device::Cuda(0)
        );
    }

    #[test]
    fn explicit_cuda_still_checks_the_index() {
        assert_eq!(
            resolve_device(DeviceRequest::Cuda, 1, ONE_GPU),
            Err(DeviceError::CudaIndexOutOfRange { gpu: 1, count: 1 })
        );
    }

    #[test]
    fn is_cuda_distinguishes_the_two() {
        assert!(!Device::Cpu.is_cuda());
        assert!(Device::Cuda(0).is_cuda());
    }

    #[test]
    fn display_names_the_device() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Cuda(1).to_string(), "cuda:1");
    }

    /// libtorch must be importable and its CPU backend usable wherever the test
    /// runs, or nothing above means anything. Device-agnostic on purpose: the
    /// agent sandbox has no GPU, the devshell has one, and both must pass.
    #[cfg(feature = "cuda")]
    #[test]
    fn libtorch_is_usable_and_the_probe_is_self_consistent() {
        let probe = probe_cuda();
        assert_eq!(
            probe.available,
            probe.device_count > 0,
            "available and device_count must not disagree"
        );
        // Every request resolves or errors deterministically against the real
        // probe, on a GPU host and a CPU-only host alike.
        for request in [DeviceRequest::Auto, DeviceRequest::Cpu, DeviceRequest::Cuda] {
            for gpu in [0, 1] {
                let _ = resolve_device(request, gpu, probe);
            }
        }
        let cpu = (tch::Kind::Double, tch::Device::Cpu);
        let a = tch::Tensor::ones([2, 3], cpu);
        let b = tch::Tensor::ones([3, 4], cpu);
        let c = a.matmul(&b);
        assert_eq!(c.size(), vec![2, 4]);
        assert!(c.sum(tch::Kind::Double).double_value(&[]) == 24.0);
    }

    #[cfg(feature = "cuda")]
    fn here() -> Device {
        auto_device()
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn f32_tensor_round_trips_a_known_buffer() {
        let d = here();
        let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let t = f32_tensor(&data, &[2, 3], d);
        assert_eq!(t.size(), vec![2, 3]);
        assert_eq!(to_f32_vec(&t), data);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn f32_tensor_handles_a_rank_one_shape() {
        let d = here();
        let t = f32_tensor(&[1.0, 2.0, 3.0], &[3], d);
        assert_eq!(t.dim(), 1);
        assert_eq!(to_f32_vec(&t), vec![1.0, 2.0, 3.0]);
    }

    /// The buffer the MoE port hands across: host values in, the same values
    /// back, and a write that refuses the wrong length rather than truncating.
    #[cfg(feature = "cuda")]
    #[test]
    fn dev_tensor_round_trips_values_through_the_device() {
        use crate::domain::expert_kernel::{MoEError, MoETensor};

        let d = here();
        let mut t = DevTensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], d);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.to_host_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        t.set_host_f32(&[9.0, 8.0, 7.0, 6.0]).unwrap();
        assert_eq!(to_f32_vec(&t.0), vec![9.0, 8.0, 7.0, 6.0]);
        assert!(matches!(t.set_host_f32(&[1.0]), Err(MoEError::Bad(_))));

        let zeros = DevTensor::zeros(&[2, 3], d);
        assert_eq!(zeros.numel(), 6);
        assert_eq!(to_f32_vec(&zeros.0), vec![0.0; 6]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn bf16_tensor_keeps_bf16_precision() {
        // bf16 holds 8 mantissa bits, so these must be exact while the gap below
        // them is below the format's resolution. The check pins the storage
        // precision rather than libtorch's arithmetic.
        let exact = [1.0f32, 0.5, 2.0, -1.0, 0.0];
        let t = bf16_tensor(&exact, &[5], here());
        assert_eq!(t.kind(), tch::Kind::BFloat16);
        assert_eq!(to_f32_vec(&t), exact.to_vec());

        let lossy = [1.0f32 / 3.0, 0.1];
        let back = to_f32_vec(&bf16_tensor(&lossy, &[2], here()));
        for (got, want) in back.iter().zip(lossy.iter()) {
            assert!(
                (got - want).abs() < 1e-2,
                "{got} is not within bf16 resolution of {want}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn a_tensor_survives_a_device_move() {
        // On a CPU host this is the identity, on a GPU host a real copy. Both
        // must give the same values, which is what makes the load path
        // device-agnostic.
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let on_host = f32_tensor(&data, &[2, 4], Device::Cpu);
        let on_device = on_host.to(here().to_tch());
        assert_eq!(to_f32_vec(&on_device), data);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn matmul_accumulates_the_row_major_layout_we_uploaded() {
        // A 2x3 times 3x2, checked exactly, so a transposed or column-major
        // upload would fail here rather than as a subtly wrong completion.
        let d = here();
        let a = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], d);
        let b = f32_tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], d);
        let c = a.matmul(&b);
        assert_eq!(to_f32_vec(&c), vec![58.0, 64.0, 139.0, 154.0]);
    }
}
