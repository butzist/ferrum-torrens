// Compute-runtime adapter: owns the device choice. Infrastructure, so these
// types never cross into the domain core.
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
    /// True for the CUDA device, the cheap guard for the NVFP4 kernel (which is
    /// GPU-only).
    pub fn is_cuda(self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

/// Inspect the host through the CUDA driver, with no libtorch in the process.
/// `init` is what distinguishes "no driver" from "driver but no visible device",
/// and the count is only meaningful once it succeeds.
#[cfg(feature = "cudarc")]
pub fn probe_cuda() -> CudaProbe {
    use crate::infrastructure::cuda_driver::Driver;

    let Ok(lib) = Driver::open() else {
        return CudaProbe::default();
    };
    if lib.init().is_err() {
        return CudaProbe::default();
    }
    let count = lib.device_count().unwrap_or(0).max(0) as usize;
    if count == 0 {
        return CudaProbe::default();
    }
    if lib.attach_primary_context(0).is_err() {
        return CudaProbe::default();
    }
    CudaProbe {
        available: count > 0,
        device_count: count,
    }
}

/// The device a test or a load should use: whatever `Auto` resolves to on this
/// host. Device-agnostic so one test body runs in the sandbox and in a devshell.
#[cfg(feature = "cudarc")]
pub fn auto_device() -> Device {
    resolve_device(DeviceRequest::Auto, 0, probe_cuda()).expect("auto never fails")
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

    #[cfg(feature = "cudarc")]
    #[test]
    fn the_cudarc_probe_is_self_consistent() {
        let probe = probe_cuda();
        assert_eq!(probe.available, probe.device_count > 0);
        for request in [DeviceRequest::Auto, DeviceRequest::Cpu, DeviceRequest::Cuda] {
            for gpu in [0, 1] {
                let _ = resolve_device(request, gpu, probe);
            }
        }
    }
}
