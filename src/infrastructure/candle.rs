//! M7.0 candle spike helpers: one CUDA context, the candle stream as a raw
//! `CUstream`, candle storage as a device pointer, and a raw-ABI launch of one
//! committed cubin on candle tensors.
//!
//! This module exists only behind the non-default `candle` feature. It is the
//! seed of the M7.1 runtime adapter, not part of the serving path: nothing here
//! is reachable from `cargo build`.

use std::sync::Arc;

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::{CudaStorageSlice, cudarc};
use candle_core::{DType, Device, Storage, Tensor};
use half::bf16;

#[cfg(feature = "candle-nn")]
use candle_nn as _;

use crate::domain::expert_kernel::MoEError;
use crate::infrastructure::cuda_driver::{CUstream, CudaError, Driver, GdnKernels};
use crate::infrastructure::kernel_abi::{
    ADD_RMSNORM, AddRmsnormArgs, LinearAbi, add_rmsnorm_args, add_rmsnorm_grid,
};

fn bad(e: impl std::fmt::Display) -> MoEError {
    MoEError::Bad(format!("candle: {e}"))
}

fn driver(e: CudaError) -> MoEError {
    MoEError::Bad(format!("driver: {e}"))
}

/// A candle CUDA device on the primary context, with its own non-legacy stream.
///
/// `new_cuda_with_stream` retains the primary context (`CudaContext::new`),
/// which is the same context `Driver::attach_primary_context` and libtorch use,
/// so a candle op and a driver-API launch share one context and one address
/// space.
pub fn device(ordinal: usize) -> Result<Device, MoEError> {
    Device::new_cuda_with_stream(ordinal).map_err(bad)
}

/// The device's stream, as cudarc and as the raw driver handle our launchers
/// take. The two views are the same object.
pub fn stream(device: &Device) -> Result<(Arc<cudarc::driver::CudaStream>, CUstream), MoEError> {
    let cuda = device.as_cuda_device().map_err(bad)?;
    let stream = cuda.cuda_stream();
    let raw = stream.cu_stream() as CUstream;
    Ok((stream, raw))
}

/// The candle stream as the raw handle the cubin launchers take.
pub fn raw_stream(device: &Device) -> Result<CUstream, MoEError> {
    stream(device).map(|(_, raw)| raw)
}

/// Disable cudarc's per-tensor event tracking for tensors created afterwards.
///
/// A captured step must not record events inside the captured region, and the
/// spike holds every captured tensor alive, so no cross-stream lifetime guard
/// is needed.
///
/// # Safety
/// The caller owns all stream ordering, as `CudaDevice::disable_event_tracking`
/// documents.
pub unsafe fn disable_event_tracking(device: &Device) -> Result<(), MoEError> {
    let cuda = device.as_cuda_device().map_err(bad)?;
    // SAFETY: the caller upholds the stream-ordering contract.
    unsafe { cuda.disable_event_tracking() };
    Ok(())
}

/// A dense bf16 tensor from the checkpoint's raw little-endian bf16 bytes.
///
/// Going through `half::bf16` keeps the upload a byte copy instead of an
/// f32 widen-then-cast, which matters for the 1 GiB embedding and head.
pub fn bf16_tensor(raw: &[u8], shape: &[usize], device: &Device) -> Result<Tensor, MoEError> {
    if !raw.len().is_multiple_of(2) {
        return Err(bad("bf16 bank has an odd byte length"));
    }
    let values: Vec<bf16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let want: usize = shape.iter().product();
    if values.len() != want {
        return Err(bad(format!(
            "{} bf16 values for shape {shape:?} ({want})",
            values.len()
        )));
    }
    Tensor::from_vec(values, shape, device).map_err(bad)
}

/// A dense tensor from checkpoint-native bytes without conversion. Narrow
/// formats (FP8, NVFP4) stay their resident width; the engine's cubins read
/// those bytes directly.
pub fn native_tensor(
    raw: &[u8],
    dtype: crate::domain::weights::DType,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor, MoEError> {
    use crate::domain::weights::DType as D;
    let want: usize = shape.iter().product();
    if raw.len() != want * dtype.element_size() {
        return Err(bad(format!(
            "{} bytes for {want} {} values",
            raw.len(),
            dtype
        )));
    }
    let t = match dtype {
        D::Bf16 => bf16_tensor(raw, shape, device)?,
        D::F16 => {
            let values: Vec<half::f16> = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            Tensor::from_vec(values, shape, device).map_err(bad)?
        }
        D::F32 => {
            let values: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_vec(values, shape, device).map_err(bad)?
        }
        D::F64 => {
            let values: Vec<f64> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_vec(values, shape, device).map_err(bad)?
        }
        D::I64 => {
            let values: Vec<i64> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_vec(values, shape, device).map_err(bad)?
        }
        D::I32 => {
            let values: Vec<i32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_vec(values, shape, device).map_err(bad)?
        }
        D::I8 => Tensor::from_slice(raw, shape, device).map_err(bad)?,
        D::U8 | D::F8E4M3 | D::F8E4M3Fnuz => {
            // A raw byte view: candle has no F8E4M3 upload from bytes without
            // a cast, and the kernel ABI only sees the address.
            Tensor::from_slice(raw, shape, device).map_err(bad)?
        }
    };
    Ok(t)
}

/// The tensor's first-element device address, plus the element offset of a
/// view into its storage. The caller owns the tensor for as long as the
/// address is used.
pub fn device_ptr(t: &Tensor) -> Result<u64, MoEError> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(st) = &*storage else {
        return Err(bad("tensor is not on a CUDA device"));
    };
    let stream = st.device.cuda_stream();
    let base = match &st.slice {
        CudaStorageSlice::BF16(s) => s.device_ptr(&stream).0,
        CudaStorageSlice::F16(s) => s.device_ptr(&stream).0,
        CudaStorageSlice::F32(s) => s.device_ptr(&stream).0,
        CudaStorageSlice::F64(s) => s.device_ptr(&stream).0,
        CudaStorageSlice::U8(s) => s.device_ptr(&stream).0,
        CudaStorageSlice::U32(s) => s.device_ptr(&stream).0,
        CudaStorageSlice::I16(s) => s.device_ptr(&stream).0,
        CudaStorageSlice::I32(s) => s.device_ptr(&stream).0,
        CudaStorageSlice::I64(s) => s.device_ptr(&stream).0,
        CudaStorageSlice::F8E4M3(s) => s.device_ptr(&stream).0,
        CudaStorageSlice::F6E2M3(s)
        | CudaStorageSlice::F6E3M2(s)
        | CudaStorageSlice::F4(s)
        | CudaStorageSlice::F8E8M0(s) => s.device_ptr(&stream).0,
    };
    let elem = match t.dtype() {
        DType::U8 => 1,
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => 1,
        DType::F8E4M3 => 1,
        DType::I16 | DType::F16 | DType::BF16 => 2,
        DType::U32 | DType::I32 | DType::F32 => 4,
        DType::I64 | DType::F64 => 8,
        _ => 1,
    };
    Ok(base + layout.start_offset() as u64 * elem)
}

/// Copy a tensor to the host as f32, the same crossing `torch::to_f32_vec`
/// provides.
pub fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>, MoEError> {
    let flat = t
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .map_err(bad)?;
    flat.to_vec1::<f32>().map_err(bad)
}

/// The committed linear-attention cubins over a driver, ready for a raw-ABI
/// launch from candle storage.
pub struct LinearKernels {
    abi: LinearAbi,
    kernels: GdnKernels,
}

impl LinearKernels {
    pub fn load(lib: Arc<Driver>) -> Result<Self, MoEError> {
        let abi = LinearAbi::committed()?;
        abi.kernel(ADD_RMSNORM)?;
        let kernels = GdnKernels::load(lib, abi.clone()).map_err(driver)?;
        Ok(Self { abi, kernels })
    }

    /// `add_rmsnorm` on raw addresses: the same cubin, grid and argument pack
    /// as `GdnDecode::add_rmsnorm`, minus the `tch::Tensor` shape checks.
    ///
    /// # Safety
    /// Every pointer must be a live device allocation in `stream`'s context,
    /// with `rows * hidden` bf16 elements, and both outputs fully writable.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn add_rmsnorm(
        &self,
        stream: CUstream,
        x: u64,
        r: u64,
        w: u64,
        sum: u64,
        out: u64,
        eps: f32,
        rows: usize,
        hidden: usize,
        stride_x: u32,
        stride_r: u32,
    ) -> Result<(), MoEError> {
        let kernel = self.abi.kernel(ADD_RMSNORM)?;
        let baked = kernel.constexpr("HIDDEN")? as usize;
        if hidden != baked {
            return Err(MoEError::Bad(format!(
                "add_rmsnorm was built for hidden {baked}, not {hidden}"
            )));
        }
        let args = AddRmsnormArgs {
            x,
            r,
            w,
            sum,
            out,
            eps,
            stride_x,
            stride_r,
        };
        let pack = add_rmsnorm_args(&self.abi, &args)?;
        let mut params = pack.kernel_params()?;
        let grid = add_rmsnorm_grid(rows);
        // SAFETY: the caller guarantees every pointer and the stream.
        unsafe {
            self.kernels
                .launch(kernel, grid, &mut params, stream)
                .map_err(driver)
        }
    }

    pub fn hidden(&self) -> Result<usize, MoEError> {
        // The norm tile is baked as the model hidden size; the manifest's
        // constexpr is the source of truth for the spike's synthetic shapes.
        let k = self.abi.kernel(ADD_RMSNORM)?;
        Ok(k.constexpr("HIDDEN")? as usize)
    }
}

/// Read a contiguous bf16 tensor back as f32 without allocating anything on
/// the device: a synchronous D2H copy through the driver, then a host-side
/// `bf16 -> f32` widen. Used by the capture gate so the readbacks between
/// replays cannot perturb candle's stream-ordered pool.
pub fn bf16_f32_via_d2h(lib: &Driver, t: &Tensor) -> Result<Vec<f32>, MoEError> {
    if t.dtype() != DType::BF16 {
        return Err(bad(format!("expected bf16, got {:?}", t.dtype())));
    }
    let n = t.elem_count();
    let mut bytes = vec![0u8; n * 2];
    let ptr = device_ptr(t)?;
    // SAFETY: `bytes` is `n * 2` bytes and `ptr` is the tensor's live base.
    unsafe { lib.copy_d2h(&mut bytes, ptr) }.map_err(|e| bad(format!("d2h: {e}")))?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
