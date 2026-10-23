// Model registry: map the FTW dir's `config.json` arch string to a concrete
// `Model` implementation. There is no arch label in the FTW index meta, so the
// FTW dir's copied config.json is the dispatch key (TODO.md section 3).

use crate::domain::decode::CapturedDecode;
use crate::domain::kvcache::CachedModel;
use crate::domain::model::Model;
use crate::domain::weights::WeightsReader;
#[cfg(any(feature = "cuda", feature = "cudarc"))]
use crate::domain::{expert_kernel::MoEExecutor, model::ModelParams};
use crate::infrastructure::ftw::FtwIndex;
use crate::infrastructure::torch::Device;
use crate::models::qwen3_6::Qwen3_6;
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("FTW dir has no config.json with architectures[0]")]
    NoArch,
    #[error("unsupported architecture {0:?}")]
    Unsupported(String),
    #[error("model construction failed: {0}")]
    Build(String),
}

/// A model plus the capture service the composition root built over its own KV
/// cache, when `model.cuda_graph` asked for one. The engine holds both.
pub struct BuiltModel {
    pub model: Box<dyn CachedModel>,
    pub captured: Option<Arc<dyn CapturedDecode>>,
    /// The resolved expert-cache slot count (explicit or auto-sized, item 12).
    pub cache_size_slots: usize,
}

/// The arch label from `config.json` `architectures[0]`, used to dispatch.
pub fn arch_of(config_json: &Value) -> Result<&str, RegistryError> {
    config_json
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .ok_or(RegistryError::NoArch)
}

/// Build a `Model` for the arch declared in `config_json` (parsed from the FTW
/// dir's `config.json`), boxing the arch-specific implementation behind the
/// engine-facing trait.
///
/// The `cuda` cargo feature picks the forward: with it the arch runs on libtorch
/// (on `device`, CPU included, decision 21), without it the scalar reference is
/// the only implementation there is. The engine cannot tell the difference --
/// both are `Box<dyn Model>` behind the same trait.
///
/// `config` supplies the `[kernel]` and `[moe]` sections, so the tower can be
/// handed the engine's MoE port when the kernel is on; with it off the tower
/// keeps widening routed experts on the host itself.
#[cfg_attr(
    not(any(feature = "cuda", feature = "cudarc")),
    allow(unused_variables)
)]
pub fn build(
    config_json: &Value,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
    device: Device,
    config: &crate::config::Config,
) -> Result<BuiltModel, RegistryError> {
    match arch_of(config_json)? {
        "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5MoeForCausalLM" => {
            let tc = config_json
                .get("text_config")
                .ok_or(RegistryError::NoArch)?;
            #[cfg(feature = "cudarc")]
            {
                // The torchless runtime has no CPU compute path in the serving
                // composition root, but the scalar reference still answers on a
                // CPU-only host, so `auto` keeps working there.
                let Device::Cuda(_) = device else {
                    return scalar_built(tc, index, reader, config);
                };
                // The torchless path widens and uploads every weight itself, so
                // it takes the parsed params plus the reader rather than the
                // scalar path's pre-loaded dense store.
                let params = crate::models::qwen3_6::config::parse_text_config(tc)
                    .map_err(|e| RegistryError::Build(e.to_string()))?;
                if config.model.cuda_graph {
                    eprintln!(
                        "warning: cuda_graph is unavailable on the torchless runtime; decode stays eager"
                    );
                }
                // The pool owns the runtime this whole path runs on: the model,
                // the dense launchers and the MoE adapter all launch on its
                // stream, so a step's writes and its next reads stay ordered.
                let cache = Arc::new(kv_cache(config, &params, device, false));
                let rt = cache.runtime().clone();
                let stream = rt.raw_stream();
                let driver = rt.driver().clone();
                let head_gemv = head_port(device, stream)?;
                let fp8_linear = fp8_port(device, stream)?;
                // Load the tower with the MoE port detached: the auto cache
                // budget (item 12) reads the free VRAM after the tower, the KV
                // pool and the ports are all resident, then sizes the expert
                // cache and attaches it.
                let mut model = crate::models::qwen3_6::forward_cudarc::CudarcQwen3_6::load(
                    params.clone(),
                    rt,
                    index.clone(),
                    reader.clone(),
                    None,
                    head_gemv,
                    fp8_linear,
                    cache,
                )
                .map_err(|e| RegistryError::Build(e.to_string()))?;
                let slots = resolve_cache_slots(config, &params, device, Some(driver.as_ref()))?;
                let moe = moe_port(config, &params, index, reader, device, stream, slots)?;
                model.attach_moe(moe);
                Ok(BuiltModel {
                    model: Box::new(model),
                    captured: None,
                    cache_size_slots: slots,
                })
            }
            #[cfg(all(feature = "cuda", not(feature = "cudarc")))]
            {
                // The libtorch path widens and uploads every weight itself, so
                // it takes the parsed params plus the reader rather than the
                // scalar path's pre-loaded dense store.
                let params = crate::models::qwen3_6::config::parse_text_config(tc)
                    .map_err(|e| RegistryError::Build(e.to_string()))?;
                // Capture needs the device-staged MoE path (the inline path reads
                // the router's ids back to the host, which a graph cannot record),
                // so it only turns on with the kernel and a CUDA device.
                let capture = config.model.cuda_graph
                    && config.kernel.triton
                    && matches!(device, Device::Cuda(_));
                let (lib, stream) = side_stream(capture, device)?;
                // The `FT_PROFILE` stage timer times the eager forward, so it is
                // only meaningful with capture off (see profile.rs).
                if let Device::Cuda(ordinal) = device {
                    let _ = crate::infrastructure::profile::init(ordinal);
                }
                let attention = attention_port(&params, device, stream)?;
                let gdn = gdn_port(&params, device, stream)?;
                let head_gemv = head_port(device, stream)?;
                let fp8_linear = fp8_port(device, stream)?;
                let cache = Arc::new(kv_cache(config, &params, device, capture));
                // Load the tower with the MoE port detached: the auto cache
                // budget (item 12) reads the free VRAM after the tower, the KV
                // pool and the ports are all resident, then sizes the expert
                // cache and attaches it.
                let mut model = crate::models::qwen3_6::forward_torch::TorchQwen3_6::load(
                    params.clone(),
                    device,
                    index.clone(),
                    reader.clone(),
                    None,
                    attention,
                    gdn,
                    head_gemv,
                    fp8_linear,
                    cache.clone(),
                )
                .map_err(|e| RegistryError::Build(e.to_string()))?;
                let slots = resolve_cache_slots(config, &params, device, lib.as_deref())?;
                let moe = moe_port(config, &params, index, reader, device, stream, slots)?;
                model.attach_moe(moe);
                let captured = match lib {
                    Some(lib) => Some(capture_decode(config, &model, cache, lib, stream, device)?),
                    None => None,
                };
                Ok(BuiltModel {
                    model: Box::new(model),
                    captured,
                    cache_size_slots: slots,
                })
            }
            #[cfg(all(not(feature = "cuda"), not(feature = "cudarc")))]
            scalar_built(tc, index, reader, config)
        }
        other => Err(RegistryError::Unsupported(other.to_string())),
    }
}

/// The scalar reference model, the answer on a host with no CUDA device and on
/// a build with no compute backend at all.
#[cfg(any(feature = "cudarc", not(feature = "cuda")))]
fn scalar_built(
    tc: &Value,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
    config: &crate::config::Config,
) -> Result<BuiltModel, RegistryError> {
    Ok(BuiltModel {
        model: Box::new(
            Qwen3_6::from_text_config(tc, index, reader)
                .map_err(|e| RegistryError::Build(e.to_string()))?,
        ),
        captured: None,
        cache_size_slots: config
            .moe
            .cache_size_slots
            .explicit()
            .unwrap_or(crate::config::DEFAULT_CACHE_SIZE_SLOTS),
    })
}

/// The process-wide driver and the non-legacy side stream a capture records on,
/// or `(None, null)` when capture is off. Creating the stream attaches the
/// primary context of `device`, which the MoE adapter then reuses.
#[cfg(all(feature = "cuda", not(feature = "cudarc")))]
fn capture_driver(
    device: Device,
) -> Result<
    (
        Arc<crate::infrastructure::cuda_driver::Driver>,
        crate::infrastructure::cuda_driver::CUstream,
    ),
    RegistryError,
> {
    use crate::infrastructure::cuda_driver::Driver;
    let Device::Cuda(ordinal) = device else {
        return Err(RegistryError::Build(
            "a decode capture needs a CUDA device".to_string(),
        ));
    };
    let build_err = |e: String| RegistryError::Build(e);
    let lib = Arc::new(Driver::open().map_err(|e| build_err(e.to_string()))?);
    lib.init().map_err(|e| build_err(e.to_string()))?;
    lib.attach_primary_context(ordinal as i32)
        .map_err(|e| build_err(e.to_string()))?;
    let stream = unsafe { lib.create_stream() }.map_err(|e| build_err(e.to_string()))?;
    Ok((lib, stream))
}

/// The side stream a capture records on, or `(None, null)` when capture is off.
#[cfg(all(feature = "cuda", not(feature = "cudarc")))]
fn side_stream(
    capture: bool,
    device: Device,
) -> Result<
    (
        Option<Arc<crate::infrastructure::cuda_driver::Driver>>,
        crate::infrastructure::cuda_driver::CUstream,
    ),
    RegistryError,
> {
    if capture {
        let (lib, stream) = capture_driver(device)?;
        Ok((Some(lib), stream))
    } else {
        Ok((None, std::ptr::null_mut()))
    }
}

/// Record one decode graph per batch size and wrap them in the engine's service.
/// The padding row claims a reserved state slot and pool location, and the
/// reservation itself is moved into the service so nothing else can take them.
#[cfg(all(feature = "cuda", not(feature = "cudarc")))]
fn capture_decode(
    config: &crate::config::Config,
    model: &crate::models::qwen3_6::forward_torch::TorchQwen3_6,
    cache: Arc<crate::infrastructure::kvcache_dev::DeviceCache>,
    lib: Arc<crate::infrastructure::cuda_driver::Driver>,
    stream: crate::infrastructure::cuda_driver::CUstream,
    device: Device,
) -> Result<Arc<dyn CapturedDecode>, RegistryError> {
    use crate::domain::decode::DecodeRow;
    use crate::domain::kvcache::KvCache;
    use crate::infrastructure::decode_buffers::DecodeBuffers;
    use crate::infrastructure::decode_graph::EngineDecodeGraphs;

    let build_err = |e: String| RegistryError::Build(e);
    let Device::Cuda(ordinal) = device else {
        return Err(build_err("a decode capture needs a CUDA device".into()));
    };
    // The capture records every torch op on the side stream.
    unsafe { crate::infrastructure::cuda_graph::set_current_stream(stream, ordinal) };

    let mut padding_cache = cache.begin();
    cache
        .reserve(&mut padding_cache, 1)
        .map_err(|e| build_err(e.to_string()))?;
    let padding = DecodeRow {
        input_id: 0,
        position: 0,
        state_slot: padding_cache.state_slot(),
        out_loc: padding_cache.tail(1)[0],
    };

    let sizes = graph_batch_sizes(config.model.max_running_requests);
    let graphs = unsafe {
        // SAFETY: `stream` is the non-legacy side stream just created on the
        // device's primary context, and the step is the strictly device-only
        // decode forward (no host copy, no allocation outside the pool, no sync).
        EngineDecodeGraphs::capture(
            lib,
            stream,
            device,
            model.params().vocab_size,
            config.model.max_seq_len,
            &sizes,
            padding,
            padding_cache,
            |bufs: &DecodeBuffers| model.forward_decode_into(bufs).map_err(|e| e.to_string()),
        )
    }
    .map_err(|e| RegistryError::Build(e.to_string()))?;
    Ok(Arc::new(graphs))
}

/// The batch sizes to capture: `1`, `2`, `4` and `max_running`, deduplicated and
/// capped at `max_running`. A step of `n` rows replays on the smallest captured
/// size at or above `n`.
#[cfg(all(feature = "cuda", not(feature = "cudarc")))]
fn graph_batch_sizes(max_running: usize) -> Vec<usize> {
    let mut sizes = vec![1usize];
    for s in [2usize, 4, max_running] {
        if (1..=max_running).contains(&s) && !sizes.contains(&s) {
            sizes.push(s);
        }
    }
    sizes.sort_unstable();
    sizes
}

/// The device KV cache the engine hands page tables out from: `[kv]` sizes it,
/// one state slot per request the scheduler may run at once, and `[kv] dtype`
/// picks the element type. The default `bf16` is half of f32, which is what lets
/// the 4.51 GiB resident tower and the pool coexist on the 8 GB device; `f32`
/// stays available because the cached forward's contract is to reproduce a
/// re-fed forward and that is the lossless option (DONE.md decision 33).
#[cfg(any(feature = "cuda", feature = "cudarc"))]
fn kv_cache(
    config: &crate::config::Config,
    params: &ModelParams,
    device: Device,
    reserve_padding: bool,
) -> crate::infrastructure::kvcache_dev::DeviceCache {
    use crate::config::KvDtype;
    use crate::infrastructure::kvcache_dev::{DeviceCache, KvDType};
    let dtype = match config.kv.dtype {
        KvDtype::F32 => KvDType::F32,
        KvDtype::Bf16 => KvDType::BFloat16,
        // Refused at config validation; a real dtype never reaches here.
        KvDtype::Fp8 => unreachable!("kv.dtype = fp8 is rejected at config validation"),
    };
    // One live state slot per running request plus one locked snapshot each,
    // plus progress headroom and the captured graph's padding row (item 10).
    let slots = crate::infrastructure::kvcache_dev::state_slots(
        config.model.max_running_requests,
        reserve_padding,
    );
    DeviceCache::new(params, config.kv_tokens(), slots, dtype, device)
}

/// The engine's MoE port for `kernel.triton = true`: the driver-API adapter over
/// the mmap banks, wrapped in the slot-cache executor that plans residency per
/// layer. `None` whenever the kernel is off, which is what leaves the tower
/// widening routed experts on the host.
///
/// Composition root work on purpose: this is the one place that knows both the
/// config and the adapter, and it is arch-agnostic -- a new MoE arch with the
/// same bank layout gets the same executor.
#[cfg(any(feature = "cuda", feature = "cudarc"))]
fn moe_port(
    config: &crate::config::Config,
    params: &ModelParams,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
    device: Device,
    stream: crate::infrastructure::cuda_driver::CUstream,
    slots: usize,
) -> Result<Option<Arc<dyn MoEExecutor>>, RegistryError> {
    use crate::application::run_experts::CachedExperts;
    use crate::domain::expert_kernel::ExpertShapes;
    use crate::infrastructure::banks::{FtwBanks, FtwPinnedBanks};
    use crate::infrastructure::cuda_moe::CudaMoE;

    if !config.kernel.triton {
        return Ok(None);
    }
    let build_err = |e: String| RegistryError::Build(e);
    let shapes = ExpertShapes::new(params.hidden_size, params.moe_intermediate_size)
        .map_err(|e| build_err(e.to_string()))?;
    // One `Mutex<CudaMoE>` is both ports: the store stages into it and the gemm
    // reads the same device banks.
    let mut adapter = CudaMoE::new_with_stream(
        device,
        Arc::new(FtwBanks::new(reader.clone(), index.clone())),
        shapes,
        slots,
        stream,
    )
    .map_err(|e| build_err(e.to_string()))?;
    // Decision 40 prerequisite 4: a page-locked copy of every expert bank that
    // the device LRU reads its misses from, so the plan never crosses to the
    // host. Loaded once and owned by the adapter, because a captured graph
    // bakes its device aliases in.
    let driver = adapter.driver();
    let pinned = FtwPinnedBanks::load(
        &driver,
        &reader,
        &index,
        shapes,
        params.num_layers,
        params.num_experts,
    )
    .map_err(|e| build_err(e.to_string()))?;
    adapter
        .enable_device_staging(Box::new(pinned), params.num_layers, params.num_experts)
        .map_err(|e| build_err(e.to_string()))?;
    // Item 8 slice 2: the whole-layer prefill double buffer borrows the cache's
    // first `2 * num_experts` slots, so it needs them to exist. With the config
    // gate on, a cache at or above that size streams whole layers; below it the
    // adapter keeps the chunked LRU (decision 67).
    if config.moe.prefill_overlap && slots >= 2 * params.num_experts {
        adapter
            .enable_prefill_overlap()
            .map_err(|e| build_err(e.to_string()))?;
    }
    let moe = Arc::new(std::sync::Mutex::new(adapter));
    Ok(Some(Arc::new(CachedExperts::with_device_staging(
        moe.clone(),
        moe,
        params.num_layers,
        params.num_experts,
        slots,
    ))))
}

/// The device expert-cache slot count. An explicit `[moe] cache_size_slots` wins;
/// `"auto"` sizes it from the free device memory right after the tower and KV
/// pool are resident (item 12): the MoE cache is the "rest of the budget", minus
/// a reserve for the captured decode graphs and transient activations. The
/// result is clamped to `[MIN_CACHE_SIZE_SLOTS, the cubin's BLOCK_C ceiling]` and
/// rounded down to a multiple of 128.
#[cfg(any(feature = "cuda", feature = "cudarc"))]
fn resolve_cache_slots(
    config: &crate::config::Config,
    params: &ModelParams,
    device: Device,
    lib: Option<&crate::infrastructure::cuda_driver::Driver>,
) -> Result<usize, RegistryError> {
    use crate::domain::expert_kernel::ExpertShapes;
    use crate::infrastructure::cuda_driver::Driver;
    use crate::infrastructure::kernel_abi::StagingAbi;

    if let Some(n) = config.moe.cache_size_slots.explicit() {
        return Ok(n);
    }
    let Device::Cuda(ordinal) = device else {
        // No device to budget from: keep the conservative explicit default.
        return Ok(crate::config::DEFAULT_CACHE_SIZE_SLOTS);
    };
    let build_err = |e: String| RegistryError::Build(e);
    // Free VRAM as the driver sees it: the tower, the KV pool and the moe ports
    // are already resident, so this is exactly the expert-cache budget.
    let owned = match lib {
        Some(_) => None,
        None => {
            let d = Driver::open().map_err(|e| build_err(e.to_string()))?;
            d.init().map_err(|e| build_err(e.to_string()))?;
            d.attach_primary_context(ordinal as i32)
                .map_err(|e| build_err(e.to_string()))?;
            Some(d)
        }
    };
    let lib = lib.or(owned.as_ref()).expect("a driver is open");
    let (free, _total) = lib.mem_info().map_err(|e| build_err(e.to_string()))?;
    let shapes = ExpertShapes::new(params.hidden_size, params.moe_intermediate_size)
        .map_err(|e| build_err(e.to_string()))?;
    let slot_bytes = crate::infrastructure::cuda_moe::expert_slot_bytes(shapes);
    let ceiling = StagingAbi::committed()
        .and_then(|abi| {
            abi.kernel(crate::infrastructure::kernel_abi::LRU_ENSURE)?
                .constexpr("BLOCK_C")
        })
        .map_err(|e| build_err(e.to_string()))? as usize;
    let slots = auto_cache_slots(free, slot_bytes, ceiling);
    if std::env::var_os("FT_SLOTS_REPORT").is_some() {
        eprintln!(
            "[slots] kv_tokens={} free={} MiB reserve={} MiB slot_bytes={} ceiling={} resolved={}",
            config.kv_tokens(),
            free >> 20,
            AUTO_CACHE_RESERVE_BYTES >> 20,
            slot_bytes,
            ceiling,
            slots,
        );
    }
    Ok(slots)
}

/// Slot count from the free VRAM after the tower and KV pool are resident,
/// rounded down to the 128-slot LRU block. `free` and `slot_bytes` are bytes.
#[cfg(any(feature = "cuda", feature = "cudarc"))]
fn auto_cache_slots(free: usize, slot_bytes: usize, ceiling: usize) -> usize {
    let budget = free.saturating_sub(AUTO_CACHE_RESERVE_BYTES);
    let slots = (budget / slot_bytes.max(1)).clamp(crate::config::MIN_CACHE_SIZE_SLOTS, ceiling);
    slots / 128 * 128
}

/// Bytes held back from the auto cache budget for the captured decode graphs,
/// transient activations and allocator fragmentation. The binding constraint
/// is the prefill scratch at `PREFILL_CHUNK_TOKENS`: at 8 GiB, 1984 slots plus
/// a 2057-token prefill runs out during the silu/cat scratch, while 1920
/// (529 MiB free after load) fits. Keeping 512 MiB resolves 1920 here; the
/// larger caches this buys cut the decode miss rate (+8% tg at 1920 vs 1664).
#[cfg(any(feature = "cuda", feature = "cudarc"))]
const AUTO_CACHE_RESERVE_BYTES: usize = 512 * 1024 * 1024;

/// The paged attention launcher, when the engine runs on a CUDA device. The
/// device decode path reads each row's KV window straight out of the pool
/// through it; a CPU device gets `None` and keeps the dense gather, which is
/// what keeps `cargo test --features cuda` meaningful without a device
/// (decision 21). `stream` is the capture side stream when the engine captures,
/// otherwise stream 0 (the legacy default stream).
#[cfg(all(feature = "cuda", not(feature = "cudarc")))]
fn attention_port(
    params: &ModelParams,
    device: Device,
    stream: crate::infrastructure::cuda_driver::CUstream,
) -> Result<Option<Arc<crate::infrastructure::cuda_attention::PagedAttention>>, RegistryError> {
    use crate::infrastructure::cuda_attention::PagedAttention;
    match device {
        Device::Cuda(_) => Ok(Some(Arc::new(
            PagedAttention::new(device, params, stream)
                .map_err(|e| RegistryError::Build(e.to_string()))?,
        ))),
        _ => Ok(None),
    }
}

/// The fused GDN decode launcher, when the engine runs on a CUDA device. The
/// device decode path folds the linear-attention mixer into one kernel per layer
/// through it; a CPU device gets `None` and keeps the op-by-op libtorch path
/// (decision 21), the same shape as `attention_port`.
#[cfg(all(feature = "cuda", not(feature = "cudarc")))]
fn gdn_port(
    params: &ModelParams,
    device: Device,
    stream: crate::infrastructure::cuda_driver::CUstream,
) -> Result<Option<Arc<crate::infrastructure::cuda_gdn::GdnDecode>>, RegistryError> {
    use crate::infrastructure::cuda_gdn::GdnDecode;
    match device {
        Device::Cuda(_) => Ok(Some(Arc::new(
            GdnDecode::new(device, params, stream)
                .map_err(|e| RegistryError::Build(e.to_string()))?,
        ))),
        _ => Ok(None),
    }
}

/// The dense NVFP4 GEMV launcher the lm_head runs, when the engine runs on a
/// CUDA device. It keeps the checkpoint's packed lm_head and streams it once per
/// token instead of widening it to bf16 (TODO.md item 18); a CPU device gets
/// `None` and keeps the widened-bf16 head, the same shape as `gdn_port`.
#[cfg(any(feature = "cuda", feature = "cudarc"))]
fn head_port(
    device: Device,
    stream: crate::infrastructure::cuda_driver::CUstream,
) -> Result<Option<Arc<crate::infrastructure::cuda_linear::DenseGemv>>, RegistryError> {
    use crate::infrastructure::cuda_linear::DenseGemv;
    match device {
        Device::Cuda(_) => Ok(Some(Arc::new(
            DenseGemv::new(device, stream).map_err(|e| RegistryError::Build(e.to_string()))?,
        ))),
        _ => Ok(None),
    }
}

/// The vendored W8A16 FP8 projection launcher (decision 82 slice 2), when the
/// engine runs on a CUDA device. With it an FP8 weight stays packed and the
/// kernel reads it; a CPU device gets `None` and widens every projection to bf16.
#[cfg(any(feature = "cuda", feature = "cudarc"))]
fn fp8_port(
    device: Device,
    stream: crate::infrastructure::cuda_driver::CUstream,
) -> Result<Option<Arc<crate::infrastructure::cuda_linear::Fp8Linear>>, RegistryError> {
    use crate::infrastructure::cuda_linear::Fp8Linear;
    match device {
        Device::Cuda(_) => Ok(Some(Arc::new(
            Fp8Linear::new(device, stream).map_err(|e| RegistryError::Build(e.to_string()))?,
        ))),
        _ => Ok(None),
    }
}

/// Always the scalar reference forward, whatever the `cuda` feature says.
///
/// The reference is kept as the oracle the libtorch forward is validated
/// against (decision 12), so its own gated test has to keep reaching it even in
/// a build where `build` hands back the torch forward.
pub fn build_scalar(
    config_json: &Value,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
) -> Result<Box<dyn Model>, RegistryError> {
    match arch_of(config_json)? {
        "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5MoeForCausalLM" => {
            let tc = config_json
                .get("text_config")
                .ok_or(RegistryError::NoArch)?;
            Ok(Box::new(
                Qwen3_6::from_text_config(tc, index, reader)
                    .map_err(|e| RegistryError::Build(e.to_string()))?,
            ))
        }
        other => Err(RegistryError::Unsupported(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::ftw::FtwIndex;
    use std::str::FromStr;

    fn empty_index() -> FtwIndex {
        FtwIndex {
            shard_limit: 1024,
            total_bytes: 0,
            tensors: Vec::new(),
            shards: Vec::new(),
            quant_format: None,
            expert_bank_num_layers: None,
            meta: Default::default(),
        }
    }

    #[test]
    fn arch_of_reads_architectures_0() {
        let v: Value =
            serde_json::from_str(r#"{"architectures": ["Qwen3_5MoeForConditionalGeneration"]}"#)
                .unwrap();
        assert_eq!(arch_of(&v).unwrap(), "Qwen3_5MoeForConditionalGeneration");
    }

    #[test]
    fn build_dispatch_on_architectures_0() {
        let v: Value = serde_json::from_str(
            r#"{
              "architectures": ["Qwen3_5MoeForConditionalGeneration"],
              "text_config": {
                "hidden_size": 2048, "num_hidden_layers": 4, "vocab_size": 100,
                "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 8, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 16, "partial_rotary_factor": 0.25,
                "eos_token_id": 5,
                "rope_parameters": {"rope_theta": 10000000.0},
                "layer_types": ["linear_attention","full_attention","linear_attention","full_attention"]
              }
            }"#,
        )
        .unwrap();
        // `build_scalar`, not `build`: with the `cuda` feature `build` hands
        // back the libtorch forward, which needs the real weights at load and so
        // cannot be built against an empty index. The params assertion is about
        // dispatch on the arch label, which both paths share.
        let m = build_scalar(&v, Arc::new(empty_index()), Arc::new(MockReader)).unwrap();
        assert_eq!(m.params().num_layers, 4);
        assert_eq!(m.params().num_full_attention_layers(), 2);
    }

    #[test]
    fn unknown_arch_errors() {
        let v: Value = serde_json::from_str(r#"{"architectures": ["Qwen3.5-VL"]}"#).unwrap();
        let config = crate::config::Config::from_str("[model]\npath = \"/tmp/q.ftw\"\n").unwrap();
        assert!(matches!(
            build(
                &v,
                Arc::new(empty_index()),
                Arc::new(MockReader),
                Device::Cpu,
                &config
            ),
            Err(RegistryError::Unsupported(_))
        ));
    }

    /// Params with only the fields `moe_port` reads, so the test below needs no
    /// checkpoint and no python.
    #[cfg(feature = "cuda")]
    fn small_params() -> crate::domain::model::ModelParams {
        let tc: Value = serde_json::from_str(
            r#"{
              "hidden_size": 2048,
              "num_hidden_layers": 2,
              "vocab_size": 128,
              "num_experts": 8,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 512,
              "shared_expert_intermediate_size": 512,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 128,
              "partial_rotary_factor": 0.5,
              "rope_parameters": {"rope_theta": 10000.0},
              "layer_types": ["linear_attention", "full_attention"],
              "eos_token_id": 7
            }"#,
        )
        .unwrap();
        crate::models::qwen3_6::config::parse_text_config(&tc).unwrap()
    }

    /// With `kernel.triton` off the tower gets no port, so it keeps widening
    /// routed experts on the host. Nothing here may reach python, which is what
    /// keeps this a CPU-only test.
    #[cfg(feature = "cuda")]
    #[test]
    fn an_unconfigured_kernel_leaves_the_moe_port_out() {
        let config = crate::config::Config::from_str("[model]\npath = \"/tmp/q.ftw\"\n").unwrap();
        assert!(!config.kernel.triton);
        let port = moe_port(
            &config,
            &small_params(),
            Arc::new(empty_index()),
            Arc::new(MockReader),
            Device::Cpu,
            std::ptr::null_mut(),
            512,
        )
        .unwrap();
        assert!(port.is_none());
    }

    #[test]
    fn missing_arch_errors() {
        let v: Value = serde_json::from_str(r#"{"model_type": "qwen3_5_moe"}"#).unwrap();
        assert!(matches!(arch_of(&v), Err(RegistryError::NoArch)));
    }

    /// The captured set is `1`, `2`, `4` and `max_running`, deduplicated and
    /// capped so a small `max_running` does not capture a size the scheduler can
    /// never run.
    #[cfg(feature = "cuda")]
    #[test]
    fn graph_batch_sizes_cover_one_two_four_and_max() {
        assert_eq!(graph_batch_sizes(1), vec![1]);
        assert_eq!(graph_batch_sizes(2), vec![1, 2]);
        assert_eq!(graph_batch_sizes(3), vec![1, 2, 3]);
        assert_eq!(graph_batch_sizes(4), vec![1, 2, 4]);
        assert_eq!(graph_batch_sizes(8), vec![1, 2, 4, 8]);
    }

    /// The 8 GiB target: 3878 MiB free after the tower and KV pool, 1.69 MiB
    /// per expert slot. 512 MiB reserved leaves 1920 slots; the next 128-slot
    /// step (2048) OOMs the 2057-token prefill scratch.
    #[cfg(feature = "cuda")]
    #[test]
    fn auto_cache_slots_leave_room_for_the_prefill_scratch() {
        let mib = 1024 * 1024;
        let slot = 1_775_616;
        assert_eq!(auto_cache_slots(3878 * mib, slot, 2048), 1920);
        assert_eq!(
            auto_cache_slots(0, slot, 2048),
            crate::config::MIN_CACHE_SIZE_SLOTS
        );
        assert_eq!(auto_cache_slots(64 * 1024 * mib, slot, 2048), 2048);
    }

    /// Reader that only exists to satisfy `build`; no weights are read there.
    struct MockReader;
    impl WeightsReader for MockReader {
        fn read_range(&self, _o: u64, _n: u64, _dst: &mut [u8]) -> Result<(), String> {
            Err("no shards in registry tests".into())
        }
    }
}
