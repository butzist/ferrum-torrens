// Model registry: map the FTW dir's `config.json` arch string to a concrete
// `Model` implementation. There is no arch label in the FTW index meta, so the
// FTW dir's copied config.json is the dispatch key (TODO.md section 3).

use crate::application::engine::VisionSource;
use crate::domain::decode::CapturedDecode;
use crate::domain::kvcache::CachedModel;
use crate::domain::model::Model;
use crate::domain::weights::WeightsReader;
#[cfg(feature = "cudarc")]
use crate::domain::{expert_kernel::MoEExecutor, model::ModelParams};
#[cfg(feature = "cudarc")]
use crate::infrastructure::decode_graph::EngineDecodeGraphs;
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
    /// The device had no free memory for the model. Kept apart from `Build`
    /// because the server answers it with a retryable 503 rather than a fault:
    /// an idle-released engine simply rebuilds once memory is available again.
    #[error("the GPU has no free memory for the model: {0}")]
    OutOfDeviceMemory(String),
}

impl RegistryError {
    /// Classify a construction failure: a device out-of-memory report is
    /// retryable, anything else is a real fault. Every device-touching site
    /// funnels through here so the distinction survives to the HTTP layer.
    pub fn build(e: impl std::fmt::Display) -> Self {
        let text = e.to_string();
        if crate::domain::expert_kernel::is_out_of_device_memory(&text) {
            Self::OutOfDeviceMemory(text)
        } else {
            Self::Build(text)
        }
    }
}

/// A model plus the capture service the composition root built over its own KV
/// cache.
///
/// `captured` must drop *before* `model`: a graph bakes the model's weight and
/// pool addresses, so the allocations it replays over have to outlive it.
/// Rust drops fields in declaration order, which is why `captured` is first.
pub struct BuiltModel {
    pub captured: Option<Arc<dyn CapturedDecode>>,
    pub recapture: Option<crate::domain::decode::Recapture>,
    pub model: Arc<dyn CachedModel>,
    /// The routed-expert executor attached to the model, when the composition
    /// root built one (cudarc device staging). `None` on a CPU build.
    pub moe: Option<Arc<dyn crate::domain::expert_kernel::MoEExecutor>>,
    /// The resolved expert-cache slot count (explicit or auto-sized, item 12).
    pub cache_size_slots: usize,
    /// Where the arch's vision tower comes from for this build: `Absent` when
    /// the checkpoint has no `vision_config`, `Refused` when `model.text_only`
    /// is set, `OnDemand` (device path) when images are supported.
    pub vision: VisionSource,
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
/// The `cudarc` cargo feature picks the CUDA forward; without it, and on a
/// non-CUDA device even with it, the scalar reference is the implementation.
/// The engine cannot tell the difference -- both are `Box<dyn Model>` behind
/// the same trait.
///
/// `config` supplies the `[kernel]` and `[moe]` sections, so the tower can be
/// handed the engine's MoE port when the kernel is on; with it off the tower
/// keeps widening routed experts on the host itself.
#[cfg_attr(not(feature = "cudarc"), allow(unused_variables))]
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
                // The cudarc runtime has no CPU compute path in the serving
                // composition root, but the scalar reference still answers on a
                // CPU-only host, so `auto` keeps working there.
                let Device::Cuda(_) = device else {
                    return scalar_built(config_json, tc, index, reader, config);
                };
                // The cudarc path widens and uploads every weight itself, so
                // it takes the parsed params plus the reader rather than the
                // scalar path's pre-loaded dense store.
                let params = crate::models::qwen3_6::config::parse_text_config(tc)
                    .map_err(RegistryError::build)?;
                let capture = config.model.cuda_graph && config.kernel.triton && device.is_cuda();
                if config.model.cuda_graph && !capture {
                    eprintln!(
                        "warning: cuda_graph needs kernel.triton = true on a CUDA device; decode stays eager"
                    );
                }
                // The pool owns the runtime this whole path runs on: the model,
                // the dense launchers and the MoE adapter all launch on its
                // stream, so a step's writes and its next reads stay ordered.
                // A capture reserves the extra state slot for its padding row.
                let cache = Arc::new(kv_cache(config, &params, device, capture));
                let rt = cache.runtime().clone();
                let stream = rt.raw_stream();
                let driver = rt.driver().clone();
                let head_gemv = head_port(device, stream)?;
                let fp8_linear = fp8_port(device, stream)?;
                // The vision tower is never resident on the device between
                // requests: a text-only deployment refuses image input, a
                // checkpoint without a `vision_config` has no tower, and an
                // image-capable deployment builds it on demand from the
                // captured runtime. The factory captures the *current* runtime
                // so it keeps working after an idle release + rebuild.
                let vision = if config.model.text_only {
                    VisionSource::Refused
                } else if config_json.get("vision_config").is_some() {
                    let cj = config_json.clone();
                    let idx = index.clone();
                    let rdr = reader.clone();
                    let rt2 = rt.clone();
                    let cache_key = config.model.path.clone();
                    VisionSource::OnDemand(Arc::new(move || {
                        crate::models::qwen3_6::vision_cudarc::build_device_vision(
                            &cj,
                            idx.clone(),
                            rdr.clone(),
                            rt2.clone(),
                            &cache_key,
                        )
                        .map(|tower| tower.expect("vision_config is present"))
                        .map_err(crate::application::engine::RebuildError::Failed)
                    }))
                } else {
                    VisionSource::Absent
                };
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
                    shared_expert_fused_port(device, stream).ok().flatten(),
                    fp8_linear,
                    cache.clone(),
                )
                .map_err(RegistryError::build)?;
                // M9: the MTP draft head loads *before* the auto cache budget so
                // its ~0.5 GB resident bank comes out of what the target's
                // expert cache may size itself to, not out of the reserve.
                if config.speculative_algorithm() == Some("mtp") {
                    if index.has_mtp == Some(true) {
                        match model.enable_draft(config.kv_tokens()) {
                            Ok(_) => {
                                match draft_moe_port(
                                    config,
                                    &params,
                                    index.clone(),
                                    reader.clone(),
                                    device,
                                    stream,
                                ) {
                                    Ok(moe) => model.attach_draft_moe(moe),
                                    Err(e) => eprintln!(
                                        "warning: the MTP draft's expert bank did not build ({e}); \
                                         speculation stays off"
                                    ),
                                }
                            }
                            Err(e) => eprintln!(
                                "warning: the MTP draft head did not load ({e}); \
                                 speculation stays off"
                            ),
                        }
                    } else {
                        eprintln!(
                            "warning: [spec] asks for mtp but this checkpoint has no draft head; \
                             decode stays plain"
                        );
                    }
                }
                let slots = resolve_cache_slots(config, &params, device, Some(driver.as_ref()))?;
                let moe = moe_port(
                    config,
                    &params,
                    index.clone(),
                    reader.clone(),
                    device,
                    stream,
                    slots,
                )?;
                let moe_for_model = moe.clone();
                model.attach_moe(moe_for_model);
                let model = Arc::new(model);
                let captured = if capture {
                    match capture_decode(config, &model, &cache, model.runtime()) {
                        Ok(graphs) => Some(Arc::new(graphs) as Arc<dyn CapturedDecode>),
                        Err(e) => {
                            // A failed capture must not break serving: report the
                            // op that refused and fall back to the eager step.
                            eprintln!(
                                "warning: decode graph capture failed ({e}); decode stays eager"
                            );
                            None
                        }
                    }
                } else {
                    None
                };
                // A verify graph is only ever replayed by `[spec] mtp`; without
                // speculation it would cost a capture and its buffers for a
                // step nothing runs.
                if capture && config.speculative_algorithm() == Some("mtp") {
                    let rows = config.speculative_num_steps() + 1;
                    match capture_verify(config, &model, &cache, model.runtime(), rows) {
                        Ok(graphs) => {
                            model.attach_captured_verify(
                                Arc::new(graphs) as Arc<dyn CapturedDecode>,
                            );
                        }
                        Err(e) => eprintln!(
                            "warning: verify graph capture failed ({e}); verify stays eager"
                        ),
                    }
                }
                // The padding row ran `lru_ensure` during capture and polluted the
                // device LRU tables (step counter + usage). Reset them so the
                // first real replay starts from a cold cache, matching the eager
                // path's slot assignment (decision 164). Only when capture ran:
                // otherwise the eager path's live cache must survive.
                if capture && let Some(moe) = &moe {
                    let _ = moe.reset_staging();
                }
                let recapture = if capture {
                    let weak_model = std::sync::Arc::downgrade(&model);
                    let weak_cache = std::sync::Arc::downgrade(&cache);
                    let config = config.clone();
                    Some(Arc::new(move || {
                        let m = weak_model
                            .upgrade()
                            .ok_or_else(|| "the model was dropped".to_string())?;
                        let c = weak_cache
                            .upgrade()
                            .ok_or_else(|| "the kv pool was dropped".to_string())?;
                        capture_decode(&config, &m, &c, m.runtime())
                            .map(|g| Arc::new(g) as Arc<dyn CapturedDecode>)
                            .map_err(|e| e.to_string())
                    })
                        as Arc<
                            dyn Fn() -> Result<Arc<dyn CapturedDecode>, String> + Send + Sync,
                        >)
                } else {
                    None
                };
                Ok(BuiltModel {
                    captured,
                    recapture,
                    model: model as Arc<dyn CachedModel>,
                    moe,
                    cache_size_slots: slots,
                    vision,
                })
            }
            #[cfg(not(feature = "cudarc"))]
            scalar_built(config_json, tc, index, reader, config)
        }
        other => Err(RegistryError::Unsupported(other.to_string())),
    }
}

/// The scalar reference model, the answer on a host with no CUDA device and on
/// a build with no compute backend at all.
fn scalar_built(
    config_json: &Value,
    tc: &Value,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
    config: &crate::config::Config,
) -> Result<BuiltModel, RegistryError> {
    if config.model.cuda_graph {
        eprintln!(
            "warning: cuda_graph needs the cudarc runtime on a CUDA device; decode stays eager"
        );
    }
    Ok(BuiltModel {
        captured: None,
        recapture: None,
        model: Arc::new(
            Qwen3_6::from_text_config(tc, index.clone(), reader.clone())
                .map_err(RegistryError::build)?,
        ),
        moe: None,
        cache_size_slots: config
            .moe
            .cache_size_slots
            .explicit()
            .unwrap_or(crate::config::DEFAULT_CACHE_SIZE_SLOTS),
        vision: if config.model.text_only {
            VisionSource::Refused
        } else {
            match load_vision(config_json, index, reader)? {
                Some(tower) => {
                    let t = tower.clone();
                    VisionSource::OnDemand(Arc::new(move || Ok(t.clone())))
                }
                None => VisionSource::Absent,
            }
        },
    })
}

/// Load the arch's vision tower when the checkpoint declares one: parse the
/// `vision_config`, load its `visual.*` weights on the host with a registry
/// error if anything is missing — this runs once at build time, so a missing
/// tensor is a hard failure, not a request-time surprise.
pub fn load_vision(
    config_json: &Value,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
) -> Result<Option<Arc<dyn crate::domain::model::VisionTower>>, RegistryError> {
    let Some(vc) = config_json.get("vision_config") else {
        return Ok(None);
    };
    let id = config_json
        .get("image_token_id")
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| {
            RegistryError::build("vision_config present but image_token_id is not a u32")
        })?;
    let vcfg =
        crate::models::qwen3_6::config::parse_vision_config(vc).map_err(RegistryError::build)?;
    let weights = crate::models::qwen3_6::weights::VisionWeights::load(&index, reader.as_ref())
        .map_err(RegistryError::build)?;
    let tower = crate::models::qwen3_6::vision::VisionTower::new(vcfg, weights);
    Ok(Some(std::sync::Arc::new(
        crate::models::qwen3_6::vision::Qwen3_6Vision::new(tower, id),
    )))
}

/// Record one decode graph per batch size and wrap them in the engine's service.
/// The padding row claims a reserved state slot and pool location, and the
/// reservation itself moves into the service so nothing else can take them.
#[cfg(feature = "cudarc")]
fn capture_decode(
    config: &crate::config::Config,
    model: &crate::models::qwen3_6::forward_cudarc::CudarcQwen3_6,
    cache: &crate::infrastructure::kvcache_dev::DeviceCache,
    rt: &crate::infrastructure::cudarc::Runtime,
) -> Result<EngineDecodeGraphs<crate::infrastructure::decode_graph::DecodeGraph>, RegistryError> {
    use crate::domain::decode::DecodeRow;
    use crate::domain::kvcache::KvCache;
    use crate::infrastructure::decode_buffers::DecodeBuffers;

    let build_err = |e: String| RegistryError::build(e);
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
    // SAFETY: `rt` is the model's non-default stream on the device's primary
    // context, and the step is the strictly device-only decode forward (no host
    // copy, no allocation outside the capture pool, no synchronization).
    unsafe {
        EngineDecodeGraphs::capture(
            rt,
            model.params().vocab_size,
            config.model.max_seq_len,
            &sizes,
            padding,
            padding_cache,
            |bufs: &DecodeBuffers| model.forward_decode_into(bufs).map_err(|e| e.to_string()),
        )
    }
    .map_err(RegistryError::build)
}

/// Record one verify graph at `rows` rows (typically `k + 1`).
#[cfg(feature = "cudarc")]
fn capture_verify(
    config: &crate::config::Config,
    model: &crate::models::qwen3_6::forward_cudarc::CudarcQwen3_6,
    cache: &crate::infrastructure::kvcache_dev::DeviceCache,
    rt: &crate::infrastructure::cudarc::Runtime,
    rows: usize,
) -> Result<EngineDecodeGraphs<crate::infrastructure::decode_graph::DecodeGraph>, RegistryError> {
    use crate::domain::decode::DecodeRow;
    use crate::domain::kvcache::KvCache;
    use crate::infrastructure::decode_buffers::DecodeBuffers;

    let build_err = |e: String| RegistryError::build(e);
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
    let sizes = vec![rows];
    unsafe {
        EngineDecodeGraphs::capture(
            rt,
            model.params().vocab_size,
            config.model.max_seq_len,
            &sizes,
            padding,
            padding_cache,
            |bufs: &DecodeBuffers| model.forward_verify_into(bufs).map_err(|e| e.to_string()),
        )
    }
    .map_err(RegistryError::build)
}

/// The batch sizes to capture: `1`, `2`, `4` and `max_running`, deduplicated and
/// capped at `max_running`. A step of `n` rows replays on the smallest captured
/// size at or above `n`.
#[cfg(feature = "cudarc")]
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
#[cfg(feature = "cudarc")]
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

/// The MTP draft's expert port: the same adapter and slot machine as the
/// target's, over a `draft_`-prefixed bank space and sized to hold every draft
/// expert (433 MiB NVFP4), so after warmup the draft never copies a bank. No
/// pinned host staging: the draft's whole bank budget is resident.
#[cfg(feature = "cudarc")]
fn draft_moe_port(
    config: &crate::config::Config,
    params: &ModelParams,
    index: Arc<FtwIndex>,
    reader: Arc<dyn WeightsReader>,
    device: Device,
    stream: crate::infrastructure::cuda_driver::CUstream,
) -> Result<Option<Arc<dyn MoEExecutor>>, RegistryError> {
    use crate::application::run_experts::CachedExperts;
    use crate::domain::expert_kernel::ExpertShapes;
    use crate::infrastructure::banks::FtwBanks;
    use crate::infrastructure::cuda_moe::CudaMoE;

    if !config.kernel.triton {
        return Ok(None);
    }
    let shapes = ExpertShapes::new(params.hidden_size, params.moe_intermediate_size)
        .map_err(|e| RegistryError::build(e.to_string()))?;
    // Every draft expert stays resident (433 MiB of NVFP4): a draft row is on
    // the critical path of every step, so a miss's 1.7 MiB read is worth
    // avoiding. `resolve_cache_slots` runs after this and sizes the target's
    // cache around whatever is left.
    let slots = params.num_experts;
    let adapter = CudaMoE::new_with_stream(
        device,
        Arc::new(FtwBanks::draft(reader, index)),
        shapes,
        slots,
        stream,
    )
    .map_err(|e| RegistryError::build(e.to_string()))?;
    let moe = Arc::new(std::sync::Mutex::new(adapter));
    Ok(Some(Arc::new(CachedExperts::new(
        moe.clone(),
        moe,
        1,
        params.num_experts,
        slots,
    ))))
}

/// The engine's MoE port for `kernel.triton = true`: the driver-API adapter over
/// the mmap banks, wrapped in the slot-cache executor that plans residency per
/// layer. `None` whenever the kernel is off, which is what leaves the tower
/// widening routed experts on the host.
///
/// Composition root work on purpose: this is the one place that knows both the
/// config and the adapter, and it is arch-agnostic -- a new MoE arch with the
/// same bank layout gets the same executor.
#[cfg(feature = "cudarc")]
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
    let build_err = |e: String| RegistryError::build(e);
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
#[cfg(feature = "cudarc")]
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
    let build_err = |e: String| RegistryError::build(e);
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
#[cfg(feature = "cudarc")]
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
#[cfg(feature = "cudarc")]
const AUTO_CACHE_RESERVE_BYTES: usize = 512 * 1024 * 1024;

/// The dense NVFP4 GEMV launcher the lm_head runs, when the engine runs on a
/// CUDA device. It keeps the checkpoint's packed lm_head and streams it once per
/// token instead of widening it to bf16 (TODO.md item 18); a CPU device gets
/// `None` and keeps the widened-bf16 head.
#[cfg(feature = "cudarc")]
fn head_port(
    device: Device,
    stream: crate::infrastructure::cuda_driver::CUstream,
) -> Result<Option<Arc<crate::infrastructure::cuda_linear::DenseGemv>>, RegistryError> {
    use crate::infrastructure::cuda_linear::DenseGemv;
    match device {
        Device::Cuda(_) => Ok(Some(Arc::new(
            DenseGemv::new(device, stream).map_err(RegistryError::build)?,
        ))),
        _ => Ok(None),
    }
}

#[cfg(feature = "cudarc")]
fn shared_expert_fused_port(
    device: Device,
    stream: crate::infrastructure::cuda_driver::CUstream,
) -> Result<Option<Arc<crate::infrastructure::cuda_linear::SharedExpertFused>>, RegistryError> {
    use crate::infrastructure::cuda_linear::SharedExpertFused;
    match device {
        Device::Cuda(_) => Ok(Some(Arc::new(
            SharedExpertFused::new(device, stream).map_err(RegistryError::build)?,
        ))),
        _ => Ok(None),
    }
}

/// The vendored W8A16 FP8 projection launcher (decision 82 slice 2), when the
/// engine runs on a CUDA device. With it an FP8 weight stays packed and the
/// kernel reads it; a CPU device gets `None` and widens every projection to bf16.
#[cfg(feature = "cudarc")]
fn fp8_port(
    device: Device,
    stream: crate::infrastructure::cuda_driver::CUstream,
) -> Result<Option<Arc<crate::infrastructure::cuda_linear::Fp8Linear>>, RegistryError> {
    use crate::infrastructure::cuda_linear::Fp8Linear;
    match device {
        Device::Cuda(_) => Ok(Some(Arc::new(
            Fp8Linear::new(device, stream).map_err(RegistryError::build)?,
        ))),
        _ => Ok(None),
    }
}

/// Always the scalar reference forward, whatever the `cuda` feature says.
///
/// The reference is kept as the oracle the scalar and CUDA forwards are
/// validated against (decision 12), so its own gated test keeps reaching it even
/// in a build where `build` hands back the CUDA forward.
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
                Qwen3_6::from_text_config(tc, index, reader).map_err(RegistryError::build)?,
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
            has_mtp: None,
            max_draft_depth: None,
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
        // `build_scalar`, not `build`: `build` hands back the CUDA forward,
        // which needs the real weights at load and so cannot be built against
        // an empty index. The params assertion is about dispatch on the arch
        // label, which both paths share.
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
    #[cfg(feature = "cudarc")]
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
    #[cfg(feature = "cudarc")]
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

    #[cfg(feature = "cudarc")]
    #[test]
    fn graph_batch_sizes_cover_one_two_four_and_the_cap() {
        assert_eq!(graph_batch_sizes(1), vec![1]);
        assert_eq!(graph_batch_sizes(2), vec![1, 2]);
        assert_eq!(graph_batch_sizes(3), vec![1, 2, 3]);
        assert_eq!(graph_batch_sizes(4), vec![1, 2, 4]);
        assert_eq!(graph_batch_sizes(8), vec![1, 2, 4, 8]);
    }

    /// The 8 GiB target: 3878 MiB free after the tower and KV pool, 1.69 MiB
    /// per expert slot. 512 MiB reserved leaves 1920 slots; the next 128-slot
    /// step (2048) OOMs the 2057-token prefill scratch.
    #[cfg(feature = "cudarc")]
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
