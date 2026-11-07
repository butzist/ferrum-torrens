// HF Qwen3.6-35B-A3B-NVFP4 -> FTW mapping: prefix stripping, Gemma norm bake,
// FP8/NVFP4 role expansion, the fused projections, and the per-layer NVFP4
// routed-expert banks. Everything is CPU-only and streams one tensor at a time.

use super::ftw::{Counts, FtwError, FtwMeta, FtwWriter};
use super::safetensors::{SafeTensorsError, TensorStore};
use crate::domain::weights::DType;
use crate::infrastructure::ftw::DEFAULT_SHARD_LIMIT;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error(transparent)]
    SafeTensors(#[from] SafeTensorsError),
    #[error(transparent)]
    Ftw(#[from] FtwError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid checkpoint: {0}")]
    Invalid(String),
}

fn invalid(msg: impl Into<String>) -> ConvertError {
    ConvertError::Invalid(msg.into())
}

#[derive(Debug, Clone)]
pub struct ConvertOptions {
    pub model_dir: PathBuf,
    pub out_dir: PathBuf,
    pub shard_limit: u64,
    /// Only the first N transformer layers (0-based count); `None` keeps all.
    pub layers: Option<usize>,
    /// Output-name globs (`*` only); `None` keeps every tensor.
    pub limit: Option<Vec<String>>,
}

impl ConvertOptions {
    pub fn new(model_dir: impl Into<PathBuf>, out_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            out_dir: out_dir.into(),
            shard_limit: DEFAULT_SHARD_LIMIT,
            layers: None,
            limit: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConvertReport {
    pub weight_count: usize,
    pub experts_bank_count: usize,
    pub total_bytes: u64,
    pub shard_count: usize,
    pub copied_metadata: Vec<String>,
    pub elapsed: Duration,
}

pub fn convert(opts: &ConvertOptions) -> Result<ConvertReport, ConvertError> {
    let start = Instant::now();
    let store = TensorStore::open(&opts.model_dir)?;
    let (pass, groups) = build_plan(&store, opts)?;
    let mut writer = FtwWriter::new(&opts.out_dir, opts.shard_limit)?;
    let mut emitter = Emitter {
        store: &store,
        writer: &mut writer,
        limit: opts.limit.as_deref(),
        weight_count: 0,
        bank_count: 0,
    };
    emitter.emit_pass(&pass)?;
    emitter.emit_groups(&groups)?;
    let bank_layers = emitter.emit_banks(opts.layers)?;
    let draft_layers = emitter.emit_draft_banks()?;
    let (weight_count, bank_count) = (emitter.weight_count, emitter.bank_count);

    let copied_metadata = copy_metadata(&opts.model_dir, &opts.out_dir)?;
    let source = opts
        .model_dir
        .canonicalize()
        .unwrap_or_else(|_| opts.model_dir.clone());
    let fingerprint = fingerprint(&source, store.shard_files());
    let fin = writer.finalize(FtwMeta {
        quant_format: bank_layers.is_some().then(|| "nvfp4".to_string()),
        expert_bank_num_layers: bank_layers,
        has_mtp: Some(draft_layers.is_some()),
        max_draft_depth: draft_layers,
        counts: Counts {
            weight: weight_count,
            experts_bank: bank_count,
        },
        source_model_path: source.to_string_lossy().into_owned(),
        fingerprint,
        copied_metadata: copied_metadata.clone(),
        side_files: Vec::new(),
    })?;

    Ok(ConvertReport {
        weight_count,
        experts_bank_count: bank_count,
        total_bytes: fin.total_bytes,
        shard_count: fin.shard_count,
        copied_metadata,
        elapsed: start.elapsed(),
    })
}

// ============================== planning ==============================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantClass {
    Fp8,
    Nvfp4,
    Concat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Weight,
    WeightScale,
    WeightScale2,
    InputScale,
}

#[derive(Debug, Default, Clone)]
struct PartPlan {
    weight: Option<String>,
    weight_scale: Option<String>,
    weight_scale_2: Option<String>,
    input_scale: Option<String>,
}

#[derive(Debug, Clone)]
struct GroupPlan {
    class: QuantClass,
    parts: BTreeMap<usize, PartPlan>,
}

struct PassPlan {
    src: String,
}

type Plan = (BTreeMap<String, PassPlan>, BTreeMap<String, GroupPlan>);

fn build_plan(store: &TensorStore, opts: &ConvertOptions) -> Result<Plan, ConvertError> {
    let mut pass: BTreeMap<String, PassPlan> = BTreeMap::new();
    let mut groups: BTreeMap<String, GroupPlan> = BTreeMap::new();
    for src in store.names() {
        if skip_source(src) {
            continue;
        }
        let out = rename(src);
        if let (Some(n), Some(layer)) = (opts.layers, layer_of(&out))
            && layer >= n
        {
            continue;
        }
        let Some((module, role)) = module_role(&out) else {
            insert_pass(&mut pass, out, src)?;
            continue;
        };
        let Some((class, target, part)) = quant_target(module) else {
            insert_pass(&mut pass, out, src)?;
            continue;
        };
        let group = groups.entry(target).or_insert_with(|| GroupPlan {
            class,
            parts: BTreeMap::new(),
        });
        if group.class != class {
            return Err(invalid(format!(
                "{module:?} maps to a {class:?} group already planned as {:?}",
                group.class
            )));
        }
        let slot = group.parts.entry(part).or_default();
        match (class, role) {
            (QuantClass::Fp8, Role::Weight) => set_slot(&mut slot.weight, src)?,
            (QuantClass::Fp8, Role::WeightScale) => set_slot(&mut slot.weight_scale, src)?,
            (QuantClass::Fp8, Role::InputScale) => set_slot(&mut slot.input_scale, src)?,
            (QuantClass::Nvfp4, Role::Weight) => set_slot(&mut slot.weight, src)?,
            (QuantClass::Nvfp4, Role::WeightScale) => set_slot(&mut slot.weight_scale, src)?,
            (QuantClass::Nvfp4, Role::WeightScale2) => set_slot(&mut slot.weight_scale_2, src)?,
            (QuantClass::Nvfp4, Role::InputScale) => {}
            (QuantClass::Concat, Role::Weight) => set_slot(&mut slot.weight, src)?,
            (class, role) => {
                return Err(invalid(format!(
                    "{src:?}: {role:?} is not a stored role for {class:?}"
                )));
            }
        }
    }
    Ok((pass, groups))
}

fn insert_pass(
    pass: &mut BTreeMap<String, PassPlan>,
    out: String,
    src: &str,
) -> Result<(), ConvertError> {
    if pass
        .insert(
            out.clone(),
            PassPlan {
                src: src.to_string(),
            },
        )
        .is_some()
    {
        return Err(invalid(format!(
            "two sources map to the dense tensor {out:?}"
        )));
    }
    Ok(())
}

fn set_slot(slot: &mut Option<String>, src: &str) -> Result<(), ConvertError> {
    if slot.is_some() {
        return Err(invalid(format!("duplicate source for {src:?}")));
    }
    *slot = Some(src.to_string());
    Ok(())
}

fn rename(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("mtp.") {
        return format!("draft.{rest}");
    }
    if name.starts_with("model.visual.") || name.starts_with("visual.") {
        return format!(
            "visual.{}",
            name.strip_prefix("model.")
                .unwrap_or(name)
                .strip_prefix("visual.")
                .unwrap_or(name)
        );
    }
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        return format!("model.{rest}");
    }
    if let Some(rest) = name.strip_prefix("language_model.") {
        return format!("model.{rest}");
    }
    name.to_string()
}

fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.layers.")?;
    let (idx, _) = rest.split_once('.')?;
    idx.parse().ok()
}

fn skip_source(name: &str) -> bool {
    name.ends_with(".k_scale")
        || name.ends_with(".v_scale")
        || name.ends_with(".q_scale")
        || name.ends_with(".prob_scale")
        || name.ends_with(".input_global_scale")
        || is_expert_piece(name)
        || (name.starts_with("mtp.") && name.contains(".mlp.experts."))
}

fn is_expert_piece(name: &str) -> bool {
    let Some(at) = name.find(".mlp.experts.") else {
        return false;
    };
    let rest = &name[at + ".mlp.experts.".len()..];
    rest.split_once('.')
        .is_some_and(|(idx, _)| idx.parse::<usize>().is_ok())
}

fn module_role(name: &str) -> Option<(&str, Role)> {
    for (suffix, role) in [
        (".weight_scale_2", Role::WeightScale2),
        (".weight_scale", Role::WeightScale),
        (".input_scale", Role::InputScale),
        (".weight", Role::Weight),
    ] {
        if let Some(module) = name.strip_suffix(suffix) {
            return Some((module, role));
        }
    }
    None
}

fn quant_target(module: &str) -> Option<(QuantClass, String, usize)> {
    // The checkpoint's quant config excludes the MTP head (`exclude_modules:
    // mtp*`), so its linears are already bf16: only the fusions apply, and they
    // pass bf16 through instead of expecting pre-packed NVFP4 or FP8 parts.
    // `module` is the renamed output name, so the draft prefix is `draft.`.
    if module.starts_with("draft.") {
        for (suffix, part) in [
            (".self_attn.q_proj", 0),
            (".self_attn.k_proj", 1),
            (".self_attn.v_proj", 2),
        ] {
            if let Some(prefix) = module.strip_suffix(suffix) {
                return Some((
                    QuantClass::Concat,
                    format!("{prefix}.self_attn.qkv_proj"),
                    part,
                ));
            }
        }
        for (suffix, part) in [
            (".mlp.shared_expert.gate_proj", 0),
            (".mlp.shared_expert.up_proj", 1),
        ] {
            if let Some(prefix) = module.strip_suffix(suffix) {
                return Some((
                    QuantClass::Concat,
                    format!("{prefix}.mlp.shared_expert.gate_up_proj"),
                    part,
                ));
            }
        }
        return None;
    }
    if let Some(prefix) = module.strip_suffix(".self_attn.q_proj") {
        return Some((QuantClass::Fp8, format!("{prefix}.self_attn.qkv_proj"), 0));
    }
    if let Some(prefix) = module.strip_suffix(".self_attn.k_proj") {
        return Some((QuantClass::Fp8, format!("{prefix}.self_attn.qkv_proj"), 1));
    }
    if let Some(prefix) = module.strip_suffix(".self_attn.v_proj") {
        return Some((QuantClass::Fp8, format!("{prefix}.self_attn.qkv_proj"), 2));
    }
    if module.ends_with(".self_attn.o_proj") {
        return Some((QuantClass::Fp8, module.to_string(), 0));
    }
    if let Some(prefix) = module.strip_suffix(".linear_attn.in_proj_qkv") {
        return Some((
            QuantClass::Fp8,
            format!("{prefix}.linear_attn.in_proj_qkvz"),
            0,
        ));
    }
    if let Some(prefix) = module.strip_suffix(".linear_attn.in_proj_z") {
        return Some((
            QuantClass::Fp8,
            format!("{prefix}.linear_attn.in_proj_qkvz"),
            1,
        ));
    }
    if module.ends_with(".linear_attn.out_proj") {
        return Some((QuantClass::Fp8, module.to_string(), 0));
    }
    if let Some(prefix) = module.strip_suffix(".linear_attn.in_proj_b") {
        return Some((
            QuantClass::Concat,
            format!("{prefix}.linear_attn.in_proj_ba"),
            0,
        ));
    }
    if let Some(prefix) = module.strip_suffix(".linear_attn.in_proj_a") {
        return Some((
            QuantClass::Concat,
            format!("{prefix}.linear_attn.in_proj_ba"),
            1,
        ));
    }
    if let Some(prefix) = module.strip_suffix(".mlp.shared_expert.gate_proj") {
        return Some((
            QuantClass::Nvfp4,
            format!("{prefix}.mlp.shared_expert.gate_up_proj"),
            0,
        ));
    }
    if let Some(prefix) = module.strip_suffix(".mlp.shared_expert.up_proj") {
        return Some((
            QuantClass::Nvfp4,
            format!("{prefix}.mlp.shared_expert.gate_up_proj"),
            1,
        ));
    }
    if module.ends_with(".mlp.shared_expert.down_proj") {
        return Some((QuantClass::Nvfp4, module.to_string(), 0));
    }
    if module == "lm_head" {
        return Some((QuantClass::Nvfp4, "lm_head".to_string(), 0));
    }
    None
}

fn is_gemma_norm(name: &str) -> bool {
    name == "model.norm.weight"
        // The MTP draft's own norms are the same GemmaPlusOne form (upstream
        // builds them as GemmaPlusOneRMSNorm), so the bake is not target-only.
        || name == "draft.norm.weight"
        || name == "draft.pre_fc_norm_embedding.weight"
        || name == "draft.pre_fc_norm_hidden.weight"
        || name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
        || name.ends_with(".self_attn.q_norm.weight")
        || name.ends_with(".self_attn.k_norm.weight")
}

pub fn glob_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let first = parts[0];
    if !text.starts_with(first) {
        return false;
    }
    let mut pos = first.len();
    for part in &parts[1..parts.len() - 1] {
        let Some(found) = text[pos..].find(part) else {
            return false;
        };
        pos += found + part.len();
    }
    let last = parts[parts.len() - 1];
    text.len() >= pos + last.len() && text[pos..].ends_with(last)
}

// ============================== emission ==============================

struct Emitter<'a> {
    store: &'a TensorStore,
    writer: &'a mut FtwWriter,
    limit: Option<&'a [String]>,
    weight_count: usize,
    bank_count: usize,
}

impl Emitter<'_> {
    fn enabled(&self, name: &str) -> bool {
        match self.limit {
            None => true,
            Some(patterns) => patterns.iter().any(|p| glob_match(p, name)),
        }
    }

    fn add(
        &mut self,
        name: &str,
        kind: &str,
        dtype: DType,
        shape: &[usize],
        bytes: &[u8],
    ) -> Result<(), ConvertError> {
        if kind == "weight" {
            self.weight_count += 1;
        } else {
            self.bank_count += 1;
        }
        self.writer.add_tensor(name, kind, dtype, shape, bytes)?;
        Ok(())
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, ConvertError> {
        Ok(self.store.read(name)?)
    }

    fn emit_pass(&mut self, pass: &BTreeMap<String, PassPlan>) -> Result<(), ConvertError> {
        for (out, plan) in pass {
            if !self.enabled(out) {
                continue;
            }
            let info = self
                .store
                .info(&plan.src)
                .ok_or_else(|| invalid(format!("missing {}", plan.src)))?;
            let (dtype, shape) = (info.dtype, info.shape.clone());
            let mut bytes = self.read(&plan.src)?;
            if is_gemma_norm(out) {
                match dtype {
                    DType::Bf16 => add_one_bf16(&mut bytes),
                    DType::F16 => add_one_f16(&mut bytes),
                    DType::F32 => add_one_f32(&mut bytes),
                    _ => {}
                }
            }
            self.add(out, "weight", dtype, &shape, &bytes)?;
        }
        Ok(())
    }

    fn emit_groups(&mut self, groups: &BTreeMap<String, GroupPlan>) -> Result<(), ConvertError> {
        for (target, group) in groups {
            match group.class {
                QuantClass::Fp8 => self.emit_fp8(target, group)?,
                QuantClass::Nvfp4 => self.emit_nvfp4(target, group)?,
                QuantClass::Concat => self.emit_concat(target, group)?,
            }
        }
        Ok(())
    }

    fn emit_fp8(&mut self, target: &str, group: &GroupPlan) -> Result<(), ConvertError> {
        let weight = format!("{target}.weight");
        if self.enabled(&weight) {
            let parts = ordered_parts(group, Role::Weight)?;
            let (shape, bytes) = concat(&parts, DType::F8E4M3, self.store)?;
            self.add(&weight, "weight", DType::F8E4M3, &shape, &bytes)?;
        }
        let scale = format!("{target}.weight_scale");
        if self.enabled(&scale) {
            let parts = ordered_parts(group, Role::WeightScale)?;
            let rows = fp8_rows(group, &parts, self.store)?;
            let bytes = fp8_scales(group, &parts, self.store)?;
            self.add(&scale, "weight", DType::F32, &[rows], &bytes)?;
        }
        let input = format!("{target}.input_scale");
        if self.enabled(&input) {
            let parts = ordered_parts(group, Role::InputScale)?;
            let value = fp8_input_scale(&parts, self.store)?;
            self.add(&input, "weight", DType::F32, &[], &value.to_le_bytes())?;
        }
        Ok(())
    }

    fn emit_nvfp4(&mut self, target: &str, group: &GroupPlan) -> Result<(), ConvertError> {
        let weight = format!("{target}.weight");
        if self.enabled(&weight) {
            let parts = ordered_parts(group, Role::Weight)?;
            let (shape, bytes) = concat(&parts, DType::U8, self.store)?;
            self.add(&weight, "weight", DType::U8, &shape, &bytes)?;
        }
        let scale = format!("{target}.weight_scale");
        if self.enabled(&scale) {
            let parts = ordered_parts(group, Role::WeightScale)?;
            let (shape, bytes) = concat(&parts, DType::F8E4M3, self.store)?;
            self.add(&scale, "weight", DType::F8E4M3, &shape, &bytes)?;
        }
        let global = format!("{target}.weight_global");
        if self.enabled(&global) {
            let parts = ordered_parts(group, Role::WeightScale2)?;
            let rows = fp8_rows(group, &parts, self.store)?;
            let bytes = nvfp4_globals(group, &parts, self.store)?;
            self.add(&global, "weight", DType::F16, &[rows], &bytes)?;
        }
        Ok(())
    }

    fn emit_concat(&mut self, target: &str, group: &GroupPlan) -> Result<(), ConvertError> {
        let weight = format!("{target}.weight");
        if !self.enabled(&weight) {
            return Ok(());
        }
        let parts = ordered_parts(group, Role::Weight)?;
        let dtype = parts
            .first()
            .and_then(|(_, src)| self.store.info(src))
            .map(|i| i.dtype)
            .ok_or_else(|| invalid(format!("{target} has no weight parts")))?;
        let (shape, bytes) = concat(&parts, dtype, self.store)?;
        self.add(&weight, "weight", dtype, &shape, &bytes)?;
        Ok(())
    }

    fn emit_banks(&mut self, layers: Option<usize>) -> Result<Option<usize>, ConvertError> {
        let Some(dims) = expert_dims(self.store) else {
            return Ok(None);
        };
        let selected: Vec<usize> = dims
            .layers
            .iter()
            .copied()
            .filter(|l| layers.is_none_or(|n| *l < n))
            .collect();
        let mut written = 0usize;
        for layer in selected {
            if self.emit_bank_layer(layer, dims.num_experts)? {
                written += 1;
            }
        }
        Ok((written > 0).then_some(written))
    }

    fn emit_bank_layer(&mut self, layer: usize, experts: usize) -> Result<bool, ConvertError> {
        let names = [
            format!("gate_up_packed#L{layer:05}"),
            format!("gate_up_scale#L{layer:05}"),
            format!("gate_up_global#L{layer:05}"),
            format!("down_packed#L{layer:05}"),
            format!("down_scale#L{layer:05}"),
            format!("down_global#L{layer:05}"),
        ];
        let enabled: Vec<bool> = names.iter().map(|n| self.enabled(n)).collect();
        if !enabled.iter().any(|&e| e) {
            return Ok(false);
        }
        let base =
            |expert: usize| format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
        let first = base(0);
        let gate = expert_info(self.store, &first, "gate_proj", "weight")?;
        let up = expert_info(self.store, &first, "up_proj", "weight")?;
        let down = expert_info(self.store, &first, "down_proj", "weight")?;
        let gate_scale = expert_info(self.store, &first, "gate_proj", "weight_scale")?;
        let down_scale = expert_info(self.store, &first, "down_proj", "weight_scale")?;
        check_dtype(gate, DType::U8, &first)?;
        check_dtype(up, DType::U8, &first)?;
        check_dtype(down, DType::U8, &first)?;
        check_dtype(gate_scale, DType::F8E4M3, &first)?;
        check_dtype(down_scale, DType::F8E4M3, &first)?;
        if gate.shape != up.shape {
            return Err(invalid(format!(
                "{first}: gate_proj and up_proj shapes differ"
            )));
        }
        let gate_rows = gate.shape[0];
        let gate_cols = gate.shape[1];
        let down_rows = down.shape[0];
        let down_cols = down.shape[1];
        if gate_scale.shape[0] != gate_rows || down_scale.shape[0] != down_rows {
            return Err(invalid(format!(
                "{first}: scale rows do not match the weight rows"
            )));
        }
        let gate_scale_cols = gate_scale.shape[1];
        let down_scale_cols = down_scale.shape[1];

        if enabled[0] {
            let mut bytes = Vec::with_capacity(experts * 2 * gate_rows * gate_cols);
            for e in 0..experts {
                let b = base(e);
                append_expert(&mut bytes, self.store, &b, "gate_proj", "weight", gate)?;
                append_expert(&mut bytes, self.store, &b, "up_proj", "weight", up)?;
            }
            let shape = [experts, 2 * gate_rows, gate_cols];
            self.add(&names[0], "experts_bank", DType::U8, &shape, &bytes)?;
        }
        if enabled[1] {
            let mut bytes = Vec::with_capacity(experts * 2 * gate_rows * gate_scale_cols);
            for e in 0..experts {
                let b = base(e);
                append_expert(
                    &mut bytes,
                    self.store,
                    &b,
                    "gate_proj",
                    "weight_scale",
                    gate_scale,
                )?;
                append_expert(
                    &mut bytes,
                    self.store,
                    &b,
                    "up_proj",
                    "weight_scale",
                    gate_scale,
                )?;
            }
            let shape = [experts, 2 * gate_rows, gate_scale_cols];
            self.add(&names[1], "experts_bank", DType::F8E4M3, &shape, &bytes)?;
        }
        if enabled[2] {
            let mut bytes = Vec::with_capacity(experts * 2 * gate_rows * 2);
            for e in 0..experts {
                let b = base(e);
                append_global(&mut bytes, self.store, &b, "gate_proj", gate_rows)?;
                append_global(&mut bytes, self.store, &b, "up_proj", gate_rows)?;
            }
            let shape = [experts, 2 * gate_rows];
            self.add(&names[2], "experts_bank", DType::F16, &shape, &bytes)?;
        }
        if enabled[3] {
            let mut bytes = Vec::with_capacity(experts * down_rows * down_cols);
            for e in 0..experts {
                let b = base(e);
                append_expert(&mut bytes, self.store, &b, "down_proj", "weight", down)?;
            }
            let shape = [experts, down_rows, down_cols];
            self.add(&names[3], "experts_bank", DType::U8, &shape, &bytes)?;
        }
        if enabled[4] {
            let mut bytes = Vec::with_capacity(experts * down_rows * down_scale_cols);
            for e in 0..experts {
                let b = base(e);
                append_expert(
                    &mut bytes,
                    self.store,
                    &b,
                    "down_proj",
                    "weight_scale",
                    down_scale,
                )?;
            }
            let shape = [experts, down_rows, down_scale_cols];
            self.add(&names[4], "experts_bank", DType::F8E4M3, &shape, &bytes)?;
        }
        if enabled[5] {
            let mut bytes = Vec::with_capacity(experts * down_rows * 2);
            for e in 0..experts {
                let b = base(e);
                append_global(&mut bytes, self.store, &b, "down_proj", down_rows)?;
            }
            let shape = [experts, down_rows];
            self.add(&names[5], "experts_bank", DType::F16, &shape, &bytes)?;
        }
        Ok(true)
    }

    fn emit_draft_banks(&mut self) -> Result<Option<usize>, ConvertError> {
        // Detect layout. The stacked Qwen export names the MTP expert tensors
        // without a `.weight` suffix; the per-expert export carries it.
        let stacked_pair = [
            (
                "mtp.layers.0.mlp.experts.gate_up_proj.weight",
                "mtp.layers.0.mlp.experts.down_proj.weight",
            ),
            (
                "mtp.layers.0.mlp.experts.gate_up_proj",
                "mtp.layers.0.mlp.experts.down_proj",
            ),
        ]
        .into_iter()
        .find(|(gate, _)| self.store.info(gate).is_some());
        let per_expert_gate = "mtp.layers.0.mlp.experts.0.gate_proj.weight";
        let num_experts = if let Some((gate, _)) = stacked_pair {
            let info = self.store.info(gate).unwrap();
            info.shape.first().copied().unwrap_or(0)
        } else if self.store.info(per_expert_gate).is_some() {
            // count experts
            let mut n = 0usize;
            while self
                .store
                .info(&format!("mtp.layers.0.mlp.experts.{n}.gate_proj.weight"))
                .is_some()
            {
                n += 1;
            }
            n
        } else {
            return Ok(None);
        };
        if num_experts == 0 {
            return Ok(None);
        }
        // Determine dims from gate_up (rows = 2*inter, cols = hidden).
        let (gate_rows, gate_cols, down_rows, down_cols) = if let Some((gate, down)) = stacked_pair
        {
            let info = self.store.info(gate).unwrap();
            // shape [E, 2I, H]
            let r = info.shape[1];
            let c = info.shape[2];
            // down
            let dinfo = self.store.info(down).unwrap();
            (r, c, dinfo.shape[1], dinfo.shape[2])
        } else {
            let ginfo = self
                .store
                .info("mtp.layers.0.mlp.experts.0.gate_proj.weight")
                .unwrap();
            let dinfo = self
                .store
                .info("mtp.layers.0.mlp.experts.0.down_proj.weight")
                .unwrap();
            // gate_proj [I, H]
            (
                ginfo.shape[0] * 2,
                ginfo.shape[1],
                dinfo.shape[0],
                dinfo.shape[1],
            )
        };
        let inter = gate_rows / 2;
        let hidden = gate_cols;

        // Build f32 matrices [E, gate_rows, gate_cols] and [E, down_rows, down_cols].
        let mut gate_up = vec![0.0f32; num_experts * gate_rows * gate_cols];
        let mut down = vec![0.0f32; num_experts * down_rows * down_cols];

        if let Some((gate, down_name)) = stacked_pair {
            // read bf16 bytes
            let bytes = self.store.read(gate).unwrap();
            bf16_bytes_to_f32(&bytes, &mut gate_up);
            let dbytes = self.store.read(down_name).unwrap();
            bf16_bytes_to_f32(&dbytes, &mut down);
        } else {
            for e in 0..num_experts {
                // gate_proj and up_proj -> concat along rows
                let gname = format!("mtp.layers.0.mlp.experts.{e}.gate_proj.weight");
                let uname = format!("mtp.layers.0.mlp.experts.{e}.up_proj.weight");
                let dname = format!("mtp.layers.0.mlp.experts.{e}.down_proj.weight");
                let gb = self.store.read(&gname).unwrap();
                let ub = self.store.read(&uname).unwrap();
                let db = self.store.read(&dname).unwrap();
                let i = inter;
                // gate: [i, hidden], up: [i, hidden] -> gate_up expert: [2i, hidden]
                let mut off = e * gate_rows * gate_cols;
                // gate rows first
                bf16_bytes_to_f32_into(&gb, &mut gate_up[off..off + i * hidden]);
                off += i * hidden;
                bf16_bytes_to_f32_into(&ub, &mut gate_up[off..off + i * hidden]);
                // down
                let doff = e * down_rows * down_cols;
                bf16_bytes_to_f32_into(&db, &mut down[doff..doff + down_rows * down_cols]);
            }
        }

        // Quantize.
        let (gu_packed, gu_scale, gu_global) =
            quantize_nvfp4_3d(&gate_up, num_experts, gate_rows, gate_cols);
        let (dn_packed, dn_scale, dn_global) =
            quantize_nvfp4_3d(&down, num_experts, down_rows, down_cols);

        // Emit bank tensors with names draft_*#L00000
        let layer = 0usize;
        let names = [
            format!("draft_gate_up_packed#L{layer:05}"),
            format!("draft_gate_up_scale#L{layer:05}"),
            format!("draft_gate_up_global#L{layer:05}"),
            format!("draft_down_packed#L{layer:05}"),
            format!("draft_down_scale#L{layer:05}"),
            format!("draft_down_global#L{layer:05}"),
        ];
        let enabled: Vec<bool> = names.iter().map(|n| self.enabled(n)).collect();
        if !enabled.iter().any(|&e| e) {
            // still count as present? We need the bank for the runtime. Emit anyway (enabled check only gates output).
        }
        // packed
        if self.enabled(&names[0]) {
            let shape = [num_experts, gate_rows, gate_cols / 2];
            self.add(&names[0], "experts_bank", DType::U8, &shape, &gu_packed)?;
        }
        if self.enabled(&names[1]) {
            let shape = [num_experts, gate_rows, gate_cols / 16];
            self.add(&names[1], "experts_bank", DType::F8E4M3, &shape, &gu_scale)?;
        }
        if self.enabled(&names[2]) {
            let mut bytes = Vec::with_capacity(num_experts * gate_rows * 2);
            for _ in 0..(num_experts * gate_rows) {
                let bits = half::f16::from_f32(gu_global).to_bits();
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
            let shape = [num_experts, gate_rows];
            self.add(&names[2], "experts_bank", DType::F16, &shape, &bytes)?;
        }
        if self.enabled(&names[3]) {
            let shape = [num_experts, down_rows, down_cols / 2];
            self.add(&names[3], "experts_bank", DType::U8, &shape, &dn_packed)?;
        }
        if self.enabled(&names[4]) {
            let shape = [num_experts, down_rows, down_cols / 16];
            self.add(&names[4], "experts_bank", DType::F8E4M3, &shape, &dn_scale)?;
        }
        if self.enabled(&names[5]) {
            let mut bytes = Vec::with_capacity(num_experts * down_rows * 2);
            for _ in 0..(num_experts * down_rows) {
                let bits = half::f16::from_f32(dn_global).to_bits();
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
            let shape = [num_experts, down_rows];
            self.add(&names[5], "experts_bank", DType::F16, &shape, &bytes)?;
        }
        Ok(Some(1))
    }
}

fn bf16_bytes_to_f32(bytes: &[u8], out: &mut [f32]) {
    for (i, chunk) in bytes.chunks(2).enumerate() {
        if i >= out.len() || chunk.len() < 2 {
            break;
        }
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out[i] = crate::domain::numerics::bf16_to_f32(bits);
    }
}

fn bf16_bytes_to_f32_into(bytes: &[u8], out: &mut [f32]) {
    bf16_bytes_to_f32(bytes, out);
}

fn f32_to_e4m3(x: f32) -> u8 {
    // round to nearest representable E4M3 by scanning all 256 codes.
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0u8..=255u8 {
        let v = crate::domain::numerics::e4m3fn_to_f32(code);
        let err = (v - x).abs();
        if err < best_err {
            best_err = err;
            best = code;
        }
    }
    best
}

fn f32_to_e2m1_code(x: f32) -> u8 {
    // nearest of the 16 e2m1 magnitudes, preserving sign.
    let sign = if x < 0.0 { 8u8 } else { 0u8 };
    let ax = x.abs();
    let mags: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut best = 0usize;
    let mut best_err = f32::INFINITY;
    for (i, &m) in mags.iter().enumerate() {
        let err = (m - ax).abs();
        if err < best_err {
            best_err = err;
            best = i;
        }
    }
    sign | (best as u8)
}

fn quantize_nvfp4_3d(data: &[f32], e: usize, rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>, f32) {
    let mut tensor_max: f32 = 0.0;
    for v in data {
        tensor_max = tensor_max.max(v.abs());
    }
    let global = if tensor_max == 0.0 {
        0.0
    } else {
        tensor_max / (6.0 * 448.0)
    };

    let mut packed = vec![0u8; e * rows * (cols / 2)];
    let mut scale = vec![0u8; e * rows * (cols / 16)];

    for ei in 0..e {
        for r in 0..rows {
            for bi in 0..(cols / 16) {
                let mut block_max: f32 = 0.0;
                for k in 0..16usize {
                    let idx = (ei * rows + r) * cols + bi * 16 + k;
                    block_max = block_max.max(data[idx].abs());
                }
                let local = if global == 0.0 || block_max == 0.0 {
                    0.0
                } else {
                    block_max / (6.0 * global)
                };
                let local_e4m3 = f32_to_e4m3(local);
                let local_f = crate::domain::numerics::e4m3fn_to_f32(local_e4m3);
                scale[(ei * rows + r) * (cols / 16) + bi] = local_e4m3;
                for k in 0..16usize {
                    let idx = (ei * rows + r) * cols + bi * 16 + k;
                    let denom = global * local_f;
                    let code = if denom == 0.0 {
                        0u8
                    } else {
                        let q = (data[idx] / denom).clamp(-6.0, 6.0);
                        f32_to_e2m1_code(q)
                    };
                    let byte_idx = (ei * rows + r) * (cols / 2) + (bi * 16 + k) / 2;
                    if k % 2 == 0 {
                        packed[byte_idx] = (packed[byte_idx] & 0xf0) | code;
                    } else {
                        packed[byte_idx] = (packed[byte_idx] & 0x0f) | (code << 4);
                    }
                }
            }
        }
    }
    (packed, scale, global)
}

fn ordered_parts(group: &GroupPlan, role: Role) -> Result<Vec<(usize, &str)>, ConvertError> {
    let mut parts = Vec::with_capacity(group.parts.len());
    for (idx, part) in &group.parts {
        let src = match role {
            Role::Weight => part.weight.as_deref(),
            Role::WeightScale => part.weight_scale.as_deref(),
            Role::WeightScale2 => part.weight_scale_2.as_deref(),
            Role::InputScale => part.input_scale.as_deref(),
        };
        let src = src.ok_or_else(|| {
            invalid(format!(
                "part {idx} is missing its {role:?} tensor for this fused module"
            ))
        })?;
        parts.push((*idx, src));
    }
    parts.sort_by_key(|(idx, _)| *idx);
    for (want, (got, _)) in parts.iter().enumerate() {
        if want != *got {
            return Err(invalid(format!(
                "fused module has part {got} but is missing part {want}"
            )));
        }
    }
    Ok(parts)
}

fn concat(
    parts: &[(usize, &str)],
    dtype: DType,
    store: &TensorStore,
) -> Result<(Vec<usize>, Vec<u8>), ConvertError> {
    let mut shape: Option<Vec<usize>> = None;
    let mut rows = 0usize;
    let mut bytes = Vec::new();
    for (_, src) in parts {
        let info = store
            .info(src)
            .ok_or_else(|| invalid(format!("missing {src}")))?;
        if info.dtype != dtype {
            return Err(invalid(format!(
                "{src} is {:?}, expected {dtype:?}",
                info.dtype
            )));
        }
        match &shape {
            None => shape = Some(info.shape.clone()),
            Some(first) => {
                if first.len() != info.shape.len() || first[1..] != info.shape[1..] {
                    return Err(invalid(format!(
                        "{src} shape {:?} cannot concatenate onto {first:?}",
                        info.shape
                    )));
                }
            }
        }
        rows += info.shape.first().copied().unwrap_or(1);
        bytes.extend_from_slice(&store.read(src)?);
    }
    let mut shape = shape.ok_or_else(|| invalid("cannot concatenate zero parts"))?;
    if shape.is_empty() {
        return Err(invalid("cannot concatenate rank-0 tensors"));
    }
    shape[0] = rows;
    Ok((shape, bytes))
}

fn fp8_rows(
    group: &GroupPlan,
    parts: &[(usize, &str)],
    store: &TensorStore,
) -> Result<usize, ConvertError> {
    let mut rows = 0usize;
    for (idx, _) in parts {
        let weight = group
            .parts
            .get(idx)
            .and_then(|p| p.weight.as_deref())
            .ok_or_else(|| invalid(format!("part {idx} has no weight")))?;
        let info = store
            .info(weight)
            .ok_or_else(|| invalid(format!("missing {weight}")))?;
        rows += *info
            .shape
            .first()
            .ok_or_else(|| invalid(format!("{weight} has no rows")))?;
    }
    Ok(rows)
}

fn fp8_scales(
    group: &GroupPlan,
    parts: &[(usize, &str)],
    store: &TensorStore,
) -> Result<Vec<u8>, ConvertError> {
    let mut out = Vec::new();
    for (idx, src) in parts {
        let weight = group
            .parts
            .get(idx)
            .and_then(|p| p.weight.as_deref())
            .ok_or_else(|| invalid(format!("part {idx} has no weight")))?;
        let rows = *store
            .info(weight)
            .and_then(|i| i.shape.first())
            .ok_or_else(|| invalid(format!("{weight} has no rows")))?;
        let info = store
            .info(src)
            .ok_or_else(|| invalid(format!("missing {src}")))?;
        if info.dtype != DType::F32 {
            return Err(invalid(format!("{src} is {:?}, expected f32", info.dtype)));
        }
        let raw = store.read(src)?;
        if raw.len() == 4 {
            let value = f32::from_le_bytes(raw[..4].try_into().unwrap());
            for _ in 0..rows {
                out.extend_from_slice(&value.to_le_bytes());
            }
        } else if raw.len() == rows * 4 {
            out.extend_from_slice(&raw);
        } else {
            return Err(invalid(format!(
                "{src} has {} scale bytes for {rows} rows",
                raw.len()
            )));
        }
    }
    Ok(out)
}

fn fp8_input_scale(parts: &[(usize, &str)], store: &TensorStore) -> Result<f32, ConvertError> {
    let mut max = f32::NEG_INFINITY;
    for (_, src) in parts {
        let info = store
            .info(src)
            .ok_or_else(|| invalid(format!("missing {src}")))?;
        if info.dtype != DType::F32 || info.nbytes != 4 {
            return Err(invalid(format!("{src} is not a scalar f32")));
        }
        let raw = store.read(src)?;
        let value = f32::from_le_bytes(raw[..4].try_into().unwrap());
        if value > max {
            max = value;
        }
    }
    Ok(max)
}

fn nvfp4_globals(
    group: &GroupPlan,
    parts: &[(usize, &str)],
    store: &TensorStore,
) -> Result<Vec<u8>, ConvertError> {
    let mut out = Vec::new();
    for (idx, src) in parts {
        let weight = group
            .parts
            .get(idx)
            .and_then(|p| p.weight.as_deref())
            .ok_or_else(|| invalid(format!("part {idx} has no weight")))?;
        let rows = *store
            .info(weight)
            .and_then(|i| i.shape.first())
            .ok_or_else(|| invalid(format!("{weight} has no rows")))?;
        let info = store
            .info(src)
            .ok_or_else(|| invalid(format!("missing {src}")))?;
        if info.dtype != DType::F32 || info.nbytes != 4 {
            return Err(invalid(format!("{src} is not a scalar f32")));
        }
        let raw = store.read(src)?;
        let bits = half::f16::from_f32(f32::from_le_bytes(raw[..4].try_into().unwrap())).to_bits();
        for _ in 0..rows {
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    Ok(out)
}

fn check_dtype(
    info: &super::safetensors::TensorInfo,
    want: DType,
    name: &str,
) -> Result<(), ConvertError> {
    if info.dtype != want {
        return Err(invalid(format!(
            "{name} has dtype {:?}, expected {want:?}",
            info.dtype
        )));
    }
    Ok(())
}

fn expert_info<'a>(
    store: &'a TensorStore,
    base: &str,
    proj: &str,
    kind: &str,
) -> Result<&'a super::safetensors::TensorInfo, ConvertError> {
    let name = format!("{base}.{proj}.{kind}");
    store
        .info(&name)
        .ok_or_else(|| invalid(format!("missing {name}")))
}

fn append_expert(
    out: &mut Vec<u8>,
    store: &TensorStore,
    base: &str,
    proj: &str,
    kind: &str,
    expected: &super::safetensors::TensorInfo,
) -> Result<(), ConvertError> {
    let name = format!("{base}.{proj}.{kind}");
    let info = store
        .info(&name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    if info.dtype != expected.dtype || info.shape != expected.shape {
        return Err(invalid(format!(
            "{name} is {:?} {:?}, expected {:?} {:?}",
            info.dtype, info.shape, expected.dtype, expected.shape
        )));
    }
    out.extend_from_slice(&store.read(&name)?);
    Ok(())
}

fn append_global(
    out: &mut Vec<u8>,
    store: &TensorStore,
    base: &str,
    proj: &str,
    rows: usize,
) -> Result<(), ConvertError> {
    let name = format!("{base}.{proj}.weight_scale_2");
    let info = store
        .info(&name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    if info.dtype != DType::F32 || info.nbytes != 4 {
        return Err(invalid(format!("{name} is not a scalar f32")));
    }
    let raw = store.read(&name)?;
    let bits = half::f16::from_f32(f32::from_le_bytes(raw[..4].try_into().unwrap())).to_bits();
    for _ in 0..rows {
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(())
}

struct ExpertDims {
    layers: BTreeSet<usize>,
    num_experts: usize,
}

fn expert_dims(store: &TensorStore) -> Option<ExpertDims> {
    let mut layers = BTreeSet::new();
    let mut num_experts = 0usize;
    for name in store.names() {
        if let Some((layer, expert)) = expert_key(name) {
            layers.insert(layer);
            num_experts = num_experts.max(expert + 1);
        }
    }
    (!layers.is_empty()).then_some(ExpertDims {
        layers,
        num_experts,
    })
}

fn expert_key(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let (layer, rest) = rest.split_once(".mlp.experts.")?;
    let (expert, rest) = rest.split_once('.')?;
    let (proj, kind) = rest.split_once('.')?;
    if !matches!(proj, "gate_proj" | "up_proj" | "down_proj") {
        return None;
    }
    if !matches!(kind, "weight" | "weight_scale" | "weight_scale_2") {
        return None;
    }
    Some((layer.parse().ok()?, expert.parse().ok()?))
}

fn add_one_bf16(bytes: &mut [u8]) {
    for chunk in bytes.as_chunks_mut::<2>().0 {
        let bits = u16::from_le_bytes(*chunk);
        let value = half::bf16::from_bits(bits).to_f32() + 1.0;
        *chunk = half::bf16::from_f32(value).to_bits().to_le_bytes();
    }
}

fn add_one_f16(bytes: &mut [u8]) {
    for chunk in bytes.as_chunks_mut::<2>().0 {
        let bits = u16::from_le_bytes(*chunk);
        let value = half::f16::from_bits(bits).to_f32() + 1.0;
        *chunk = half::f16::from_f32(value).to_bits().to_le_bytes();
    }
}

fn add_one_f32(bytes: &mut [u8]) {
    for chunk in bytes.as_chunks_mut::<4>().0 {
        *chunk = (f32::from_le_bytes(*chunk) + 1.0).to_le_bytes();
    }
}

// ============================== metadata ==============================

fn copy_metadata(model_dir: &Path, out_dir: &Path) -> Result<Vec<String>, ConvertError> {
    let out_abs = out_dir
        .canonicalize()
        .unwrap_or_else(|_| out_dir.to_path_buf());
    let mut copied = Vec::new();
    walk_metadata(model_dir, model_dir, out_dir, &out_abs, &mut copied)?;
    copied.sort();
    Ok(copied)
}

fn walk_metadata(
    root: &Path,
    dir: &Path,
    out_dir: &Path,
    out_abs: &Path,
    copied: &mut Vec<String>,
) -> Result<(), ConvertError> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == ".cache" || name == ".git" {
                continue;
            }
            let resolved = path.canonicalize().unwrap_or_else(|_| path.clone());
            if resolved == out_abs {
                continue;
            }
            walk_metadata(root, &path, out_dir, out_abs, copied)?;
            continue;
        }
        if excluded_metadata(&name) {
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(&path);
        let dst = out_dir.join(rel);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&path, &dst)?;
        copied.push(rel.to_string_lossy().replace('\\', "/"));
    }
    Ok(())
}

fn excluded_metadata(name: &str) -> bool {
    name.ends_with(".safetensors")
        || name == "model.safetensors.index.json"
        || name == "README.md"
        || name == ".gitattributes"
        || name == ".quant_summary.txt"
}

fn fingerprint(model_dir: &Path, shards: &[String]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    mix(model_dir.to_string_lossy().as_bytes());
    for shard in shards {
        mix(shard.as_bytes());
        if let Ok(meta) = std::fs::metadata(model_dir.join(shard)) {
            mix(&meta.len().to_le_bytes());
        }
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converter::testing::{SynthTensor, TempDir, write_checkpoint};
    use crate::infrastructure::ftw::FtwIndex;
    use std::os::unix::fs::FileExt;

    fn f32s(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bf16s(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read(root: &Path, index: &FtwIndex, name: &str) -> Vec<u8> {
        let t = index.entry(name).unwrap();
        let mut out = vec![0u8; t.nbytes as usize];
        let mut done = 0usize;
        for p in index.pieces(t.global_off, t.nbytes).unwrap() {
            let f = std::fs::File::open(root.join(&p.file)).unwrap();
            f.read_exact_at(&mut out[done..done + p.length as usize], p.file_off)
                .unwrap();
            done += p.length as usize;
        }
        out
    }

    fn tiny_checkpoint() -> Vec<SynthTensor<'static>> {
        let mut tensors = vec![
            SynthTensor {
                name: "model.language_model.norm.weight",
                dtype: "BF16",
                shape: vec![2],
                data: bf16s(&[0x0000, 0x3f00]),
            },
            SynthTensor {
                name: "model.language_model.embed_tokens.weight",
                dtype: "BF16",
                shape: vec![2, 2],
                data: bf16s(&[0x3f80, 0x3f00, 0x0000, 0xbf80]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.input_layernorm.weight",
                dtype: "BF16",
                shape: vec![2],
                data: bf16s(&[0x0000, 0x3f00]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.linear_attn.norm.weight",
                dtype: "BF16",
                shape: vec![2],
                data: bf16s(&[0x0000, 0x3f00]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.q_proj.weight",
                dtype: "F8_E4M3",
                shape: vec![2, 4],
                data: (1..=8).collect(),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.k_proj.weight",
                dtype: "F8_E4M3",
                shape: vec![2, 4],
                data: (9..=16).collect(),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.v_proj.weight",
                dtype: "F8_E4M3",
                shape: vec![2, 4],
                data: (17..=24).collect(),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.q_proj.weight_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.5]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.k_proj.weight_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.25]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.v_proj.weight_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.125]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.q_proj.input_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.1]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.k_proj.input_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.3]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.v_proj.input_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.2]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.o_proj.weight",
                dtype: "F8_E4M3",
                shape: vec![2, 8],
                data: (25..=40).collect(),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.o_proj.weight_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.5]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.o_proj.input_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.25]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                dtype: "U8",
                shape: vec![2, 8],
                data: (41..=56).collect(),
            },
            SynthTensor {
                name: "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight_scale",
                dtype: "F8_E4M3",
                shape: vec![2, 1],
                data: vec![10, 11],
            },
            SynthTensor {
                name: "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight_scale_2",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.5]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.mlp.shared_expert.gate_proj.input_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[9.0]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
                dtype: "U8",
                shape: vec![2, 8],
                data: (57..=72).collect(),
            },
            SynthTensor {
                name: "model.language_model.layers.0.mlp.shared_expert.up_proj.weight_scale",
                dtype: "F8_E4M3",
                shape: vec![2, 1],
                data: vec![12, 13],
            },
            SynthTensor {
                name: "model.language_model.layers.0.mlp.shared_expert.up_proj.weight_scale_2",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.25]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
                dtype: "U8",
                shape: vec![4, 4],
                data: (73..=88).collect(),
            },
            SynthTensor {
                name: "model.language_model.layers.0.mlp.shared_expert.down_proj.weight_scale",
                dtype: "F8_E4M3",
                shape: vec![4, 1],
                data: vec![20, 21, 22, 23],
            },
            SynthTensor {
                name: "model.language_model.layers.0.mlp.shared_expert.down_proj.weight_scale_2",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[1.0]),
            },
            SynthTensor {
                name: "lm_head.weight",
                dtype: "U8",
                shape: vec![3, 2],
                data: (89..=94).collect(),
            },
            SynthTensor {
                name: "lm_head.weight_scale",
                dtype: "F8_E4M3",
                shape: vec![3, 1],
                data: vec![30, 31, 32],
            },
            SynthTensor {
                name: "lm_head.weight_scale_2",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[0.5]),
            },
            SynthTensor {
                name: "lm_head.input_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[7.0]),
            },
            SynthTensor {
                name: "model.visual.blocks.0.attn.proj.bias",
                dtype: "BF16",
                shape: vec![2],
                data: bf16s(&[0x3f80, 0xbf80]),
            },
            SynthTensor {
                name: "mtp.fc.weight",
                dtype: "BF16",
                shape: vec![1],
                data: bf16s(&[0x3f80]),
            },
            SynthTensor {
                name: "model.language_model.layers.0.self_attn.k_scale",
                dtype: "F32",
                shape: vec![],
                data: f32s(&[1.0]),
            },
        ];
        for e in 0..2u8 {
            let g = format!("model.language_model.layers.0.mlp.experts.{e}");
            tensors.extend([
                SynthTensor {
                    name: Box::leak(format!("{g}.gate_proj.weight").into_boxed_str()),
                    dtype: "U8",
                    shape: vec![2, 8],
                    data: (10 + e * 16..26 + e * 16).collect(),
                },
                SynthTensor {
                    name: Box::leak(format!("{g}.gate_proj.weight_scale").into_boxed_str()),
                    dtype: "F8_E4M3",
                    shape: vec![2, 1],
                    data: vec![40 + e, 41 + e],
                },
                SynthTensor {
                    name: Box::leak(format!("{g}.gate_proj.weight_scale_2").into_boxed_str()),
                    dtype: "F32",
                    shape: vec![],
                    data: f32s(&[0.5]),
                },
                SynthTensor {
                    name: Box::leak(format!("{g}.up_proj.weight").into_boxed_str()),
                    dtype: "U8",
                    shape: vec![2, 8],
                    data: (50 + e * 16..66 + e * 16).collect(),
                },
                SynthTensor {
                    name: Box::leak(format!("{g}.up_proj.weight_scale").into_boxed_str()),
                    dtype: "F8_E4M3",
                    shape: vec![2, 1],
                    data: vec![50 + e, 51 + e],
                },
                SynthTensor {
                    name: Box::leak(format!("{g}.up_proj.weight_scale_2").into_boxed_str()),
                    dtype: "F32",
                    shape: vec![],
                    data: f32s(&[0.25]),
                },
                SynthTensor {
                    name: Box::leak(format!("{g}.down_proj.weight").into_boxed_str()),
                    dtype: "U8",
                    shape: vec![2, 8],
                    data: (90 + e * 16..106 + e * 16).collect(),
                },
                SynthTensor {
                    name: Box::leak(format!("{g}.down_proj.weight_scale").into_boxed_str()),
                    dtype: "F8_E4M3",
                    shape: vec![2, 1],
                    data: vec![60 + e, 61 + e],
                },
                SynthTensor {
                    name: Box::leak(format!("{g}.down_proj.weight_scale_2").into_boxed_str()),
                    dtype: "F32",
                    shape: vec![],
                    data: f32s(&[0.5]),
                },
            ]);
        }
        tensors
    }

    #[test]
    fn converts_a_tiny_checkpoint_end_to_end() {
        let dir = TempDir::new("tiny-src");
        let out = TempDir::new("tiny-out");
        let tensors = tiny_checkpoint();
        write_checkpoint(dir.path(), &tensors);
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();

        let opts = ConvertOptions::new(dir.path(), out.path());
        let report = convert(&opts).unwrap();
        assert_eq!(report.weight_count, 21);
        assert_eq!(report.experts_bank_count, 6);
        assert_eq!(report.copied_metadata, vec!["config.json"]);

        let index = FtwIndex::parse(
            &std::fs::read_to_string(out.path().join("freetoken_weight.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(index.tensors.len(), 27);
        assert_eq!(
            index.quant_format,
            Some(crate::domain::weights::QuantFormat::Nvfp4)
        );
        assert_eq!(index.expert_bank_num_layers, Some(1));

        let qkv: Vec<u8> = (1..=24).collect();
        assert_eq!(
            read(
                out.path(),
                &index,
                "model.layers.0.self_attn.qkv_proj.weight"
            ),
            qkv
        );
        assert_eq!(
            read(
                out.path(),
                &index,
                "model.layers.0.self_attn.qkv_proj.weight_scale"
            ),
            f32s(&[0.5, 0.5, 0.25, 0.25, 0.125, 0.125])
        );
        assert_eq!(
            read(
                out.path(),
                &index,
                "model.layers.0.self_attn.qkv_proj.input_scale"
            ),
            f32s(&[0.3])
        );
        assert_eq!(
            read(out.path(), &index, "model.layers.0.self_attn.o_proj.weight"),
            (25..=40).collect::<Vec<u8>>()
        );
        let gate_up: Vec<u8> = (41..=72).collect();
        assert_eq!(
            read(
                out.path(),
                &index,
                "model.layers.0.mlp.shared_expert.gate_up_proj.weight"
            ),
            gate_up
        );
        assert_eq!(
            read(
                out.path(),
                &index,
                "model.layers.0.mlp.shared_expert.gate_up_proj.weight_global"
            ),
            bf16s(&[0x3800, 0x3800, 0x3400, 0x3400])
        );
        assert_eq!(
            read(
                out.path(),
                &index,
                "model.layers.0.mlp.shared_expert.down_proj.weight_global"
            ),
            bf16s(&[0x3c00, 0x3c00, 0x3c00, 0x3c00])
        );
        assert_eq!(
            read(out.path(), &index, "lm_head.weight_global"),
            bf16s(&[0x3800, 0x3800, 0x3800])
        );
        assert_eq!(
            read(out.path(), &index, "model.norm.weight"),
            bf16s(&[0x3f80, 0x3fc0])
        );
        assert_eq!(
            read(out.path(), &index, "model.layers.0.linear_attn.norm.weight"),
            bf16s(&[0x0000, 0x3f00])
        );
        assert_eq!(
            read(out.path(), &index, "visual.blocks.0.attn.proj.bias"),
            bf16s(&[0x3f80, 0xbf80])
        );
        let mut gate_up = Vec::new();
        for e in 0..2u8 {
            gate_up.extend(10 + e * 16..26 + e * 16);
            gate_up.extend(50 + e * 16..66 + e * 16);
        }
        assert_eq!(read(out.path(), &index, "gate_up_packed#L00000"), gate_up);
        assert_eq!(
            read(out.path(), &index, "gate_up_global#L00000"),
            bf16s(&[
                0x3800, 0x3800, 0x3400, 0x3400, 0x3800, 0x3800, 0x3400, 0x3400
            ])
        );
        assert_eq!(
            read(out.path(), &index, "down_packed#L00000"),
            (90..122).collect::<Vec<u8>>()
        );
        assert_eq!(
            read(out.path(), &index, "down_global#L00000"),
            bf16s(&[0x3800, 0x3800, 0x3800, 0x3800])
        );
        assert!(index.entry("model.embed_tokens.weight").is_some());
        assert!(index.entry("mtp.fc.weight").is_none());

        assert!(matches!(
            convert(&opts),
            Err(ConvertError::Ftw(FtwError::AlreadyExists(_)))
        ));
    }

    #[test]
    fn limit_globs_select_output_names() {
        let dir = TempDir::new("limit-src");
        let out = TempDir::new("limit-out");
        write_checkpoint(dir.path(), &tiny_checkpoint());
        let mut opts = ConvertOptions::new(dir.path(), out.path());
        opts.limit = Some(vec!["lm_head.*".into(), "*#L00000".into()]);
        let report = convert(&opts).unwrap();
        assert_eq!(report.weight_count, 3);
        assert_eq!(report.experts_bank_count, 6);
    }

    #[test]
    fn glob_matching_and_name_mapping() {
        assert!(glob_match("lm_head.*", "lm_head.weight"));
        assert!(glob_match("*#L00000", "gate_up_packed#L00000"));
        assert!(glob_match(
            "model.layers.0.*",
            "model.layers.0.mlp.gate.weight"
        ));
        assert!(!glob_match("lm_head.*", "model.lm_head.weight"));
        assert!(!glob_match("*.bias", "visual.blocks.0.norm1.weight"));

        assert_eq!(
            rename("model.language_model.layers.0.mlp.gate.weight"),
            "model.layers.0.mlp.gate.weight"
        );
        assert_eq!(
            rename("model.visual.blocks.0.attn.qkv.bias"),
            "visual.blocks.0.attn.qkv.bias"
        );
        assert_eq!(rename("lm_head.weight"), "lm_head.weight");
        assert!(is_gemma_norm("model.norm.weight"));
        assert!(is_gemma_norm("model.layers.3.self_attn.q_norm.weight"));
        assert!(is_gemma_norm("draft.norm.weight"));
        assert!(is_gemma_norm("draft.pre_fc_norm_embedding.weight"));
        assert!(is_gemma_norm("draft.pre_fc_norm_hidden.weight"));
        assert!(is_gemma_norm("draft.layers.0.input_layernorm.weight"));
        assert!(!is_gemma_norm("draft.layers.0.mlp.gate.weight"));
        assert!(!is_gemma_norm("model.layers.0.linear_attn.norm.weight"));
        assert_eq!(
            quant_target("model.layers.3.self_attn.k_proj").unwrap().2,
            1
        );
        assert_eq!(
            quant_target("model.layers.3.self_attn.k_proj").unwrap().1,
            "model.layers.3.self_attn.qkv_proj"
        );
    }
}
