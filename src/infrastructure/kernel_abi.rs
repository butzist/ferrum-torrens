//! The committed kernels' launch ABI: what each cubin takes, in what order.
//!
//! Triton authors the NVFP4 MoE kernel and `ferrum_kernel.aot` freezes it into
//! `kernel/*.cubin` plus a manifest; this module is the Rust side of that
//! contract. Nothing here calls CUDA, which is the point: the argument layout is
//! where a hand-written launcher goes wrong silently, so it is a pure function
//! of the manifest and can be checked against the committed PTX on any machine.
//!
//! The manifest is the source of truth, not the code below, and the PTX is the
//! oracle the manifest is checked against. A kernel edited without re-running
//! the AOT build moves the PTX out from under the manifest, and `cargo test`
//! says so instead of a launch corrupting an argument buffer.
//!
//! Why the trailing null pointers: triton's launcher always appends a global and
//! a profile scratch pointer to the entry signature. Both sizes are 0 for these
//! kernels, so null is what the kernel expects, but the parameters still exist
//! in the signature and a launcher that omits them shifts every argument.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::domain::expert_kernel::MoEError;

const MANIFEST_JSON: &str = include_str!("../../kernel/manifest.json");

/// How wide one launch parameter is, which is also how the driver reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgKind {
    /// A device address.
    Ptr,
    U32,
    U64,
    /// A 4-byte float scalar (the driver reads the raw bits from the slot, the
    /// same way it reads a `u32`).
    F32,
}

impl ArgKind {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "ptr" => Some(Self::Ptr),
            "u32" => Some(Self::U32),
            "u64" => Some(Self::U64),
            "f32" => Some(Self::F32),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct Param {
    index: usize,
    kind: String,
}

#[derive(Debug, Deserialize)]
struct KernelEntry {
    entry: String,
    params: Vec<Param>,
    declared_args: Vec<String>,
    scratch_params: usize,
    num_warps: u32,
    shared: u32,
    global_scratch_size: u32,
    profile_scratch_size: u32,
    n: usize,
    block: usize,
}

#[derive(Debug, Deserialize)]
struct Shapes {
    hidden: usize,
    inter: usize,
    top_k: usize,
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    triton: String,
    shapes: Shapes,
    kernels: BTreeMap<String, KernelEntry>,
}

/// One kernel as the launcher sees it.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: &'static str,
    pub entry: String,
    /// Declared arguments in launch order; the trailing scratch pointers are
    /// not here, `scratch_params` counts them.
    pub args: Vec<ArgKind>,
    pub scratch_params: usize,
    pub num_warps: u32,
    pub shared: u32,
    /// The kernel's output columns, a constexpr trip count baked into the cubin.
    pub n: usize,
    /// The columns one program covers, also a constexpr. The grid is
    /// `cdiv(n, block)` in y, so a launcher that gets this wrong either skips
    /// columns or reads past them.
    pub block: usize,
}

/// The committed kernels, parsed. Every access is fallible because a stale
/// manifest (a kernel renamed, a shape changed without a rebuild) has to be a
/// clear error at load time, not a panic in the middle of a request.
#[derive(Debug, Clone)]
pub struct Abi {
    pub triton: String,
    pub hidden: usize,
    pub inter: usize,
    pub top_k: usize,
    pub dtype: String,
    kernels: BTreeMap<&'static str, Kernel>,
}

/// The kernel names `grouped_moe` launches, in the order it launches them.
pub const GEMV_GATE_UP: &str = "gemv_gate_up";
pub const GEMV_DOWN: &str = "gemv_down";
pub const SILU_MUL: &str = "silu_mul";
pub const SUM_REDUCE: &str = "sum_reduce";

/// Validate one manifest kernel's parameter list, returning the declared
/// argument kinds (the trailing scratch pointers stripped).
///
/// The grouped-MoE and staging sections describe the same triton launch
/// contract, so a malformed entry has to fail the same way in either.
fn parse_args(
    name: &str,
    params: &[Param],
    scratch_params: usize,
    declared_len: usize,
    global_scratch: u32,
    profile_scratch: u32,
) -> Result<Vec<ArgKind>, MoEError> {
    let mut args = Vec::with_capacity(params.len());
    for param in params {
        let kind = ArgKind::parse(&param.kind).ok_or_else(|| {
            MoEError::Bad(format!("kernel {name}: unknown param kind {}", param.kind))
        })?;
        if param.index != args.len() {
            return Err(MoEError::Bad(format!(
                "kernel {name}: param {} is out of order",
                param.index
            )));
        }
        args.push(kind);
    }
    if scratch_params > args.len() {
        return Err(MoEError::Bad(format!(
            "kernel {name}: {scratch_params} scratch params for {} parameters",
            args.len()
        )));
    }
    // The scratch pointers are pointers whatever the manifest says, and they
    // are not arguments the caller supplies.
    let tail = args.split_off(args.len() - scratch_params);
    if tail.iter().any(|k| *k != ArgKind::Ptr) {
        return Err(MoEError::Bad(format!(
            "kernel {name}: scratch params are not all pointers"
        )));
    }
    if params.len() != declared_len + scratch_params {
        return Err(MoEError::Bad(format!(
            "kernel {name}: {} params for {declared_len} declared args and {scratch_params} scratch",
            params.len()
        )));
    }
    if global_scratch != 0 || profile_scratch != 0 {
        return Err(MoEError::Bad(format!(
            "kernel {name}: scratch sizes {global_scratch}/{profile_scratch} need a real \
             allocation, which the driver-API launcher does not do yet"
        )));
    }
    Ok(args)
}

impl Abi {
    pub fn parse(json: &str) -> Result<Self, MoEError> {
        let m: Manifest = serde_json::from_str(json)
            .map_err(|e| MoEError::Bad(format!("kernel manifest: {e}")))?;
        let mut kernels = BTreeMap::new();
        for (name, entry) in m.kernels {
            let args = parse_args(
                &name,
                &entry.params,
                entry.scratch_params,
                entry.declared_args.len(),
                entry.global_scratch_size,
                entry.profile_scratch_size,
            )?;
            let name: &'static str = NAMES
                .iter()
                .copied()
                .find(|n| *n == name)
                .ok_or_else(|| MoEError::Bad(format!("unknown kernel {name}")))?;
            kernels.insert(
                name,
                Kernel {
                    name,
                    entry: entry.entry,
                    args,
                    scratch_params: entry.scratch_params,
                    num_warps: entry.num_warps,
                    shared: entry.shared,
                    n: entry.n,
                    block: entry.block,
                },
            );
        }
        for name in NAMES {
            if !kernels.contains_key(name) {
                return Err(MoEError::Bad(format!("manifest is missing kernel {name}")));
            }
        }
        Ok(Self {
            triton: m.triton,
            hidden: m.shapes.hidden,
            inter: m.shapes.inter,
            top_k: m.shapes.top_k,
            dtype: m.shapes.dtype,
            kernels,
        })
    }

    /// The manifest as committed.
    pub fn committed() -> Result<Self, MoEError> {
        Self::parse(MANIFEST_JSON)
    }

    /// The kernel names, in the order `grouped_moe` launches them.
    pub fn kernel_names() -> &'static [&'static str] {
        &NAMES
    }

    pub fn kernel(&self, name: &str) -> Result<&Kernel, MoEError> {
        self.kernels
            .get(name)
            .ok_or_else(|| MoEError::Bad(format!("no committed kernel {name}")))
    }

    /// Threads per block: one warp per `num_warps`, NVIDIA warp size 32.
    pub fn block_threads(kernel: &Kernel) -> u32 {
        kernel.num_warps * 32
    }
}

const NAMES: [&str; 4] = [GEMV_GATE_UP, GEMV_DOWN, SILU_MUL, SUM_REDUCE];

/// The cubin bytes, by name. `include_bytes!` so a missing or renamed artifact
/// is a compile error rather than a runtime surprise.
pub fn cubin(name: &str) -> Result<&'static [u8], MoEError> {
    Ok(match name {
        GEMV_GATE_UP => include_bytes!("../../kernel/gemv_gate_up.cubin"),
        GEMV_DOWN => include_bytes!("../../kernel/gemv_down.cubin"),
        SILU_MUL => include_bytes!("../../kernel/silu_mul.cubin"),
        SUM_REDUCE => include_bytes!("../../kernel/sum_reduce.cubin"),
        other => return Err(MoEError::Bad(format!("no committed cubin for {other}"))),
    })
}

/// The staging kernels `CudaMoE` launches around the grouped passes, in order.
pub const LRU_ENSURE: &str = "lru_ensure";
pub const INDEX_COPY: &str = "index_copy";

const STAGING_NAMES: [&str; 2] = [LRU_ENSURE, INDEX_COPY];

#[derive(Debug, Deserialize)]
struct StagingEntry {
    entry: String,
    params: Vec<Param>,
    declared_args: Vec<String>,
    scratch_params: usize,
    num_warps: u32,
    shared: u32,
    global_scratch_size: u32,
    profile_scratch_size: u32,
    /// The fixed launch shape. Both staging kernels are data-independent, so
    /// the grid is a constant of the cubin, not a function of the route count.
    grid: [u32; 3],
    /// The constexprs baked into the cubin, by name. All staging constexprs are
    /// non-negative counts, so `u64` is the natural type.
    constexprs: BTreeMap<String, u64>,
}

#[derive(Debug, Deserialize)]
struct StagingManifest {
    staging: BTreeMap<String, StagingEntry>,
}

/// One staging kernel as its launcher sees it.
#[derive(Debug, Clone)]
pub struct StagingKernel {
    pub name: &'static str,
    pub entry: String,
    pub args: Vec<ArgKind>,
    pub scratch_params: usize,
    pub num_warps: u32,
    pub shared: u32,
    pub grid: [u32; 3],
    constexprs: BTreeMap<String, u64>,
}

impl StagingKernel {
    /// Threads per block: one warp per `num_warps`, NVIDIA warp size 32.
    pub fn block_threads(&self) -> u32 {
        self.num_warps * 32
    }

    /// A constexpr baked into the cubin, by name. The launcher reads the plan
    /// cap and cache ceiling from here so it cannot drift from the build.
    pub fn constexpr(&self, name: &str) -> Result<u64, MoEError> {
        self.constexprs
            .get(name)
            .copied()
            .ok_or_else(|| MoEError::Bad(format!("kernel {}: no constexpr {name}", self.name)))
    }
}

/// The committed staging kernels, parsed from the manifest's `staging` section.
///
/// Separate from `Abi`: the staging pair is not part of `grouped_moe`, it runs
/// on its own streams with a fixed grid, and keeping the two contracts apart
/// means a generic-Abi change cannot silently alter a staging launch.
#[derive(Debug, Clone)]
pub struct StagingAbi {
    kernels: BTreeMap<&'static str, StagingKernel>,
}

impl StagingAbi {
    pub fn parse(json: &str) -> Result<Self, MoEError> {
        let m: StagingManifest = serde_json::from_str(json)
            .map_err(|e| MoEError::Bad(format!("kernel manifest: {e}")))?;
        let mut kernels = BTreeMap::new();
        for (name, entry) in m.staging {
            let args = parse_args(
                &name,
                &entry.params,
                entry.scratch_params,
                entry.declared_args.len(),
                entry.global_scratch_size,
                entry.profile_scratch_size,
            )?;
            let name: &'static str = STAGING_NAMES
                .iter()
                .copied()
                .find(|n| *n == name)
                .ok_or_else(|| MoEError::Bad(format!("unknown staging kernel {name}")))?;
            kernels.insert(
                name,
                StagingKernel {
                    name,
                    entry: entry.entry,
                    args,
                    scratch_params: entry.scratch_params,
                    num_warps: entry.num_warps,
                    shared: entry.shared,
                    grid: entry.grid,
                    constexprs: entry.constexprs,
                },
            );
        }
        for name in STAGING_NAMES {
            if !kernels.contains_key(name) {
                return Err(MoEError::Bad(format!(
                    "manifest is missing staging kernel {name}"
                )));
            }
        }
        Ok(Self { kernels })
    }

    /// The manifest as committed.
    pub fn committed() -> Result<Self, MoEError> {
        Self::parse(MANIFEST_JSON)
    }

    /// The staging kernel names, in launch order.
    pub fn kernel_names() -> &'static [&'static str] {
        &STAGING_NAMES
    }

    pub fn kernel(&self, name: &str) -> Result<&StagingKernel, MoEError> {
        self.kernels
            .get(name)
            .ok_or_else(|| MoEError::Bad(format!("no committed staging kernel {name}")))
    }
}

/// The staging cubin bytes, by name.
pub fn staging_cubin(name: &str) -> Result<&'static [u8], MoEError> {
    Ok(match name {
        LRU_ENSURE => include_bytes!("../../kernel/lru_ensure.cubin"),
        INDEX_COPY => include_bytes!("../../kernel/index_copy.cubin"),
        other => return Err(MoEError::Bad(format!("no committed cubin for {other}"))),
    })
}

/// The committed staging PTX, by name. Only the ABI tests read this.
#[cfg(test)]
fn staging_ptx(name: &str) -> &'static str {
    match name {
        LRU_ENSURE => include_str!("../../kernel/lru_ensure.ptx"),
        INDEX_COPY => include_str!("../../kernel/index_copy.ptx"),
        other => panic!("no committed ptx for {other}"),
    }
}

/// The paged GQA decode attention kernel `MhaPool` launches.
pub const PAGED_ATTENTION: &str = "paged_attention";

/// The bf16-KV pool variant (TODO.md item 14): same body, bf16 K/V pointers.
pub const PAGED_ATTENTION_BF16: &str = "paged_attention_bf16";

const ATTENTION_NAMES: [&str; 2] = [PAGED_ATTENTION, PAGED_ATTENTION_BF16];

#[derive(Debug, Deserialize)]
struct AttentionEntry {
    entry: String,
    params: Vec<Param>,
    declared_args: Vec<String>,
    scratch_params: usize,
    num_warps: u32,
    shared: u32,
    global_scratch_size: u32,
    profile_scratch_size: u32,
    /// The constexprs baked into the cubin, by name: the head geometry and the
    /// key tile. All are non-negative counts, so `u64` is the natural type.
    constexprs: BTreeMap<String, u64>,
}

#[derive(Debug, Deserialize)]
struct AttentionManifest {
    attention: BTreeMap<String, AttentionEntry>,
}

/// One attention kernel as its launcher sees it.
#[derive(Debug, Clone)]
pub struct AttentionKernel {
    pub name: &'static str,
    pub entry: String,
    pub args: Vec<ArgKind>,
    pub scratch_params: usize,
    pub num_warps: u32,
    pub shared: u32,
    constexprs: BTreeMap<String, u64>,
}

impl AttentionKernel {
    /// Threads per block: one warp per `num_warps`, NVIDIA warp size 32.
    pub fn block_threads(&self) -> u32 {
        self.num_warps * 32
    }

    /// A constexpr baked into the cubin, by name. The launcher reads the head
    /// geometry from here so it cannot drift from the build.
    pub fn constexpr(&self, name: &str) -> Result<u64, MoEError> {
        self.constexprs
            .get(name)
            .copied()
            .ok_or_else(|| MoEError::Bad(format!("kernel {}: no constexpr {name}", self.name)))
    }
}

/// The committed attention kernels, parsed from the manifest's `attention`
/// section.
///
/// Separate from `Abi` and `StagingAbi`: the attention launch grid is
/// `(rows, num_q_heads, 1)` -- runtime in x, a baked constexpr in y -- which is
/// neither the grouped `n`/`block` tile nor the staging pair's fixed grid, and
/// keeping the contracts apart means a change to one cannot alter another.
#[derive(Debug, Clone)]
pub struct AttentionAbi {
    kernels: BTreeMap<&'static str, AttentionKernel>,
}

impl AttentionAbi {
    pub fn parse(json: &str) -> Result<Self, MoEError> {
        let m: AttentionManifest = serde_json::from_str(json)
            .map_err(|e| MoEError::Bad(format!("kernel manifest: {e}")))?;
        let mut kernels = BTreeMap::new();
        for (name, entry) in m.attention {
            let args = parse_args(
                &name,
                &entry.params,
                entry.scratch_params,
                entry.declared_args.len(),
                entry.global_scratch_size,
                entry.profile_scratch_size,
            )?;
            let name: &'static str = ATTENTION_NAMES
                .iter()
                .copied()
                .find(|n| *n == name)
                .ok_or_else(|| MoEError::Bad(format!("unknown attention kernel {name}")))?;
            kernels.insert(
                name,
                AttentionKernel {
                    name,
                    entry: entry.entry,
                    args,
                    scratch_params: entry.scratch_params,
                    num_warps: entry.num_warps,
                    shared: entry.shared,
                    constexprs: entry.constexprs,
                },
            );
        }
        for name in ATTENTION_NAMES {
            if !kernels.contains_key(name) {
                return Err(MoEError::Bad(format!(
                    "manifest is missing attention kernel {name}"
                )));
            }
        }
        Ok(Self { kernels })
    }

    /// The manifest as committed.
    pub fn committed() -> Result<Self, MoEError> {
        Self::parse(MANIFEST_JSON)
    }

    /// The attention kernel names, in launch order.
    pub fn kernel_names() -> &'static [&'static str] {
        &ATTENTION_NAMES
    }

    pub fn kernel(&self, name: &str) -> Result<&AttentionKernel, MoEError> {
        self.kernels
            .get(name)
            .ok_or_else(|| MoEError::Bad(format!("no committed attention kernel {name}")))
    }
}

/// The attention cubin bytes, by name.
pub fn attention_cubin(name: &str) -> Result<&'static [u8], MoEError> {
    Ok(match name {
        PAGED_ATTENTION => include_bytes!("../../kernel/paged_attention.cubin"),
        PAGED_ATTENTION_BF16 => include_bytes!("../../kernel/paged_attention_bf16.cubin"),
        other => return Err(MoEError::Bad(format!("no committed cubin for {other}"))),
    })
}

/// The committed attention PTX, by name. Only the ABI tests read this.
#[cfg(test)]
fn attention_ptx(name: &str) -> &'static str {
    match name {
        PAGED_ATTENTION => include_str!("../../kernel/paged_attention.ptx"),
        PAGED_ATTENTION_BF16 => include_str!("../../kernel/paged_attention_bf16.ptx"),
        other => panic!("no committed ptx for {other}"),
    }
}

/// The fused GDN decode kernel the linear-attention mixer launches.
pub const GDN_DECODE: &str = "gdn_decode";

/// The dense NVFP4 GEMV the lm_head runs (TODO.md section 8 item 18).
pub const NVFP4_DENSE_GEMV: &str = "nvfp4_dense_gemv";

/// The vendored W8A16 FP8 projection kernels (TODO.md item 17).
pub const FP8_GEMV: &str = "fp8_gemv";
pub const FP8_SPLITK_REDUCE: &str = "fp8_splitk_reduce";
pub const FP8_GEMM: &str = "fp8_gemm";

/// The fused single-token GDN causal conv + silu (TODO.md item 19).
pub const CONV_DECODE: &str = "conv_decode";

/// The fused residual-add + RMSNorm (OPTIMIZATION.md R6: the norm launch tax).
pub const ADD_RMSNORM: &str = "add_rmsnorm";

/// The prefill chunk triangular solve (OPTIMIZATION.md R9.3).
pub const GDN_TRI_SOLVE: &str = "gdn_tri_solve";

const LINEAR_NAMES: [&str; 8] = [
    GDN_DECODE,
    CONV_DECODE,
    ADD_RMSNORM,
    GDN_TRI_SOLVE,
    NVFP4_DENSE_GEMV,
    FP8_GEMV,
    FP8_SPLITK_REDUCE,
    FP8_GEMM,
];

#[derive(Debug, Deserialize)]
struct LinearEntry {
    entry: String,
    params: Vec<Param>,
    declared_args: Vec<String>,
    scratch_params: usize,
    num_warps: u32,
    shared: u32,
    global_scratch_size: u32,
    profile_scratch_size: u32,
    /// The constexprs baked into the cubin, by name: the GDN head geometry and
    /// the key/value tiles. All are non-negative counts, so `u64` is natural.
    constexprs: BTreeMap<String, u64>,
}

#[derive(Debug, Deserialize)]
struct LinearManifest {
    linear: BTreeMap<String, LinearEntry>,
}

/// One linear-attention kernel as its launcher sees it.
#[derive(Debug, Clone)]
pub struct LinearKernel {
    pub name: &'static str,
    pub entry: String,
    pub args: Vec<ArgKind>,
    pub scratch_params: usize,
    pub num_warps: u32,
    pub shared: u32,
    constexprs: BTreeMap<String, u64>,
}

impl LinearKernel {
    /// Threads per block: one warp per `num_warps`, NVIDIA warp size 32.
    pub fn block_threads(&self) -> u32 {
        self.num_warps * 32
    }

    /// A constexpr baked into the cubin, by name. The launcher reads the GDN
    /// geometry from here so it cannot drift from the build.
    pub fn constexpr(&self, name: &str) -> Result<u64, MoEError> {
        self.constexprs
            .get(name)
            .copied()
            .ok_or_else(|| MoEError::Bad(format!("kernel {}: no constexpr {name}", self.name)))
    }
}

/// The committed linear-attention kernels, parsed from the manifest's `linear`
/// section.
///
/// Separate from `AttentionAbi`: the fused GDN decode is a fourth contract, with
/// `num_warps = 1` and a `(1, cdiv(V, BV), rows * NUM_V_HEADS)` grid, and keeping
/// it apart means an attention change cannot silently move a GDN launch.
#[derive(Debug, Clone)]
pub struct LinearAbi {
    kernels: BTreeMap<&'static str, LinearKernel>,
}

impl LinearAbi {
    pub fn parse(json: &str) -> Result<Self, MoEError> {
        let m: LinearManifest = serde_json::from_str(json)
            .map_err(|e| MoEError::Bad(format!("kernel manifest: {e}")))?;
        let mut kernels = BTreeMap::new();
        for (name, entry) in m.linear {
            let args = parse_args(
                &name,
                &entry.params,
                entry.scratch_params,
                entry.declared_args.len(),
                entry.global_scratch_size,
                entry.profile_scratch_size,
            )?;
            let name: &'static str = LINEAR_NAMES
                .iter()
                .copied()
                .find(|n| *n == name)
                .ok_or_else(|| MoEError::Bad(format!("unknown linear kernel {name}")))?;
            kernels.insert(
                name,
                LinearKernel {
                    name,
                    entry: entry.entry,
                    args,
                    scratch_params: entry.scratch_params,
                    num_warps: entry.num_warps,
                    shared: entry.shared,
                    constexprs: entry.constexprs,
                },
            );
        }
        for name in LINEAR_NAMES {
            if !kernels.contains_key(name) {
                return Err(MoEError::Bad(format!(
                    "manifest is missing linear kernel {name}"
                )));
            }
        }
        Ok(Self { kernels })
    }

    /// The manifest as committed.
    pub fn committed() -> Result<Self, MoEError> {
        Self::parse(MANIFEST_JSON)
    }

    /// The linear kernel names, in launch order.
    pub fn kernel_names() -> &'static [&'static str] {
        &LINEAR_NAMES
    }

    pub fn kernel(&self, name: &str) -> Result<&LinearKernel, MoEError> {
        self.kernels
            .get(name)
            .ok_or_else(|| MoEError::Bad(format!("no committed linear kernel {name}")))
    }
}

/// The linear-attention cubin bytes, by name.
pub fn linear_cubin(name: &str) -> Result<&'static [u8], MoEError> {
    Ok(match name {
        GDN_DECODE => include_bytes!("../../kernel/gdn_decode.cubin"),
        CONV_DECODE => include_bytes!("../../kernel/conv_decode.cubin"),
        ADD_RMSNORM => include_bytes!("../../kernel/add_rmsnorm.cubin"),
        GDN_TRI_SOLVE => include_bytes!("../../kernel/gdn_tri_solve.cubin"),
        NVFP4_DENSE_GEMV => include_bytes!("../../kernel/nvfp4_dense_gemv.cubin"),
        FP8_GEMV => include_bytes!("../../kernel/fp8_gemv.cubin"),
        FP8_SPLITK_REDUCE => include_bytes!("../../kernel/fp8_splitk_reduce.cubin"),
        FP8_GEMM => include_bytes!("../../kernel/fp8_gemm.cubin"),
        other => return Err(MoEError::Bad(format!("no committed cubin for {other}"))),
    })
}

/// The committed linear-attention PTX, by name. Only the ABI tests read this.
#[cfg(test)]
fn linear_ptx(name: &str) -> &'static str {
    match name {
        GDN_DECODE => include_str!("../../kernel/gdn_decode.ptx"),
        CONV_DECODE => include_str!("../../kernel/conv_decode.ptx"),
        ADD_RMSNORM => include_str!("../../kernel/add_rmsnorm.ptx"),
        GDN_TRI_SOLVE => include_str!("../../kernel/gdn_tri_solve.ptx"),
        NVFP4_DENSE_GEMV => include_str!("../../kernel/nvfp4_dense_gemv.ptx"),
        FP8_GEMV => include_str!("../../kernel/fp8_gemv.ptx"),
        FP8_SPLITK_REDUCE => include_str!("../../kernel/fp8_splitk_reduce.ptx"),
        FP8_GEMM => include_str!("../../kernel/fp8_gemm.ptx"),
        other => panic!("no committed ptx for {other}"),
    }
}

/// The committed PTX, by name. Only the ABI tests read this.
#[cfg(test)]
fn ptx(name: &str) -> &'static str {
    match name {
        GEMV_GATE_UP => include_str!("../../kernel/gemv_gate_up.ptx"),
        GEMV_DOWN => include_str!("../../kernel/gemv_down.ptx"),
        SILU_MUL => include_str!("../../kernel/silu_mul.ptx"),
        SUM_REDUCE => include_str!("../../kernel/sum_reduce.ptx"),
        other => panic!("no committed ptx for {other}"),
    }
}

/// The parameter list of a PTX entry, as the driver would read it.
///
/// `.param .u64 .ptr .global .align 1 name_param_0`: the width gives the size,
/// `.ptr` the address space, and the trailing index is triton's positional name.
#[cfg(test)]
fn ptx_entry_args(src: &str) -> Vec<ArgKind> {
    fn entry_args(src: &str) -> Option<Vec<ArgKind>> {
        let entry = src
            .split_once(".visible .entry ")?
            .1
            .split_once('(')?
            .1
            .split_once(')')?
            .0;
        let mut out: Vec<(usize, ArgKind)> = Vec::new();
        for raw in entry.split(',') {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            let fields: Vec<&str> = raw.split_whitespace().collect();
            let (width, is_float) = fields
                .iter()
                .find_map(|f| {
                    f.strip_prefix(".u")
                        .map(|w| (w, false))
                        .or_else(|| f.strip_prefix(".f").map(|w| (w, true)))
                })
                .and_then(|(w, f)| w.parse::<u32>().ok().map(|v| (v, f)))
                .unwrap_or_else(|| panic!("PTX param without a width: {raw}"));
            let index: usize = fields
                .last()
                .and_then(|f| f.rsplit_once("_param_"))
                .and_then(|(_, i)| i.parse().ok())
                .unwrap_or_else(|| panic!("PTX param without an index: {raw}"));
            let kind = if fields.contains(&".ptr") {
                ArgKind::Ptr
            } else if is_float {
                ArgKind::F32
            } else if width == 32 {
                ArgKind::U32
            } else {
                ArgKind::U64
            };
            out.push((index, kind));
        }
        out.sort_by_key(|(i, _)| *i);
        Some(out.into_iter().map(|(_, k)| k).collect())
    }
    entry_args(src).expect("no .visible .entry in the committed PTX")
}

/// One launch argument value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arg {
    Ptr(u64),
    U32(u32),
    U64(u64),
    /// An f32 scalar carried as its raw bit pattern.
    F32(u32),
}

/// The value buffer handed to `cuLaunchKernel`.
///
/// `cuLaunchKernel` takes an array of pointers to the individual arguments, not a
/// packed blob, so every value gets its own 8-byte slot: a `u32` parameter is
/// read as 4 bytes from an 8-byte-aligned address, which is what the driver
/// expects, and there is no alignment arithmetic to get wrong per kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgPack {
    kinds: Vec<ArgKind>,
    declared: usize,
    slots: Vec<u64>,
    next: usize,
}

impl ArgPack {
    fn with_args(args: &[ArgKind], scratch_params: usize) -> Self {
        let mut pack = Self {
            kinds: args.to_vec(),
            declared: args.len(),
            slots: vec![0; args.len() + scratch_params],
            next: 0,
        };
        pack.kinds
            .extend(std::iter::repeat_n(ArgKind::Ptr, scratch_params));
        pack
    }

    /// Sized for `kernel`, with the trailing scratch pointers already zeroed.
    pub fn for_kernel(kernel: &Kernel) -> Self {
        Self::with_args(&kernel.args, kernel.scratch_params)
    }

    /// Sized for a staging `kernel`, scratch pointers already zeroed.
    pub fn for_staging(kernel: &StagingKernel) -> Self {
        Self::with_args(&kernel.args, kernel.scratch_params)
    }

    /// Sized for an attention `kernel`, scratch pointers already zeroed.
    pub fn for_attention(kernel: &AttentionKernel) -> Self {
        Self::with_args(&kernel.args, kernel.scratch_params)
    }

    /// Sized for a linear-attention `kernel`, scratch pointers already zeroed.
    pub fn for_linear(kernel: &LinearKernel) -> Self {
        Self::with_args(&kernel.args, kernel.scratch_params)
    }

    /// Append the next argument. The kind is checked against the manifest so a
    /// reordered or mistyped argument is an error here rather than a silently
    /// misaligned launch.
    pub fn push(&mut self, arg: Arg) -> Result<(), MoEError> {
        let want = self.kinds[self.next];
        let ok = matches!(
            (want, arg),
            (ArgKind::Ptr, Arg::Ptr(_))
                | (ArgKind::U32, Arg::U32(_))
                | (ArgKind::U64, Arg::U64(_))
                | (ArgKind::F32, Arg::F32(_))
        );
        if !ok {
            let (index, want) = (self.next, want);
            return Err(MoEError::Bad(format!(
                "kernel argument {index} wants {want:?}, got {arg:?}"
            )));
        }
        self.slots[self.next] = match arg {
            Arg::Ptr(a) | Arg::U64(a) => a,
            Arg::U32(v) | Arg::F32(v) => u64::from(v),
        };
        self.next += 1;
        Ok(())
    }

    pub fn ptr(&mut self, addr: u64) -> Result<(), MoEError> {
        self.push(Arg::Ptr(addr))
    }

    pub fn u32(&mut self, v: u32) -> Result<(), MoEError> {
        self.push(Arg::U32(v))
    }

    pub fn u64(&mut self, v: u64) -> Result<(), MoEError> {
        self.push(Arg::U64(v))
    }

    pub fn f32(&mut self, v: f32) -> Result<(), MoEError> {
        self.push(Arg::F32(v.to_bits()))
    }

    fn finish(&self) -> Result<(), MoEError> {
        if self.next != self.declared {
            return Err(MoEError::Bad(format!(
                "packed {} of {} kernel arguments",
                self.next, self.declared
            )));
        }
        Ok(())
    }

    /// The `kernelParams` array: one pointer per parameter, into `slots`.
    ///
    /// The pointers are `*mut` only because the CUDA header's
    /// `cuLaunchKernel` takes `void**`; the driver reads the argument values and
    /// writes none of them, and each points into `slots`, which this borrows.
    pub fn kernel_params(&self) -> Result<Vec<*mut std::ffi::c_void>, MoEError> {
        self.finish()?;
        Ok(self
            .slots
            .iter()
            .map(|s| (s as *const u64).cast_mut().cast())
            .collect())
    }

    /// The packed values, for tests and for a debug print.
    pub fn slots(&self) -> &[u64] {
        &self.slots
    }
}

/// Device addresses and shapes for one layer's four launches. Everything is a
/// raw address: the caller owns the memory (libtorch tensors, or the slot cache),
/// and this module never allocates.
///
/// `Default` is all zeroes so a caller that runs one kernel of the four only has
/// to name the buffers that kernel reads.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoEArgs {
    pub tokens: usize,
    pub routes: usize,
    pub x: u64,
    pub gu_packed: u64,
    pub gu_scale: u64,
    pub gu_global: u64,
    pub dn_packed: u64,
    pub dn_scale: u64,
    pub dn_global: u64,
    pub route_slot: u64,
    pub route_token: u64,
    pub route_weight: u64,
    pub lut: u64,
    pub ic1: u64,
    pub ic2: u64,
    pub ic3: u64,
    pub out: u64,
}

fn div_ceil(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

/// `x[token] @ gate_up^T` into `ic1[route]`, one row per route.
pub fn gemv_gate_up_args(abi: &Abi, a: &MoEArgs) -> Result<ArgPack, MoEError> {
    let k = abi.kernel(GEMV_GATE_UP)?;
    let (hidden, inter) = (abi.hidden, abi.inter);
    let rows = 2 * inter;
    let mut p = ArgPack::for_kernel(k);
    p.ptr(a.x)?;
    p.ptr(a.gu_packed)?;
    p.ptr(a.gu_scale)?;
    p.ptr(a.gu_global)?;
    p.ptr(a.ic1)?;
    p.ptr(a.route_token)?;
    p.ptr(a.route_slot)?;
    p.ptr(a.route_weight)?;
    p.ptr(a.lut)?;
    // A_ROW_IS_ROUTE is false, so the activation matrix is x [tokens, hidden]
    // and M counts tokens while the grid runs over routes.
    p.u32(a.tokens as u32)?;
    p.u32(k.n as u32)?;
    p.u32(hidden as u32)?;
    p.u32(1)?;
    // The packed bank is read as int32 words (8 e2m1 codes each), so its row /
    // slot strides are in words, not bytes (OPTIMIZATION.md R1).
    p.u64((rows * (hidden / 8)) as u64)?;
    p.u32((hidden / 8) as u32)?;
    p.u32(1)?;
    p.u64((rows * (hidden / 16)) as u64)?;
    p.u32((hidden / 16) as u32)?;
    p.u32(1)?;
    p.u64(rows as u64)?;
    p.u32(1)?;
    p.u64((2 * inter) as u64)?;
    p.u32(1)?;
    Ok(p)
}

/// `ic2[route] @ down^T` into `ic3[route]`, route weight applied.
pub fn gemv_down_args(abi: &Abi, a: &MoEArgs) -> Result<ArgPack, MoEError> {
    let k = abi.kernel(GEMV_DOWN)?;
    let (hidden, inter) = (abi.hidden, abi.inter);
    let rows = hidden;
    let mut p = ArgPack::for_kernel(k);
    p.ptr(a.ic2)?;
    p.ptr(a.dn_packed)?;
    p.ptr(a.dn_scale)?;
    p.ptr(a.dn_global)?;
    p.ptr(a.ic3)?;
    p.ptr(a.route_token)?;
    p.ptr(a.route_slot)?;
    p.ptr(a.route_weight)?;
    p.ptr(a.lut)?;
    // A_ROW_IS_ROUTE is true: the activations are already one row per route.
    p.u32(a.routes as u32)?;
    p.u32(k.n as u32)?;
    p.u32(inter as u32)?;
    p.u32(1)?;
    p.u64((rows * (inter / 8)) as u64)?;
    p.u32((inter / 8) as u32)?;
    p.u32(1)?;
    p.u64((rows * (inter / 16)) as u64)?;
    p.u32((inter / 16) as u32)?;
    p.u32(1)?;
    p.u64(rows as u64)?;
    p.u32(1)?;
    p.u64(hidden as u64)?;
    p.u32(1)?;
    Ok(p)
}

/// `silu(gate) * up` from `ic1[r, :I]` and `ic1[r, I:]` into `ic2[r]`.
pub fn silu_mul_args(abi: &Abi, a: &MoEArgs) -> Result<ArgPack, MoEError> {
    let k = abi.kernel(SILU_MUL)?;
    let inter = abi.inter;
    let mut p = ArgPack::for_kernel(k);
    p.ptr(a.ic1)?;
    p.ptr(a.ic2)?;
    p.u32(inter as u32)?;
    p.u32((2 * inter) as u32)?;
    p.u32(1)?;
    p.u32(inter as u32)?;
    p.u32(1)?;
    Ok(p)
}

/// Weighted sum of the `top_k` routes per token into `out[token]`.
pub fn sum_reduce_args(abi: &Abi, a: &MoEArgs) -> Result<ArgPack, MoEError> {
    let k = abi.kernel(SUM_REDUCE)?;
    let hidden = abi.hidden;
    let mut p = ArgPack::for_kernel(k);
    p.ptr(a.ic3)?;
    p.ptr(a.out)?;
    p.u32(hidden as u32)?;
    p.u64(hidden as u64)?;
    p.u32(1)?;
    p.u64(hidden as u64)?;
    p.u32(1)?;
    Ok(p)
}

/// Device addresses and counts for the LRU admission kernel. Every buffer is a
/// raw address the caller owns; `routes` is the number of routed ids this step
/// (`<= BLOCK_K`), `num_cached` the slot ceiling and `id_base` the layer offset.
#[derive(Debug, Clone, Copy, Default)]
pub struct LruArgs {
    pub experts: u64,
    pub slot_for_id: u64,
    pub id_of_slot: u64,
    pub usage: u64,
    pub step: u64,
    pub misses: u64,
    pub slots: u64,
    pub num: u64,
    pub routes: u32,
    pub num_cached: u32,
    pub id_base: u32,
}

/// Device addresses for the fused multi-bank copy. `dst_ptrs`/`src_ptrs` are
/// device arrays of per-bank base addresses and `feat_bytes` the per-bank row
/// size; `num` is the device plan count.
#[derive(Debug, Clone, Copy, Default)]
pub struct CopyArgs {
    pub dst_ptrs: u64,
    pub src_ptrs: u64,
    pub feat_bytes: u64,
    pub dst_idx: u64,
    pub src_idx: u64,
    pub num: u64,
}

/// The LRU admission launch: rewrites `experts` to slots in place and emits the
/// `(miss, slot)` plan the copy follows.
pub fn lru_ensure_args(abi: &StagingAbi, a: &LruArgs) -> Result<ArgPack, MoEError> {
    let k = abi.kernel(LRU_ENSURE)?;
    let mut p = ArgPack::for_staging(k);
    p.ptr(a.experts)?;
    p.ptr(a.slot_for_id)?;
    p.ptr(a.id_of_slot)?;
    p.ptr(a.usage)?;
    p.ptr(a.step)?;
    p.ptr(a.misses)?;
    p.ptr(a.slots)?;
    p.ptr(a.num)?;
    p.u32(a.routes)?;
    p.u32(a.num_cached)?;
    p.u32(a.id_base)?;
    Ok(p)
}

/// The fused host->slot copy launch, one program per (bank, group).
pub fn index_copy_args(abi: &StagingAbi, a: &CopyArgs) -> Result<ArgPack, MoEError> {
    let k = abi.kernel(INDEX_COPY)?;
    let mut p = ArgPack::for_staging(k);
    p.ptr(a.dst_ptrs)?;
    p.ptr(a.src_ptrs)?;
    p.ptr(a.feat_bytes)?;
    p.ptr(a.dst_idx)?;
    p.ptr(a.src_idx)?;
    p.ptr(a.num)?;
    Ok(p)
}

/// Device addresses for one paged GQA decode launch. Every buffer is a raw
/// address the caller owns: `q`/`out` are `[rows, q_heads, head_dim]`, `k`/`v`
/// the layer's KV pool viewed as `[num_tokens, kv_heads, head_dim]`, `locs` the
/// `[rows, window]` pool locations the engine staged (head-aligned, padded) and
/// `positions` the rows' newest absolute positions. `window` is the location
/// buffer's row stride, a runtime value because the engine's window is not baked
/// into the cubin.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttentionArgs {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub locs: u64,
    pub positions: u64,
    pub out: u64,
    pub window: u32,
}

/// The paged attention launch argument pack, in the cubin's declared order.
/// `name` selects the f32 or bf16-pool cubin.
pub fn paged_attention_args(
    abi: &AttentionAbi,
    name: &str,
    a: &AttentionArgs,
) -> Result<ArgPack, MoEError> {
    let kernel = abi.kernel(name)?;
    let mut p = ArgPack::for_attention(kernel);
    p.ptr(a.q)?;
    p.ptr(a.k)?;
    p.ptr(a.v)?;
    p.ptr(a.locs)?;
    p.ptr(a.positions)?;
    p.ptr(a.out)?;
    p.u32(a.window)?;
    Ok(p)
}

/// The attention launch grid: one program per (row, q_head). The y extent is the
/// baked `NUM_Q_HEADS`; the x extent is the step's row count, which is why this
/// cannot come from the cubin alone.
pub fn attention_grid(rows: usize, kernel: &AttentionKernel) -> Result<(u32, u32, u32), MoEError> {
    Ok((rows as u32, kernel.constexpr("NUM_Q_HEADS")? as u32, 1))
}

/// Device addresses for one fused GDN decode launch. `ba` is `[rows, 2 * v_heads]`
/// f32 (`b` then `a`), `a_log`/`dt_bias` are `[v_heads]`, `q`/`k` are
/// `[rows, k_heads, K]`, `v` is `[rows, v_heads, V]`, `state` is
/// `[rows, v_heads, K, V]` f32 (updated in place) and `out` is
/// `[rows, v_heads, V]`. The three strides are the q/k/v row strides: the forward
/// slices q/k/v out of one conv buffer, so their rows are not packed.
#[derive(Debug, Clone, Copy, Default)]
pub struct GdnArgs {
    pub ba: u64,
    pub a_log: u64,
    pub dt_bias: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub state: u64,
    pub out: u64,
    pub stride_q: u32,
    pub stride_k: u32,
    pub stride_v: u32,
}

/// The GDN decode launch argument pack, in the cubin's declared order.
pub fn gdn_decode_args(abi: &LinearAbi, a: &GdnArgs) -> Result<ArgPack, MoEError> {
    let kernel = abi.kernel(GDN_DECODE)?;
    let mut p = ArgPack::for_linear(kernel);
    p.ptr(a.ba)?;
    p.ptr(a.a_log)?;
    p.ptr(a.dt_bias)?;
    p.ptr(a.q)?;
    p.ptr(a.k)?;
    p.ptr(a.v)?;
    p.ptr(a.state)?;
    p.ptr(a.out)?;
    p.u32(a.stride_q)?;
    p.u32(a.stride_k)?;
    p.u32(a.stride_v)?;
    Ok(p)
}

/// The GDN decode launch grid: `(NK, cdiv(V, BV), rows * NUM_V_HEADS)`. `NK` is
/// always 1 because `BK = next_power_of_2(K)` covers every key head at once, so
/// the x extent is the value tiling and the z extent the (row, value-head) pairs.
pub fn gdn_grid(rows: usize, kernel: &LinearKernel) -> Result<(u32, u32, u32), MoEError> {
    let v = kernel.constexpr("V")?;
    let bv = kernel.constexpr("BV")?;
    let v_heads = kernel.constexpr("NUM_V_HEADS")?;
    let tiles = v.div_ceil(bv) as u32;
    Ok((1, tiles, (rows as u64 * v_heads) as u32))
}

/// Device addresses for one fused GDN causal conv + silu launch. `x` is
/// `[rows, conv_dim]` f32 (possibly strided), `w` `[conv_dim, K]` f32, `state`
/// `[rows, conv_dim, K-1]` f32 (updated in place) and `out` `[rows, conv_dim]`.
/// The three strides are the row strides: `x` is a narrow of the qkvz projection
/// so its rows are not packed.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConvArgs {
    pub x: u64,
    pub w: u64,
    pub state: u64,
    pub out: u64,
    pub stride_x: u32,
    pub stride_state: u32,
    pub stride_out: u32,
}

/// The conv decode launch argument pack, in the cubin's declared order.
pub fn conv_decode_args(abi: &LinearAbi, a: &ConvArgs) -> Result<ArgPack, MoEError> {
    let kernel = abi.kernel(CONV_DECODE)?;
    let mut p = ArgPack::for_linear(kernel);
    p.ptr(a.x)?;
    p.ptr(a.w)?;
    p.ptr(a.state)?;
    p.ptr(a.out)?;
    p.u32(a.stride_x)?;
    p.u32(a.stride_state)?;
    p.u32(a.stride_out)?;
    Ok(p)
}

/// The conv decode launch grid: one program per `(row, feature block)`. `rows`
/// is the step's batch, so it is runtime; the feature tile is the baked
/// constexpr.
pub fn conv_grid(rows: usize, kernel: &LinearKernel) -> Result<(u32, u32, u32), MoEError> {
    let conv_dim = kernel.constexpr("CONV_DIM")?;
    let block_n = kernel.constexpr("BLOCK_N")?;
    Ok((
        rows as u32,
        (conv_dim as usize).div_ceil(block_n as usize) as u32,
        1,
    ))
}

/// Device addresses for one fused add + RMSNorm launch. `x`/`r` are
/// `[rows, hidden]` model dtype (either may be strided), `w` is `[hidden]`, and
/// `sum`/`out` are the contiguous `[rows, hidden]` outputs: the new residual
/// and the normalized row. `eps` is the f32 rms-norm epsilon.
#[derive(Debug, Clone, Copy)]
pub struct AddRmsnormArgs {
    pub x: u64,
    pub r: u64,
    pub w: u64,
    pub sum: u64,
    pub out: u64,
    pub eps: f32,
    pub stride_x: u32,
    pub stride_r: u32,
}

/// The fused add + RMSNorm launch argument pack, in the cubin's declared order.
pub fn add_rmsnorm_args(abi: &LinearAbi, a: &AddRmsnormArgs) -> Result<ArgPack, MoEError> {
    let kernel = abi.kernel(ADD_RMSNORM)?;
    let mut p = ArgPack::for_linear(kernel);
    p.ptr(a.x)?;
    p.ptr(a.r)?;
    p.ptr(a.w)?;
    p.ptr(a.sum)?;
    p.ptr(a.out)?;
    p.f32(a.eps)?;
    p.u32(a.stride_x)?;
    p.u32(a.stride_r)?;
    Ok(p)
}

/// The fused add + RMSNorm launch grid: one program per row; the whole hidden
/// row is one baked tile.
pub fn add_rmsnorm_grid(rows: usize) -> (u32, u32, u32) {
    (rows as u32, 1, 1)
}

/// Device addresses and shapes for one prefill triangular-solve launch. `a` is
/// `[hv, n, n]` f32 strictly lower, `rhs`/`out` are `[hv, n, dv]` f32 contiguous;
/// the strides are the caller's (the forward passes contiguous tensors). `n` is
/// the chunk length, at most the baked tile `N`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GdnTriSolveArgs {
    pub a: u64,
    pub rhs: u64,
    pub out: u64,
    pub n: u32,
    pub stride_a_h: u32,
    pub stride_a_r: u32,
    pub stride_rhs_h: u32,
    pub stride_rhs_r: u32,
    pub stride_out_h: u32,
    pub stride_out_r: u32,
}

/// The prefill triangular-solve launch argument pack, in the cubin's order.
pub fn gdn_tri_solve_args(abi: &LinearAbi, a: &GdnTriSolveArgs) -> Result<ArgPack, MoEError> {
    let kernel = abi.kernel(GDN_TRI_SOLVE)?;
    let mut p = ArgPack::for_linear(kernel);
    p.ptr(a.a)?;
    p.ptr(a.rhs)?;
    p.ptr(a.out)?;
    p.u32(a.n)?;
    p.u32(a.stride_a_h)?;
    p.u32(a.stride_a_r)?;
    p.u32(a.stride_rhs_h)?;
    p.u32(a.stride_rhs_r)?;
    p.u32(a.stride_out_h)?;
    p.u32(a.stride_out_r)?;
    Ok(p)
}

/// The prefill triangular-solve grid: one program per `(value head, value tile)`.
/// The value tile is the baked `BV`.
pub fn gdn_tri_solve_grid(
    value_heads: usize,
    dv: usize,
    kernel: &LinearKernel,
) -> Result<(u32, u32, u32), MoEError> {
    let bv = kernel.constexpr("BV")?;
    Ok((value_heads as u32, (dv as u64).div_ceil(bv) as u32, 1))
}

/// Device addresses and shapes for one dense NVFP4 GEMV launch. `a` is
/// `[M, K]` bf16 activations, `packed`/`scale`/`global` are the NVFP4 weight
/// banks (`[N, K // 2]` u8 read as int32 words, `[N, K // 16]` e4m3 bytes,
/// `[N]` f16 bytes) and `c` is the `[M, N]` fp32 output. The packed strides are
/// in int32 words (OPTIMIZATION.md R2).
#[derive(Debug, Clone, Copy, Default)]
pub struct DenseGemvArgs {
    pub a: u64,
    pub packed: u64,
    pub scale: u64,
    pub global: u64,
    pub c: u64,
    pub m: u32,
    pub n: u32,
    pub k_words: u32,
    pub stride_am: u32,
    pub stride_ak: u32,
    pub stride_pn: u32,
    pub stride_pkw: u32,
    pub stride_sn: u32,
    pub stride_sblk: u32,
    pub stride_cm: u32,
    pub stride_cn: u32,
}

/// The dense NVFP4 GEMV launch argument pack, in the cubin's declared order.
pub fn nvfp4_dense_gemv_args(abi: &LinearAbi, a: &DenseGemvArgs) -> Result<ArgPack, MoEError> {
    let kernel = abi.kernel(NVFP4_DENSE_GEMV)?;
    let mut p = ArgPack::for_linear(kernel);
    p.ptr(a.a)?;
    p.ptr(a.packed)?;
    p.ptr(a.scale)?;
    p.ptr(a.global)?;
    p.ptr(a.c)?;
    p.u32(a.m)?;
    p.u32(a.n)?;
    p.u32(a.k_words)?;
    p.u32(a.stride_am)?;
    p.u32(a.stride_ak)?;
    p.u32(a.stride_pn)?;
    p.u32(a.stride_pkw)?;
    p.u32(a.stride_sn)?;
    p.u32(a.stride_sblk)?;
    p.u32(a.stride_cm)?;
    p.u32(a.stride_cn)?;
    Ok(p)
}

/// The dense NVFP4 GEMV grid: one program per `(row, N block)`. The block is a
/// constexpr baked into the cubin, so the y extent comes from the manifest.
pub fn nvfp4_dense_gemv_grid(
    m: usize,
    n: usize,
    kernel: &LinearKernel,
) -> Result<(u32, u32, u32), MoEError> {
    let block_n = kernel.constexpr("BLOCK_N")?;
    Ok((m as u32, n.div_ceil(block_n as usize) as u32, 1))
}

/// Device addresses and shapes for the W8A16 split-K GEMV (M == 1). `a` is
/// `[K]` bf16, `w` `[N, K]` e4m3 bytes, `part` `[split_k, N]` fp32 scratch.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8GemvArgs {
    pub a: u64,
    pub w: u64,
    pub part: u64,
    pub n: u32,
    pub k: u32,
    pub n_kb: u32,
    pub kb_per: u32,
    pub stride_ak: u32,
    pub stride_wn: u32,
    pub stride_wk: u32,
    pub stride_pk: u32,
    pub stride_pn: u32,
}

/// The messages are in the cubin's declared order.
pub fn fp8_gemv_args(abi: &LinearAbi, a: &Fp8GemvArgs) -> Result<ArgPack, MoEError> {
    let kernel = abi.kernel(FP8_GEMV)?;
    let mut p = ArgPack::for_linear(kernel);
    p.ptr(a.a)?;
    p.ptr(a.w)?;
    p.ptr(a.part)?;
    p.u32(a.n)?;
    p.u32(a.k)?;
    p.u32(a.n_kb)?;
    p.u32(a.kb_per)?;
    p.u32(a.stride_ak)?;
    p.u32(a.stride_wn)?;
    p.u32(a.stride_wk)?;
    p.u32(a.stride_pk)?;
    p.u32(a.stride_pn)?;
    Ok(p)
}

/// The split-K reduce launch args ([part], [scale], [out]).
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8ReduceArgs {
    pub part: u64,
    pub scale: u64,
    pub out: u64,
    pub n: u32,
    pub split_k: u32,
    pub stride_pk: u32,
    pub stride_pn: u32,
}

pub fn fp8_splitk_reduce_args(abi: &LinearAbi, a: &Fp8ReduceArgs) -> Result<ArgPack, MoEError> {
    let kernel = abi.kernel(FP8_SPLITK_REDUCE)?;
    let mut p = ArgPack::for_linear(kernel);
    p.ptr(a.part)?;
    p.ptr(a.scale)?;
    p.ptr(a.out)?;
    p.u32(a.n)?;
    p.u32(a.split_k)?;
    p.u32(a.stride_pk)?;
    p.u32(a.stride_pn)?;
    Ok(p)
}

/// Device addresses and shapes for the W8A16 tiled GEMM (M > 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8GemmArgs {
    pub a: u64,
    pub w: u64,
    pub scale: u64,
    pub c: u64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub stride_am: u32,
    pub stride_ak: u32,
    pub stride_wn: u32,
    pub stride_wk: u32,
    pub stride_cm: u32,
    pub stride_cn: u32,
}

pub fn fp8_gemm_args(abi: &LinearAbi, a: &Fp8GemmArgs) -> Result<ArgPack, MoEError> {
    let kernel = abi.kernel(FP8_GEMM)?;
    let mut p = ArgPack::for_linear(kernel);
    p.ptr(a.a)?;
    p.ptr(a.w)?;
    p.ptr(a.scale)?;
    p.ptr(a.c)?;
    p.u32(a.m)?;
    p.u32(a.n)?;
    p.u32(a.k)?;
    p.u32(a.stride_am)?;
    p.u32(a.stride_ak)?;
    p.u32(a.stride_wn)?;
    p.u32(a.stride_wk)?;
    p.u32(a.stride_cm)?;
    p.u32(a.stride_cn)?;
    Ok(p)
}

/// FreeToken's split-K choice: fill up to 1536 CTAs, rounded down to a power of
/// two so the reduction order is stable.
pub fn fp8_split_k(n_tiles: usize, n_kb: usize) -> usize {
    let split = (1536 / n_tiles.max(1)).min(n_kb).max(1);
    1usize << (usize::BITS - 1 - split.leading_zeros())
}

/// The split-K GEMV grid: `(cdiv(N, BLOCK_N), split_k)`.
pub fn fp8_gemv_grid(
    n: usize,
    split_k: usize,
    kernel: &LinearKernel,
) -> Result<(u32, u32, u32), MoEError> {
    let block_n = kernel.constexpr("BLOCK_N")? as usize;
    Ok((n.div_ceil(block_n) as u32, split_k as u32, 1))
}

/// The split-K reduce grid: one `BLOCK`-wide program per output slice.
pub fn fp8_reduce_grid(n: usize, kernel: &LinearKernel) -> Result<(u32, u32, u32), MoEError> {
    let block = kernel.constexpr("BLOCK")? as usize;
    Ok((n.div_ceil(block) as u32, 1, 1))
}

/// The W8A16 GEMM grid: one program per `(m, n)` tile.
pub fn fp8_gemm_grid(
    m: usize,
    n: usize,
    kernel: &LinearKernel,
) -> Result<(u32, u32, u32), MoEError> {
    let bm = kernel.constexpr("BLOCK_M")? as usize;
    let bn = kernel.constexpr("BLOCK_N")? as usize;
    Ok((m.div_ceil(bm) as u32, n.div_ceil(bn) as u32, 1))
}

/// The launch grid for a kernel: one program per row (a route, or a token for
/// the reduce) and one per `block`-column tile.
///
/// All four kernels are shaped this way, which is why the y extent comes from
/// the cubin's own trip counts rather than from the caller: `n` and `block` are
/// both constexpr, so the grid cannot be derived from the runtime shapes alone.
pub fn grid(rows: usize, kernel: &Kernel) -> (u32, u32, u32) {
    (rows as u32, div_ceil(kernel.n, kernel.block), 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn abi() -> Abi {
        Abi::committed().expect("the committed manifest must parse")
    }

    #[test]
    fn the_manifest_lists_exactly_the_four_launches() {
        let abi = abi();
        assert_eq!(abi.hidden, 2048);
        assert_eq!(abi.inter, 512);
        assert_eq!(abi.top_k, 8);
        assert_eq!(abi.dtype, "bf16");
        for name in NAMES {
            assert!(abi.kernel(name).is_ok(), "{name} missing");
        }
        assert_eq!(abi.kernel(GEMV_GATE_UP).unwrap().n, 2 * abi.inter);
        assert_eq!(abi.kernel(GEMV_DOWN).unwrap().n, abi.hidden);
    }

    /// The whole point of the module: the manifest and the PTX must agree, or
    /// the manifest is describing a cubin that is not the committed one.
    #[test]
    fn the_manifest_agrees_with_the_committed_ptx() {
        let abi = abi();
        for name in NAMES {
            let kernel = abi.kernel(name).unwrap();
            let from_ptx = ptx_entry_args(ptx(name));
            let mut want = kernel.args.clone();
            want.extend(std::iter::repeat_n(ArgKind::Ptr, kernel.scratch_params));
            assert_eq!(
                from_ptx, want,
                "{name}: manifest does not match the PTX entry"
            );
            assert!(!from_ptx.is_empty());
        }
    }

    #[test]
    fn every_argument_builder_matches_its_kernel_signature() {
        let abi = abi();
        let args = MoEArgs {
            tokens: 2,
            routes: 16,
            x: 0x1000,
            gu_packed: 0x2000,
            gu_scale: 0x3000,
            gu_global: 0x4000,
            dn_packed: 0x5000,
            dn_scale: 0x6000,
            dn_global: 0x7000,
            route_slot: 0x8000,
            route_token: 0x9000,
            route_weight: 0xa000,
            lut: 0xb000,
            ic1: 0xc000,
            ic2: 0xd000,
            ic3: 0xe000,
            out: 0xf000,
        };
        let built = [
            (GEMV_GATE_UP, gemv_gate_up_args(&abi, &args)),
            (GEMV_DOWN, gemv_down_args(&abi, &args)),
            (SILU_MUL, silu_mul_args(&abi, &args)),
            (SUM_REDUCE, sum_reduce_args(&abi, &args)),
        ];
        for (name, pack) in built {
            let pack = pack.unwrap_or_else(|e| panic!("{name}: {e}"));
            let kernel = abi.kernel(name).unwrap();
            // kernel_params checks the count, so this fails on a builder that
            // pushes too few or too many arguments.
            let params = pack
                .kernel_params()
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                params.len(),
                kernel.args.len() + kernel.scratch_params,
                "{name}"
            );
            // The scratch slots stay null.
            for slot in &pack.slots()[kernel.args.len()..] {
                assert_eq!(*slot, 0, "{name}: scratch slot is not null");
            }
        }
    }

    #[test]
    fn the_wrong_argument_kind_is_an_error() {
        let abi = abi();
        let kernel = abi.kernel(SILU_MUL).unwrap();
        let mut pack = ArgPack::for_kernel(kernel);
        pack.ptr(1).unwrap();
        pack.ptr(2).unwrap();
        let err = pack.u64(3).unwrap_err().to_string();
        assert!(err.contains("wants U32"), "{err}");
        assert!(err.contains("got U64"), "{err}");
    }

    #[test]
    fn a_short_argument_list_is_an_error() {
        let abi = abi();
        let kernel = abi.kernel(SUM_REDUCE).unwrap();
        let mut pack = ArgPack::for_kernel(kernel);
        pack.ptr(1).unwrap();
        let err = pack.kernel_params().unwrap_err().to_string();
        assert!(err.contains("packed 1 of"), "{err}");
    }

    /// Every stride the kernels index with has to come out of the manifest
    /// shapes, so the committed cubins' constexpr trip counts stay consistent
    /// with the shapes the launchers compute from.
    #[test]
    fn the_shapes_divide_the_committed_trip_counts() {
        let abi = abi();
        for name in Abi::kernel_names() {
            let kernel = abi.kernel(name).unwrap();
            assert_eq!(
                kernel.n % kernel.block,
                0,
                "{name}: the tile does not divide n"
            );
        }
        // The e4m3 scale bank is K/16 bytes per row, so K must be a multiple of 16.
        assert_eq!(abi.hidden % 16, 0);
        assert_eq!(abi.inter % 16, 0);
    }

    fn edit_manifest(f: impl FnOnce(&mut serde_json::Value)) -> Result<Abi, MoEError> {
        let mut v: serde_json::Value = serde_json::from_str(MANIFEST_JSON).expect("manifest");
        f(&mut v);
        Abi::parse(&v.to_string())
    }

    #[test]
    fn a_manifest_with_an_unknown_kernel_is_rejected() {
        let err = edit_manifest(|v| {
            let down = v["kernels"]
                .as_object_mut()
                .unwrap()
                .remove(GEMV_DOWN)
                .unwrap();
            v["kernels"]["gemv_sideways"] = down;
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("unknown kernel gemv_sideways"), "{err}");
    }

    #[test]
    fn a_manifest_missing_a_kernel_is_rejected() {
        let err = edit_manifest(|v| {
            v["kernels"].as_object_mut().unwrap().remove(SUM_REDUCE);
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("missing kernel sum_reduce"), "{err}");
    }

    #[test]
    fn a_manifest_with_reversed_scratch_sizes_is_rejected() {
        let err = edit_manifest(|v| {
            v["kernels"][SUM_REDUCE]["global_scratch_size"] = serde_json::json!(64);
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("need a real allocation"), "{err}");
    }

    #[test]
    fn a_manifest_with_attention_scratch_to_allocate_is_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(MANIFEST_JSON).unwrap();
        v["attention"][PAGED_ATTENTION]["global_scratch_size"] = serde_json::json!(64);
        let err = AttentionAbi::parse(&v.to_string()).unwrap_err().to_string();
        assert!(err.contains("need a real allocation"), "{err}");
    }

    #[test]
    fn the_grids_cover_every_route_and_column() {
        let abi = abi();
        let gate_up = abi.kernel(GEMV_GATE_UP).unwrap();
        let routes = 3 * abi.top_k;
        // The routed GEMV tiles 16 output columns per program (OPTIMIZATION.md
        // R1): gate_up N=1024, down N=2048. The silu tile is 256, the reduce 128.
        assert_eq!(gate_up.block, 16);
        assert_eq!(grid(routes, gate_up), (routes as u32, 64, 1));
        assert_eq!(
            grid(routes, abi.kernel(GEMV_DOWN).unwrap()),
            (routes as u32, 128, 1)
        );
        assert_eq!(
            grid(routes, abi.kernel(SILU_MUL).unwrap()),
            (routes as u32, 2, 1)
        );
        assert_eq!(grid(abi.top_k, abi.kernel(SUM_REDUCE).unwrap()), (8, 16, 1));
    }

    #[test]
    fn the_staging_manifest_lists_the_two_launches() {
        let abi = StagingAbi::committed().expect("the committed manifest must parse");
        for name in STAGING_NAMES {
            assert!(abi.kernel(name).is_ok(), "{name} missing");
        }
        let lru = abi.kernel(LRU_ENSURE).unwrap();
        assert_eq!(lru.grid, [1, 1, 1]);
        assert_eq!(lru.block_threads(), 128);
        assert_eq!(lru.constexpr("BLOCK_K").unwrap(), 128);
        assert_eq!(lru.constexpr("BLOCK_C").unwrap(), 2048);
        let copy = abi.kernel(INDEX_COPY).unwrap();
        // One program per (bank, segment): `COPY_SEG_I32`-word segments of a row,
        // each walking the plan's `n` rows (device-read) in a loop.
        assert_eq!(copy.grid, [6, 64, 1]);
        assert_eq!(copy.block_threads(), 128);
        assert_eq!(copy.constexpr("NUM_BANKS").unwrap(), 6);
        assert_eq!(copy.constexpr("SEG_I32").unwrap(), 4096);
    }

    #[test]
    fn the_staging_manifest_agrees_with_the_committed_ptx() {
        let abi = StagingAbi::committed().unwrap();
        for name in STAGING_NAMES {
            let kernel = abi.kernel(name).unwrap();
            let from_ptx = ptx_entry_args(staging_ptx(name));
            let mut want = kernel.args.clone();
            want.extend(std::iter::repeat_n(ArgKind::Ptr, kernel.scratch_params));
            assert_eq!(
                from_ptx, want,
                "{name}: manifest does not match the PTX entry"
            );
            assert!(!from_ptx.is_empty());
        }
    }

    #[test]
    fn every_staging_argument_builder_matches_its_kernel_signature() {
        let abi = StagingAbi::committed().unwrap();
        let lru = LruArgs {
            experts: 0x1000,
            slot_for_id: 0x2000,
            id_of_slot: 0x3000,
            usage: 0x4000,
            step: 0x5000,
            misses: 0x6000,
            slots: 0x7000,
            num: 0x8000,
            routes: 32,
            num_cached: 512,
            id_base: 2048,
        };
        let copy = CopyArgs {
            dst_ptrs: 0x1000,
            src_ptrs: 0x2000,
            feat_bytes: 0x3000,
            dst_idx: 0x4000,
            src_idx: 0x5000,
            num: 0x6000,
        };
        let built = [
            (LRU_ENSURE, lru_ensure_args(&abi, &lru)),
            (INDEX_COPY, index_copy_args(&abi, &copy)),
        ];
        for (name, pack) in built {
            let pack = pack.unwrap_or_else(|e| panic!("{name}: {e}"));
            let kernel = abi.kernel(name).unwrap();
            let params = pack
                .kernel_params()
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                params.len(),
                kernel.args.len() + kernel.scratch_params,
                "{name}"
            );
            for slot in &pack.slots()[kernel.args.len()..] {
                assert_eq!(*slot, 0, "{name}: scratch slot is not null");
            }
        }
    }

    #[test]
    fn a_manifest_with_an_unknown_attention_kernel_is_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(MANIFEST_JSON).unwrap();
        let entry = v["attention"]
            .as_object_mut()
            .unwrap()
            .remove(PAGED_ATTENTION)
            .unwrap();
        v["attention"]["paged_sideways"] = entry;
        let err = AttentionAbi::parse(&v.to_string()).unwrap_err().to_string();
        assert!(
            err.contains("unknown attention kernel paged_sideways"),
            "{err}"
        );
    }

    #[test]
    fn a_manifest_missing_a_staging_kernel_is_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(MANIFEST_JSON).unwrap();
        v["staging"].as_object_mut().unwrap().remove(LRU_ENSURE);
        let err = StagingAbi::parse(&v.to_string()).unwrap_err().to_string();
        assert!(err.contains("missing staging kernel lru_ensure"), "{err}");
    }

    /// The grouped section must still parse unchanged next to the new staging
    /// section, and the two must not see each other's kernels.
    #[test]
    fn the_two_sections_are_disjoint() {
        let abi = abi();
        assert!(abi.kernel(LRU_ENSURE).is_err());
        let staging = StagingAbi::committed().unwrap();
        assert!(staging.kernel(GEMV_GATE_UP).is_err());
    }

    #[test]
    fn the_attention_manifest_lists_the_paged_kernel() {
        let abi = AttentionAbi::committed().expect("the committed manifest must parse");
        let kernel = abi.kernel(PAGED_ATTENTION).unwrap();
        // 8 warps spread the `[BLOCK_N, HEAD_DIM]` f32 tile so it does not spill.
        assert_eq!(kernel.block_threads(), 256);
        assert_eq!(kernel.args.len(), 7);
        assert_eq!(kernel.args[..6], [ArgKind::Ptr; 6]);
        assert_eq!(kernel.args[6], ArgKind::U32);
        assert_eq!(kernel.constexpr("NUM_Q_HEADS").unwrap(), 16);
        assert_eq!(kernel.constexpr("NUM_KV_HEADS").unwrap(), 2);
        assert_eq!(kernel.constexpr("HEAD_DIM").unwrap(), 256);
        assert_eq!(kernel.constexpr("BLOCK_N").unwrap(), 64);
    }

    #[test]
    fn the_attention_manifest_agrees_with_the_committed_ptx() {
        let abi = AttentionAbi::committed().unwrap();
        for name in ATTENTION_NAMES {
            let kernel = abi.kernel(name).unwrap();
            let from_ptx = ptx_entry_args(attention_ptx(name));
            let mut want = kernel.args.clone();
            want.extend(std::iter::repeat_n(ArgKind::Ptr, kernel.scratch_params));
            assert_eq!(
                from_ptx, want,
                "{name}: manifest does not match the PTX entry"
            );
            assert!(!from_ptx.is_empty());
        }
    }

    #[test]
    fn the_attention_argument_builder_matches_its_kernel_signature() {
        let abi = AttentionAbi::committed().unwrap();
        let args = AttentionArgs {
            q: 0x1000,
            k: 0x2000,
            v: 0x3000,
            locs: 0x4000,
            positions: 0x5000,
            out: 0x6000,
            window: 7,
        };
        let pack = paged_attention_args(&abi, PAGED_ATTENTION, &args).unwrap();
        let kernel = abi.kernel(PAGED_ATTENTION).unwrap();
        let params = pack.kernel_params().unwrap();
        assert_eq!(params.len(), kernel.args.len() + kernel.scratch_params);
        for slot in &pack.slots()[kernel.args.len()..] {
            assert_eq!(*slot, 0, "scratch slot is not null");
        }
    }

    #[test]
    fn the_bf16_attention_variant_packs_like_the_f32_one() {
        // Same body, only the K/V pointer types differ, so the arg kinds match.
        let abi = AttentionAbi::committed().unwrap();
        let args = AttentionArgs {
            q: 0x1000,
            k: 0x2000,
            v: 0x3000,
            locs: 0x4000,
            positions: 0x5000,
            out: 0x6000,
            window: 7,
        };
        let f32 = paged_attention_args(&abi, PAGED_ATTENTION, &args).unwrap();
        let bf16 = paged_attention_args(&abi, PAGED_ATTENTION_BF16, &args).unwrap();
        assert_eq!(f32.slots(), bf16.slots());
        let kernel = abi.kernel(PAGED_ATTENTION_BF16).unwrap();
        assert_eq!(kernel.constexpr("NUM_KV_HEADS").unwrap(), 2);
    }

    #[test]
    fn the_attention_grid_uses_the_baked_head_count() {
        let abi = AttentionAbi::committed().unwrap();
        let kernel = abi.kernel(PAGED_ATTENTION).unwrap();
        assert_eq!(attention_grid(3, kernel).unwrap(), (3, 16, 1));
    }

    #[test]
    fn a_manifest_missing_the_paged_attention_kernel_is_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(MANIFEST_JSON).unwrap();
        v["attention"]
            .as_object_mut()
            .unwrap()
            .remove(PAGED_ATTENTION);
        let err = AttentionAbi::parse(&v.to_string()).unwrap_err().to_string();
        assert!(
            err.contains("missing attention kernel paged_attention"),
            "{err}"
        );
    }

    #[test]
    fn the_manifest_sections_are_disjoint() {
        let abi = abi();
        assert!(abi.kernel(PAGED_ATTENTION).is_err());
        assert!(abi.kernel(GDN_DECODE).is_err());
        let staging = StagingAbi::committed().unwrap();
        assert!(staging.kernel(PAGED_ATTENTION).is_err());
        assert!(staging.kernel(GDN_DECODE).is_err());
        let attention = AttentionAbi::committed().unwrap();
        assert!(attention.kernel(GEMV_GATE_UP).is_err());
        assert!(attention.kernel(LRU_ENSURE).is_err());
        assert!(attention.kernel(GDN_DECODE).is_err());
        let linear = LinearAbi::committed().unwrap();
        assert!(linear.kernel(GEMV_GATE_UP).is_err());
        assert!(linear.kernel(PAGED_ATTENTION).is_err());
        assert!(linear.kernel(LRU_ENSURE).is_err());
    }

    #[test]
    fn the_linear_manifest_records_the_gdn_geometry() {
        let abi = LinearAbi::committed().unwrap();
        let kernel = abi.kernel(GDN_DECODE).unwrap();
        assert_eq!(kernel.constexpr("NUM_K_HEADS").unwrap(), 16);
        assert_eq!(kernel.constexpr("NUM_V_HEADS").unwrap(), 32);
        assert_eq!(kernel.constexpr("K").unwrap(), 128);
        assert_eq!(kernel.constexpr("V").unwrap(), 128);
        assert_eq!(kernel.constexpr("BV").unwrap(), 32);
        assert_eq!(kernel.num_warps, 4);
    }

    #[test]
    fn the_linear_manifest_agrees_with_the_committed_ptx() {
        let abi = LinearAbi::committed().unwrap();
        for name in LINEAR_NAMES {
            let kernel = abi.kernel(name).unwrap();
            let from_ptx = ptx_entry_args(linear_ptx(name));
            let mut want = kernel.args.clone();
            want.extend(std::iter::repeat_n(ArgKind::Ptr, kernel.scratch_params));
            assert_eq!(
                from_ptx, want,
                "{name}: manifest does not match the PTX entry"
            );
            assert!(!from_ptx.is_empty());
        }
    }

    #[test]
    fn the_gdn_argument_builder_matches_its_kernel_signature() {
        let abi = LinearAbi::committed().unwrap();
        let args = GdnArgs {
            ba: 0x1000,
            a_log: 0x2000,
            dt_bias: 0x3000,
            q: 0x4000,
            k: 0x5000,
            v: 0x6000,
            state: 0x7000,
            out: 0x8000,
            stride_q: 8192,
            stride_k: 8192,
            stride_v: 8192,
        };
        let pack = gdn_decode_args(&abi, &args).unwrap();
        let kernel = abi.kernel(GDN_DECODE).unwrap();
        let params = pack.kernel_params().unwrap();
        assert_eq!(params.len(), kernel.args.len() + kernel.scratch_params);
        for slot in &pack.slots()[kernel.args.len()..] {
            assert_eq!(*slot, 0, "scratch slot is not null");
        }
    }

    #[test]
    fn the_gdn_grid_tiles_the_value_heads() {
        let abi = LinearAbi::committed().unwrap();
        let kernel = abi.kernel(GDN_DECODE).unwrap();
        // 128 value dims / BV 32 = 4 tiles in y; 2 rows * 32 value heads = 64 in z.
        assert_eq!(gdn_grid(2, kernel).unwrap(), (1, 4, 64));
    }

    #[test]
    fn a_manifest_missing_the_gdn_kernel_is_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(MANIFEST_JSON).unwrap();
        v["linear"].as_object_mut().unwrap().remove(GDN_DECODE);
        let err = LinearAbi::parse(&v.to_string()).unwrap_err().to_string();
        assert!(err.contains("missing linear kernel gdn_decode"), "{err}");
    }

    #[test]
    fn the_conv_manifest_records_the_shape() {
        let abi = LinearAbi::committed().unwrap();
        let kernel = abi.kernel(CONV_DECODE).unwrap();
        // conv_dim = 2 * 16 * 128 + 32 * 128 = 8192, the checkpoint's conv width.
        assert_eq!(kernel.constexpr("CONV_DIM").unwrap(), 8192);
        assert_eq!(kernel.constexpr("K").unwrap(), 4);
        assert_eq!(kernel.constexpr("TAPS").unwrap(), 4);
        assert_eq!(kernel.constexpr("BLOCK_N").unwrap(), 256);
        assert_eq!(kernel.num_warps, 4);
    }

    #[test]
    fn the_conv_argument_builder_matches_its_kernel_signature() {
        let abi = LinearAbi::committed().unwrap();
        let args = ConvArgs {
            x: 0x1000,
            w: 0x2000,
            state: 0x3000,
            out: 0x4000,
            stride_x: 12288,
            stride_state: 24576,
            stride_out: 8192,
        };
        let pack = conv_decode_args(&abi, &args).unwrap();
        let kernel = abi.kernel(CONV_DECODE).unwrap();
        let params = pack.kernel_params().unwrap();
        assert_eq!(params.len(), kernel.args.len() + kernel.scratch_params);
        for slot in &pack.slots()[kernel.args.len()..] {
            assert_eq!(*slot, 0, "scratch slot is not null");
        }
    }

    #[test]
    fn the_conv_grid_tiles_the_features() {
        let abi = LinearAbi::committed().unwrap();
        let kernel = abi.kernel(CONV_DECODE).unwrap();
        // 8192 features / 256 tile = 32 blocks in y, one program per row in x.
        assert_eq!(conv_grid(2, kernel).unwrap(), (2, 32, 1));
    }

    #[test]
    fn a_manifest_missing_the_conv_kernel_is_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(MANIFEST_JSON).unwrap();
        v["linear"].as_object_mut().unwrap().remove(CONV_DECODE);
        let err = LinearAbi::parse(&v.to_string()).unwrap_err().to_string();
        assert!(err.contains("missing linear kernel conv_decode"), "{err}");
    }

    #[test]
    fn the_add_rmsnorm_argument_builder_matches_its_kernel_signature() {
        let abi = LinearAbi::committed().unwrap();
        let args = AddRmsnormArgs {
            x: 0x1000,
            r: 0x2000,
            w: 0x3000,
            sum: 0x4000,
            out: 0x5000,
            eps: 1e-6,
            stride_x: 2048,
            stride_r: 2048,
        };
        let pack = add_rmsnorm_args(&abi, &args).unwrap();
        let kernel = abi.kernel(ADD_RMSNORM).unwrap();
        assert_eq!(
            kernel.args,
            vec![
                ArgKind::Ptr,
                ArgKind::Ptr,
                ArgKind::Ptr,
                ArgKind::Ptr,
                ArgKind::Ptr,
                ArgKind::F32,
                ArgKind::U32,
                ArgKind::U32,
            ]
        );
        let params = pack.kernel_params().unwrap();
        assert_eq!(params.len(), kernel.args.len() + kernel.scratch_params);
        // The f32 slot carries the raw bit pattern, so a mistyped arg pack
        // would change the bits the driver reads, not just the width check.
        assert_eq!(pack.slots()[5], u64::from(1e-6f32.to_bits()));
        for slot in &pack.slots()[kernel.args.len()..] {
            assert_eq!(*slot, 0, "scratch slot is not null");
        }
    }

    #[test]
    fn the_add_rmsnorm_grid_is_one_program_per_row() {
        assert_eq!(add_rmsnorm_grid(3), (3, 1, 1));
    }

    #[test]
    fn the_tri_solve_manifest_records_the_tiles() {
        let abi = LinearAbi::committed().unwrap();
        let kernel = abi.kernel(GDN_TRI_SOLVE).unwrap();
        assert_eq!(kernel.constexpr("N").unwrap(), 64);
        assert_eq!(kernel.constexpr("BV").unwrap(), 32);
        assert_eq!(kernel.constexpr("DV").unwrap(), 128);
        assert_eq!(kernel.num_warps, 4);
        // 128 value dims / BV 32 = 4 tiles in y, 32 value heads in x.
        assert_eq!(gdn_tri_solve_grid(32, 128, kernel).unwrap(), (32, 4, 1));
        // A value dim that is not a multiple of the tile rounds up.
        assert_eq!(gdn_tri_solve_grid(2, 5, kernel).unwrap(), (2, 1, 1));
    }

    #[test]
    fn the_tri_solve_argument_builder_matches_its_kernel_signature() {
        let abi = LinearAbi::committed().unwrap();
        let args = GdnTriSolveArgs {
            a: 0x1000,
            rhs: 0x2000,
            out: 0x3000,
            n: 64,
            stride_a_h: 4096,
            stride_a_r: 64,
            stride_rhs_h: 8192,
            stride_rhs_r: 128,
            stride_out_h: 8192,
            stride_out_r: 128,
        };
        let pack = gdn_tri_solve_args(&abi, &args).unwrap();
        let kernel = abi.kernel(GDN_TRI_SOLVE).unwrap();
        assert_eq!(
            kernel.args,
            vec![
                ArgKind::Ptr,
                ArgKind::Ptr,
                ArgKind::Ptr,
                ArgKind::U32,
                ArgKind::U32,
                ArgKind::U32,
                ArgKind::U32,
                ArgKind::U32,
                ArgKind::U32,
                ArgKind::U32,
            ]
        );
        assert_eq!(pack.slots()[3], u64::from(64u32));
        let params = pack.kernel_params().unwrap();
        assert_eq!(params.len(), kernel.args.len() + kernel.scratch_params);
        for slot in &pack.slots()[kernel.args.len()..] {
            assert_eq!(*slot, 0, "scratch slot is not null");
        }
    }

    #[test]
    fn a_manifest_missing_the_tri_solve_kernel_is_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(MANIFEST_JSON).unwrap();
        v["linear"].as_object_mut().unwrap().remove(GDN_TRI_SOLVE);
        let err = LinearAbi::parse(&v.to_string()).unwrap_err().to_string();
        assert!(err.contains("missing linear kernel gdn_tri_solve"), "{err}");
    }

    #[test]
    fn a_mistyped_argpack_slot_is_rejected() {
        let abi = LinearAbi::committed().unwrap();
        let kernel = abi.kernel(ADD_RMSNORM).unwrap();
        let mut p = ArgPack::for_linear(kernel);
        for _ in 0..5 {
            p.ptr(0x1000).unwrap();
        }
        // The sixth declared parameter is the f32 eps; a u32 must not pass.
        assert!(p.u32(1).is_err());
    }

    #[test]
    fn the_dense_gemv_manifest_records_the_tile() {
        let abi = LinearAbi::committed().unwrap();
        let kernel = abi.kernel(NVFP4_DENSE_GEMV).unwrap();
        assert_eq!(kernel.constexpr("BLOCK_N").unwrap(), 32);
        assert_eq!(kernel.constexpr("BLOCK_KW").unwrap(), 16);
        assert_eq!(kernel.num_warps, 4);
    }

    #[test]
    fn the_dense_gemv_argument_builder_matches_its_kernel_signature() {
        let abi = LinearAbi::committed().unwrap();
        let args = DenseGemvArgs {
            a: 0x1000,
            packed: 0x2000,
            scale: 0x3000,
            global: 0x4000,
            c: 0x5000,
            m: 1,
            n: 248320,
            k_words: 256,
            stride_am: 2048,
            stride_ak: 1,
            // K-major resident head: N is contiguous, K steps by N.
            stride_pn: 1,
            stride_pkw: 248320,
            stride_sn: 1,
            stride_sblk: 248320,
            stride_cm: 248320,
            stride_cn: 1,
        };
        let pack = nvfp4_dense_gemv_args(&abi, &args).unwrap();
        let kernel = abi.kernel(NVFP4_DENSE_GEMV).unwrap();
        let params = pack.kernel_params().unwrap();
        assert_eq!(params.len(), kernel.args.len() + kernel.scratch_params);
        for slot in &pack.slots()[kernel.args.len()..] {
            assert_eq!(*slot, 0, "scratch slot is not null");
        }
    }

    #[test]
    fn the_dense_gemv_grid_tiles_the_vocab() {
        let abi = LinearAbi::committed().unwrap();
        let kernel = abi.kernel(NVFP4_DENSE_GEMV).unwrap();
        // 248320 vocab / 32 tile = 7760 blocks in y, one program per row in x.
        assert_eq!(
            nvfp4_dense_gemv_grid(1, 248320, kernel).unwrap(),
            (1, 7760, 1)
        );
    }

    #[test]
    fn the_fp8_manifest_records_the_tiles() {
        let abi = LinearAbi::committed().unwrap();
        let g = abi.kernel(FP8_GEMV).unwrap();
        assert_eq!(g.constexpr("BLOCK_N").unwrap(), 16);
        assert_eq!(g.constexpr("BLOCK_K").unwrap(), 128);
        assert_eq!(g.num_warps, 1);
        let r = abi.kernel(FP8_SPLITK_REDUCE).unwrap();
        assert_eq!(r.constexpr("BLOCK").unwrap(), 256);
        let gm = abi.kernel(FP8_GEMM).unwrap();
        assert_eq!(gm.constexpr("BLOCK_M").unwrap(), 64);
    }

    #[test]
    fn the_fp8_split_k_matches_freetoken() {
        // qkvz: N=12288 -> 768 tiles of 16, K=2048 -> 16 chunks, min(2, 16) -> 2.
        assert_eq!(fp8_split_k(768, 16), 2);
        // out_proj: N=2048 -> 128 tiles, K=4096 -> 32 chunks, min(12, 32) -> 8.
        assert_eq!(fp8_split_k(128, 32), 8);
        assert_eq!(fp8_split_k(1024, 1), 1);
    }

    #[test]
    fn the_fp8_argument_builders_match_their_kernel_signatures() {
        let abi = LinearAbi::committed().unwrap();
        let g = fp8_gemv_args(
            &abi,
            &Fp8GemvArgs {
                a: 0x1,
                w: 0x2,
                part: 0x3,
                n: 2048,
                k: 2048,
                n_kb: 16,
                kb_per: 8,
                stride_ak: 1,
                stride_wn: 2048,
                stride_wk: 1,
                stride_pk: 2048,
                stride_pn: 1,
            },
        )
        .unwrap();
        let kg = abi.kernel(FP8_GEMV).unwrap();
        assert_eq!(
            g.kernel_params().unwrap().len(),
            kg.args.len() + kg.scratch_params
        );
        let r = fp8_splitk_reduce_args(
            &abi,
            &Fp8ReduceArgs {
                part: 0x1,
                scale: 0x2,
                out: 0x3,
                n: 2048,
                split_k: 8,
                stride_pk: 2048,
                stride_pn: 1,
            },
        )
        .unwrap();
        let kr = abi.kernel(FP8_SPLITK_REDUCE).unwrap();
        assert_eq!(
            r.kernel_params().unwrap().len(),
            kr.args.len() + kr.scratch_params
        );
        let gm = fp8_gemm_args(
            &abi,
            &Fp8GemmArgs {
                a: 0x1,
                w: 0x2,
                scale: 0x3,
                c: 0x4,
                m: 64,
                n: 2048,
                k: 2048,
                stride_am: 2048,
                stride_ak: 1,
                stride_wn: 2048,
                stride_wk: 1,
                stride_cm: 2048,
                stride_cn: 1,
            },
        )
        .unwrap();
        let kgm = abi.kernel(FP8_GEMM).unwrap();
        assert_eq!(
            gm.kernel_params().unwrap().len(),
            kgm.args.len() + kgm.scratch_params
        );
    }
}
