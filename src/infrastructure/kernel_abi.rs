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
}

impl ArgKind {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "ptr" => Some(Self::Ptr),
            "u32" => Some(Self::U32),
            "u64" => Some(Self::U64),
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

const ATTENTION_NAMES: [&str; 1] = [PAGED_ATTENTION];

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
        other => return Err(MoEError::Bad(format!("no committed cubin for {other}"))),
    })
}

/// The committed attention PTX, by name. Only the ABI tests read this.
#[cfg(test)]
fn attention_ptx(name: &str) -> &'static str {
    match name {
        PAGED_ATTENTION => include_str!("../../kernel/paged_attention.ptx"),
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
            let width = fields
                .iter()
                .find_map(|f| f.strip_prefix(".u"))
                .and_then(|w| w.parse::<u32>().ok())
                .unwrap_or_else(|| panic!("PTX param without a width: {raw}"));
            let index: usize = fields
                .last()
                .and_then(|f| f.rsplit_once("_param_"))
                .and_then(|(_, i)| i.parse().ok())
                .unwrap_or_else(|| panic!("PTX param without an index: {raw}"));
            let kind = if fields.contains(&".ptr") {
                ArgKind::Ptr
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

    /// Append the next argument. The kind is checked against the manifest so a
    /// reordered or mistyped argument is an error here rather than a silently
    /// misaligned launch.
    pub fn push(&mut self, arg: Arg) -> Result<(), MoEError> {
        let want = self.kinds[self.next];
        let ok = matches!(
            (want, arg),
            (ArgKind::Ptr, Arg::Ptr(_)) | (ArgKind::U32, Arg::U32(_)) | (ArgKind::U64, Arg::U64(_))
        );
        if !ok {
            let (index, want) = (self.next, want);
            return Err(MoEError::Bad(format!(
                "kernel argument {index} wants {want:?}, got {arg:?}"
            )));
        }
        self.slots[self.next] = match arg {
            Arg::Ptr(a) | Arg::U64(a) => a,
            Arg::U32(v) => u64::from(v),
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
    p.u64((rows * (hidden / 2)) as u64)?;
    p.u32((hidden / 2) as u32)?;
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
    p.u64((rows * (inter / 2)) as u64)?;
    p.u32((inter / 2) as u32)?;
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
pub fn paged_attention_args(abi: &AttentionAbi, a: &AttentionArgs) -> Result<ArgPack, MoEError> {
    let kernel = abi.kernel(PAGED_ATTENTION)?;
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
        assert_eq!(grid(routes, gate_up), (routes as u32, 8, 1));
        assert_eq!(
            grid(routes, abi.kernel(GEMV_DOWN).unwrap()),
            (routes as u32, 16, 1)
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
        assert_eq!(lru.constexpr("BLOCK_C").unwrap(), 512);
        let copy = abi.kernel(INDEX_COPY).unwrap();
        assert_eq!(copy.grid, [6, 128, 1]);
        assert_eq!(copy.block_threads(), 128);
        assert_eq!(copy.constexpr("NUM_BANKS").unwrap(), 6);
        assert_eq!(copy.constexpr("GROUPS").unwrap(), 128);
        // A `BLOCK_K`-route plan of the 1 MiB `gate_up_packed` row.
        assert_eq!(copy.constexpr("STEPS").unwrap(), 128);
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
        assert_eq!(kernel.block_threads(), 128);
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
        let pack = paged_attention_args(&abi, &args).unwrap();
        let kernel = abi.kernel(PAGED_ATTENTION).unwrap();
        let params = pack.kernel_params().unwrap();
        assert_eq!(params.len(), kernel.args.len() + kernel.scratch_params);
        for slot in &pack.slots()[kernel.args.len()..] {
            assert_eq!(*slot, 0, "scratch slot is not null");
        }
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
    fn the_three_sections_are_disjoint() {
        let abi = abi();
        assert!(abi.kernel(PAGED_ATTENTION).is_err());
        let staging = StagingAbi::committed().unwrap();
        assert!(staging.kernel(PAGED_ATTENTION).is_err());
        let attention = AttentionAbi::committed().unwrap();
        assert!(attention.kernel(GEMV_GATE_UP).is_err());
        assert!(attention.kernel(LRU_ENSURE).is_err());
    }
}
