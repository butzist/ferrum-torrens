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

impl Abi {
    pub fn parse(json: &str) -> Result<Self, MoEError> {
        let m: Manifest = serde_json::from_str(json)
            .map_err(|e| MoEError::Bad(format!("kernel manifest: {e}")))?;
        let mut kernels = BTreeMap::new();
        for (name, entry) in m.kernels {
            let mut args = Vec::with_capacity(entry.params.len());
            for param in &entry.params {
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
            // The scratch pointers are pointers whatever the manifest says, and
            // they are not arguments the caller supplies.
            let tail = args.split_off(args.len() - entry.scratch_params);
            if tail.iter().any(|k| *k != ArgKind::Ptr) {
                return Err(MoEError::Bad(format!(
                    "kernel {name}: scratch params are not all pointers"
                )));
            }
            if entry.params.len() != entry.declared_args.len() + entry.scratch_params {
                return Err(MoEError::Bad(format!(
                    "kernel {name}: {} params for {} declared args and {} scratch",
                    entry.params.len(),
                    entry.declared_args.len(),
                    entry.scratch_params
                )));
            }
            if entry.global_scratch_size != 0 || entry.profile_scratch_size != 0 {
                return Err(MoEError::Bad(format!(
                    "kernel {name}: scratch sizes {}/{} need a real allocation, \
                     which the driver-API launcher does not do yet",
                    entry.global_scratch_size, entry.profile_scratch_size
                )));
            }
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
    /// Sized for `kernel`, with the trailing scratch pointers already zeroed.
    pub fn for_kernel(kernel: &Kernel) -> Self {
        let mut pack = Self {
            kinds: kernel.args.clone(),
            declared: kernel.args.len(),
            slots: vec![0; kernel.args.len() + kernel.scratch_params],
            next: 0,
        };
        pack.kinds
            .extend(std::iter::repeat_n(ArgKind::Ptr, kernel.scratch_params));
        pack
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
    pub fn kernel_params(&self) -> Result<Vec<*const std::ffi::c_void>, MoEError> {
        self.finish()?;
        Ok(self
            .slots
            .iter()
            .map(|s| s as *const u64 as *const std::ffi::c_void)
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
#[derive(Debug, Clone, Copy)]
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

/// `(grid_x, grid_y, 1)` for a GEMV: one program per route, one per N block.
pub fn gemv_grid(routes: usize, kernel: &Kernel) -> (u32, u32, u32) {
    (routes as u32, div_ceil(kernel.n, abi_block_n()), 1)
}

/// The silu launch: one program per route, one per intermediate block.
pub fn silu_grid(routes: usize, inter: usize) -> (u32, u32, u32) {
    (routes as u32, div_ceil(inter, 256), 1)
}

/// The reduce launch: one program per token, one per hidden block.
pub fn sum_reduce_grid(tokens: usize, hidden: usize, block_n: usize) -> (u32, u32, u32) {
    (tokens as u32, div_ceil(hidden, block_n), 1)
}

/// `BLOCK_N` is a constexpr in the cubin, so the grid has to use the same value
/// the AOT build baked in. It is fixed for decode (upstream's offline sweep),
/// and the manifest's `n` plus this must divide evenly or the tail block runs
/// masked; the manifest test pins both.
const fn abi_block_n() -> usize {
    128
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
        for name in [GEMV_GATE_UP, GEMV_DOWN] {
            let kernel = abi.kernel(name).unwrap();
            assert_eq!(
                kernel.n % abi_block_n(),
                0,
                "{name}: N is not a multiple of BLOCK_N"
            );
        }
        assert_eq!(
            abi.inter % 256,
            0,
            "silu tile does not divide the intermediate size"
        );
        assert_eq!(
            abi.hidden % abi_block_n(),
            0,
            "reduce tile does not divide hidden"
        );
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
    fn a_manifest_with_scratch_to_allocate_is_rejected() {
        let json = MANIFEST_JSON.replacen(
            "\"global_scratch_size\": 0",
            "\"global_scratch_size\": 64",
            1,
        );
        let err = Abi::parse(&json).unwrap_err().to_string();
        assert!(err.contains("need a real allocation"), "{err}");
    }

    #[test]
    fn the_grids_cover_every_route_and_column() {
        let abi = abi();
        let gate_up = abi.kernel(GEMV_GATE_UP).unwrap();
        let routes = 3 * abi.top_k;
        assert_eq!(gemv_grid(routes, gate_up), (routes as u32, 8, 1));
        assert_eq!(silu_grid(routes, abi.inter), (routes as u32, 2, 1));
        assert_eq!(
            sum_reduce_grid(abi.top_k, abi.hidden, abi_block_n()),
            (8, 16, 1)
        );
    }
}
