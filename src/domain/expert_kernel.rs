// MoE expert-compute ports: slot-resident NVFP4 experts plus the grouped GEMM
// that consumes them. Pure data and traits, no CUDA and no IO. `grouped_moe_reference`
// is the arithmetic oracle and mirrors the scalar tower's `moe_experts` exactly;
// adapters implement the two ports with host buffers (the CPU path) or the
// Triton kernel plus a device slot cache (the serving path).

use crate::domain::numerics::{e4m3fn_to_f32, f16_to_f32, fp4_nibble};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MoEError {
    #[error("moe: {0}")]
    Bad(String),
    #[error("moe: slot {slot} out of range (capacity {capacity})")]
    SlotOutOfRange { slot: usize, capacity: usize },
    /// The checkpoint has no such expert-bank entry.
    #[error("moe: no expert bank entry {entry}")]
    Missing { entry: String },
    /// The bytes behind an entry could not be read; `why` is the reader port's
    /// own message, kept verbatim so the shard/file detail survives.
    #[error("moe: read {entry}: {why}")]
    Read { entry: String, why: String },
}

/// Which half of an expert the two GEMMs read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    GateUp,
    Down,
}

/// One expert slab's bytes, already sliced to that expert's rows (decision 23
/// layouts): `packed [rows, cols/2]` u8 with the low nibble holding the even
/// k, `scale [rows, cols/16]` e4m3fn, `global [rows]` f16.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Slab {
    pub packed: Vec<u8>,
    pub scale: Vec<u8>,
    pub global: Vec<u8>,
}

/// Host bytes for every expert of one layer; mmap-backed in production.
pub trait ExpertSlabSource: Send + Sync {
    fn slab(&self, layer: usize, expert: u32, part: Part) -> Result<Slab, MoEError>;
}

/// The slot cache's write side: stage one expert's bytes into `slot`.
#[cfg_attr(test, mockall::automock)]
pub trait ExpertSlotStore: Send + Sync {
    /// Declare the layer that is about to be staged. Returns the layer whose
    /// device bytes were dropped to make room, if any, so the slot machine can
    /// forget that layer BEFORE `ExpertCache::ensure_experts` plans the new
    /// one -- otherwise the plan sees stale residency and never copies
    /// (decision 27). Idempotent per layer.
    fn begin_layer(&mut self, layer: usize) -> Result<Option<usize>, MoEError>;
    fn load(&mut self, layer: usize, expert: u32, slot: usize) -> Result<(), MoEError>;
    fn slots(&self) -> usize;
}

/// One layer's grouped MoE GEMM input: every `(token, top_k)` route is one
/// expert GEMV, `silu(gate) * up` sits between the two, and the routes are
/// summed weighted. `routes` holds slot ids (`-1` = not resident, contributes
/// nothing). The output stays a separate `&mut`, so nothing can alias it.
pub struct MoEGroup<'a> {
    pub hidden: usize,
    pub inter: usize,
    pub x: &'a [f32],
    pub top_k: usize,
    pub routes: &'a [i32],
    pub weights: &'a [f32],
}

#[cfg_attr(test, mockall::automock)]
pub trait ExpertGemm: Send + Sync {
    fn grouped_moe<'a>(&self, req: &MoEGroup<'a>, out: &mut [f32]) -> Result<(), MoEError>;
}

/// Slot-major banks for the reference path: `packed [slots, rows, cols/2]`, the
/// scales and globals the same way. This is exactly what a device slot cache
/// holds, so the Triton kernel sees the same byte layout.
pub struct HostBanks<'a> {
    pub gu_packed: &'a [u8],
    pub gu_scale: &'a [u8],
    pub gu_global: &'a [u8],
    pub dn_packed: &'a [u8],
    pub dn_scale: &'a [u8],
    pub dn_global: &'a [u8],
    pub slots: usize,
    pub gu_rows: usize,
    pub gu_cols: usize,
    pub dn_rows: usize,
    pub dn_cols: usize,
}

/// `out[t] = sum_j w[t][j] * down^T(silu_mul(gate_up^T x[t]))` over the resident
/// slots, in f32 and in the same order the scalar tower accumulates it.
pub fn grouped_moe_reference(
    banks: &HostBanks<'_>,
    req: &MoEGroup<'_>,
    out: &mut [f32],
) -> Result<(), MoEError> {
    let h = req.hidden;
    let i = req.inter;
    if banks.gu_rows != 2 * i
        || banks.gu_cols != h
        || banks.dn_rows != h
        || banks.dn_cols != i
        || h == 0
        || i == 0
        || !h.is_multiple_of(16)
        || !i.is_multiple_of(16)
    {
        return Err(MoEError::Bad(format!(
            "banks gu {}x{} dn {}x{} vs hidden {h} inter {i}",
            banks.gu_rows, banks.gu_cols, banks.dn_rows, banks.dn_cols
        )));
    }
    let slots = banks.slots;
    let gu_pb = banks.gu_rows * h / 2;
    let gu_sb = banks.gu_rows * h / 16;
    let gu_gb = banks.gu_rows * 2;
    let dn_pb = h * i / 2;
    let dn_sb = h * i / 16;
    let dn_gb = h * 2;
    if banks.gu_packed.len() != slots * gu_pb
        || banks.gu_scale.len() != slots * gu_sb
        || banks.gu_global.len() != slots * gu_gb
        || banks.dn_packed.len() != slots * dn_pb
        || banks.dn_scale.len() != slots * dn_sb
        || banks.dn_global.len() != slots * dn_gb
    {
        return Err(MoEError::Bad("slot bank byte lengths".into()));
    }

    let tokens = req.x.len() / h;
    let routes = tokens * req.top_k;
    if req.x.len() != tokens * h
        || req.routes.len() < routes
        || req.weights.len() < routes
        || out.len() < tokens * h
    {
        return Err(MoEError::Bad(format!(
            "request buffer lengths: x {} routes {} weights {} out {}",
            req.x.len(),
            req.routes.len(),
            req.weights.len(),
            out.len()
        )));
    }

    let out = &mut out[..tokens * h];
    out.fill(0.0);
    let mut ic1 = vec![0.0f32; 2 * i];
    let mut ic2 = vec![0.0f32; i];
    let mut ic3 = vec![0.0f32; h];

    for t in 0..tokens {
        let xrow = &req.x[t * h..(t + 1) * h];
        for j in 0..req.top_k {
            let slot = req.routes[t * req.top_k + j];
            if slot < 0 {
                continue;
            }
            let slot = slot as usize;
            if slot >= slots {
                return Err(MoEError::SlotOutOfRange {
                    slot,
                    capacity: slots,
                });
            }
            let w = req.weights[t * req.top_k + j];

            gemv_rows(
                xrow,
                &banks.gu_packed[slot * gu_pb..(slot + 1) * gu_pb],
                &banks.gu_scale[slot * gu_sb..(slot + 1) * gu_sb],
                &banks.gu_global[slot * gu_gb..(slot + 1) * gu_gb],
                2 * i,
                h,
                &mut ic1,
            );
            for k in 0..i {
                let g = ic1[k] / (1.0 + (-ic1[k]).exp());
                ic2[k] = g * ic1[i + k];
            }
            gemv_rows(
                &ic2,
                &banks.dn_packed[slot * dn_pb..(slot + 1) * dn_pb],
                &banks.dn_scale[slot * dn_sb..(slot + 1) * dn_sb],
                &banks.dn_global[slot * dn_gb..(slot + 1) * dn_gb],
                h,
                i,
                &mut ic3,
            );
            for d in 0..h {
                out[t * h + d] += w * ic3[d];
            }
        }
    }
    Ok(())
}

/// `y[r] = (sum_k x[k] * fp4(packed, r, k) * e4m3(scale, r, k / 16)) *
/// f16(global, r)`, the same order the scalar tower's `nvfp4_gemv` accumulates.
fn gemv_rows(
    x: &[f32],
    packed: &[u8],
    scale: &[u8],
    global: &[u8],
    rows: usize,
    cols: usize,
    y: &mut [f32],
) {
    let cols2 = cols / 2;
    let cols16 = cols / 16;
    for r in 0..rows {
        let g = if global.len() >= 2 * rows {
            f16_to_f32(u16::from_le_bytes([global[2 * r], global[2 * r + 1]]))
        } else {
            1.0
        };
        let mut acc = 0.0f32;
        for b in 0..cols16 {
            let s = e4m3fn_to_f32(scale[r * cols16 + b]);
            for j in 0..16 {
                let k = b * 16 + j;
                acc += x[k] * fp4_nibble(packed[r * cols2 + k / 2], k % 2 == 1) * s;
            }
        }
        y[r] = acc * g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: usize = 16;
    const I: usize = 16;

    const E4M3_ONE: u8 = 0x38;
    const E4M3_TWO: u8 = 0x40;

    /// Only the round values the tests use.
    fn f16_bits(v: f32) -> [u8; 2] {
        let bits: u16 = match v {
            0.5 => 0x3800,
            1.0 => 0x3c00,
            4.0 => 0x4400,
            _ => panic!("unsupported f16 literal"),
        };
        bits.to_le_bytes()
    }

    /// One slab whose every packed byte is `packed` and every scale is `scale`.
    fn slab(rows: usize, cols: usize, packed: u8, scale: u8, global: f32) -> Slab {
        Slab {
            packed: vec![packed; rows * cols / 2],
            scale: vec![scale; rows * cols / 16],
            global: vec![f16_bits(global); rows].concat(),
        }
    }

    /// Slot-major bytes, exactly what a device slot cache holds.
    fn flat(slabs: &[Slab]) -> Vec<u8> {
        slabs
            .iter()
            .flat_map(|s| s.packed.iter().copied())
            .collect()
    }
    fn flat_scale(slabs: &[Slab]) -> Vec<u8> {
        slabs.iter().flat_map(|s| s.scale.iter().copied()).collect()
    }
    fn flat_global(slabs: &[Slab]) -> Vec<u8> {
        slabs
            .iter()
            .flat_map(|s| s.global.iter().copied())
            .collect()
    }

    fn close(a: f32, b: f32) {
        assert!(
            (a - b).abs() <= 1e-4 * b.abs().max(1.0),
            "got {a}, want {b}"
        );
    }

    #[test]
    fn hand_computed_case_matches() {
        // Byte 0x10 is lo-nibble code 1 (0.5) and hi-nibble code 0 (0.0), so
        // every weight row is 0.5 on even k and 0.0 on odd k. With x all ones
        // and unit scales: ic1[r] = 8 * 0.5 = 4, gate and up are both 4,
        // ic2 = silu(4) * 4 everywhere, ic3[d] = 8 * 0.5 * ic2.
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let want = 0.5 * (8.0 * 0.5 * silu(4.0) * 4.0);

        let gu = flat(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let gu_s = flat_scale(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let gu_g = flat_global(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let dn = flat(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let dn_s = flat_scale(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let dn_g = flat_global(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 1,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let mut out = vec![0.0f32; H];
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 1,
                routes: &[0],
                weights: &[0.5],
            },
            &mut out,
        )
        .unwrap();
        for v in out {
            close(v, want);
        }
    }

    #[test]
    fn non_resident_route_contributes_nothing() {
        let gu = flat(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let gu_s = flat_scale(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let gu_g = flat_global(&[slab(2 * I, H, 0x10, E4M3_ONE, 1.0)]);
        let dn = flat(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let dn_s = flat_scale(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let dn_g = flat_global(&[slab(H, I, 0x10, E4M3_ONE, 1.0)]);
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 1,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let mut out = vec![7.0f32; H];
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 1,
                routes: &[-1],
                weights: &[1.0],
            },
            &mut out,
        )
        .unwrap();
        assert_eq!(out, vec![0.0; H]);
    }

    #[test]
    fn slot_selects_its_own_expert() {
        // Slot 0 is 0x01 (0.5 on even k), slot 1 the 0x10 one (0.5 on odd k),
        // so alternating x tells the two slots apart.
        let gus = [
            slab(2 * I, H, 0x01, E4M3_ONE, 1.0),
            slab(2 * I, H, 0x10, E4M3_ONE, 1.0),
        ];
        let dns = [
            slab(H, I, 0x01, E4M3_ONE, 1.0),
            slab(H, I, 0x10, E4M3_ONE, 1.0),
        ];
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));

        let run = |banks: &HostBanks<'_>, slot: i32| {
            let x: Vec<f32> = (0..H).map(|k| if k % 2 == 0 { 1.0 } else { 2.0 }).collect();
            let mut out = vec![0.0f32; H];
            grouped_moe_reference(
                banks,
                &MoEGroup {
                    hidden: H,
                    inter: I,
                    x: &x,
                    top_k: 1,
                    routes: &[slot],
                    weights: &[1.0],
                },
                &mut out,
            )
            .unwrap();
            out
        };

        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 2,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let silu = |v: f32| v / (1.0 + (-v).exp());
        // 8 nonzero weights per row: ic1 = 8 * 0.5 * 1 = 4 on slot 0 and
        // 8 * 0.5 * 2 = 8 on slot 1, then ic2 = silu(ic1) * ic1.
        let want0 = 8.0 * 0.5 * silu(4.0) * 4.0;
        let want1 = 8.0 * 0.5 * silu(8.0) * 8.0;
        for v in run(&banks, 0) {
            close(v, want0);
        }
        for v in run(&banks, 1) {
            close(v, want1);
        }

        // Zeroing slot 1's bytes must zero the slot-1 route only.
        let mut gus = gus.clone();
        let mut dns = dns.clone();
        gus[1] = Slab {
            packed: vec![0; 2 * I * H / 2],
            scale: vec![0; 2 * I * H / 16],
            global: vec![0; 4 * I],
        };
        dns[1] = Slab {
            packed: vec![0; H * I / 2],
            scale: vec![0; H * I / 16],
            global: vec![0; 2 * H],
        };
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 2,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        assert_eq!(run(&banks, 1), vec![0.0; H]);
        assert!(run(&banks, 0).iter().all(|&v| v > 0.0));
    }

    #[test]
    fn global_scale_applies_once_after_the_k_reduction() {
        // gate_up global 4x and down global 0.5x: the down half scales
        // linearly, the gate_up half goes through silu(4 * ic1).
        let gus = [slab(2 * I, H, 0x10, E4M3_ONE, 4.0)];
        let dns = [slab(H, I, 0x10, E4M3_ONE, 0.5)];
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 1,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let mut out = vec![0.0f32; H];
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 1,
                routes: &[0],
                weights: &[1.0],
            },
            &mut out,
        )
        .unwrap();
        // ic1 = 8 * 0.5 * 4 = 16, so ic2 = silu(16) * 16, ic3 = 8 * 0.5 * ic2 * 0.5.
        let ic2 = (16.0 / (1.0 + (-16.0f32).exp())) * 16.0;
        let want = 8.0 * 0.5 * ic2 * 0.5;
        for v in out {
            close(v, want);
        }
    }

    #[test]
    fn block_scale_is_per_sixteen_k() {
        // cols = 32: two e4m3 blocks per row, the second one 2x.
        let s0 = slab(2 * I, 32, 0x10, E4M3_ONE, 1.0);
        let mut s1 = slab(2 * I, 32, 0x10, E4M3_TWO, 1.0);
        s1.scale = (0..2 * I * 32 / 16)
            .map(|b| if b % 2 == 1 { E4M3_TWO } else { E4M3_ONE })
            .collect();
        let gus = [s0, s1];
        let dns = [
            slab(32, I, 0x10, E4M3_ONE, 1.0),
            slab(32, I, 0x10, E4M3_ONE, 1.0),
        ];
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 2,
            gu_rows: 2 * I,
            gu_cols: 32,
            dn_rows: 32,
            dn_cols: I,
        };
        let x = vec![1.0f32; 32];
        let mut out = vec![0.0f32; 32];
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: 32,
                inter: I,
                x: &x,
                top_k: 1,
                routes: &[1],
                weights: &[1.0],
            },
            &mut out,
        )
        .unwrap();
        // 16 k per row: block 0 carries the even k at scale 1 (8 * 0.5) and
        // block 1 the odd k at scale 2 (8 * 0.5 * 2), so ic1 = 12.
        let ic2 = (12.0 / (1.0 + (-12.0f32).exp())) * 12.0;
        let want = 8.0 * 0.5 * ic2;
        for v in out {
            close(v, want);
        }
    }

    #[test]
    fn routes_sum_weighted_over_top_k() {
        let gus = [
            slab(2 * I, H, 0x10, E4M3_ONE, 1.0),
            slab(2 * I, H, 0x11, E4M3_ONE, 1.0),
        ];
        let dns = [
            slab(H, I, 0x10, E4M3_ONE, 1.0),
            slab(H, I, 0x11, E4M3_ONE, 1.0),
        ];
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 2,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let one = |slot: i32, w: f32| {
            let mut out = vec![0.0f32; H];
            grouped_moe_reference(
                &banks,
                &MoEGroup {
                    hidden: H,
                    inter: I,
                    x: &x,
                    top_k: 1,
                    routes: &[slot],
                    weights: &[w],
                },
                &mut out,
            )
            .unwrap();
            out
        };
        let mut both = vec![0.0f32; H];
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 2,
                routes: &[0, 1],
                weights: &[0.25, 0.75],
            },
            &mut both,
        )
        .unwrap();
        let a = one(0, 0.25);
        let b = one(1, 0.75);
        for i in 0..H {
            close(both[i], a[i] + b[i]);
        }
    }

    #[test]
    fn shape_and_slot_mismatches_are_errors() {
        let gus = [slab(2 * I, H, 0x10, E4M3_ONE, 1.0)];
        let dns = [slab(H, I, 0x10, E4M3_ONE, 1.0)];
        let (gu, gu_s, gu_g) = (flat(&gus), flat_scale(&gus), flat_global(&gus));
        let (dn, dn_s, dn_g) = (flat(&dns), flat_scale(&dns), flat_global(&dns));
        let banks = HostBanks {
            gu_packed: &gu,
            gu_scale: &gu_s,
            gu_global: &gu_g,
            dn_packed: &dn,
            dn_scale: &dn_s,
            dn_global: &dn_g,
            slots: 1,
            gu_rows: 2 * I,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        let x = vec![1.0f32; H];
        let mut out = vec![0.0f32; H];
        let wide = MoEGroup {
            hidden: H + 16,
            inter: I,
            x: &x,
            top_k: 1,
            routes: &[0],
            weights: &[1.0],
        };
        assert!(grouped_moe_reference(&banks, &wide, &mut out).is_err());

        let ok = MoEGroup { hidden: H, ..wide };
        let overflow = MoEGroup { routes: &[5], ..ok };
        assert!(matches!(
            grouped_moe_reference(&banks, &overflow, &mut out),
            Err(MoEError::SlotOutOfRange {
                slot: 5,
                capacity: 1
            })
        ));
    }

    #[test]
    fn reference_matches_the_scalar_gemv_order() {
        // Same bytes through the scalar tower's `nvfp4_gemv` + silu-mul path,
        // which is the oracle the Triton kernel will be compared against.
        let rows = 2 * I;
        let mut packed = vec![0u8; rows * H / 2];
        let mut scale = vec![0u8; rows * H / 16];
        let mut global = vec![0u8; rows * 2];
        for (i, b) in packed.iter_mut().enumerate() {
            *b = (i * 37 + 11) as u8;
        }
        for (i, b) in scale.iter_mut().enumerate() {
            *b = (i * 7 + 3) as u8;
        }
        for r in 0..rows {
            global[2 * r..2 * r + 2].copy_from_slice(&f16_bits(1.0));
        }
        let dn = slab(H, I, 0x21, E4M3_ONE, 1.0);
        let dns = flat(std::slice::from_ref(&dn));
        let dns_s = flat_scale(std::slice::from_ref(&dn));
        let dns_g = flat_global(std::slice::from_ref(&dn));

        let x: Vec<f32> = (0..H).map(|k| ((k * 13 % 9) as f32) * 0.25 - 1.0).collect();
        let mut out = vec![0.0f32; H];
        let banks = HostBanks {
            gu_packed: &packed,
            gu_scale: &scale,
            gu_global: &global,
            dn_packed: &dns,
            dn_scale: &dns_s,
            dn_global: &dns_g,
            slots: 1,
            gu_rows: rows,
            gu_cols: H,
            dn_rows: H,
            dn_cols: I,
        };
        grouped_moe_reference(
            &banks,
            &MoEGroup {
                hidden: H,
                inter: I,
                x: &x,
                top_k: 1,
                routes: &[0],
                weights: &[1.0],
            },
            &mut out,
        )
        .unwrap();

        let ic1 =
            crate::models::qwen3_6::forward::nvfp4_gemv(&x, &packed, &scale, &global, rows, H);
        let mut ic2 = vec![0.0f32; I];
        for k in 0..I {
            let g = ic1[k] / (1.0 + (-ic1[k]).exp());
            ic2[k] = g * ic1[I + k];
        }
        let ic3 = crate::models::qwen3_6::forward::nvfp4_gemv(&ic2, &dns, &dns_s, &dns_g, H, I);
        for i in 0..H {
            close(out[i], ic3[i]);
        }
    }

    #[test]
    fn dequant_helpers_agree_with_the_reference_scales() {
        // Pins the two scale decoders the kernel has to mirror bit for bit.
        assert_eq!(e4m3fn_to_f32(E4M3_ONE), 1.0);
        assert_eq!(e4m3fn_to_f32(E4M3_TWO), 2.0);
        assert_eq!(f16_to_f32(u16::from_le_bytes(f16_bits(4.0))), 4.0);
        // Low nibble is the even k (decision 23).
        assert_eq!(fp4_nibble(0x10, false), 0.0);
        assert_eq!(fp4_nibble(0x10, true), 0.5);
        assert_eq!(fp4_nibble(0x01, false), 0.5);
        assert_eq!(fp4_nibble(0x01, true), 0.0);
    }
}
