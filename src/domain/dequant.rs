// Weight dequantization to f32, the storage-format half of a forward pass.
//
// libtorch has no e4m3 `mm` and no NVFP4 GEMM on any backend, so every
// checkpoint tensor is widened to f32 once at load (or per expert, for the
// banks) and the math runs on plain dense tensors. The packing rules here are
// the ones the converter wrote and FreeToken's Triton kernels read; they are the
// numerically delicate part of the port, so they live in the domain where they
// are unit-tested without libtorch.

use crate::domain::numerics::{bf16_to_f32, e4m3fn_to_f32, f16_to_f32, fp4_nibble};

/// Read a little-endian f16 array.
pub fn f16_array(bytes: &[u8], n: usize) -> Vec<f32> {
    (0..n.min(bytes.len() / 2))
        .map(|i| f16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])))
        .collect()
}

/// Read a little-endian bf16 array.
pub fn bf16_array(bytes: &[u8], n: usize) -> Vec<f32> {
    (0..n.min(bytes.len() / 2))
        .map(|i| bf16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])))
        .collect()
}

/// Per-output-row FP8 (e4m3fn) with an f32 row scale: `w[o][k] = e4m3 * scale[o]`.
///
/// `scale` holds one f32 per row. The checkpoint's separate per-tensor
/// `*_input_scale` is deliberately not applied: FreeToken's FP8 tensor path
/// ignores it too, so applying it here would drift from the oracle.
pub fn e4m3_with_row_scale(packed: &[u8], rows: usize, cols: usize, scale: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = scale.get(r).copied().unwrap_or(1.0);
        let base = r * cols;
        for k in 0..cols {
            out[base + k] = e4m3fn_to_f32(packed[base + k]) * s;
        }
    }
    out
}

/// NVFP4 block-scaled weights to f32, row-major `[rows, cols]`, **without** the
/// per-row f16 global.
///
/// Layout, as written by the converter and read by FreeToken's Triton kernels:
/// * `packed[rows, cols/2]` u8, **low nibble = even k**, high nibble = odd k.
/// * `scale[rows, cols/16]` e4m3fn, one per block of 16 along k.
///
/// An empty `global` means no per-row scale, which is how a `weight_scale`-only
/// tensor is expressed.
pub fn nvfp4(packed: &[u8], rows: usize, cols: usize, scale: &[u8], global: &[u8]) -> Vec<f32> {
    let mut out = nvfp4_block(packed, rows, cols, scale);
    let globals = nvfp4_globals(global, rows);
    if !globals.is_empty() {
        for r in 0..rows {
            let g = globals[r];
            for k in 0..cols {
                out[r * cols + k] *= g;
            }
        }
    }
    out
}

/// NVFP4 with the block scale folded in but the per-row f16 global left out.
///
/// This split is what makes bf16 storage lossless: `fp4 * e4m3` needs at most
/// four mantissa bits, well inside bf16's seven, so the widened value is
/// represented exactly. The f16 global has ten mantissa bits and would not be,
/// so it is returned separately by `nvfp4_globals` and applied as a per-output-row
/// factor *after* the matmul -- which is also what the reference's packed gemm
/// does, and what its Triton kernel does.
pub fn nvfp4_block(packed: &[u8], rows: usize, cols: usize, scale: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let pbase = r * (cols / 2);
        let sbase = r * (cols / 16);
        for k in 0..cols {
            out[r * cols + k] = fp4_nibble(packed[pbase + k / 2], k % 2 == 1)
                * e4m3fn_to_f32(scale[sbase + k / 16]);
        }
    }
    out
}

/// The per-row f16 globals of an NVFP4 tensor, or empty when the tensor has
/// none. Always apply after the matmul, never fold into the block scale
/// (folding underflows).
pub fn nvfp4_globals(global: &[u8], rows: usize) -> Vec<f32> {
    if global.len() < rows * 2 {
        return Vec::new();
    }
    (0..rows)
        .map(|r| f16_to_f32(u16::from_le_bytes([global[2 * r], global[2 * r + 1]])))
        .collect()
}

/// NVFP4 for one expert slab: `packed`/`scale` are already sliced to that
/// expert's rows, so this is a plain `nvfp4_block` with no per-expert offsets.
pub fn nvfp4_expert(packed: &[u8], scale: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    nvfp4_block(packed, rows, cols, scale)
}

/// Raw e4m3fn values, no row scale. Split out for the same reason as
/// `nvfp4_block`: an e4m3 value needs three mantissa bits, so bf16 holds it
/// exactly and the f32 row scale can be applied after the matmul.
pub fn e4m3_values(packed: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for (i, &b) in packed.iter().take(rows * cols).enumerate() {
        out[i] = e4m3fn_to_f32(b);
    }
    out
}

/// Bake the Gemma `1 + w` norm convention. The checkpoint stores the bare `w`;
/// every Gemma-style norm in this model multiplies by `1 + w`, and doing it here
/// keeps the norm ops themselves unremarkable.
pub fn gemma_bake(w: &mut [f32]) {
    for v in w.iter_mut() {
        *v += 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// e4m3fn code for a value, so the tests read in dequantized units.
    fn e4m3(v: f32) -> u8 {
        for code in 0u16..256 {
            let b = code as u8;
            if e4m3fn_to_f32(b).to_bits() == v.to_bits() {
                return b;
            }
        }
        panic!("no e4m3fn code for {v}");
    }

    /// Little-endian f16 bytes for a value, so tests read in dequantized units.
    fn f16(v: f32) -> [u8; 2] {
        for bits in 0u16..=u16::MAX {
            if f16_to_f32(bits) == v {
                return bits.to_le_bytes();
            }
        }
        panic!("no f16 code for {v}");
    }

    #[test]
    fn e4m3_rows_take_their_scale() {
        // Row 0: 1.0, 1.5, -1.0, 0.0 under a 2.0 scale -> 2.0, 3.0, -2.0, 0.0.
        // Row 1: the same codes under a 0.5 scale.
        let packed = [
            e4m3(1.0),
            e4m3(1.5),
            e4m3(-1.0),
            e4m3(0.0),
            e4m3(1.0),
            e4m3(1.5),
            e4m3(-1.0),
            e4m3(0.0),
        ];
        let out = e4m3_with_row_scale(&packed, 2, 4, &[2.0, 0.5]);
        assert_eq!(&out[0..4], &[2.0, 3.0, -2.0, 0.0]);
        assert_eq!(&out[4..8], &[0.5, 0.75, -0.5, 0.0]);
    }

    #[test]
    fn e4m3_missing_scale_defaults_to_one() {
        let packed = [e4m3(1.5)];
        let out = e4m3_with_row_scale(&packed, 1, 1, &[]);
        assert_eq!(out, vec![1.5]);
    }

    #[test]
    fn nvfp4_low_nibble_is_even_k() {
        // One row, 32 cols. Byte 0 = 0x21: low nibble 1 -> 0.5 at k=0, high
        // nibble 2 -> 1.0 at k=1. A swapped order would transpose these.
        let mut packed = vec![0u8; 16];
        packed[0] = 0x21;
        // Block scale 2.0 for both blocks, per-row global 0.5 -> effective 1.0.
        let scale = vec![e4m3(2.0); 2];
        let global = f16(0.5).to_vec();

        let out = nvfp4(&packed, 1, 32, &scale, &global);
        assert_eq!(out[0], 0.5, "k=0 comes from the low nibble");
        assert_eq!(out[1], 1.0, "k=1 comes from the high nibble");
        assert_eq!(out[2], 0.0, "unwritten bytes decode to zero");
    }

    #[test]
    fn nvfp4_global_is_per_row_and_applied_after_the_block_scale() {
        // Two rows, 16 cols, so 16 packed bytes. Row 0 byte 0 = 0x10, row 1
        // byte 0 = 0x10: low nibble 0 -> 0.0 at k=0, high nibble 1 -> 0.5 at k=1.
        // Block scale 4.0, globals 1.0 and 0.25 -> k=1 is 2.0 in row 0, 0.5 in
        // row 1.
        let mut packed = vec![0u8; 16];
        packed[0] = 0x10;
        packed[8] = 0x10;
        let scale = vec![e4m3(4.0); 2];
        let mut global = f16(1.0).to_vec();
        global.extend_from_slice(&f16(0.25));

        let out = nvfp4(&packed, 2, 16, &scale, &global);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[16 + 1], 0.5);
    }

    #[test]
    fn nvfp4_without_a_global_scale_is_the_identity() {
        let mut packed = vec![0u8; 8];
        packed[0] = 0x01; // low nibble 1 -> 0.5 at k=0
        let scale = vec![e4m3(1.0)];
        let out = nvfp4(&packed, 1, 16, &scale, &[]);
        assert_eq!(out[0], 0.5);
    }

    #[test]
    fn nvfp4_block_scale_actually_scales_its_own_block_of_16() {
        // 32 cols = 2 blocks; give block 1 a zero scale and check it zeroes only
        // k in [16, 32). This is the property the "fold the global in" bug loses.
        let packed = vec![0x11u8; 16]; // every k nonzero
        let scale = vec![e4m3(1.0), e4m3(0.0)];
        let out = nvfp4(&packed, 1, 32, &scale, &[]);
        assert_eq!(out[0], 0.5, "block 0 is unscaled");
        assert_eq!(out[15], 0.5);
        assert_eq!(out[16], 0.0, "block 1 has a zero scale");
        assert_eq!(out[31], 0.0);
    }

    #[test]
    fn nvfp4_covers_all_sixteen_fp4_codes() {
        // Every nibble value must survive the pack/unpack round trip: k = code,
        // so the code lands in the low nibble of byte k/2 when k is even and in
        // the high nibble when k is odd.
        let mut packed = vec![0u8; 8];
        for code in 0u8..16 {
            let nib = if code % 2 == 0 { code } else { code << 4 };
            packed[(code / 2) as usize] |= nib;
        }
        let scale = vec![e4m3(1.0)];
        let out = nvfp4(&packed, 1, 16, &scale, &[]);
        for code in 0u8..16 {
            let want = crate::domain::numerics::FP4_E2M1_LUT[code as usize];
            assert_eq!(out[code as usize], want, "code {code}");
        }
    }

    #[test]
    fn expert_slab_is_a_plain_nvfp4() {
        let packed = vec![0x21u8; 8];
        let scale = vec![e4m3(1.0)];
        let out = nvfp4_expert(&packed, &scale, 1, 16);
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], 1.0);
    }

    #[test]
    fn the_split_block_and_global_form_recompose_exactly() {
        // This split is what makes bf16 storage lossless, so it has to
        // recompose bit for bit.
        let (rows, cols) = (8, 64);
        let packed: Vec<u8> = (0..rows * cols / 2).map(|i| (i * 37 % 251) as u8).collect();
        let scale: Vec<u8> = (0..rows * cols / 16)
            .map(|i| e4m3(0.5 + (i % 8) as f32 * 0.25))
            .collect();
        let global: Vec<u8> = (0..rows)
            .flat_map(|r| f16(0.5 + (r % 4) as f32 * 0.5))
            .collect();

        let whole = nvfp4(&packed, rows, cols, &scale, &global);
        let blocks = nvfp4_block(&packed, rows, cols, &scale);
        let globals = nvfp4_globals(&global, rows);
        assert_eq!(globals.len(), rows);
        for r in 0..rows {
            for k in 0..cols {
                let recomposed = blocks[r * cols + k] * globals[r];
                assert_eq!(recomposed, whole[r * cols + k], "row {r} col {k}");
            }
        }
    }

    #[test]
    fn the_block_product_is_exactly_representable_in_bf16() {
        // The whole reason the global is split out: fp4 x e4m3 must survive a
        // bf16 round trip with no error, or bf16 storage would quietly lose
        // weight precision.
        let (rows, cols) = (4, 32);
        let packed: Vec<u8> = (0..rows * cols / 2).map(|i| (i * 11 % 256) as u8).collect();
        let scale: Vec<u8> = (0..rows * cols / 16)
            .map(|i| e4m3(0.5 + (i % 8) as f32 * 0.25))
            .collect();
        let blocks = nvfp4_block(&packed, rows, cols, &scale);
        for (i, &v) in blocks.iter().enumerate() {
            let round = bf16_to_f32((v.to_bits() >> 16) as u16);
            assert_eq!(round, v, "block value {i} ({v}) loses precision in bf16");
        }
    }

    #[test]
    fn raw_e4m3_values_survive_bf16_exactly() {
        // Same argument for the FP8 path: every e4m3fn code must be exact in bf16
        // so the row scale can be applied after the matmul.
        let values = e4m3_values(&[0x38, 0x3c, 0xb8, 0x01, 0x7e, 0x00], 2, 3);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 1.5);
        assert_eq!(values[2], -1.0);
        assert_eq!(values[3], 2.0f32.powi(-9));
        assert_eq!(values[4], 448.0);
        assert_eq!(values[5], 0.0);
        for &v in &values {
            assert_eq!(bf16_to_f32((v.to_bits() >> 16) as u16), v, "{v}");
        }
    }

    #[test]
    fn gemma_bake_shifts_by_one() {
        let mut w = vec![-1.0, 0.0, 0.5];
        gemma_bake(&mut w);
        assert_eq!(w, vec![0.0, 1.0, 1.5]);
    }

    #[test]
    fn bf16_and_f16_arrays_decode_little_endian() {
        // bf16 1.0 = 0x3f80, f16 1.0 = 0x3c00, both little-endian.
        assert_eq!(bf16_array(&[0x80, 0x3f, 0x00, 0xc0], 2), vec![1.0, -2.0]);
        assert_eq!(f16_array(&[0x00, 0x3c, 0x00, 0xc0], 2), vec![1.0, -2.0]);
    }
}
