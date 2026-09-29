// Dtype decoders used by the reference forward (and later by GPU stub tests).
// The engine runs fp32 internally; every stored dtype decodes to f32.

/// bf16 -> f32 (top 16 bits of an f32).
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// IEEE f16 -> f32 with round-to-nearest-even through a widening sign/exp path.
/// f16 values with all-ones exponent (inf/nan) map to the f32 counterparts.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let man = (bits & 0x3ff) as u32;
    if exp == 0 {
        if man == 0 {
            f32::from_bits(sign) // +/- 0.0
        } else {
            // subnormal: value = man * 2^-24 (man is exact in the f32 mantissa)
            (man as f32) * 2.0f32.powi(-24)
        }
    } else if exp == 0x1f {
        f32::from_bits(sign | 0x7f80_0000 | (man << 13)) // inf/nan
    } else {
        f32::from_bits(sign | ((exp as u32 + 112) << 23) | (man << 13))
    }
}

/// NVIDIA float8 e4m3fn -> f32. Exponent field 0b1111 is NaN only with mantissa
/// 111, otherwise a normal with exponent 8 (this is how the max 448 is encoded).
#[inline]
pub fn e4m3fn_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 3) & 0x0f;
    let man = bits & 0x07;
    let mag = if exp == 0 {
        // subnormal: man * 2^-9
        man as f32 * 2.0f32.powi(-9)
    } else if exp == 0x0f {
        // e4m3fn stores no infinity; exponent 15 is NaN only with man 111,
        // otherwise a normal with exponent 8 (this is how 448 is encoded).
        if man == 0x07 {
            return f32::NAN;
        }
        (1.0 + man as f32 / 8.0) * 2.0f32.powi(8)
    } else {
        (1.0 + man as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7)
    };
    sign * mag
}

/// NVFP4 e2m1 code -> f32 via the standard LUT (the nibble that the experts,
/// shared experts and lm_head store; byte[0] is the low nibble of the first
/// packed column pair in the row ordering the FTW writer produced).
/// Matches ``kernel/triton/nvfp4_dequant.py``'s table.
pub const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, //
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode one packed element at 4-bit position ``half`` (0 = low, 1 = high).
#[inline]
pub fn fp4_nibble(byte: u8, half: bool) -> f32 {
    let code = if half { byte >> 4 } else { byte & 0x0f } as usize;
    FP4_E2M1_LUT[code]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip() {
        for (bits, want) in [
            (0x3f80u16, 1.0f32),
            (0xc000, -2.0),
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x7f80, f32::INFINITY), // bf16 inf -> f32 inf
            (0x3f00, 0.5),
        ] {
            assert_eq!(bf16_to_f32(bits), want, "bits {bits:#06x}");
        }
        // 1.0 + 1 ulp is representable; check a nontrivial value.
        let x: f32 = 1.0 / 3.0;
        let b = (x.to_bits() >> 16) as u16;
        let roundtrip = bf16_to_f32(b);
        assert!((roundtrip - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn f16_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        // smallest subnormal: 2^-24
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        // 0.5 exactly
        assert_eq!(f16_to_f32(0x3800), 0.5);
    }

    #[test]
    fn e4m3fn_values() {
        assert_eq!(e4m3fn_to_f32(0x00), 0.0);
        assert_eq!(e4m3fn_to_f32(0x80), -0.0);
        // 1.0: 0x38 (exp 0111, mantissa 0)
        assert_eq!(e4m3fn_to_f32(0x38), 1.0);
        // 1.5: 0x3c (mantissa 100)
        assert_eq!(e4m3fn_to_f32(0x3c), 1.5);
        // -1.0
        assert_eq!(e4m3fn_to_f32(0xb8), -1.0);
        // 240.0: largest normal in the exp-14 binade (0x77)
        assert_eq!(e4m3fn_to_f32(0x77), 240.0);
        // 448.0: exponent-15 normal (1.75 * 2^8), 0x7e; only 0x7f is NaN
        assert_eq!(e4m3fn_to_f32(0x7e), 448.0);
        assert!(e4m3fn_to_f32(0x7f).is_nan());
        assert!(e4m3fn_to_f32(0xff).is_nan());
        // smallest subnormal: 2^-9 (0x01)
        assert_eq!(e4m3fn_to_f32(0x01), 2.0f32.powi(-9));
        assert_eq!(e4m3fn_to_f32(0x03), 3.0 * 2.0f32.powi(-9));
    }

    #[test]
    fn fp4_lut_covers_all_codes() {
        for (code, &want) in FP4_E2M1_LUT.iter().enumerate() {
            let byte = (code as u8) & 0x0f;
            assert_eq!(fp4_nibble(byte, false), want);
            let byteh = (code as u8) << 4;
            assert_eq!(fp4_nibble(byteh, true), want);
        }
    }
}
