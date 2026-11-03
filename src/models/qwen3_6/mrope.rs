// Multimodal RoPE (mRoPE) for Qwen3.5/3.6 vision. Positions are triples
// (temporal, height, width); the rotary dimension is split into sections
// assigned to each component, interleaved.

/// Compute the per-dimension position index for mRoPE: for rotary dim `d`
/// (0..rotary_dim), return which of the three components (0=t,1=h,2=w) it
/// belongs to, given mrope_section = [t_len, h_len, w_len] (in rotary-dim
/// units, each doubled for sin/cos pairs).
pub fn mrope_component(d: usize, section: &[usize]) -> usize {
    // section gives half-dim counts per component (pairs). Work in pair units.
    let pair = d / 2;
    let mut acc = 0usize;
    for (i, &s) in section.iter().enumerate() {
        if pair < acc + s {
            return i;
        }
        acc += s;
    }
    section.len().saturating_sub(1)
}

/// Build the position array for a sequence of visual tokens: each token gets
/// (t, h, w). For a grid of (gh, gw) tokens, t is constant, h varies by row,
/// w varies by column.
pub fn visual_positions(gh: usize, gw: usize, t: i64) -> Vec<(i64, i64, i64)> {
    let mut out = Vec::with_capacity(gh * gw);
    for y in 0..gh {
        for x in 0..gw {
            out.push((t, y as i64, x as i64));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_assignment() {
        let section = [11usize, 11, 10];
        // first 11 pairs -> temporal
        assert_eq!(mrope_component(0, &section), 0);
        assert_eq!(mrope_component(20, &section), 0); // pair 10
        // next 11 -> height
        assert_eq!(mrope_component(22, &section), 1); // pair 11
        // last 10 -> width
        assert_eq!(mrope_component(60, &section), 2);
    }

    #[test]
    fn visual_positions_count() {
        let p = visual_positions(7, 7, 0);
        assert_eq!(p.len(), 49);
        assert_eq!(p[0], (0, 0, 0));
        assert_eq!(p[7], (0, 1, 0));
    }
}
