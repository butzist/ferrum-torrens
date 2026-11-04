// Multimodal RoPE (mRoPE) for Qwen3.5/3.6 vision. Positions are triples
// (temporal, height, width); the rotary pairs are assigned to the three axes
// and each pair rotates with its own axis's position at the pair's frequency.
//
// Semantics follow HF transformers `modeling_qwen3_vl.py`:
// `get_rope_index` (the per-run position counter) and
// `Qwen3VLTextRotaryEmbedding::recomposition_frequencies` (the pair map).

/// Which of the three mrope axes (0=temporal, 1=height, 2=width) rotary pair
/// `pair` rotates with. Interleaved layout, `mrope_section = [t_len, h_len,
/// w_len]` in pairs: temporal everywhere first, then the height axis takes
/// every pair at offset 1 stride 3 below `h_len * 3`, the width axis every
/// pair at offset 2 stride 3 below `w_len * 3`, and the rest stays temporal.
pub fn mrope_component(pair: usize, section: &[usize]) -> usize {
    let Some(&h_len) = section.get(1) else {
        return 0;
    };
    if pair % 3 == 1 && pair < h_len * 3 {
        return 1;
    }
    let Some(&w_len) = section.get(2) else {
        return 0;
    };
    if pair % 3 == 2 && pair < w_len * 3 {
        return 2;
    }
    0
}

/// The axis index (0, 1 or 2) for every one of `pairs` rotary pairs, from the
/// `mrope_section` config. The interleaved layout recomposes the axes
/// pairwise; the contiguous layout gives the first `section[0]` pairs to
/// temporal, the next to height and the rest to width.
pub fn mrope_axis_map(section: &[usize], pairs: usize, interleaved: bool) -> Vec<usize> {
    if !interleaved {
        return (0..pairs)
            .map(|pair| {
                let mut acc = 0usize;
                for (i, &s) in section.iter().enumerate() {
                    if pair < acc + s {
                        return i;
                    }
                    acc += s;
                }
                section.len().saturating_sub(1)
            })
            .collect();
    }
    (0..pairs)
        .map(|pair| mrope_component(pair, section))
        .collect()
}

/// One image run inside an expanded prompt: the run's first global token index
/// and the image's merged LLM grid (spatial merge applied, so the run holds
/// `gh * gw` tokens laid out row-major).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRun {
    pub start: usize,
    pub gh: usize,
    pub gw: usize,
}

/// The mrope position triples for a whole expanded prompt of `total_len`
/// tokens: text runs count every axis up together, an image run holds its
/// temporal position constant at the counter and spreads height/width over the
/// grid row-major, and the counter advances by `max(gh, gw)` per image (HF
/// `get_rope_index`). Also returns the rope delta for decode continuations:
/// the token at sequence index `start` past the prompt sits at
/// `start + delta` on every axis, where `delta = max position + 1 -
/// total_len`. Runs must be sorted and non-overlapping, and together with the
/// text between them cover all `total_len` tokens.
pub fn prompt_triples(total_len: usize, runs: &[ImageRun]) -> (Vec<[u32; 3]>, i64) {
    let mut out = Vec::with_capacity(total_len);
    let mut st = 0i64;
    let mut at = 0usize;
    for run in runs {
        debug_assert!(
            run.start >= at,
            "image runs must be sorted and non-overlapping"
        );
        let text = run.start - at;
        for i in 0..text {
            out.push([u_pos(st + i as i64); 3]);
        }
        st += text as i64;
        at += text;
        let tokens = run.gh * run.gw;
        for i in 0..tokens {
            let row = (i / run.gw) as i64;
            let col = (i % run.gw) as i64;
            out.push([u_pos(st), u_pos(st + row), u_pos(st + col)]);
        }
        st += run.gh.max(run.gw) as i64;
        at += tokens;
    }
    while at < total_len {
        out.push([u_pos(st); 3]);
        st += 1;
        at += 1;
    }
    let max_pos = out
        .iter()
        .flat_map(|t| t.iter())
        .copied()
        .max()
        .map(i64::from)
        .unwrap_or(0);
    let delta = max_pos + 1 - total_len as i64;
    (out, delta)
}

fn u_pos(v: i64) -> u32 {
    u32::try_from(v).unwrap_or_else(|_| panic!("mrope position {v} does not fit u32"))
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
    fn interleaved_pair_map_splits_by_offset_and_stride() {
        let section = [11usize, 11, 10];
        let map = mrope_axis_map(&section, 32, true);
        assert_eq!(map.len(), 32);
        let height: Vec<usize> = (1..33).step_by(3).collect();
        let width: Vec<usize> = (2..33).step_by(3).filter(|p| *p < 30).collect();
        for &p in &height {
            assert_eq!(map[p], 1, "pair {p} is height");
        }
        for &p in &width {
            assert_eq!(map[p], 2, "pair {p} is width");
        }
        for (p, &axis) in map.iter().enumerate() {
            if !height.contains(&p) && !width.contains(&p) {
                assert_eq!(axis, 0, "pair {p} is temporal");
            }
        }
        // The old contiguous helper disagreed exactly here: pair 10 is height
        // interleaved, temporal contiguous.
        assert_eq!(mrope_component(10, &section), 1);
        assert_eq!(mrope_component(9, &section), 0);
        assert_eq!(mrope_component(29, &section), 2);
        assert_eq!(mrope_component(30, &section), 0);
        assert_eq!(mrope_component(32, &section), 0);
        assert_eq!(map.iter().filter(|&&a| a == 0).count(), 11);
        assert_eq!(map.iter().filter(|&&a| a == 1).count(), 11);
        assert_eq!(map.iter().filter(|&&a| a == 2).count(), 10);
    }

    #[test]
    fn contiguous_pair_map_keeps_the_sections_together() {
        let map = mrope_axis_map(&[11usize, 11, 10], 32, false);
        assert!(map[..11].iter().all(|&a| a == 0));
        assert!(map[11..22].iter().all(|&a| a == 1));
        assert!(map[22..32].iter().all(|&a| a == 2));
        // Pairs past the section total stay on the last axis.
        let map = mrope_axis_map(&[11usize, 11, 10], 40, false);
        assert!(map[32..].iter().all(|&a| a == 2));
        // Interleaved pairs past the section total stay temporal.
        let map = mrope_axis_map(&[11usize, 11, 10], 40, true);
        assert!(map[33..].iter().all(|&a| a == 0));
    }

    #[test]
    fn text_prompt_is_flat_positions_with_zero_delta() {
        let (triples, delta) = prompt_triples(5, &[]);
        assert_eq!(
            triples,
            vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
        );
        assert_eq!(delta, 0);
    }

    /// The hand example: 5 text tokens, then a 7x7 image. The image run takes
    /// temporal 5 for every token, height 5 + row, width 5 + col, and the
    /// counter advances by max(7, 7) = 7 -- not by the 49 tokens the run
    /// splices.
    #[test]
    fn image_run_spreads_height_and_width_over_the_grid() {
        let (triples, delta) = prompt_triples(
            54,
            &[ImageRun {
                start: 5,
                gh: 7,
                gw: 7,
            }],
        );
        assert_eq!(triples.len(), 54);
        for (i, tr) in triples[..5].iter().enumerate() {
            assert_eq!(*tr, [i as u32; 3]);
        }
        assert_eq!(triples[5], [5, 5, 5], "first image token: (st, st, st)");
        assert_eq!(triples[6], [5, 5, 6], "row-major: width first");
        assert_eq!(triples[11], [5, 5, 11], "end of the first grid row");
        assert_eq!(triples[12], [5, 6, 5], "second grid row starts");
        assert_eq!(triples[53], [5, 11, 11], "last image token");
        // Max position is 11 (height or width of the last grid row), so the
        // decode delta is 12 - 54.
        assert_eq!(delta, -42);
        // The first decode token sits at the counter the prompt ended on.
        assert_eq!(54 + delta, 12);
    }

    /// Text after an image continues at the advanced counter. The counter
    /// trails the token count (the run splices gh*gw tokens but advances by
    /// max(gh, gw)), so the delta is negative even with trailing text.
    #[test]
    fn text_after_an_image_continues_at_the_counter() {
        let run = ImageRun {
            start: 2,
            gh: 2,
            gw: 4,
        };
        let (triples, delta) = prompt_triples(13, &[run]);
        assert_eq!(triples.len(), 13);
        assert_eq!(triples[0], [0, 0, 0]);
        assert_eq!(triples[1], [1, 1, 1]);
        assert_eq!(triples[2], [2, 2, 2]);
        assert_eq!(triples[5], [2, 2, 5], "first grid row, last column");
        assert_eq!(triples[6], [2, 3, 2], "second grid row");
        assert_eq!(
            triples[9],
            [2, 3, 5],
            "last image token: 8 tokens row-major"
        );
        assert_eq!(triples[10], [6, 6, 6], "text resumes at 2 + max(2, 4)");
        assert_eq!(triples[12], [8, 8, 8]);
        assert_eq!(delta, 9 - 13);
    }

    #[test]
    fn two_images_stack_their_counters() {
        let runs = [
            ImageRun {
                start: 1,
                gh: 2,
                gw: 2,
            },
            ImageRun {
                start: 9,
                gh: 2,
                gw: 3,
            },
        ];
        // 1 text + 4 tokens + 4 text + 6 tokens + 1 text = 16.
        let (triples, delta) = prompt_triples(16, &runs);
        assert_eq!(triples.len(), 16);
        assert_eq!(triples[0], [0, 0, 0]);
        assert_eq!(triples[1], [1, 1, 1]);
        assert_eq!(triples[4], [1, 2, 2]);
        assert_eq!(triples[5], [3, 3, 3], "counter advanced by max(2, 2)");
        assert_eq!(triples[8], [6, 6, 6], "four text tokens, positions 3..6");
        assert_eq!(triples[9], [7, 7, 7], "counter advanced by 4 more to 7");
        assert_eq!(triples[14], [7, 8, 9]);
        assert_eq!(
            triples[15],
            [10, 10, 10],
            "last text token at 7 + max(2, 3)"
        );
        assert_eq!(delta, 11 - 16);
    }

    #[test]
    fn visual_positions_count() {
        let p = visual_positions(7, 7, 0);
        assert_eq!(p.len(), 49);
        assert_eq!(p[0], (0, 0, 0));
        assert_eq!(p[7], (0, 1, 0));
    }
}
