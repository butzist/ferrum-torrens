// Image preprocessing for the multimodal path: decode, resize, normalize to
// the patch grid. Uses the `image` crate.

use image::GenericImageView;

/// The image processor's normalization, from `preprocessor_config.json`
/// (mean and std 0.5 per channel for this checkpoint).
pub const IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Decode image bytes and return (width, height, rgb f32 in [0,1]).
pub fn decode_rgb(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>), String> {
    let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for px in rgb.pixels() {
        out.push(px[0] as f32 / 255.0);
        out.push(px[1] as f32 / 255.0);
        out.push(px[2] as f32 / 255.0);
    }
    Ok((w, h, out))
}

/// Channel-wise normalization over a CHW buffer: each of the `channels`
/// planes `(min(3, len))` is `(v - mean[c]) / std[c]`.
pub fn normalize_rgb(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let plane = w * h;
    (0..3)
        .flat_map(move |c| {
            src[c * plane..(c + 1) * plane]
                .iter()
                .map(move |v| (v - IMAGE_MEAN[c]) / IMAGE_STD[c])
        })
        .collect()
}

/// Interleaved HWC RGB to channel-plane CHW, the tower's pixel layout.
pub fn to_chw(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let plane = w * h;
    let mut out = vec![0f32; src.len()];
    for c in 0..3usize {
        for i in 0..plane {
            out[c * plane + i] = src[i * 3 + c];
        }
    }
    out
}

/// Resize an RGB buffer to (nw, nh) using bilinear filtering.
pub fn resize_rgb(src: &[f32], sw: u32, sh: u32, nw: u32, nh: u32) -> Vec<f32> {
    let mut dst = vec![0f32; (nw * nh * 3) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let sx = x as f32 * (sw as f32 - 1.0) / (nw as f32 - 1.0).max(1.0);
            let sy = y as f32 * (sh as f32 - 1.0) / (nh as f32 - 1.0).max(1.0);
            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            for c in 0..3usize {
                let a = src[((y0 * sw + x0) * 3 + c as u32) as usize];
                let b = src[((y0 * sw + x1) * 3 + c as u32) as usize];
                let cc = src[((y1 * sw + x0) * 3 + c as u32) as usize];
                let d = src[((y1 * sw + x1) * 3 + c as u32) as usize];
                dst[((y * nw + x) * 3 + c as u32) as usize] = a * (1.0 - fx) * (1.0 - fy)
                    + b * fx * (1.0 - fy)
                    + cc * (1.0 - fx) * fy
                    + d * fx * fy;
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_shape() {
        let src = vec![0.5f32; 4 * 4 * 3];
        let dst = resize_rgb(&src, 4, 4, 2, 2);
        assert_eq!(dst.len(), 2 * 2 * 3);
    }

    #[test]
    fn normalize_is_zero_centered_on_mid_gray() {
        let src = vec![0.5f32; 3 * 2 * 2];
        assert!(normalize_rgb(&src, 2, 2).iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn normalize_scale_is_two() {
        // One plane is all 1.0 -> +1; the next all 0.0 -> -1.
        let mut src = vec![1.0f32; 4];
        src.extend(vec![0.0f32; 4]);
        src.extend(vec![0.5f32; 4]);
        assert_eq!(
            normalize_rgb(&src, 2, 2),
            vec![
                1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
    }

    #[test]
    fn to_chw_transposes_the_planes() {
        let src: Vec<f32> = (0..6).map(|i| i as f32).collect();
        assert_eq!(to_chw(&src, 2, 1), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn decode_png() {
        // 2x2 red png generated in-memory via image crate
        let mut img = image::RgbImage::new(2, 2);
        for p in img.pixels_mut() {
            *p = image::Rgb([255, 0, 0]);
        }
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let (w, h, px) = decode_rgb(&buf).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(px.len(), 12);
        assert!((px[0] - 1.0).abs() < 1e-6);
    }
}
