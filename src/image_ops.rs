use image::Rgb;
use image::RgbImage;
use ndarray::ArrayView4;

use crate::error::Result;
use crate::error::XqVisionError;
use crate::fast_path::TensorShape;
use crate::fast_path::normalize_rgb_to_chw;
use crate::geometry::Mat3;
use crate::geometry::invert;

#[derive(Debug, Default, Clone)]
pub(crate) struct TensorScratch {
    data: Vec<f32>,
    shape: Option<TensorShape>,
}

impl TensorScratch {
    pub(crate) fn with_capacity(pixels: usize) -> Self {
        let mut data = Vec::with_capacity(pixels * 3);
        data.resize(pixels * 3, 0.0);
        Self { data, shape: None }
    }

    pub(crate) fn normalize_image(&mut self, image: &RgbImage) {
        let shape =
            normalize_rgb_to_chw(image.as_raw(), image.width() as usize, image.height() as usize, &mut self.data);
        self.shape = Some(shape);
    }

    pub(crate) fn view(&self) -> Result<ArrayView4<'_, f32>> {
        let shape = self.shape.ok_or(XqVisionError::InvalidGeometry("tensor scratch is empty"))?;
        ArrayView4::from_shape((shape.n, shape.c, shape.h, shape.w), &self.data)
            .map_err(|_| XqVisionError::InvalidGeometry("tensor scratch shape mismatch"))
    }
}

pub(crate) fn resize_center_crop_rgb(
    image: &RgbImage, crop_width: u32, crop_height: u32, out_width: u32, out_height: u32,
) -> Result<RgbImage> {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 || out_width == 0 || out_height == 0 {
        return Err(XqVisionError::InvalidGeometry("image dimensions must be positive"));
    }

    let crop_width = crop_width.min(width);
    let crop_height = crop_height.min(height);
    let start_x = (width - crop_width) / 2;
    let start_y = (height - crop_height) / 2;

    let mut out = RgbImage::new(out_width, out_height);
    let src = image.as_raw();
    let dst = out.as_mut();
    let scale_x = crop_width as f32 / out_width as f32;
    let scale_y = crop_height as f32 / out_height as f32;
    // Hoist the row/column origin so the inner loop reduces to `origin + i * scale`
    // (2 ops) instead of `start + (i + 0.5) * scale - 0.5` (3 ops). Per-pixel
    // multiply form keeps the computation exact (no accumulated FP drift).
    let src_x_origin = start_x as f32 + 0.5 * scale_x - 0.5;
    let src_y_origin = start_y as f32 + 0.5 * scale_y - 0.5;

    let mut pixel_index = 0usize;
    for y in 0..out_height {
        let src_y = src_y_origin + y as f32 * scale_y;
        for x in 0..out_width {
            let src_x = src_x_origin + x as f32 * scale_x;
            // SAFETY: `sample_bilinear_rgb` validates and clamps source coordinates;
            // pixel_index increments monotonically over an `out_width * out_height`
            // buffer, so it stays in-bounds.
            unsafe {
                let rgb = sample_bilinear_rgb(src.as_ptr(), width, height, src_x, src_y);
                write_rgb_unchecked(dst.as_mut_ptr(), pixel_index, rgb);
            }
            pixel_index += 1;
        }
    }
    Ok(out)
}

pub(crate) fn warp_rgb(image: &RgbImage, forward: &Mat3, dst_width: u32, dst_height: u32) -> Result<RgbImage> {
    if image.width() == 0 || image.height() == 0 || dst_width == 0 || dst_height == 0 {
        return Err(XqVisionError::InvalidGeometry("image dimensions must be positive"));
    }

    let inverse = invert(forward)?;
    let mut out = RgbImage::new(dst_width, dst_height);
    let src = image.as_raw();
    let dst = out.as_mut();
    let width = image.width();
    let height = image.height();

    // Per-pixel inlining of `apply_mat3` plus a single reciprocal multiply
    // (vs. two divisions). Row origin (`u0`, `v0`, `w0`) is hoisted once per
    // row; per-pixel uses one mul-add per coordinate (multiply form, not
    // running sum — avoids accumulated FP drift across long rows).
    //
    // Degenerate case mirrors `apply_mat3`: when `|w| <= EPSILON` we emit a
    // black pixel rather than divide. Non-finite source coordinates produced
    // by other ill-conditioned matrices are caught inside `sample_bilinear_rgb`.
    let m = &inverse;
    let mut pixel_index = 0usize;
    for y in 0..dst_height {
        let yf = y as f32;
        let u0 = m[0][1] * yf + m[0][2];
        let v0 = m[1][1] * yf + m[1][2];
        let w0 = m[2][1] * yf + m[2][2];
        for x in 0..dst_width {
            let xf = x as f32;
            let u = u0 + m[0][0] * xf;
            let v = v0 + m[1][0] * xf;
            let w = w0 + m[2][0] * xf;
            let rgb = if w.abs() > f32::EPSILON {
                let inv_w = 1.0 / w;
                // SAFETY: `sample_bilinear_rgb` validates and clamps source coordinates.
                unsafe { sample_bilinear_rgb(src.as_ptr(), width, height, u * inv_w, v * inv_w) }
            } else {
                Rgb([0, 0, 0])
            };
            // SAFETY: pixel_index increments monotonically over a `dst_width *
            // dst_height` RGB buffer; stays in-bounds.
            unsafe { write_rgb_unchecked(dst.as_mut_ptr(), pixel_index, rgb) };
            pixel_index += 1;
        }
    }
    Ok(out)
}

unsafe fn write_rgb_unchecked(dst: *mut u8, pixel_index: usize, rgb: Rgb<u8>) {
    let base = pixel_index * 3;
    // SAFETY: caller guarantees `dst` points at an RGB buffer large enough for
    // `pixel_index`; all writes target the current pixel's three bytes.
    unsafe {
        *dst.add(base) = rgb[0];
        *dst.add(base + 1) = rgb[1];
        *dst.add(base + 2) = rgb[2];
    }
}

unsafe fn sample_bilinear_rgb(raw: *const u8, width: u32, height: u32, x: f32, y: f32) -> Rgb<u8> {
    // SAFETY: forwarded to the f32-returning sibling; same preconditions.
    let rgb = unsafe { sample_bilinear_rgb_f32(raw, width, height, x, y) };
    Rgb([
        rgb[0].round().clamp(0.0, 255.0) as u8,
        rgb[1].round().clamp(0.0, 255.0) as u8,
        rgb[2].round().clamp(0.0, 255.0) as u8,
    ])
}

// Bilinear sample without u8 quantization. `wx_inv` and `wy_inv` are hoisted
// once (vs. recomputed inside a per-channel loop). The per-channel arithmetic
// preserves the original `(top, bottom, blend)` order so the result is bit-
// identical to `sample_bilinear_rgb`'s pre-optimization output for any input.
unsafe fn sample_bilinear_rgb_f32(raw: *const u8, width: u32, height: u32, x: f32, y: f32) -> [f32; 3] {
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return [0.0, 0.0, 0.0];
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;
    let wx_inv = 1.0 - wx;
    let wy_inv = 1.0 - wy;

    // SAFETY: x/y are clamped to image bounds above, and x1/y1 are min-clamped.
    let p00 = unsafe { load_rgb_unchecked(raw, width, x0, y0) };
    // SAFETY: same as p00.
    let p10 = unsafe { load_rgb_unchecked(raw, width, x1, y0) };
    // SAFETY: same as p00.
    let p01 = unsafe { load_rgb_unchecked(raw, width, x0, y1) };
    // SAFETY: same as p00.
    let p11 = unsafe { load_rgb_unchecked(raw, width, x1, y1) };

    let mut out = [0.0_f32; 3];
    for channel in 0..3 {
        let top = p00[channel] as f32 * wx_inv + p10[channel] as f32 * wx;
        let bottom = p01[channel] as f32 * wx_inv + p11[channel] as f32 * wx;
        out[channel] = top * wy_inv + bottom * wy;
    }
    out
}

unsafe fn load_rgb_unchecked(raw: *const u8, width: u32, x: u32, y: u32) -> [u8; 3] {
    let base = ((y * width + x) * 3) as usize;
    // SAFETY: caller guarantees x/y are in-bounds for the source RGB buffer.
    unsafe { [*raw.add(base), *raw.add(base + 1), *raw.add(base + 2)] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point2f;

    #[test]
    fn resize_center_crop_keeps_requested_size() -> Result<()> {
        let image = RgbImage::from_pixel(4, 4, Rgb([10, 20, 30]));
        let resized = resize_center_crop_rgb(&image, 2, 2, 8, 6)?;
        assert_eq!(resized.dimensions(), (8, 6));
        assert_eq!(*resized.get_pixel(0, 0), Rgb([10, 20, 30]));
        Ok(())
    }

    // Verify the bilinear interpolation produces values matching a naive
    // hand-computed reference for a non-uniform 4×4 input. Sampling at the
    // midpoint between two pixels should average them; this guards against
    // accidental changes to the weight/order arithmetic in
    // `sample_bilinear_rgb_f32`.
    #[test]
    fn bilinear_matches_naive_reference() {
        let mut image = RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                image.put_pixel(x, y, Rgb([x as u8 * 60, y as u8 * 60, 0]));
            }
        }
        let raw = image.as_raw();
        for fy in 0..7 {
            for fx in 0..7 {
                let x = fx as f32 * 0.5;
                let y = fy as f32 * 0.5;
                // SAFETY: x/y stay within [0, 3] and the helper clamps anyway.
                let actual = unsafe { sample_bilinear_rgb_f32(raw.as_ptr(), 4, 4, x, y) };
                let expected = naive_bilinear(raw, 4, 4, x, y);
                for c in 0..3 {
                    assert!(
                        (actual[c] - expected[c]).abs() <= 1e-4,
                        "channel {} at ({}, {}): actual={} expected={}",
                        c,
                        x,
                        y,
                        actual[c],
                        expected[c]
                    );
                }
            }
        }
    }

    fn naive_bilinear(raw: &[u8], width: u32, height: u32, x: f32, y: f32) -> [f32; 3] {
        if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
            return [0.0, 0.0, 0.0];
        }
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);
        let wx = x - x0 as f32;
        let wy = y - y0 as f32;
        let read = |xi: u32, yi: u32, c: usize| raw[((yi * width + xi) * 3) as usize + c] as f32;
        let mut out = [0.0_f32; 3];
        for (c, item) in out.iter_mut().enumerate() {
            let v00 = read(x0, y0, c);
            let v10 = read(x1, y0, c);
            let v01 = read(x0, y1, c);
            let v11 = read(x1, y1, c);
            *item = v00 * (1.0 - wx) * (1.0 - wy) + v10 * wx * (1.0 - wy) + v01 * (1.0 - wx) * wy + v11 * wx * wy;
        }
        out
    }

    // Verify the inlined warp arithmetic is equivalent to the
    // `apply_mat3 + sample_bilinear_rgb` reference for several non-trivial
    // matrices. This is the key regression test for T5.
    #[test]
    fn warp_matches_apply_mat3_reference() -> Result<()> {
        let mut image = RgbImage::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                image.put_pixel(x, y, Rgb([(x * 30) as u8, (y * 30) as u8, ((x + y) * 15) as u8]));
            }
        }

        // Cases: identity, scale-up, slight perspective.
        let cases: &[Mat3] = &[
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.5, 0.0, -1.0], [0.0, 1.5, -1.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.1, 0.5], [0.05, 1.0, -0.2], [0.001, 0.0005, 1.0]],
        ];
        for (case_idx, forward) in cases.iter().enumerate() {
            let warped = warp_rgb(&image, forward, 8, 8)?;
            let inverse = crate::geometry::invert(forward)?;
            for y in 0..8 {
                for x in 0..8 {
                    let source = crate::geometry::apply_mat3(&inverse, Point2f::new(x as f32, y as f32));
                    let expected = if let Some(p) = source {
                        // SAFETY: matches the path inside warp_rgb.
                        unsafe { sample_bilinear_rgb(image.as_raw().as_ptr(), 8, 8, p.x, p.y) }
                    } else {
                        Rgb([0, 0, 0])
                    };
                    let actual = *warped.get_pixel(x, y);
                    for c in 0..3 {
                        let diff = (actual[c] as i16 - expected[c] as i16).abs();
                        assert!(
                            diff <= 1,
                            "case {} at ({}, {}) channel {}: actual={:?} expected={:?}",
                            case_idx,
                            x,
                            y,
                            c,
                            actual,
                            expected
                        );
                    }
                }
            }
        }
        Ok(())
    }

    // End-to-end regression: a gradient image resized must preserve monotonic
    // structure (left side darker than right, top darker than bottom) and stay
    // within the input value range. Guards against arithmetic regressions in
    // the resize loop's coordinate handling.
    #[test]
    fn resize_preserves_gradient_structure() -> Result<()> {
        let mut image = RgbImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                image.put_pixel(x, y, Rgb([(x * 16) as u8, (y * 16) as u8, 0]));
            }
        }
        let out = resize_center_crop_rgb(&image, 16, 16, 8, 8)?;
        for y in 0..8 {
            for x in 0..8 {
                let p = out.get_pixel(x, y);
                if x > 0 {
                    assert!(p[0] >= out.get_pixel(x - 1, y)[0], "R monotone failed at ({}, {})", x, y);
                }
                if y > 0 {
                    assert!(p[1] >= out.get_pixel(x, y - 1)[1], "G monotone failed at ({}, {})", x, y);
                }
                assert_eq!(p[2], 0, "B should remain zero at ({}, {})", x, y);
            }
        }
        Ok(())
    }

    #[test]
    fn tensor_scratch_uses_chw_shape() -> Result<()> {
        let image = RgbImage::from_pixel(3, 2, Rgb([10, 20, 30]));
        let mut scratch = TensorScratch::default();
        scratch.normalize_image(&image);
        assert_eq!(scratch.view()?.shape(), &[1, 3, 2, 3]);
        Ok(())
    }
}
