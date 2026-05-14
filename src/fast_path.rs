pub(crate) const IMAGENET_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
pub(crate) const IMAGENET_STD: [f32; 3] = [58.395, 57.12, 57.375];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TensorShape {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

pub(crate) fn normalize_rgb_to_chw(raw: &[u8], width: usize, height: usize, out: &mut Vec<f32>) -> TensorShape {
    let pixels = width * height;
    debug_assert_eq!(raw.len(), pixels * 3);
    out.resize(pixels * 3, 0.0);
    dispatch_normalize(raw, width, height, pixels, out);
    TensorShape { n: 1, c: 3, h: height, w: width }
}

#[cfg(all(feature = "fast-path", target_arch = "aarch64"))]
fn dispatch_normalize(raw: &[u8], _width: usize, _height: usize, pixels: usize, out: &mut [f32]) {
    // SAFETY: NEON is part of the aarch64 baseline; buffers sized as documented.
    unsafe { normalize_rgb_to_chw_neon(raw.as_ptr(), pixels, out.as_mut_ptr()) };
}

#[cfg(all(feature = "fast-path", any(target_arch = "x86", target_arch = "x86_64")))]
fn dispatch_normalize(raw: &[u8], _width: usize, _height: usize, pixels: usize, out: &mut [f32]) {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2 + FMA detected at runtime; buffers sized as documented.
        unsafe { normalize_rgb_to_chw_avx2(raw.as_ptr(), pixels, out.as_mut_ptr()) };
    } else {
        // SAFETY: scalar fallback when AVX2/FMA is unavailable; buffers sized as documented.
        unsafe { normalize_rgb_to_chw_unchecked(raw.as_ptr(), pixels, out.as_mut_ptr()) };
    }
}

#[cfg(all(feature = "fast-path", not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))))]
fn dispatch_normalize(raw: &[u8], _width: usize, _height: usize, pixels: usize, out: &mut [f32]) {
    // SAFETY: scalar fallback for unsupported architectures; buffers sized as documented.
    unsafe { normalize_rgb_to_chw_unchecked(raw.as_ptr(), pixels, out.as_mut_ptr()) };
}

#[cfg(not(feature = "fast-path"))]
fn dispatch_normalize(raw: &[u8], width: usize, height: usize, _pixels: usize, out: &mut [f32]) {
    normalize_rgb_to_chw_safe(raw, width, height, out);
}

#[cfg(any(test, not(feature = "fast-path")))]
pub(crate) fn normalize_rgb_to_chw_safe(raw: &[u8], width: usize, height: usize, out: &mut [f32]) {
    let pixels = width * height;
    debug_assert_eq!(raw.len(), pixels * 3);
    debug_assert_eq!(out.len(), pixels * 3);

    let (red, rest) = out.split_at_mut(pixels);
    let (green, blue) = rest.split_at_mut(pixels);
    for i in 0..pixels {
        let base = i * 3;
        red[i] = (raw[base] as f32 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        green[i] = (raw[base + 1] as f32 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        blue[i] = (raw[base + 2] as f32 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
}

#[cfg(all(feature = "fast-path", not(target_arch = "aarch64")))]
unsafe fn normalize_rgb_to_chw_unchecked(raw: *const u8, pixels: usize, out: *mut f32) {
    let red = out;
    // SAFETY: caller guarantees `out` has three contiguous planes of `pixels`.
    let green = unsafe { out.add(pixels) };
    // SAFETY: caller guarantees `out` has three contiguous planes of `pixels`.
    let blue = unsafe { out.add(pixels * 2) };

    for i in 0..pixels {
        // SAFETY: caller guarantees `raw` has `pixels * 3` initialized bytes.
        let src = unsafe { raw.add(i * 3) };
        // SAFETY: pointers are in-bounds for the same reason documented above.
        unsafe {
            *red.add(i) = (*src as f32 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            *green.add(i) = (*src.add(1) as f32 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            *blue.add(i) = (*src.add(2) as f32 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }
    }
}

// AVX2 + FMA: 8 pixels per iteration (24 RGB bytes), tail scalar.
// Algorithm: two 16-byte SSE loads at offsets 0 and 12 each cover 4 RGB pixels;
// `_mm_shuffle_epi8` deinterleaves each load into 4 R/G/B values; the two halves
// are merged into 8-lane vectors with `_mm_unpacklo_epi32`, widened to 8 × f32,
// and fused as `x * (1/std) + (-mean/std)` via `_mm256_fmadd_ps`.
#[cfg(all(feature = "fast-path", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn normalize_rgb_to_chw_avx2(raw: *const u8, pixels: usize, out: *mut f32) {
    use std::arch::x86_64::*;

    let red = out;
    // SAFETY: caller guarantees `out` has three contiguous planes of `pixels`.
    let green = unsafe { out.add(pixels) };
    // SAFETY: same as above.
    let blue = unsafe { out.add(pixels * 2) };

    // SAFETY: AVX2 + FMA target features enabled by attribute; these intrinsics
    // operate on by-value scalars and produce vector constants.
    let (recip_std_r, recip_std_g, recip_std_b, neg_bias_r, neg_bias_g, neg_bias_b) = unsafe {
        (
            _mm256_set1_ps(1.0 / IMAGENET_STD[0]),
            _mm256_set1_ps(1.0 / IMAGENET_STD[1]),
            _mm256_set1_ps(1.0 / IMAGENET_STD[2]),
            _mm256_set1_ps(-IMAGENET_MEAN[0] / IMAGENET_STD[0]),
            _mm256_set1_ps(-IMAGENET_MEAN[1] / IMAGENET_STD[1]),
            _mm256_set1_ps(-IMAGENET_MEAN[2] / IMAGENET_STD[2]),
        )
    };

    // Shuffle masks: pick R/G/B bytes from a 16-byte load covering 4 RGB pixels
    // plus 4 spillover bytes. High bit (0x80) zeroes the destination byte.
    // SAFETY: same as above.
    let (mask_r, mask_g, mask_b) = unsafe {
        (
            _mm_setr_epi8(0, 3, 6, 9, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128),
            _mm_setr_epi8(1, 4, 7, 10, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128),
            _mm_setr_epi8(2, 5, 8, 11, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128),
        )
    };

    let mut i = 0usize;
    // We process 8 pixels (24 RGB bytes) per iteration but load via two 16-byte
    // SSE loads at offsets 0 and 12 from `raw + i*3`. The second load reads up
    // to byte `i*3 + 27`, so the precondition is `i*3 + 28 <= pixels*3`, i.e.,
    // `i + 10 <= pixels` (because 28/3 ≈ 9.33, rounded up).
    while i + 10 <= pixels {
        // SAFETY: all AVX2/SSE intrinsics here require the AVX2+FMA target features,
        // which the function attribute guarantees; bounds for loads/stores follow
        // from the loop condition and caller contracts.
        unsafe {
            let load1 = _mm_loadu_si128(raw.add(i * 3) as *const __m128i);
            let load2 = _mm_loadu_si128(raw.add(i * 3 + 12) as *const __m128i);

            let r1 = _mm_shuffle_epi8(load1, mask_r); // [r0,r1,r2,r3, 0×12]
            let g1 = _mm_shuffle_epi8(load1, mask_g);
            let b1 = _mm_shuffle_epi8(load1, mask_b);
            let r2 = _mm_shuffle_epi8(load2, mask_r); // [r4,r5,r6,r7, 0×12]
            let g2 = _mm_shuffle_epi8(load2, mask_g);
            let b2 = _mm_shuffle_epi8(load2, mask_b);

            // unpacklo_epi32 interleaves 32-bit elements from the low halves; with
            // each input's r/g/b packed into low 4 bytes, the result's low 8 bytes
            // become [r0..r3, r4..r7], ready for cvtepu8_epi32 → 8 × i32 → 8 × f32.
            let r_packed = _mm_unpacklo_epi32(r1, r2);
            let g_packed = _mm_unpacklo_epi32(g1, g2);
            let b_packed = _mm_unpacklo_epi32(b1, b2);

            let r_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r_packed));
            let g_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g_packed));
            let b_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b_packed));

            let r_out = _mm256_fmadd_ps(r_f32, recip_std_r, neg_bias_r);
            let g_out = _mm256_fmadd_ps(g_f32, recip_std_g, neg_bias_g);
            let b_out = _mm256_fmadd_ps(b_f32, recip_std_b, neg_bias_b);

            _mm256_storeu_ps(red.add(i), r_out);
            _mm256_storeu_ps(green.add(i), g_out);
            _mm256_storeu_ps(blue.add(i), b_out);
        }
        i += 8;
    }

    // Scalar tail.
    while i < pixels {
        // SAFETY: i < pixels and raw has pixels*3 bytes.
        let src = unsafe { raw.add(i * 3) };
        // SAFETY: writing within the f32 planes.
        unsafe {
            *red.add(i) = (*src as f32 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            *green.add(i) = (*src.add(1) as f32 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            *blue.add(i) = (*src.add(2) as f32 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }
        i += 1;
    }
}

// NEON: 16 pixels per iteration via hardware deinterleave `vld3q_u8`, tail scalar.
#[cfg(all(feature = "fast-path", target_arch = "aarch64"))]
unsafe fn normalize_rgb_to_chw_neon(raw: *const u8, pixels: usize, out: *mut f32) {
    use std::arch::aarch64::vdupq_n_f32;
    use std::arch::aarch64::vld3q_u8;

    let red = out;
    // SAFETY: caller guarantees three contiguous f32 planes of `pixels` each.
    let green = unsafe { out.add(pixels) };
    // SAFETY: same as above.
    let blue = unsafe { out.add(pixels * 2) };

    // SAFETY: NEON intrinsics are part of the aarch64 baseline.
    let (recip_std_r, recip_std_g, recip_std_b, neg_bias_r, neg_bias_g, neg_bias_b) = unsafe {
        (
            vdupq_n_f32(1.0 / IMAGENET_STD[0]),
            vdupq_n_f32(1.0 / IMAGENET_STD[1]),
            vdupq_n_f32(1.0 / IMAGENET_STD[2]),
            vdupq_n_f32(-IMAGENET_MEAN[0] / IMAGENET_STD[0]),
            vdupq_n_f32(-IMAGENET_MEAN[1] / IMAGENET_STD[1]),
            vdupq_n_f32(-IMAGENET_MEAN[2] / IMAGENET_STD[2]),
        )
    };

    let mut i = 0usize;
    while i + 16 <= pixels {
        // SAFETY: caller guarantees raw has `pixels*3` initialized bytes;
        // 16 pixels = 48 bytes, fits because i + 16 <= pixels.
        let rgb = unsafe { vld3q_u8(raw.add(i * 3)) };

        // SAFETY: `red/green/blue + i + 16` lies inside the f32 planes.
        unsafe {
            store_chunk_neon(rgb.0, red.add(i), recip_std_r, neg_bias_r);
            store_chunk_neon(rgb.1, green.add(i), recip_std_g, neg_bias_g);
            store_chunk_neon(rgb.2, blue.add(i), recip_std_b, neg_bias_b);
        }
        i += 16;
    }

    // Scalar tail.
    while i < pixels {
        // SAFETY: i < pixels and raw has pixels*3 bytes.
        let src = unsafe { raw.add(i * 3) };
        // SAFETY: writing within the f32 planes.
        unsafe {
            *red.add(i) = (*src as f32 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            *green.add(i) = (*src.add(1) as f32 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            *blue.add(i) = (*src.add(2) as f32 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }
        i += 1;
    }
}

#[cfg(all(feature = "fast-path", target_arch = "aarch64"))]
#[inline]
unsafe fn store_chunk_neon(
    src: std::arch::aarch64::uint8x16_t, dst: *mut f32, recip_std: std::arch::aarch64::float32x4_t,
    neg_bias: std::arch::aarch64::float32x4_t,
) {
    use std::arch::aarch64::vcvtq_f32_u32;
    use std::arch::aarch64::vfmaq_f32;
    use std::arch::aarch64::vget_high_u8;
    use std::arch::aarch64::vget_high_u16;
    use std::arch::aarch64::vget_low_u8;
    use std::arch::aarch64::vget_low_u16;
    use std::arch::aarch64::vmovl_u8;
    use std::arch::aarch64::vmovl_u16;
    use std::arch::aarch64::vst1q_f32;

    // SAFETY: all NEON intrinsics below are part of the aarch64 baseline and
    // operate on data passed by value; `dst..dst+16` is in-bounds per contract.
    // `vfmaq_f32(a, b, c)` = a + b*c = neg_bias + x*recip_std = (x - mean)/std.
    unsafe {
        let lo16 = vmovl_u8(vget_low_u8(src));
        let hi16 = vmovl_u8(vget_high_u8(src));

        let u32_0 = vmovl_u16(vget_low_u16(lo16));
        let u32_1 = vmovl_u16(vget_high_u16(lo16));
        let u32_2 = vmovl_u16(vget_low_u16(hi16));
        let u32_3 = vmovl_u16(vget_high_u16(hi16));

        let r0 = vfmaq_f32(neg_bias, vcvtq_f32_u32(u32_0), recip_std);
        let r1 = vfmaq_f32(neg_bias, vcvtq_f32_u32(u32_1), recip_std);
        let r2 = vfmaq_f32(neg_bias, vcvtq_f32_u32(u32_2), recip_std);
        let r3 = vfmaq_f32(neg_bias, vcvtq_f32_u32(u32_3), recip_std);

        vst1q_f32(dst, r0);
        vst1q_f32(dst.add(4), r1);
        vst1q_f32(dst.add(8), r2);
        vst1q_f32(dst.add(12), r3);
    }
}

pub(crate) fn argmax_f32(values: &[f32]) -> (usize, f32) {
    debug_assert!(!values.is_empty());

    #[cfg(all(feature = "fast-path", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is part of the aarch64 baseline and the slice is non-empty.
        unsafe { argmax_f32_neon(values) }
    }

    #[cfg(not(all(feature = "fast-path", target_arch = "aarch64")))]
    {
        #[cfg(all(feature = "fast-path", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 was detected at runtime and the slice is non-empty.
                return unsafe { argmax_f32_avx2(values) };
            }
        }

        argmax_f32_safe(values)
    }
}

#[cfg(any(test, not(all(feature = "fast-path", target_arch = "aarch64"))))]
pub(crate) fn argmax_f32_safe(values: &[f32]) -> (usize, f32) {
    debug_assert!(!values.is_empty());
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().copied().enumerate() {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    (best_index, best_value)
}

// AVX2 argmax: 8 lanes of per-position max value + index kept in registers; one
// scalar fold across the eight lanes at the end. _CMP_GT_OQ is ordered/quiet:
// returns false when either operand is NaN, matching scalar `value > best`
// semantics. Lane-internally we use strict `>`, so earliest argmax wins; the
// scalar fold preserves earliest index on ties via the `idx < best_idx` check.
#[cfg(all(feature = "fast-path", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn argmax_f32_avx2(values: &[f32]) -> (usize, f32) {
    use std::arch::x86_64::__m256i;
    use std::arch::x86_64::_CMP_GT_OQ;
    use std::arch::x86_64::_mm256_add_epi32;
    use std::arch::x86_64::_mm256_blendv_epi8;
    use std::arch::x86_64::_mm256_blendv_ps;
    use std::arch::x86_64::_mm256_castps_si256;
    use std::arch::x86_64::_mm256_cmp_ps;
    use std::arch::x86_64::_mm256_loadu_ps;
    use std::arch::x86_64::_mm256_set1_epi32;
    use std::arch::x86_64::_mm256_set1_ps;
    use std::arch::x86_64::_mm256_setr_epi32;
    use std::arch::x86_64::_mm256_setzero_si256;
    use std::arch::x86_64::_mm256_storeu_ps;
    use std::arch::x86_64::_mm256_storeu_si256;

    // SAFETY: AVX2 target feature enabled by attribute; these intrinsics build
    // register-resident vector constants from immediate operands.
    let (mut best_vec, mut idx_vec, mut pos_vec, step_vec) = unsafe {
        (
            _mm256_set1_ps(f32::NEG_INFINITY),
            _mm256_setzero_si256(),
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
            _mm256_set1_epi32(8),
        )
    };

    let mut i = 0usize;
    while i + 8 <= values.len() {
        // SAFETY: loop condition guarantees eight f32 lanes available; lane-wise
        // compare-and-blend keeps the per-lane max and its earliest index.
        unsafe {
            let v = _mm256_loadu_ps(values.as_ptr().add(i));
            let mask_ps = _mm256_cmp_ps::<_CMP_GT_OQ>(v, best_vec);
            let mask_si = _mm256_castps_si256(mask_ps);
            best_vec = _mm256_blendv_ps(best_vec, v, mask_ps);
            idx_vec = _mm256_blendv_epi8(idx_vec, pos_vec, mask_si);
            pos_vec = _mm256_add_epi32(pos_vec, step_vec);
        }
        i += 8;
    }

    let mut lane_vals = [0.0_f32; 8];
    let mut lane_idxs = [0_i32; 8];
    // SAFETY: writing 32 bytes each into 8-element local arrays.
    unsafe {
        _mm256_storeu_ps(lane_vals.as_mut_ptr(), best_vec);
        _mm256_storeu_si256(lane_idxs.as_mut_ptr() as *mut __m256i, idx_vec);
    }

    let mut best_index = lane_idxs[0] as usize;
    let mut best_value = lane_vals[0];
    for k in 1..8 {
        let idx = lane_idxs[k] as usize;
        if lane_vals[k] > best_value || (lane_vals[k] == best_value && idx < best_index) {
            best_value = lane_vals[k];
            best_index = idx;
        }
    }

    while i < values.len() {
        let value = values[i];
        if value > best_value {
            best_index = i;
            best_value = value;
        }
        i += 1;
    }
    (best_index, best_value)
}

#[cfg(all(feature = "fast-path", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn argmax_f32_avx2(values: &[f32]) -> (usize, f32) {
    use std::arch::x86::__m256i;
    use std::arch::x86::_CMP_GT_OQ;
    use std::arch::x86::_mm256_add_epi32;
    use std::arch::x86::_mm256_blendv_epi8;
    use std::arch::x86::_mm256_blendv_ps;
    use std::arch::x86::_mm256_castps_si256;
    use std::arch::x86::_mm256_cmp_ps;
    use std::arch::x86::_mm256_loadu_ps;
    use std::arch::x86::_mm256_set1_epi32;
    use std::arch::x86::_mm256_set1_ps;
    use std::arch::x86::_mm256_setr_epi32;
    use std::arch::x86::_mm256_setzero_si256;
    use std::arch::x86::_mm256_storeu_ps;
    use std::arch::x86::_mm256_storeu_si256;

    // SAFETY: same as the x86_64 sibling above.
    let (mut best_vec, mut idx_vec, mut pos_vec, step_vec) = unsafe {
        (
            _mm256_set1_ps(f32::NEG_INFINITY),
            _mm256_setzero_si256(),
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
            _mm256_set1_epi32(8),
        )
    };

    let mut i = 0usize;
    while i + 8 <= values.len() {
        // SAFETY: same as the x86_64 sibling above.
        unsafe {
            let v = _mm256_loadu_ps(values.as_ptr().add(i));
            let mask_ps = _mm256_cmp_ps::<_CMP_GT_OQ>(v, best_vec);
            let mask_si = _mm256_castps_si256(mask_ps);
            best_vec = _mm256_blendv_ps(best_vec, v, mask_ps);
            idx_vec = _mm256_blendv_epi8(idx_vec, pos_vec, mask_si);
            pos_vec = _mm256_add_epi32(pos_vec, step_vec);
        }
        i += 8;
    }

    let mut lane_vals = [0.0_f32; 8];
    let mut lane_idxs = [0_i32; 8];
    // SAFETY: writing 32 bytes each into 8-element local arrays.
    unsafe {
        _mm256_storeu_ps(lane_vals.as_mut_ptr(), best_vec);
        _mm256_storeu_si256(lane_idxs.as_mut_ptr() as *mut __m256i, idx_vec);
    }

    let mut best_index = lane_idxs[0] as usize;
    let mut best_value = lane_vals[0];
    for k in 1..8 {
        let idx = lane_idxs[k] as usize;
        if lane_vals[k] > best_value || (lane_vals[k] == best_value && idx < best_index) {
            best_value = lane_vals[k];
            best_index = idx;
        }
    }

    while i < values.len() {
        let value = values[i];
        if value > best_value {
            best_index = i;
            best_value = value;
        }
        i += 1;
    }
    (best_index, best_value)
}

// NEON argmax: 4 lanes of per-position max value + index kept in registers; one
// scalar fold across the four lanes at the end. No round-trip through stack.
// `vcgtq_f32` returns false for NaN, so NaN values cannot displace a finite max
// — matching scalar `value > best_value` semantics. Equal values do not displace
// the existing one, so the earliest index wins (same as scalar).
#[cfg(all(feature = "fast-path", target_arch = "aarch64"))]
unsafe fn argmax_f32_neon(values: &[f32]) -> (usize, f32) {
    use std::arch::aarch64::vaddq_u32;
    use std::arch::aarch64::vbslq_f32;
    use std::arch::aarch64::vbslq_u32;
    use std::arch::aarch64::vcgtq_f32;
    use std::arch::aarch64::vdupq_n_f32;
    use std::arch::aarch64::vdupq_n_u32;
    use std::arch::aarch64::vgetq_lane_f32;
    use std::arch::aarch64::vgetq_lane_u32;
    use std::arch::aarch64::vld1q_f32;
    use std::arch::aarch64::vld1q_u32;

    let mut i = 0usize;
    // SAFETY: NEON intrinsics are part of the aarch64 baseline; no unaligned or
    // out-of-bounds memory access is performed by these scalar broadcasts.
    let (mut best_vec, mut idx_vec, mut pos_vec, step_vec) = unsafe {
        (vdupq_n_f32(f32::NEG_INFINITY), vdupq_n_u32(0), vld1q_u32([0u32, 1, 2, 3].as_ptr()), vdupq_n_u32(4))
    };

    while i + 4 <= values.len() {
        // SAFETY: loop condition guarantees four f32 lanes are available.
        let v = unsafe { vld1q_f32(values.as_ptr().add(i)) };
        // SAFETY: register-only operations.
        unsafe {
            let mask = vcgtq_f32(v, best_vec);
            best_vec = vbslq_f32(mask, v, best_vec);
            idx_vec = vbslq_u32(mask, pos_vec, idx_vec);
            pos_vec = vaddq_u32(pos_vec, step_vec);
        }
        i += 4;
    }

    // SAFETY: lane extractions are register operations.
    let (lane_vals, lane_idxs) = unsafe {
        (
            [
                vgetq_lane_f32::<0>(best_vec),
                vgetq_lane_f32::<1>(best_vec),
                vgetq_lane_f32::<2>(best_vec),
                vgetq_lane_f32::<3>(best_vec),
            ],
            [
                vgetq_lane_u32::<0>(idx_vec),
                vgetq_lane_u32::<1>(idx_vec),
                vgetq_lane_u32::<2>(idx_vec),
                vgetq_lane_u32::<3>(idx_vec),
            ],
        )
    };

    let mut best_index = lane_idxs[0] as usize;
    let mut best_value = lane_vals[0];
    for k in 1..4 {
        if lane_vals[k] > best_value || (lane_vals[k] == best_value && (lane_idxs[k] as usize) < best_index) {
            best_value = lane_vals[k];
            best_index = lane_idxs[k] as usize;
        }
    }

    while i < values.len() {
        let value = values[i];
        if value > best_value {
            best_index = i;
            best_value = value;
        }
        i += 1;
    }
    (best_index, best_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_fast_path_matches_safe_path() {
        let raw = [10_u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let mut safe = vec![0.0; 12];
        let mut fast = Vec::new();
        normalize_rgb_to_chw_safe(&raw, 2, 2, &mut safe);
        normalize_rgb_to_chw(&raw, 2, 2, &mut fast);
        assert_eq!(safe, fast);
    }

    // Exercise the SIMD main loop (NEON: ≥ 16 pixels per batch; AVX2: ≥ 8).
    // A 64×64 image guarantees both architectures hit the vector path with a
    // non-trivial tail. SIMD-FMA paths fuse `(x - mean)/std` as `x*(1/std)
    // + (-mean/std)`, which differs from the scalar two-step expression by a
    // handful of ulps, so this test uses absolute tolerance rather than `==`.
    #[test]
    fn normalize_fast_path_matches_safe_path_large() {
        let width = 64;
        let height = 64;
        let mut raw = vec![0_u8; width * height * 3];
        for (idx, byte) in raw.iter_mut().enumerate() {
            *byte = (idx % 251) as u8; // covers all u8 values across the buffer
        }
        let mut safe = vec![0.0; raw.len()];
        let mut fast = Vec::new();
        normalize_rgb_to_chw_safe(&raw, width, height, &mut safe);
        normalize_rgb_to_chw(&raw, width, height, &mut fast);
        assert_eq!(safe.len(), fast.len());
        for (i, (s, f)) in safe.iter().zip(fast.iter()).enumerate() {
            assert!((s - f).abs() <= 1e-5, "mismatch at {}: safe={} fast={}", i, s, f);
        }
    }

    // Deterministic xorshift fuzz: 32 rounds × varied sizes (covering scalar
    // tail, exact SIMD multiples, and odd remainders).
    #[test]
    fn normalize_fast_path_fuzz_matches_safe_path() {
        let mut state: u64 = 0x000C_0FFE_EBAD_F00D;
        let next = |s: &mut u64| -> u8 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s & 0xFF) as u8
        };
        for round in 0..32 {
            let pixels_count = 1 + (round * 7) % 257; // 1..=257, hits SIMD + tail
            let width = pixels_count;
            let height = 1;
            let mut raw = vec![0_u8; pixels_count * 3];
            for byte in raw.iter_mut() {
                *byte = next(&mut state);
            }
            let mut safe = vec![0.0; raw.len()];
            let mut fast = Vec::new();
            normalize_rgb_to_chw_safe(&raw, width, height, &mut safe);
            normalize_rgb_to_chw(&raw, width, height, &mut fast);
            for (i, (s, f)) in safe.iter().zip(fast.iter()).enumerate() {
                assert!(
                    (s - f).abs() <= 1e-5,
                    "round {} pixels={} idx {}: safe={} fast={}",
                    round,
                    pixels_count,
                    i,
                    s,
                    f
                );
            }
        }
    }

    #[test]
    fn argmax_fast_path_matches_safe_path() {
        let values = [-10.0, 1.0, 8.0, 3.0, 8.5, 2.0, 0.0, 9.0, 7.0, 8.0];
        assert_eq!(argmax_f32_safe(&values), argmax_f32(&values));
    }

    // Exercise SIMD main loop (NEON 4-lane, AVX2 8-lane), tail, and edge sizes.
    #[test]
    fn argmax_fast_path_handles_varied_lengths() {
        for &n in &[1_usize, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1000] {
            let mut values = vec![0.0_f32; n];
            for (i, v) in values.iter_mut().enumerate() {
                *v = ((i * 31) % 7) as f32 - 3.0;
            }
            assert_eq!(argmax_f32_safe(&values), argmax_f32(&values), "n={}", n);
        }
    }

    // Earliest-argmax tie-break must match scalar: when several positions share
    // the same max value, the lowest index wins (both within and across SIMD lanes).
    #[test]
    fn argmax_fast_path_returns_earliest_on_ties() {
        let cases: &[&[f32]] = &[
            &[5.0, 5.0, 5.0, 5.0],                                         // 4-lane all-tie
            &[1.0, 5.0, 5.0, 1.0, 1.0, 5.0, 1.0, 1.0],                     // 8-lane, multiple lane ties
            &[3.0, 5.0, 5.0, 3.0, 5.0, 3.0, 3.0, 3.0],                     // AVX2 lane updates
            &[f32::NEG_INFINITY; 16],                                      // all -inf
            &[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], // negative ties
        ];
        for (case_idx, values) in cases.iter().enumerate() {
            assert_eq!(argmax_f32_safe(values), argmax_f32(values), "case {}", case_idx);
        }
    }

    // NaN must not displace finite values (matches scalar `>`).
    #[test]
    fn argmax_fast_path_ignores_nan() {
        let nan = f32::NAN;
        let values = [1.0, nan, 2.0, nan, 3.0, nan, 0.5, nan, 4.0, nan, nan, nan];
        let (idx_safe, val_safe) = argmax_f32_safe(&values);
        let (idx_fast, val_fast) = argmax_f32(&values);
        assert_eq!(idx_safe, idx_fast);
        assert_eq!(val_safe.to_bits(), val_fast.to_bits());
        assert_eq!(idx_safe, 8);
    }

    // Deterministic xorshift fuzz over varied lengths.
    #[test]
    fn argmax_fast_path_fuzz_matches_safe_path() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let next_f32 = |s: &mut u64| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            // Map low 16 bits into a signed range; coarse enough to produce ties.
            (((*s & 0x1FF) as i32) - 256) as f32
        };
        for round in 0..32 {
            let n = 1 + (round * 11) % 257;
            let mut values = vec![0.0_f32; n];
            for v in values.iter_mut() {
                *v = next_f32(&mut state);
            }
            assert_eq!(argmax_f32_safe(&values), argmax_f32(&values), "round {} n={}", round, n);
        }
    }
}
