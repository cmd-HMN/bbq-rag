//! Quantization
//! This file contain quantization logic only for 128 dim vectors
//! Using llama.cpp block apporch

use super::blocks::{QI8, QParmas};
use core::arch::x86_64::*;

/// qf32_to_qi8
///
/// # Arguments
///
/// * `values` - Vector of f32 values
///
/// # Returns
///
/// * `QData`
///
///
/// # Note
/// FBGEMM (1 float per iteration)
/// Other apparch was used in llama.cpp has 4 floats per iteration 
/// As the quantization is done for 128 dim vectors so and 
/// 128 / 32 = 4 blocks and it is proper multiple of 32 so llama.cpp apporch is used here.
#[cfg(target_feature = "avx2")]
#[inline(never)]
fn qf32_to_qi8(src: &[f32; QParmas::BLOCK], dst: &mut QI8) {
    let ptr = src.as_ptr();
    unsafe {
        let mut v0 = _mm256_loadu_ps(ptr);
        let mut v1 = _mm256_loadu_ps(ptr.add(8));
        let mut v2 = _mm256_loadu_ps(ptr.add(16));
        let mut v3 = _mm256_loadu_ps(ptr.add(24));

        let sbit: __m256 = _mm256_set1_ps(-0.0f32);

        // computing maximum
        let mut mx: __m256 = _mm256_andnot_ps(sbit, v0);
        mx = _mm256_max_ps(mx, _mm256_andnot_ps(sbit, v1));
        mx = _mm256_max_ps(mx, _mm256_andnot_ps(sbit, v2));
        mx = _mm256_max_ps(mx, _mm256_andnot_ps(sbit, v3));

        let mut maxv: __m128 = _mm_max_ps(_mm256_extractf128_ps(mx, 1), _mm256_castps256_ps128(mx));
        maxv = _mm_max_ps(maxv, _mm_movehl_ps(maxv, maxv));
        maxv = _mm_max_ss(maxv, _mm_movehdup_ps(maxv));

        let maximum: f32 = _mm_cvtss_f32(maxv);

        //scaling
        let scale: f32 = maximum / 127.0f32;
        dst.scale = scale;
        let iscale: f32 = if maximum > 1e-12 { 127.0 / maximum } else { 0.0 };
        let mul: __m256 = _mm256_set1_ps(iscale);

        // quantization
        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        // rounding
        v0 = _mm256_round_ps(v0, _MM_FROUND_TO_NEAREST_INT);
        v1 = _mm256_round_ps(v1, _MM_FROUND_TO_NEAREST_INT);
        v2 = _mm256_round_ps(v2, _MM_FROUND_TO_NEAREST_INT);
        v3 = _mm256_round_ps(v3, _MM_FROUND_TO_NEAREST_INT);

        // clamping
        v0 = _mm256_min_ps(v0, _mm256_set1_ps(127.0));
        v1 = _mm256_min_ps(v1, _mm256_set1_ps(127.0));
        v2 = _mm256_min_ps(v2, _mm256_set1_ps(127.0));
        v3 = _mm256_min_ps(v3, _mm256_set1_ps(127.0));

        // convert to i32
        let mut i0: __m256i = _mm256_cvtps_epi32(v0);
        let i1: __m256i = _mm256_cvtps_epi32(v1);
        let mut i2: __m256i = _mm256_cvtps_epi32(v2);
        let i3: __m256i = _mm256_cvtps_epi32(v3);

        // packing
        i0 = _mm256_packus_epi32(i0, i1);
        i2 = _mm256_packus_epi32(i2, i3);

        i0 = _mm256_packs_epi16(i0, i2);

        // permute
        i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

        // store
        _mm256_storeu_si256(dst.data.as_mut_ptr() as *mut __m256i, i0);

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq32_to_sq8(src: &[f32; 32], dst: &mut [i8; 32], scale: &mut f32) {
        // Scalar implemeation of the quantization logic
        let mut max: f32 = 0.0;
        for i in 0..32 {
            max = max.max(src[i]);
        }

        //scale
        *scale = max / 127.0f32;
        //get inverse scale
        let iscale = if max > 1e-12 { 127.0 / max } else { 0.0 };
        for i in 0..32 {
            dst[i] = (src[i] * iscale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    fn checking_logic(dst1: &[i8; 32], dst2: &[i8; 32], s1: &mut f32, s2: &mut f32) {
        // for scale
        let sdiff: bool = ((*s1 - *s2).abs()) > 1e-12;
        assert!(!sdiff, "Scale Mismatch {} {}", *s1, *s2);

        // for destination
        for i in 0..32 {
            assert_eq!(dst1[i], dst2[i], "Destination Mismatch {} {} at index {}", dst1[i], dst2[i], i);
        }
    }
    
    // Assumming all values are in 32 blocks
    // helper temoate for qi8
    fn ht_qi8(value: &[f32; 32]){
        let mut sdst: [i8; 32] = [0; 32];
        let mut ss: f32 = 0.0;
        //scaler part
        sq32_to_sq8(value, &mut sdst, &mut ss);


        // avx2 part
        let mut block: QI8 = QI8 { scale: 0.0, data: [0; 32]};
        qf32_to_qi8(value, &mut block);
        checking_logic(&block.data, &sdst, &mut block.scale, &mut ss);
    }

    #[test]
    fn test_qf32_to_qi8_correctness() {
        let input: [f32; 32] = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,    
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0,
        ];

        ht_qi8(&input);
    }

    #[test]
    fn test_qf32_to_qi8_neg_ext(){
        let mut input = [0.0f32; 32];
        input[0] = -1.0;
        input[1] = 1.0;
        input[2] = 0.0;
        input[3] = -0.5;
        ht_qi8(&input);
    }
}
