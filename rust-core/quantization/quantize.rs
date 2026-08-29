//! Quantization
//! This file contain quantization logic only for 128 dim vectors
//! Using llama.cpp block apporch
//! I have only use i8 quantization, only on intel x86_64 arch
use super::blocks::{QBlock, QData, QI8, QParmas, QTYPE};
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
#[target_feature(enable = "avx2")]
#[inline]
fn qf32_to_qi8_d128(src: &[f32; QParmas::BLOCK], dst: &mut QI8) {
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
        let iscale: f32 = if maximum > 1e-12 {
            127.0 / maximum
        } else {
            0.0
        };
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
        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);

        i0 = _mm256_packs_epi16(i0, i2);

        // permute
        i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

        // store
        _mm256_storeu_si256(dst.data.as_mut_ptr() as *mut __m256i, i0);
    }
}

pub mod qnt {
    use super::{QBlock, QData, QI8, QParmas, QTYPE, qf32_to_qi8_d128};

    #[inline(always)]
    pub fn qf32_i8_d128(src: &[f32]) -> QBlock {
        let mut blocks = [QI8 {
            scale: 0.0,
            data: [0; 32],
        }; QParmas::BLOCK_SIZE];

        let ptr = src.as_ptr();

        unsafe {
            // making chunks of src
            let arr: [&[f32; QParmas::BLOCK]; QParmas::BLOCK_SIZE] = [
                &*(ptr as *const [f32; 32]),
                &*(ptr.add(32) as *const [f32; 32]),
                &*(ptr.add(64) as *const [f32; 32]),
                &*(ptr.add(96) as *const [f32; 32]),
            ];

            qf32_to_qi8_d128(&arr[0], &mut blocks[0]);
            qf32_to_qi8_d128(&arr[1], &mut blocks[1]);
            qf32_to_qi8_d128(&arr[2], &mut blocks[2]);
            qf32_to_qi8_d128(&arr[3], &mut blocks[3]);
        }
        QBlock {
            qtype: QTYPE::Int8,
            qdata: QData::Int8(blocks),
        }
    }

    // moving scaler here
    pub fn sq32_to_sq8(src: &[f32; 32], dst: &mut [i8; 32], scale: &mut f32) {
        // Scalar implemeation of the quantization logic
        let mut max: f32 = 0.0;
        for i in 0..32 {
            max = max.max(src[i].abs());
        }

        //scale
        *scale = max / 127.0f32;
        //get inverse scale
        let iscale = if max > 1e-12 { 127.0 / max } else { 0.0 };
        for i in 0..32 {
            // round vs round_ties_even
            // round change like 29.5 t0 30 but round_ties_even changes to 29 and its matches our
            // simd implemeation
            // FIX: during the build failure
            // thread 'quantization::quantize::tests::test_qf32_to_qi8_d128_rad' (6782)
            // panicked at rust-core/quantization/quantize.rs:207:13:
            // assertion `left == right` failed: Destination Mismatch 49 48
            // left: 49 round was changing this to 49 instead of 48
            // right: 48
            dst[i] = (src[i] * iscale).round_ties_even().clamp(-128.0, 127.0) as i8;
        }
    }

    pub fn sq128x32_sq8(src: &[f32; 128], dst: &mut [i8; 128]) {
        for b in 0..4 {
            let mut scale: f32 = 0.0;
            let chunk: &[f32; QParmas::BLOCK] = src[b * 32..(b + 1) * 32].try_into().unwrap();
            let mut block_dst = [0i8; QParmas::BLOCK];
            sq32_to_sq8(chunk, &mut block_dst, &mut scale);
            dst[b * 32..(b + 1) * 32].copy_from_slice(&block_dst);
        }
    }

    #[inline(always)]
    pub fn qf32_i8_d128_to_array(src: &[f32; QParmas::BASE_DIMS]) -> [i8; QParmas::BASE_DIMS] {
            qf32_i8_d128(src).to_array().unwrap()
    }
}
