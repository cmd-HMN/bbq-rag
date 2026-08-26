//! Quantization
//! This file contain quantization logic only for 128 dim vectors
//! Using llama.cpp block apporch
//! I have only use i8 quantization, only on intel x86_64 arch
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
#[inline(always)]
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
    use crate::quantization::blocks::{QBlock, QData, QI8, QTYPE};
    pub use super::*;

    pub fn qf32_i8_d128(src: &[f32]) -> QBlock {
        let mut blocks = [QI8 {
            scale: 0.0, 
            data: [0; 32],
        }; QParmas::BLOCK_SIZE];

        let ptr = src.as_ptr();

        // making chunks of src
        let arr: [&[f32; QParmas::BLOCK]; QParmas::BLOCK_SIZE] = unsafe {
            [
                &*(ptr as *const [f32; 32]),
                &*(ptr.add(32) as *const [f32; 32]),
                &*(ptr.add(64) as *const [f32; 32]),
                &*(ptr.add(96) as *const [f32; 32]),
            ]
        };

        qf32_to_qi8_d128(&arr[0], &mut blocks[0]);
        qf32_to_qi8_d128(&arr[1], &mut blocks[1]);
        qf32_to_qi8_d128(&arr[2], &mut blocks[2]);
        qf32_to_qi8_d128(&arr[3], &mut blocks[3]);

        QBlock {
            qtype: QTYPE::Int8,
            qdata: QData::Int8(blocks),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::quantization::quantize::qnt::qf32_i8_d128;

    use super::*;

    fn sq32_to_sq8(src: &[f32; 32], dst: &mut [i8; 32], scale: &mut f32) {
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
            dst[i] = (src[i] * iscale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    fn sq128x32_sq8(src: &[f32; 128], dst: &mut [i8; 128]) {
        for b in 0..4 {
            let mut scale: f32 = 0.0;
            let chunk: &[f32; QParmas::BLOCK] = src[b * 32..(b + 1) * 32].try_into().unwrap();
            let mut block_dst = [0i8; QParmas::BLOCK];
            sq32_to_sq8(chunk, &mut block_dst, &mut scale);
            dst[b * 32..(b + 1) * 32].copy_from_slice(&block_dst);
        }
    }

    fn checking_logic(dst1: &[i8; 32], dst2: &[i8; 32], s1: &mut f32, s2: &mut f32) {
        // for scale
        let sdiff: bool = ((*s1 - *s2).abs()) > 1e-12;
        assert!(!sdiff, "Scale Mismatch {} {}", *s1, *s2);

        // for destination
        for i in 0..32 {
            assert_eq!(
                dst1[i], dst2[i],
                "Destination Mismatch {} {} at index {}",
                dst1[i], dst2[i], i
            );
        }
    }

    // Assumming all values are in 32 blocks
    // helper temoate for qi8 for internal funciton
    fn ht_qi8(value: &[f32; 32]) {
        let mut sdst: [i8; 32] = [0; 32];
        let mut ss: f32 = 0.0;
        //scaler part
        sq32_to_sq8(value, &mut sdst, &mut ss);

        // avx2 part
        let mut block: QI8 = QI8 {
            scale: 0.0,
            data: [0; 32],
        };
        qf32_to_qi8_d128(value, &mut block);
        checking_logic(&block.data, &sdst, &mut block.scale, &mut ss);
    }

    // helper templete for qnt function
    fn ht_q128x32_i8(value: &[f32; 128]){
        let mut sdst = [0i8; QParmas::BASE_DIMS];

        sq128x32_sq8(value, &mut sdst);
       
        let qblock = qf32_i8_d128(value);
        let qdst: [i8; QParmas::BASE_DIMS] = qblock.to_array().unwrap();

        for i in 0..QParmas::BASE_DIMS {
            assert_eq!(sdst[i], qdst[i], "Destination Mismatch {} {}", sdst[i], qdst[i]);
        }

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
    fn test_qf32_to_qi8_neg_ext() {
        let mut input = [0.0f32; 32];
        input[0] = -1.0;
        input[1] = 1.0;
        input[2] = 0.0;
        input[3] = -0.5;
        ht_qi8(&input);
    }

    #[test]
    fn test_qf32_to_qi8_zero() {
        let input = [0.0f32; 32];
        // now need to run scalar one but what can we do now
        ht_qi8(&input);
    }

    #[test]
    fn test_qf32_to_qi8_rad() {
        use rand::{Rng, thread_rng};
        let mut rng = thread_rng();
        let mut input = [0.0f32; 32];
        for _ in 0..100 {
            for i in 0..32 {
                input[i] = rng.gen_range(-1.0..1.0);
            }
            ht_qi8(&input);
        }
    }


    //======== Main Fuction ===========\\
        
    #[test]
    fn test_qf32_to_qi8_d128_correctness() {
        let input = [0.0f32; 128];
        ht_q128x32_i8(&input);
    }

    #[test]
    fn test_qf32_to_qi8_d128_rad() {
        use rand::{Rng, thread_rng};
        let mut rng = thread_rng();
        let mut input = [0.0f32; 128];
        for _ in 0..100 {
            for i in 0..128 {
                input[i] = rng.gen_range(-1.0..1.0);
            }
            ht_q128x32_i8(&input);
        }
    }

    #[test]
    fn test_qf32_to_qi8_d128_type_mismatch_error() {
        let input = [1.0f32; QParmas::BASE_DIMS];
        let qblock = qnt::qf32_i8_d128(&input);

        // Asking to a big heart
        let res: Result<[f32; QParmas::BASE_DIMS], &'static str> = qblock.to_array();

        // println!("res: {}", res.is_err());
        // Damn stone heart
        assert!(res.is_err(), "Must return Err when requesting Float32 from an Int8 QBlock!");
        assert_eq!(
            res.unwrap_err(),
            "You are initializing with wrong datatype, its not Float32"
        );
    }


    // Teseting if the scales are independent
    #[test]
    fn test_qf32_to_qi8_d128_independent_block_scales() {
        let mut input = [0.0f32; QParmas::BASE_DIMS];

        for i in 0..32 { input[i] = 100.0; }
        for i in 32..64 { input[i] = 0.01; }
        for i in 64..96 { input[i] = 0.0; }
        for i in 96..128 { input[i] = -0.01; }

        ht_q128x32_i8(&input);
    }

    #[test]
    fn test_qf32_to_qi8_d128_sequence() {
        let mut input = [0.0f32; QParmas::BASE_DIMS];
        for i in 0..QParmas::BASE_DIMS {
            input[i] = (i as f32) * 0.5;
        }
        ht_q128x32_i8(&input);
    }
}
