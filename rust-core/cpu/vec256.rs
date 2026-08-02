// Copyright 2024 cmd-HMN
//
// This file includes some or all code from the maxsim-cpu library.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! MaxSim
//!
//! Requires either:
//! - x86_64 with AVX2

pub mod simd {
    use core::f32;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline(always)]
    pub fn load_vec256(data: &[f32], offset: usize) -> __m256 {
        unsafe { _mm256_loadu_ps(data.as_ptr().add(offset)) }
    }

    #[inline(always)]
    pub fn horizontal_sum(a: __m256) -> f32 {
        unsafe {
            let _high = _mm256_extractf128_ps(a, 1);
            let _low = _mm256_castps256_ps128(a);
            let sum128 = _mm_add_ps(_low, _high);

            let sh1 = _mm_shuffle_ps(sum128, sum128, 0b01_00_11_10);
            let sum64 = _mm_add_ps(sum128, sh1);

            let sh2 = _mm_shuffle_ps(sum64, sum64, 0b10_11_00_01);
            let sum32 = _mm_add_ps(sum64, sh2);
            _mm_cvtss_f32(sum32)
        }
    }

    pub fn horizontal_max(a: __m256) -> f32 {
        unsafe {
            let _high = _mm256_extractf128_ps(a, 1);
            let _low = _mm256_castps256_ps128(a);
            let max128 = _mm_max_ps(_low, _high);

            let sh1 = _mm_shuffle_ps(max128, max128, 0b01_00_11_10);
            let max64 = _mm_max_ps(max128, sh1);

            let sh2 = _mm_shuffle_ps(max64, max64, 0b10_11_00_01);
            let max32 = _mm_max_ps(max64, sh2);
            _mm_cvtss_f32(max32)
        }
    }

    #[inline(always)]
    pub fn max_avx2(_a: &[f32]) -> f32 {
        let mut _len = _a.len();
        let mut i = 0;

        if _len < 8 {
            return _a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }

        // acc
        unsafe {
            // using same apprach as maxsim-cpu
            let mut _acc0 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut _acc1 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut _acc2 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut _acc3 = _mm256_set1_ps(f32::NEG_INFINITY);

            while i + 32 <= _len {
                _mm_prefetch(_a.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);

                let _a1 = load_vec256(_a, i);
                let _a2 = load_vec256(_a, i + 8);
                let _a3 = load_vec256(_a, i + 16);
                let _a4 = load_vec256(_a, i + 24);

                _acc0 = _mm256_max_ps(_acc0, _a1);
                _acc1 = _mm256_max_ps(_acc1, _a2);
                _acc2 = _mm256_max_ps(_acc2, _a3);
                _acc3 = _mm256_max_ps(_acc3, _a4);

                i += 32
            }

            while i + 8 <= _len {
                _mm_prefetch(_a.as_ptr().add(i + 8) as *const i8, _MM_HINT_T0);
                let _a1 = load_vec256(_a, i);
                _acc0 = _mm256_max_ps(_acc0, _a1);
                i += 8
            }

            _acc0 = _mm256_max_ps(_acc0, _acc1);
            _acc2 = _mm256_max_ps(_acc2, _acc3);
            _acc0 = _mm256_max_ps(_acc0, _acc2);
            let mut result = horizontal_max(_acc0);

            for i in i.._len {
                result = f32::max(result, _a[i]);
            }

            result
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn max_scalar(a: &[f32]) -> f32 {
    a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}


    fn assert_max_eq(input: &[f32]) {
        let expected = max_scalar(input);
        let got = simd::max_avx2(input);
        assert_eq!(
            got, expected,
            "max mismatch for input {:?}\n  expected: {}\n  got:      {}",
            input, expected, got
        );
    }

    #[test]
    fn empty_slice() {
        assert_max_eq(&[]);
    }

    #[test]
    fn single_element() {
        assert_max_eq(&[42.0]);
        assert_max_eq(&[-3.14]);
    }

    #[test]
    fn two_elements() {
        assert_max_eq(&[1.0, 2.0]);
        assert_max_eq(&[2.0, 1.0]);
    }

    #[test]
    fn all_same() {
        assert_max_eq(&[5.0; 100]);
    }

    #[test]
    fn strictly_increasing() {
        let v: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert_max_eq(&v);
    }

    #[test]
    fn strictly_decreasing() {
        let v: Vec<f32> = (0..100).map(|i| 99.0 - i as f32).collect();
        assert_max_eq(&v);
    }

    #[test]
    fn negatives_and_positives() {
        assert_max_eq(&[-1.0, -5.0, -2.0, 0.0, -10.0]);
        assert_max_eq(&[-100.0, 100.0, -50.0, 50.0]);
    }

    #[test]
    fn around_alignment_boundaries() {
        // 1..40 covers every possible misalignment mod 32
        for len in 1..=40 {
            let v: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();
            assert_max_eq(&v);
        }
    }

    // #[test]
    // fn large_random() {
    //     use rand::Rng;
    //     let mut rng = rand::thread_rng();
    //     for _ in 0..10 {
    //         let v: Vec<f32> = (0..10_000).map(|_| rng.gen::<f32>()).collect();
    //         assert_max_eq(&v);
    //     }
    // }

    #[test]
    fn contains_neg_inf() {
        assert_max_eq(&[f32::NEG_INFINITY, 1.0, 2.0]);
        assert_max_eq(&[f32::NEG_INFINITY; 50]);
    }

    #[test]
    fn contains_infinity() {
        assert_max_eq(&[1.0, f32::INFINITY, 2.0]);
    }

    #[test]
    fn contains_nan() {
        let expected = max_scalar(&[f32::NAN, 1.0, 2.0]);
        let got = simd::max_avx2(&[f32::NAN, 1.0, 2.0]);
        assert_eq!(got.is_nan(), expected.is_nan(), "NaN handling mismatch");
    }
}
