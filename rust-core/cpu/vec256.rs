// Copyright 2024 cmd-HMN
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


    // ---------------------------------------
    // f32 dot product functions
    //
    // # Observation
    // These aren't a good approach for matrix multiplication
    // But they are useful for vector dot product
    // 1-D dot product
    // ---------------------------------------
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

    // f32 dot product
    #[inline(always)]
    pub fn dot_f32_2acc(a: &[f32], b: &[f32]) -> f32 {
        let _len = a.len();
        let mut i = 0;
        unsafe {
            let mut _acc0 = _mm256_setzero_ps();
            let mut _acc1 = _mm256_setzero_ps();

            while i + 16 <= _len {
                // only using 2 _acc register
                let _a0 = load_vec256(a, i);
                let _b0 = load_vec256(b, i);

                let _a1 = load_vec256(a, i + 8);
                let _b1 = load_vec256(b, i + 8);

                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(_a0, _b0));
                _acc1 = _mm256_add_ps(_acc1, _mm256_mul_ps(_a1, _b1));
                i += 16;
            }

            // left overs
            while i + 8 < _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                i += 8;
            }

            let _acc = _mm256_add_ps(_acc0, _acc1);
            let mut sum = horizontal_sum(_acc);

            while i < _len {
                sum += a[i] * b[i];
                i += 1;
            }
            // return
            sum
        }
    }

    #[inline(always)]
    pub fn dot_f32_4acc(a: &[f32], b: &[f32]) -> f32 {
        let _len = a.len();
        let mut i = 0;
        unsafe {
            let mut _acc0 = _mm256_setzero_ps();
            let mut _acc1 = _mm256_setzero_ps();
            let mut _acc2 = _mm256_setzero_ps();
            let mut _acc3 = _mm256_setzero_ps();

            while i + 32 <= _len {
                // only using 4 _acc register
                let _a0 = load_vec256(a, i);
                let _a1 = load_vec256(a, i + 8);
                let _a2 = load_vec256(a, i + 16);
                let _a3 = load_vec256(a, i + 24);

                let _b0 = load_vec256(b, i);
                let _b1 = load_vec256(b, i + 8);
                let _b2 = load_vec256(b, i + 16);
                let _b3 = load_vec256(b, i + 24);

                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(_a0, _b0));
                _acc1 = _mm256_add_ps(_acc1, _mm256_mul_ps(_a1, _b1));
                _acc2 = _mm256_add_ps(_acc2, _mm256_mul_ps(_a2, _b2));
                _acc3 = _mm256_add_ps(_acc3, _mm256_mul_ps(_a3, _b3));
                i += 32;
            }

            //left overs
            while i + 16 <= _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                _acc1 = _mm256_add_ps(
                    _acc1,
                    _mm256_mul_ps(load_vec256(a, i + 8), load_vec256(b, i + 8)),
                );
                i += 16;
            }

            while i + 8 <= _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                i += 8;
            }

            let _acc = _mm256_add_ps(_acc0, _mm256_add_ps(_acc1, _mm256_add_ps(_acc2, _acc3)));
            let mut sum = horizontal_sum(_acc);

            while i < _len {
                sum += a[i] * b[i];
                i += 1;
            }
            // return
            sum
        }
    }

    #[inline(always)]
    pub fn dot_f32_6acc(a: &[f32], b: &[f32]) -> f32 {
        let _len = a.len();
        let mut i = 0;
        unsafe {
            let mut _acc0 = _mm256_setzero_ps();
            let mut _acc1 = _mm256_setzero_ps();
            let mut _acc2 = _mm256_setzero_ps();
            let mut _acc3 = _mm256_setzero_ps();
            let mut _acc4 = _mm256_setzero_ps();
            let mut _acc5 = _mm256_setzero_ps();

            while i + 48 <= _len {
                // only using 6 _acc register
                let _a0 = load_vec256(a, i);
                let _a1 = load_vec256(a, i + 8);
                let _a2 = load_vec256(a, i + 16);
                let _a3 = load_vec256(a, i + 24);
                let _a4 = load_vec256(a, i + 32);
                let _a5 = load_vec256(a, i + 40);

                let _b0 = load_vec256(b, i);
                let _b1 = load_vec256(b, i + 8);
                let _b2 = load_vec256(b, i + 16);
                let _b3 = load_vec256(b, i + 24);
                let _b4 = load_vec256(b, i + 32);
                let _b5 = load_vec256(b, i + 40);

                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(_a0, _b0));
                _acc1 = _mm256_add_ps(_acc1, _mm256_mul_ps(_a1, _b1));
                _acc2 = _mm256_add_ps(_acc2, _mm256_mul_ps(_a2, _b2));
                _acc3 = _mm256_add_ps(_acc3, _mm256_mul_ps(_a3, _b3));
                _acc4 = _mm256_add_ps(_acc4, _mm256_mul_ps(_a4, _b4));
                _acc5 = _mm256_add_ps(_acc5, _mm256_mul_ps(_a5, _b5));
                i += 48;
            }

            while i + 32 <= _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                _acc1 = _mm256_add_ps(
                    _acc1,
                    _mm256_mul_ps(load_vec256(a, i + 8), load_vec256(b, i + 8)),
                );
                _acc2 = _mm256_add_ps(
                    _acc2,
                    _mm256_mul_ps(load_vec256(a, i + 16), load_vec256(b, i + 16)),
                );
                _acc3 = _mm256_add_ps(
                    _acc3,
                    _mm256_mul_ps(load_vec256(a, i + 24), load_vec256(b, i + 24)),
                );
                i += 32;
            }

            while i + 16 <= _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                _acc1 = _mm256_add_ps(
                    _acc1,
                    _mm256_mul_ps(load_vec256(a, i + 8), load_vec256(b, i + 8)),
                );
                i += 16;
            }

            while i + 8 <= _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                i += 8;
            }

            let _acc = _mm256_add_ps(
                _acc0,
                _mm256_add_ps(
                    _acc1,
                    _mm256_add_ps(_acc2, _mm256_add_ps(_acc3, _mm256_add_ps(_acc4, _acc5))),
                ),
            );
            let mut sum = horizontal_sum(_acc);

            while i < _len {
                sum += a[i] * b[i];
                i += 1;
            }
            // return
            sum
        }
    }

    #[inline(always)]
    pub fn dot_f32_8acc(a: &[f32], b: &[f32]) -> f32 {
        let _len = a.len();
        let mut i = 0;
        unsafe {
            let mut _acc0 = _mm256_setzero_ps();
            let mut _acc1 = _mm256_setzero_ps();
            let mut _acc2 = _mm256_setzero_ps();
            let mut _acc3 = _mm256_setzero_ps();
            let mut _acc4 = _mm256_setzero_ps();
            let mut _acc5 = _mm256_setzero_ps();
            let mut _acc6 = _mm256_setzero_ps();
            let mut _acc7 = _mm256_setzero_ps();

            while i + 64 <= _len {
                // only using 8 _acc register
                let _a0 = load_vec256(a, i);
                let _a1 = load_vec256(a, i + 8);
                let _a2 = load_vec256(a, i + 16);
                let _a3 = load_vec256(a, i + 24);
                let _a4 = load_vec256(a, i + 32);
                let _a5 = load_vec256(a, i + 40);
                let _a6 = load_vec256(a, i + 48);
                let _a7 = load_vec256(a, i + 56);

                let _b0 = load_vec256(b, i);
                let _b1 = load_vec256(b, i + 8);
                let _b2 = load_vec256(b, i + 16);
                let _b3 = load_vec256(b, i + 24);
                let _b4 = load_vec256(b, i + 32);
                let _b5 = load_vec256(b, i + 40);
                let _b6 = load_vec256(b, i + 48);
                let _b7 = load_vec256(b, i + 56);

                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(_a0, _b0));
                _acc1 = _mm256_add_ps(_acc1, _mm256_mul_ps(_a1, _b1));
                _acc2 = _mm256_add_ps(_acc2, _mm256_mul_ps(_a2, _b2));
                _acc3 = _mm256_add_ps(_acc3, _mm256_mul_ps(_a3, _b3));
                _acc4 = _mm256_add_ps(_acc4, _mm256_mul_ps(_a4, _b4));
                _acc5 = _mm256_add_ps(_acc5, _mm256_mul_ps(_a5, _b5));
                _acc6 = _mm256_add_ps(_acc6, _mm256_mul_ps(_a6, _b6));
                _acc7 = _mm256_add_ps(_acc7, _mm256_mul_ps(_a7, _b7));
                i += 64;
            }

            while i + 48 <= _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                _acc1 = _mm256_add_ps(
                    _acc1,
                    _mm256_mul_ps(load_vec256(a, i + 8), load_vec256(b, i + 8)),
                );
                _acc2 = _mm256_add_ps(
                    _acc2,
                    _mm256_mul_ps(load_vec256(a, i + 16), load_vec256(b, i + 16)),
                );
                _acc3 = _mm256_add_ps(
                    _acc3,
                    _mm256_mul_ps(load_vec256(a, i + 24), load_vec256(b, i + 24)),
                );
                _acc4 = _mm256_add_ps(
                    _acc4,
                    _mm256_mul_ps(load_vec256(a, i + 32), load_vec256(b, i + 32)),
                );
                _acc5 = _mm256_add_ps(
                    _acc5,
                    _mm256_mul_ps(load_vec256(a, i + 40), load_vec256(b, i + 40)),
                );
                i += 48;
            }

            while i + 32 <= _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                _acc1 = _mm256_add_ps(
                    _acc1,
                    _mm256_mul_ps(load_vec256(a, i + 8), load_vec256(b, i + 8)),
                );
                _acc2 = _mm256_add_ps(
                    _acc2,
                    _mm256_mul_ps(load_vec256(a, i + 16), load_vec256(b, i + 16)),
                );
                _acc3 = _mm256_add_ps(
                    _acc3,
                    _mm256_mul_ps(load_vec256(a, i + 24), load_vec256(b, i + 24)),
                );
                i += 32;
            }

            while i + 16 <= _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                _acc1 = _mm256_add_ps(
                    _acc1,
                    _mm256_mul_ps(load_vec256(a, i + 8), load_vec256(b, i + 8)),
                );
                i += 16;
            }

            while i + 8 <= _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                i += 8;
            }

            let _acc = _mm256_add_ps(
                _acc0,
                _mm256_add_ps(
                    _acc1,
                    _mm256_add_ps(
                        _acc2,
                        _mm256_add_ps(
                            _acc3,
                            _mm256_add_ps(_acc4, _mm256_add_ps(_acc5, _mm256_add_ps(_acc6, _acc7))),
                        ),
                    ),
                ),
            );
            let mut sum = horizontal_sum(_acc);

            while i < _len {
                sum += a[i] * b[i];
                i += 1;
            }
            // return
            sum
        }
    }
}

mod scalar {
    pub fn dot_f32(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(c, d)| c * d).sum()
    }
}

// TODO
// Move testing to separate module

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f32_matches_scalar_2acc() {
        let _lenghts = [1, 7, 8, 9, 15, 16, 17, 24, 100, 1005];
        for len in _lenghts {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 13.0 % 7.0) - 3.5).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 14.0 % 8.0) - 1.5).collect();

            let expected = scalar::dot_f32(&a, &b);
            let actual = simd::dot_f32_2acc(&a, &b);

            assert!((expected - actual).abs() < 1e-2);
        }
    }

    #[test]
    fn test_dot_f32_matches_scalar_4acc() {
        let _lenghts = [1, 7, 8, 9, 15, 16, 17, 24, 100, 1005];
        for len in _lenghts {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 13.0 % 7.0) - 3.5).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 14.0 % 8.0) - 1.5).collect();

            let expected = scalar::dot_f32(&a, &b);
            let actual = simd::dot_f32_4acc(&a, &b);

            assert!((expected - actual).abs() < 1e-2);
        }
    }

    #[test]
    fn test_dot_f32_matches_scalar_6acc() {
        let _lenghts = [1, 7, 8, 9, 15, 16, 17, 24, 100, 1005];
        for len in _lenghts {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 13.0 % 7.0) - 3.5).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 14.0 % 8.0) - 1.5).collect();

            let expected = scalar::dot_f32(&a, &b);
            let actual = simd::dot_f32_6acc(&a, &b);

            assert!((expected - actual).abs() < 1e-2);
        }
    }

    #[test]
    fn test_dot_f32_matches_scalar_8acc() {
        let _lenghts = [1, 7, 8, 9, 15, 16, 17, 24, 100, 1005];
        for len in _lenghts {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 13.0 % 7.0) - 3.5).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 14.0 % 8.0) - 1.5).collect();

            let expected = scalar::dot_f32(&a, &b);
            let actual = simd::dot_f32_8acc(&a, &b);

            assert!((expected - actual).abs() < 1e-2);
        }
    }

    // testing this is an absurb idea but here am i
    #[test]
    fn test_horizontal_sum() {
        if is_x86_feature_detected!("avx2") {
            use std::arch::x86_64::_mm256_loadu_ps;
            unsafe {
                let data: [f32; 8] = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
                let vec = _mm256_loadu_ps(data.as_ptr());

                let expected: f32 = data.iter().sum();
                let actual = simd::horizontal_sum(vec);

                assert!((expected - actual).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_mkl_sgemm_matches_dot_f32_8acc() {
        use crate::blas::mkl_blas;

        let dim = 16;
        let m = 3;
        let n = 2;

        let a: Vec<f32> = (0..m * dim)
            .map(|i| (i as f32 * 13.0 % 7.0) - 3.5)
            .collect();
        let b: Vec<f32> = (0..n * dim)
            .map(|i| (i as f32 * 14.0 % 8.0) - 1.5)
            .collect();

        let mut c = vec![0.0f32; m * n];
        unsafe {
            mkl_blas::sgemm(
                b'T', b'N', m as i32, n as i32, dim as i32, 1.0, &a, dim as i32, &b, dim as i32,
                0.0, &mut c, m as i32,
            );
        }

        for j in 0..n {
            for i in 0..m {
                let a_vec = &a[i * dim..i * dim + dim];
                let b_vec = &b[j * dim..j * dim + dim];
                let expected = simd::dot_f32_8acc(a_vec, b_vec);
                let actual = c[j * m + i];
                assert!(
                    (expected - actual).abs() < 1e-2,
                    "mismatch at ({i},{j}): {expected} vs {actual}"
                );
            }
        }
    }

    #[test]
    fn test_load_vec256() {
        if is_x86_feature_detected!("avx2") {
            use std::arch::x86_64::_mm256_storeu_ps;
            unsafe {
                let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
                let vec = simd::load_vec256(&data, 0);

                let mut out = [0.0; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), vec);

                for i in 0..8 {
                    assert_eq!(data[i], out[i]);
                }
            }
        }
    }
}
