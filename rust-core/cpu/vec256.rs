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
    pub unsafe fn horizontal_sum(a: __m256) -> f32 {
        unsafe {
            let high = _mm256_extractf128_ps(a, 1);
            let low = _mm256_castps256_ps128(a);
            let sum128 = _mm_add_ps(low, high);
            let sh1 = _mm_movehl_ps(sum128, sum128);
            let sum64 = _mm_add_ps(sum128, sh1);
            let sh2 = _mm_shuffle_ps(sum64, sum64, 1);
            let sum32 = _mm_add_ss(sum64, sh2);
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
            let mut _acc0 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut _acc1 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut _acc2 = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut _acc3 = _mm256_set1_ps(f32::NEG_INFINITY);

            while i + 32 <= _len {
                let _a1 = load_vec256(_a, i);
                let _a2 = load_vec256(_a, i + 8);
                let _a3 = load_vec256(_a, i + 16);
                let _a4 = load_vec256(_a, i + 24);

                _acc0 = _mm256_max_ps(_acc0, _a1);
                _acc1 = _mm256_max_ps(_acc1, _a2);
                _acc2 = _mm256_max_ps(_acc2, _a3);
                _acc3 = _mm256_max_ps(_acc3, _a4);

                i += 32;
            }

            while i + 8 <= _len {
                let _a1 = load_vec256(_a, i);
                _acc0 = _mm256_max_ps(_acc0, _a1);
                i += 8;
            }

            _acc0 = _mm256_max_ps(_acc0, _acc1);
            _acc2 = _mm256_max_ps(_acc2, _acc3);
            _acc0 = _mm256_max_ps(_acc0, _acc2);
            let mut result = horizontal_max(_acc0);

            for idx in i.._len {
                result = f32::max(result, _a[idx]);
            }

            result
        }
    }

    /// Colbert Style fused dot-product and max reduction for 128-dimensional vectors.
    /// 
    /// # Arguments
    /// 
    /// * `q` - The query vector.
    /// * `d` - The document vector.
    /// * `q_len` - The length of the query vector.
    /// * `d_len` - The length of the document vector.
    ///
    /// # Returns
    /// 
    /// The fused dot-product and max reduction for 128-dimensional vectors.
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub unsafe fn fused_dot_max_dim128_avx2(
        q: &[f32],
        d: &[f32],
        q_len: usize,
        d_len: usize,
    ) -> f32 {

        // have done some tested and based on those choosing 4 * 2(more info in next commit)
        
        if q_len == 0 || d_len == 0 {
            return 0.0;
        }

        let q_ptr = q.as_ptr();
        let d_ptr = d.as_ptr();
        let mut total_score = 0.0f32;

        unsafe {
            let mut qi = 0;
            while qi + 4 <= q_len {
                // accumulators
                let q0 = q_ptr.add(qi * 128);
                let q1 = q_ptr.add((qi + 1) * 128);
                let q2 = q_ptr.add((qi + 2) * 128);
                let q3 = q_ptr.add((qi + 3) * 128);

                let mut m0 = f32::NEG_INFINITY;
                let mut m1 = f32::NEG_INFINITY;
                let mut m2 = f32::NEG_INFINITY;
                let mut m3 = f32::NEG_INFINITY;

                for di in 0..d_len {
                    // pre-fetch
                    let curr_d_ptr = d_ptr.add(di * 128);

                    let mut acc0_a = _mm256_setzero_ps();
                    let mut acc1_a = _mm256_setzero_ps();
                    let mut acc2_a = _mm256_setzero_ps();
                    let mut acc3_a = _mm256_setzero_ps();

                    let mut acc0_b = _mm256_setzero_ps();
                    let mut acc1_b = _mm256_setzero_ps();
                    let mut acc2_b = _mm256_setzero_ps();
                    let mut acc3_b = _mm256_setzero_ps();

                    // moves to 16 cuz 128 / 8 = 16 yeah
                    for k in (0..16).step_by(2) {
                        let va = _mm256_loadu_ps(curr_d_ptr.add(k * 8));
                        let vb = _mm256_loadu_ps(curr_d_ptr.add((k + 1) * 8));

                        acc0_a = _mm256_fmadd_ps(va, _mm256_loadu_ps(q0.add(k * 8)), acc0_a);
                        acc0_b = _mm256_fmadd_ps(vb, _mm256_loadu_ps(q0.add(k * 8 + 8)), acc0_b);

                        acc1_a = _mm256_fmadd_ps(va, _mm256_loadu_ps(q1.add(k * 8)), acc1_a);
                        acc1_b = _mm256_fmadd_ps(vb, _mm256_loadu_ps(q1.add(k * 8 + 8)), acc1_b);

                        acc2_a = _mm256_fmadd_ps(va, _mm256_loadu_ps(q2.add(k * 8)), acc2_a);
                        acc2_b = _mm256_fmadd_ps(vb, _mm256_loadu_ps(q2.add(k * 8 + 8)), acc2_b);

                        acc3_a = _mm256_fmadd_ps(va, _mm256_loadu_ps(q3.add(k * 8)), acc3_a);
                        acc3_b = _mm256_fmadd_ps(vb, _mm256_loadu_ps(q3.add(k * 8 + 8)), acc3_b);
                    }

                    let acc0 = _mm256_add_ps(acc0_a, acc0_b);
                    let acc1 = _mm256_add_ps(acc1_a, acc1_b);
                    let acc2 = _mm256_add_ps(acc2_a, acc2_b);
                    let acc3 = _mm256_add_ps(acc3_a, acc3_b);

                    let dot0 = horizontal_sum(acc0);
                    let dot1 = horizontal_sum(acc1);
                    let dot2 = horizontal_sum(acc2);
                    let dot3 = horizontal_sum(acc3);

                    if dot0 > m0 {
                        m0 = dot0;
                    }
                    if dot1 > m1 {
                        m1 = dot1;
                    }
                    if dot2 > m2 {
                        m2 = dot2;
                    }
                    if dot3 > m3 {
                        m3 = dot3;
                    }
                }

                total_score += m0 + m1 + m2 + m3;
                qi += 4;
            }


            // left over
            while qi < q_len {
                let curr_q = q_ptr.add(qi * 128);
                let mut max_val = f32::NEG_INFINITY;

                for di in 0..d_len {
                    let curr_d_ptr = d_ptr.add(di * 128);
                    let mut acc = _mm256_setzero_ps();

                    for k in 0..16 {
                        let vd = _mm256_loadu_ps(curr_d_ptr.add(k * 8));
                        let vq = _mm256_loadu_ps(curr_q.add(k * 8));
                        acc = _mm256_fmadd_ps(vd, vq, acc);
                    }

                    let dot = horizontal_sum(acc);
                    if dot > max_val {
                        max_val = dot;
                    }
                }

                total_score += max_val;
                qi += 1;
            }
        }

        total_score
    }

    //TODO
    //This isn't optimzied at all, and haven't planning on doing so as the aim is make a colbert
    //style typ shit so this is the best I can do
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub unsafe fn fused_dot_max_generic_avx2(
        q: &[f32],
        d: &[f32],
        q_len: usize,
        d_len: usize,
        dim: usize,
    ) -> f32 {
        if q_len == 0 || d_len == 0 || dim == 0 {
            return 0.0;
        }

        let mut max_scores_buf = [f32::NEG_INFINITY; 128];
        let mut max_scores_vec;
        let max_scores: &mut [f32] = if q_len <= 128 {
            &mut max_scores_buf[..q_len]
        } else {
            max_scores_vec = vec![f32::NEG_INFINITY; q_len];
            &mut max_scores_vec[..]
        };

        let q_ptr = q.as_ptr();
        let d_ptr = d.as_ptr();
        let num_vecs = dim / 8;
        let rem = dim % 8;

        unsafe {
            for di in 0..d_len {
                let curr_d_ptr = d_ptr.add(di * dim);

                for qi in 0..q_len {
                    let curr_q_ptr = q_ptr.add(qi * dim);
                    let mut acc = _mm256_setzero_ps();

                    for k in 0..num_vecs {
                        let vd = _mm256_loadu_ps(curr_d_ptr.add(k * 8));
                        let vq = _mm256_loadu_ps(curr_q_ptr.add(k * 8));
                        acc = _mm256_fmadd_ps(vd, vq, acc);
                    }

                    let mut dot = horizontal_sum(acc);

                    if rem > 0 {
                        let rem_offset = num_vecs * 8;
                        for r in 0..rem {
                            dot +=
                                *curr_d_ptr.add(rem_offset + r) * *curr_q_ptr.add(rem_offset + r);
                        }
                    }

                    if dot > max_scores[qi] {
                        max_scores[qi] = dot;
                    }
                }
            }
        }

        let mut total = 0.0f32;
        for &s in max_scores.iter() {
            total += s;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, thread_rng};
    

    pub fn naive_maxsim_dim128(q: &[f32], d: &[f32], q_len: usize, d_len: usize) -> f32 {
        if q_len == 0 || d_len == 0 {
            return 0.0;
        }

        let mut total_score = 0.0;
        for qi in 0..q_len {
            let mut max_doc_score = f32::NEG_INFINITY;
            for di in 0..d_len {
                let mut dot = 0.0;
                // Standard linear dot product
                for k in 0..128 {
                    dot += q[qi * 128 + k] * d[di * 128 + k];
                }
                if dot > max_doc_score {
                    max_doc_score = dot;
                }
            }
            total_score += max_doc_score;
        }
        total_score
    }

    fn max_scalar(a: &[f32]) -> f32 {
        a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    fn generate_data(len: usize) -> Vec<f32> {
        let mut rng = thread_rng();
        (0..(len * 128)).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    fn assert_approx_eq(a: f32, b: f32) {
        let epsilon = 1e-4; // Margin of error for floating point reordering
        assert!((a - b).abs() < epsilon, "Mismatch: SIMD {} vs Naive {}", a, b);
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
    fn test_vec256_empty_slice() {
        assert_max_eq(&[]);
    }

    #[test]
    fn test_vec256_single_element() {
        assert_max_eq(&[42.0]);
        assert_max_eq(&[-3.14]);
    }

    #[test]
    fn test_vec256_two_elements() {
        assert_max_eq(&[1.0, 2.0]);
        assert_max_eq(&[2.0, 1.0]);
    }

    #[test]
    fn test_vec256_all_same() {
        assert_max_eq(&[5.0; 100]);
    }

    #[test]
    fn test_vec256_strictly_increasing() {
        let v: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert_max_eq(&v);
    }

    #[test]
    fn test_vec256_strictly_decreasing() {
        let v: Vec<f32> = (0..100).map(|i| 99.0 - i as f32).collect();
        assert_max_eq(&v);
    }

    #[test]
    fn test_vec256_negatives_and_positives() {
        assert_max_eq(&[-1.0, -5.0, -2.0, 0.0, -10.0]);
        assert_max_eq(&[-100.0, 100.0, -50.0, 50.0]);
    }

    #[test]
    fn test_vec256_around_alignment_boundaries() {
        for len in 1..=40 {
            let v: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();
            assert_max_eq(&v);
        }
    }

    #[test]
    fn test_vec256_contains_neg_inf() {
        assert_max_eq(&[f32::NEG_INFINITY, 1.0, 2.0]);
        assert_max_eq(&[f32::NEG_INFINITY; 50]);
    }

    #[test]
    fn test_vec256_contains_infinity() {
        assert_max_eq(&[1.0, f32::INFINITY, 2.0]);
    }

    #[test]
    fn test_vec256_contains_nan() {
        let expected = max_scalar(&[f32::NAN, 1.0, 2.0]);
        let got = simd::max_avx2(&[f32::NAN, 1.0, 2.0]);
        assert_eq!(got.is_nan(), expected.is_nan(), "NaN handling mismatch");
    }

    fn reference_maxsim(q: &[f32], d: &[f32], q_len: usize, d_len: usize, dim: usize) -> f32 {
        let mut total = 0.0f32;
        for qi in 0..q_len {
            let mut max_dot = f32::NEG_INFINITY;
            for di in 0..d_len {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += q[qi * dim + k] * d[di * dim + k];
                }
                if dot > max_dot {
                    max_dot = dot;
                }
            }
            if d_len > 0 {
                total += max_dot;
            }
        }
        total
    }

    #[test]
    fn test_fused_dot_max_dim128() {
        let q_len = 10;
        let d_len = 25;
        let dim = 128;
        let q: Vec<f32> = (0..q_len * dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();
        let d: Vec<f32> = (0..d_len * dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.1)
            .collect();

        let expected = reference_maxsim(&q, &d, q_len, d_len, dim);
        let got_128 = unsafe { simd::fused_dot_max_dim128_avx2(&q, &d, q_len, d_len) };
        let got_generic = unsafe { simd::fused_dot_max_generic_avx2(&q, &d, q_len, d_len, dim) };

        assert!(
            (got_128 - expected).abs() < 1e-4,
            "got_128: {}, expected: {}",
            got_128,
            expected
        );
        assert!(
            (got_generic - expected).abs() < 1e-4,
            "got_generic: {}, expected: {}",
            got_generic,
            expected
        );
    }

    #[test]
    fn test_simd_correctness_standard_batch() {
        let q_len = 32; // Divides perfectly by 4
        let d_len = 100;
        let q = generate_data(q_len);
        let d = generate_data(d_len);

        let naive_score = naive_maxsim_dim128(&q, &d, q_len, d_len);
        let simd_score = unsafe { simd::fused_dot_max_dim128_avx2(&q, &d, q_len, d_len) };
        assert_approx_eq(simd_score, naive_score);
    }

    #[test]
    fn test_simd_correctness_leftovers() {
        let q_len = 5; // Forces the "leftover" while loop to run (4 + 1)
        let d_len = 10;
        let q = generate_data(q_len);
        let d = generate_data(d_len);

        let naive_score = naive_maxsim_dim128(&q, &d, q_len, d_len);
        let simd_score = unsafe { simd::fused_dot_max_dim128_avx2(&q, &d, q_len, d_len) };
        assert_approx_eq(simd_score, naive_score);
    }

    #[test]
    fn test_simd_correctness_empty() {
        let q: Vec<f32> = vec![];
        let d: Vec<f32> = vec![];
        let naive_score = naive_maxsim_dim128(&q, &d, 0, 0);
        let simd_score = unsafe { simd::fused_dot_max_dim128_avx2(&q, &d, 0, 0) };
        assert_eq!(simd_score, 0.0);
        assert_eq!(naive_score, 0.0);
    }

    #[test]
    fn test_fused_dot_max_arbitrary_dim() {
        for dim in [1, 7, 8, 15, 16, 64, 130] {
            let q_len = 5;
            let d_len = 12;
            let q: Vec<f32> = (0..q_len * dim)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
                .collect();
            let d: Vec<f32> = (0..d_len * dim)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
                .collect();

            let expected = reference_maxsim(&q, &d, q_len, d_len, dim);
            let got = unsafe { simd::fused_dot_max_generic_avx2(&q, &d, q_len, d_len, dim) };
            assert!(
                (got - expected).abs() < 1e-4,
                "dim: {}, got: {}, expected: {}",
                dim,
                got,
                expected
            );
        }
    }
}
