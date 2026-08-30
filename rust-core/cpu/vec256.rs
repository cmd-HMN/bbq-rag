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
    fn load_vec256(data: &[f32], offset: usize) -> __m256 {
        unsafe { _mm256_loadu_ps(data.as_ptr().add(offset)) }
    }

    #[inline(always)]
    unsafe fn horizontal_sum(a: __m256) -> f32 {
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

    fn horizontal_max(a: __m256) -> f32 {
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

    #[inline(always)]
    fn hs_4x(a0: __m256, a1: __m256, a2: __m256, a3: __m256) -> (f32, f32, f32, f32) {
        unsafe {
            let l0 = _mm256_castps256_ps128(a0);
            let h0 = _mm256_extractf128_ps(a0, 1);
            let s0 = _mm_add_ps(l0, h0);

            let l1 = _mm256_castps256_ps128(a1);
            let h1 = _mm256_extractf128_ps(a1, 1);
            let s1 = _mm_add_ps(l1, h1);

            let l2 = _mm256_castps256_ps128(a2);
            let h2 = _mm256_extractf128_ps(a2, 1);
            let s2 = _mm_add_ps(l2, h2);

            let l3 = _mm256_castps256_ps128(a3);
            let h3 = _mm256_extractf128_ps(a3, 1);
            let s3 = _mm_add_ps(l3, h3);

            let h01 = _mm_hadd_ps(s0, s1);
            let h23 = _mm_hadd_ps(s2, s3);

            let h0123 = _mm_hadd_ps(h01, h23);

            let mut output = [f32::NEG_INFINITY; 4];
            _mm_storeu_ps(output.as_mut_ptr(), h0123);
            (output[0], output[1], output[2], output[3])
        }
    }

    //TODO
    //Change the desing from AoS to SoA
    //Furhter more cache tiling
    /// Kernel for fused_dot_max_dim128_avx2
    /// Size 4 * 2
    /// Description: This is the kernel for the fused dot-product and max reduction for 128-dimensional vectors.
    ///
    /// The kernel 6 * 16 is not good for this case
    /// Left overs calculations + a lot of if else checking + horizontal sum problem(cus of 6 way)
    /// Haven't test the 6 * 16 btw, ;) so maybe it will be good(betting everything on a lousy
    /// asumption, further this [6*16 or 6*2] can be better like bud it uses in the gotoblass)
    ///
    ///
    /// Tried using the kernel without max, the performace regressed so added support with max
    macro_rules! kernel_4x2_dim128_with_max_handling {
        ($q0:ident, $q1:ident, $q2:ident, $q3:ident, $d0:ident, $d1:ident, $m0:ident, $m1:ident, $m2:ident, $m3:ident) => {{
            
            let mut acc00 = _mm256_setzero_ps();
            let mut acc01 = _mm256_setzero_ps();

            let mut acc10 = _mm256_setzero_ps();
            let mut acc11 = _mm256_setzero_ps();

            let mut acc20 = _mm256_setzero_ps();
            let mut acc21 = _mm256_setzero_ps();

            let mut acc30 = _mm256_setzero_ps();
            let mut acc31 = _mm256_setzero_ps();
            
            for k in 0..16 {
                let offset = k * 8;

                let vd0 = _mm256_loadu_ps($d0.add(offset)); 
                let vd1 = _mm256_loadu_ps($d1.add(offset));

                // query
                let vq0 = _mm256_loadu_ps($q0.add(offset));
                let vq1 = _mm256_loadu_ps($q1.add(offset));
                let vq2 = _mm256_loadu_ps($q2.add(offset));
                let vq3 = _mm256_loadu_ps($q3.add(offset));

                acc00 = _mm256_fmadd_ps(vd0, vq0, acc00);
                acc01 = _mm256_fmadd_ps(vd1, vq0, acc01);

                acc10 = _mm256_fmadd_ps(vd0, vq1, acc10);
                acc11 = _mm256_fmadd_ps(vd1, vq1, acc11);

                acc20 = _mm256_fmadd_ps(vd0, vq2, acc20);
                acc21 = _mm256_fmadd_ps(vd1, vq2, acc21);

                acc30 = _mm256_fmadd_ps(vd0, vq3, acc30);
                acc31 = _mm256_fmadd_ps(vd1, vq3, acc31);
            }


            $m0 = $m0.max(horizontal_sum(acc00)).max(horizontal_sum(acc01));
            $m1 = $m1.max(horizontal_sum(acc10)).max(horizontal_sum(acc11));
            $m2 = $m2.max(horizontal_sum(acc20)).max(horizontal_sum(acc21));
            $m3 = $m3.max(horizontal_sum(acc30)).max(horizontal_sum(acc31));
        }};
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
    pub unsafe fn dotmax128_f32avx2(q: &[f32], d: &[f32], q_len: usize, d_len: usize) -> f32 {
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
                    let d0 = d_ptr.add((di * 0) + 128);
                    let d1 = d_ptr.add((di * 1) + 128);

                    kernel_4x2_dim128_with_max_handling!(q0, q1, q2, q3, d0, d1, m0, m1, m2, m3);
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

    /// Dot Max function with tiled
    /// For less than 500 docs leave no space 
    pub unsafe fn dotmaxt128_f32avx2(q: &[f32], d: &[f32], q_len: usize, d_len: usize) -> f32 {
        0.0
    }

    //TODO
    //This isn't optimzied at all, and haven't planning on doing so as the aim is make a colbert
    //style typ shit so this is the best I can do
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub unsafe fn dotmaxg_f32avx2(
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

    pub unsafe fn dotmaxg_i8avx2(
        q: &[i8],
        d: &[i8],
        q_scale: &[f32],
        d_scale: &[f32],
        q_len: usize,
        d_len: usize,
    ) {
    }

    pub fn ref_maxsimd128(q: &[f32], d: &[f32], q_len: usize, d_len: usize) -> f32 {
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

    pub fn ref_maxsim(q: &[f32], d: &[f32], q_len: usize, d_len: usize, dim: usize) -> f32 {
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
}
