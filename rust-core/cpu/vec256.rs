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
    unsafe fn h_sum_f32(a: __m256) -> f32 {
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

    #[inline(always)]
    unsafe fn h_sum_i32(a: __m256i) -> i32 {
        unsafe {
            let hi = _mm256_extracti128_si256(a, 1);
            let lo = _mm256_castsi256_si128(a);
            let sum128 = _mm_add_epi32(lo, hi);
            let sh1 = _mm_shuffle_epi32(sum128, 0b01_00_11_10);
            let sum64 = _mm_add_epi32(sum128, sh1);
            let sh2 = _mm_shuffle_epi32(sum64, 0b00_00_00_01);
            let sum32 = _mm_add_epi32(sum64, sh2);
            _mm_cvtsi128_si32(sum32)
        }
    }

    #[inline(always)]
    unsafe fn dot32_i8(a: __m256i, b: __m256i) -> i32 {
        unsafe {
            let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
            let b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
            let prod_lo = _mm256_madd_epi16(a_lo, b_lo);

            let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
            let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));
            let prod_hi = _mm256_madd_epi16(a_hi, b_hi);

            h_sum_i32(_mm256_add_epi32(prod_lo, prod_hi))
        }
    }

    // fn h_max_f32(a: __m256) -> f32 {
    //     unsafe {
    //         let _high = _mm256_extractf128_ps(a, 1);
    //         let _low = _mm256_castps256_ps128(a);
    //         let max128 = _mm_max_ps(_low, _high);
    //
    //         let sh1 = _mm_shuffle_ps(max128, max128, 0b01_00_11_10);
    //         let max64 = _mm_max_ps(max128, sh1);
    //
    //         let sh2 = _mm_shuffle_ps(max64, max64, 0b10_11_00_01);
    //         let max32 = _mm_max_ps(max64, sh2);
    //         _mm_cvtss_f32(max32)
    //     }
    // }

    /// After all the testing this is what is choose
    macro_rules! kdot_max_f32 {
        // 4 * 2 - 128
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

            $m0 = $m0.max(h_sum_f32(acc00)).max(h_sum_f32(acc01));
            $m1 = $m1.max(h_sum_f32(acc10)).max(h_sum_f32(acc11));
            $m2 = $m2.max(h_sum_f32(acc20)).max(h_sum_f32(acc21));
            $m3 = $m3.max(h_sum_f32(acc30)).max(h_sum_f32(acc31));
        }};

        // 4 * 1 - 128
        ($q0:ident, $q1:ident, $q2:ident, $q3:ident, $d0:ident, $m0:ident, $m1:ident, $m2:ident, $m3:ident) => {{
            let mut acc00 = _mm256_setzero_ps();
            let mut acc01 = _mm256_setzero_ps();

            let mut acc10 = _mm256_setzero_ps();
            let mut acc11 = _mm256_setzero_ps();

            for k in 0..16 {
                let offset = k * 8;

                let vd0 = _mm256_loadu_ps($d0.add(offset));

                acc00 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q0.add(offset)), acc00);
                acc01 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q1.add(offset)), acc01);

                acc10 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q2.add(offset)), acc10);
                acc11 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q3.add(offset)), acc11);
            }

            $m0 = $m0.max(h_sum_f32(acc00));
            $m1 = $m1.max(h_sum_f32(acc01));
            $m2 = $m2.max(h_sum_f32(acc10));
            $m3 = $m3.max(h_sum_f32(acc11));
        }};

        // 1 * 2 - 128
        ($q0:ident, $d0:ident, $d1:ident, $m0:ident) => {{
            let mut acc00 = _mm256_setzero_ps();
            let mut acc01 = _mm256_setzero_ps();

            for k in 0..16 {
                let offset = k * 8;

                let vd0 = _mm256_loadu_ps($d0.add(offset));
                let vd1 = _mm256_loadu_ps($d1.add(offset));

                acc00 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q0.add(offset)), acc00);
                acc01 = _mm256_fmadd_ps(vd1, _mm256_loadu_ps($q0.add(offset)), acc01);
            }

            $m0 = $m0.max(h_sum_f32(acc00).max(h_sum_f32(acc01)));
        }};

        // 1 * 1 - 128
        ($q0:ident, $d0:ident, $m0:ident) => {{
            let mut acc00 = _mm256_setzero_ps();

            for k in 0..16 {
                let offset = k * 8;

                let vd0 = _mm256_loadu_ps($d0.add(offset));

                acc00 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q0.add(offset)), acc00);
            }

            $m0 = $m0.max(h_sum_f32(acc00));
        }};

        // 4 * 2 generic
        ($num_vecs:expr, $rem:expr, $q0:ident, $q1:ident, $q2:ident, $q3:ident, $d0:ident, $d1:ident, $m0:ident, $m1:ident, $m2:ident, $m3:ident) => {{
            let mut acc00 = _mm256_setzero_ps();
            let mut acc01 = _mm256_setzero_ps();

            let mut acc10 = _mm256_setzero_ps();
            let mut acc11 = _mm256_setzero_ps();

            let mut acc20 = _mm256_setzero_ps();
            let mut acc21 = _mm256_setzero_ps();

            let mut acc30 = _mm256_setzero_ps();
            let mut acc31 = _mm256_setzero_ps();

            for k in 0..$num_vecs {
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

            let mut dd0 = h_sum_f32(acc00);
            let mut dd1 = h_sum_f32(acc01);

            let mut dd2 = h_sum_f32(acc10);
            let mut dd3 = h_sum_f32(acc11);

            let mut dd4 = h_sum_f32(acc20);
            let mut dd5 = h_sum_f32(acc21);

            let mut dd6 = h_sum_f32(acc30);
            let mut dd7 = h_sum_f32(acc31);

            if $rem > 0 {
                let rem_offset = $num_vecs * 8;
                for r in 0..$rem {
                    let rf = rem_offset + r;
                    let dv0 = *$d0.add(rf);
                    let dv1 = *$d1.add(rf);

                    dd0 += dv0 * *$q0.add(rf);
                    dd1 += dv1 * *$q0.add(rf);
                    dd2 += dv0 * *$q1.add(rf);
                    dd3 += dv1 * *$q1.add(rf);
                    dd4 += dv0 * *$q2.add(rf);
                    dd5 += dv1 * *$q2.add(rf);
                    dd6 += dv0 * *$q3.add(rf);
                    dd7 += dv1 * *$q3.add(rf);
                }
            }

            $m0 = $m0.max(dd0).max(dd1);
            $m1 = $m1.max(dd2).max(dd3);
            $m2 = $m2.max(dd4).max(dd5);
            $m3 = $m3.max(dd6).max(dd7);
        }};
        // 4 * 1 generic
        ($num_vecs:expr, $rem:expr, $q0:ident, $q1:ident, $q2:ident, $q3:ident, $d0:ident, $m0:ident, $m1:ident, $m2:ident, $m3:ident) => {{
            let mut acc00 = _mm256_setzero_ps();

            let mut acc01 = _mm256_setzero_ps();

            let mut acc10 = _mm256_setzero_ps();
            let mut acc11 = _mm256_setzero_ps();

            for k in 0..$num_vecs {
                let offset = k * 8;

                let vd0 = _mm256_loadu_ps($d0.add(offset));

                acc00 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q0.add(offset)), acc00);
                acc01 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q1.add(offset)), acc01);

                acc10 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q2.add(offset)), acc10);
                acc11 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q3.add(offset)), acc11);
            }

            let mut dd0 = h_sum_f32(acc00);
            let mut dd1 = h_sum_f32(acc01);

            let mut dd2 = h_sum_f32(acc10);
            let mut dd3 = h_sum_f32(acc11);

            if $rem > 0 {
                let rem_offset = $num_vecs * 8;
                for r in 0..$rem {
                    let rf = rem_offset + r;
                    let dv = *$d0.add(rf);

                    dd0 += dv * *$q0.add(rf);
                    dd1 += dv * *$q1.add(rf);
                    dd2 += dv * *$q2.add(rf);
                    dd3 += dv * *$q3.add(rf);
                }
            }

            $m0 = $m0.max(dd0);
            $m1 = $m1.max(dd1);
            $m2 = $m2.max(dd2);
            $m3 = $m3.max(dd3);
        }};

        // 1 * 1 generic
        ($num_vecs:expr, $rem:expr, $q0:ident, $d0:ident, $m0:ident) => {{
            let mut acc00 = _mm256_setzero_ps();

            for k in 0..$num_vecs {
                let offset = k * 8;

                let vd0 = _mm256_loadu_ps($d0.add(offset));

                acc00 = _mm256_fmadd_ps(vd0, _mm256_loadu_ps($q0.add(offset)), acc00);
            }

            let mut dd0 = h_sum_f32(acc00);
            if $rem > 0 {
                let rem_offset = $num_vecs * 8;
                for r in 0..$rem {
                    let rf = rem_offset + r;
                    let dv = *$d0.add(rf);
                    dd0 += dv * *$q0.add(rf);
                }
            }

            $m0 = $m0.max(dd0);
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
    pub unsafe fn dotmax128_f32(q: &[f32], d: &[f32], q_len: usize, d_len: usize) -> f32 {
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

                let mut di = 0;

                while di + 2 <= d_len {
                    let d0 = d_ptr.add((di + 0) * 128);
                    let d1 = d_ptr.add((di + 1) * 128);

                    kdot_max_f32!(q0, q1, q2, q3, d0, d1, m0, m1, m2, m3);

                    di += 2;
                }

                while di < d_len {
                    let d0 = d_ptr.add(di * 128);

                    kdot_max_f32!(q0, q1, q2, q3, d0, m0, m1, m2, m3);

                    di += 1;
                }

                total_score += m0 + m1 + m2 + m3;
                qi += 4;
            }

            // left over
            while qi < q_len {
                let q0 = q_ptr.add(qi * 128);

                let mut max_val = f32::NEG_INFINITY;
                let mut di = 0;

                while di + 2 <= d_len {
                    let d0 = d_ptr.add(di * 128);
                    let d1 = d_ptr.add((di + 1) * 128);

                    kdot_max_f32!(q0, d0, d1, max_val);

                    di += 2
                }

                while di < d_len {
                    let d0 = d_ptr.add(di * 128);

                    kdot_max_f32!(q0, d0, max_val);

                    di += 1;
                }

                total_score += max_val;
                qi += 1;
            }
        }

        total_score
    }

    /// Dot Max function with tiled
    /// It can take any dim
    /// For less than 500 docs leave no space
    pub unsafe fn dotmaxtg_f32(
        q: &[f32],
        d: &[f32],
        q_len: usize,
        d_len: usize,
        dim: usize,
    ) -> f32 {
        if q_len == 0 || d_len == 0 || dim == 0 {
            return 0.0;
        }

        let q_ptr = q.as_ptr();
        let d_ptr = d.as_ptr();

        let num_vecs = dim / 8;
        let rem = dim % 8;

        // stack buffer
        let mut max_scores_buf = [f32::NEG_INFINITY; 128];
        let mut max_vec_buf;

        let score: &mut [f32] = if q_len <= 128 {
            &mut max_scores_buf[0..q_len]
        } else {
            max_vec_buf = vec![f32::NEG_INFINITY; q_len];
            max_vec_buf.as_mut_slice()
        };

        // data blocks
        let db = (16384 / (dim * 4)).max(1);

        unsafe {
            let mut ds = 0;
            while ds < d_len {
                let de = (ds + db).min(d_len);

                let mut qi = 0;
                while qi + 4 <= q_len {
                    let q0 = q_ptr.add(qi * dim);
                    let q1 = q_ptr.add((qi + 1) * dim);
                    let q2 = q_ptr.add((qi + 2) * dim);
                    let q3 = q_ptr.add((qi + 3) * dim);

                    let mut m0 = score[qi + 0];
                    let mut m1 = score[qi + 1];
                    let mut m2 = score[qi + 2];
                    let mut m3 = score[qi + 3];

                    let mut di = ds;
                    while di + 2 <= de {
                        let d0 = d_ptr.add(di * dim);
                        let d1 = d_ptr.add((di + 1) * dim);

                        kdot_max_f32!(num_vecs, rem, q0, q1, q2, q3, d0, d1, m0, m1, m2, m3);

                        di += 2;
                    }

                    while di < de {
                        let d0 = d_ptr.add(di * dim);

                        kdot_max_f32!(num_vecs, rem, q0, q1, q2, q3, d0, m0, m1, m2, m3);

                        di += 1;
                    }

                    score[qi + 0] = m0;
                    score[qi + 1] = m1;
                    score[qi + 2] = m2;
                    score[qi + 3] = m3;
                    qi += 4;
                }

                while qi < q_len {
                    let q0 = q_ptr.add(qi * dim);

                    let mut max_val = f32::NEG_INFINITY;
                    let mut di = ds;
                    while di < de {
                        let d0 = d_ptr.add(di * dim);

                        kdot_max_f32!(num_vecs, rem, q0, d0, max_val);

                        di += 1;
                    }

                    score[qi] = max_val;
                    qi += 1;
                }

                ds += db;
            }
            let mut total_score = 0.0f32;

            for i in 0..q_len {
                total_score += score[i];
            }

            total_score
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub unsafe fn dotmaxg_f32(q: &[f32], d: &[f32], q_len: usize, d_len: usize, dim: usize) -> f32 {
        if q_len == 0 || d_len == 0 || dim == 0 {
            return 0.0;
        }

        let q_ptr = q.as_ptr();
        let d_ptr = d.as_ptr();
        let num_vecs = dim / 8;
        let rem = dim % 8;

        let mut qi = 0;
        let mut total = 0.0f32;

        unsafe {
            while qi + 4 <= q_len {
                let q0 = q_ptr.add(qi * dim);
                let q1 = q_ptr.add((qi + 1) * dim);
                let q2 = q_ptr.add((qi + 2) * dim);
                let q3 = q_ptr.add((qi + 3) * dim);

                let mut m0 = f32::NEG_INFINITY;
                let mut m1 = f32::NEG_INFINITY;
                let mut m2 = f32::NEG_INFINITY;
                let mut m3 = f32::NEG_INFINITY;

                let mut di = 0;
                while di + 2 <= d_len {
                    let d0 = d_ptr.add(di * dim);
                    let d1 = d_ptr.add((di + 1) * dim);

                    kdot_max_f32!(num_vecs, rem, q0, q1, q2, q3, d0, d1, m0, m1, m2, m3);

                    di += 2;
                }

                while di < d_len {
                    let d0 = d_ptr.add(di * dim);

                    kdot_max_f32!(num_vecs, rem, q0, q1, q2, q3, d0, m0, m1, m2, m3);

                    di += 1;
                }

                total += m0 + m1 + m2 + m3;
                qi += 4;
            }

            while qi < q_len {
                let q0 = q_ptr.add(qi * dim);

                let mut max_val = f32::NEG_INFINITY;
                let mut di = 0;
                while di < d_len {
                    let d0 = d_ptr.add(di * dim);

                    kdot_max_f32!(num_vecs, rem, q0, d0, max_val);

                    di += 1;
                }

                total += max_val;
                qi += 1;
            }
        }

        total
    }

    macro_rules! kdot_max_i8 {
        (
            $num_blocks:expr,
            $q0:ident, $q1:ident, $q2:ident, $q3:ident,
            $qs0:ident, $qs1:ident, $qs2:ident, $qs3:ident,
            $d0:ident, $d1:ident,
            $ds0:ident, $ds1:ident,
            $m0:ident, $m1:ident, $m2:ident, $m3:ident
        ) => {{
            let (mut s00, mut s01) = (0.0f32, 0.0f32);
            let (mut s10, mut s11) = (0.0f32, 0.0f32);
            let (mut s20, mut s21) = (0.0f32, 0.0f32);
            let (mut s30, mut s31) = (0.0f32, 0.0f32);

            for b in 0..$num_blocks {
                let offset = b * 32;

                let vq0 = _mm256_loadu_si256($q0.add(offset) as *const __m256i);
                let vq1 = _mm256_loadu_si256($q1.add(offset) as *const __m256i);
                let vq2 = _mm256_loadu_si256($q2.add(offset) as *const __m256i);
                let vq3 = _mm256_loadu_si256($q3.add(offset) as *const __m256i);

                let vd0 = _mm256_loadu_si256($d0.add(offset) as *const __m256i);
                let vd1 = _mm256_loadu_si256($d1.add(offset) as *const __m256i);

                let d_sc0 = *$ds0.add(b);
                let d_sc1 = *$ds1.add(b);

                s00 += (dot32_i8(vq0, vd0) as f32) * (*$qs0.add(b) * d_sc0);
                s01 += (dot32_i8(vq0, vd1) as f32) * (*$qs0.add(b) * d_sc1);

                s10 += (dot32_i8(vq1, vd0) as f32) * (*$qs1.add(b) * d_sc0);
                s11 += (dot32_i8(vq1, vd1) as f32) * (*$qs1.add(b) * d_sc1);

                s20 += (dot32_i8(vq2, vd0) as f32) * (*$qs2.add(b) * d_sc0);
                s21 += (dot32_i8(vq2, vd1) as f32) * (*$qs2.add(b) * d_sc1);

                s30 += (dot32_i8(vq3, vd0) as f32) * (*$qs3.add(b) * d_sc0);
                s31 += (dot32_i8(vq3, vd1) as f32) * (*$qs3.add(b) * d_sc1);
            }

            $m0 = $m0.max(s00).max(s01);
            $m1 = $m1.max(s10).max(s11);
            $m2 = $m2.max(s20).max(s21);
            $m3 = $m3.max(s30).max(s31);
        }};

        (
            $num_blocks:expr,
            $q0:ident, $q1:ident, $q2:ident, $q3:ident,
            $qs0:ident, $qs1:ident, $qs2:ident, $qs3:ident,
            $d0:ident, $ds0:ident,
            $m0:ident, $m1:ident, $m2:ident, $m3:ident
        ) => {{
            let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

            for b in 0..$num_blocks {
                let offset = b * 32;

                let vq0 = _mm256_loadu_si256($q0.add(offset) as *const __m256i);
                let vq1 = _mm256_loadu_si256($q1.add(offset) as *const __m256i);
                let vq2 = _mm256_loadu_si256($q2.add(offset) as *const __m256i);
                let vq3 = _mm256_loadu_si256($q3.add(offset) as *const __m256i);

                let vd0 = _mm256_loadu_si256($d0.add(offset) as *const __m256i);
                let d_sc0 = *$ds0.add(b);

                s0 += (dot32_i8(vq0, vd0) as f32) * (*$qs0.add(b) * d_sc0);
                s1 += (dot32_i8(vq1, vd0) as f32) * (*$qs1.add(b) * d_sc0);
                s2 += (dot32_i8(vq2, vd0) as f32) * (*$qs2.add(b) * d_sc0);
                s3 += (dot32_i8(vq3, vd0) as f32) * (*$qs3.add(b) * d_sc0);
            }

            $m0 = $m0.max(s0);
            $m1 = $m1.max(s1);
            $m2 = $m2.max(s2);
            $m3 = $m3.max(s3);
        }};

        (
            $num_blocks:expr,
            $q0:ident, $qs0:ident,
            $d0:ident, $ds0:ident,
            $m0:ident
        ) => {{
            let mut s0 = 0.0f32;

            for b in 0..$num_blocks {
                let offset = b * 32;
                let vq0 = _mm256_loadu_si256($q0.add(offset) as *const __m256i);
                let vd0 = _mm256_loadu_si256($d0.add(offset) as *const __m256i);
                let d_sc0 = *$ds0.add(b);

                s0 += (dot32_i8(vq0, vd0) as f32) * (*$qs0.add(b) * d_sc0);
            }

            $m0 = $m0.max(s0);
        }};
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn dotmaxg_i8(
        q: &[i8],
        d: &[i8],
        q_scale: &[f32],
        d_scale: &[f32],
        q_len: usize,
        d_len: usize,
    ) -> f32 {
        if q_len == 0 || d_len == 0 {
            return 0.0;
        }

        let dim = q.len() / q_len;
        let num_blocks = dim / 32;

        let q_ptr = q.as_ptr();
        let d_ptr = d.as_ptr();
        let qs_ptr = q_scale.as_ptr();
        let ds_ptr = d_scale.as_ptr();

        let mut total_score = 0.0f32;
        let mut qi = 0;

        unsafe {
            while qi + 4 <= q_len {
                let q0 = q_ptr.add((qi + 0) * dim);
                let q1 = q_ptr.add((qi + 1) * dim);
                let q2 = q_ptr.add((qi + 2) * dim);
                let q3 = q_ptr.add((qi + 3) * dim);

                let qs0 = qs_ptr.add((qi + 0) * num_blocks);
                let qs1 = qs_ptr.add((qi + 1) * num_blocks);
                let qs2 = qs_ptr.add((qi + 2) * num_blocks);
                let qs3 = qs_ptr.add((qi + 3) * num_blocks);

                let mut m0 = f32::NEG_INFINITY;
                let mut m1 = f32::NEG_INFINITY;
                let mut m2 = f32::NEG_INFINITY;
                let mut m3 = f32::NEG_INFINITY;

                let mut di = 0;

                // 1A. Step by 2 Docs (4x2 kernel)
                while di + 2 <= d_len {
                    let d0 = d_ptr.add((di + 0) * dim);
                    let d1 = d_ptr.add((di + 1) * dim);

                    let ds0 = ds_ptr.add((di + 0) * num_blocks);
                    let ds1 = ds_ptr.add((di + 1) * num_blocks);

                    kdot_max_i8!(
                        num_blocks, q0, q1, q2, q3, qs0, qs1, qs2, qs3, d0, d1, ds0, ds1, m0, m1,
                        m2, m3
                    );

                    di += 2;
                }

                // 1B. Leftover 1 Doc (4x1 kernel)
                while di < d_len {
                    let d0 = d_ptr.add(di * dim);
                    let ds0 = ds_ptr.add(di * num_blocks);

                    kdot_max_i8!(
                        num_blocks, q0, q1, q2, q3, qs0, qs1, qs2, qs3, d0, ds0, m0, m1, m2, m3
                    );

                    di += 1;
                }

                total_score += m0 + m1 + m2 + m3;
                qi += 4;
            }

            // 2. Leftover Queries (1x1 kernel)
            while qi < q_len {
                let q0 = q_ptr.add(qi * dim);
                let qs0 = qs_ptr.add(qi * num_blocks);
                let mut max_val = f32::NEG_INFINITY;

                let mut di = 0;
                while di < d_len {
                    let d0 = d_ptr.add(di * dim);
                    let ds0 = ds_ptr.add(di * num_blocks);

                    kdot_max_i8!(num_blocks, q0, qs0, d0, ds0, max_val);

                    di += 1;
                }

                total_score += max_val;
                qi += 1;
            }

            total_score
        }
    }

    pub fn ref_maxsimd128_f32(q: &[f32], d: &[f32], q_len: usize, d_len: usize) -> f32 {
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

    pub fn ref_maxsim_f32(q: &[f32], d: &[f32], q_len: usize, d_len: usize, dim: usize) -> f32 {
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

    pub fn ref_maxsim_i8(
        q: &[i8],
        d: &[i8],
        qs: &[f32],
        ds: &[f32],
        q_len: usize,
        d_len: usize,
        dim: usize,
    ) -> f32 {
        if q_len == 0 || d_len == 0 {
            return 0.0;
        }
        let num_blocks = dim / 32;
        let mut total = 0.0f32;

        for qi in 0..q_len {
            let mut max_val = f32::NEG_INFINITY;
            for di in 0..d_len {
                let mut dot = 0.0f32;
                for b in 0..num_blocks {
                    let mut b_dot = 0i32;
                    for k in 0..32 {
                        b_dot +=
                            (q[qi * dim + b * 32 + k] as i32) * (d[di * dim + b * 32 + k] as i32);
                    }
                    dot += (b_dot as f32) * (qs[qi * num_blocks + b] * ds[di * num_blocks + b]);
                }
                max_val = max_val.max(dot);
            }
            total += max_val;
        }
        total
    }
}
