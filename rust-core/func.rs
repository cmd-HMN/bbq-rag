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

mod function {
    use crate::cpu::vec256::simd::max_avx2;
    use std::cell::RefCell;

    #[cfg(all(target_arch = "x86_64", feature = "mkl"))]
    use crate::blas::mkl_blas::sgemm;

    #[cfg(not(feature = "mkl"))]
    use crate::blas::custom::sgemm;

    thread_local! {
        static BUFFER: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    }

    pub fn pro_sgl_doc(_q: &[f32], _d: &[f32], _q_len: usize, _d_len: usize, _dim: usize) -> f32 {
        unsafe {
            BUFFER.with(|buffer| {
                let mut buffer = buffer.borrow_mut();
                if buffer.len() < _d_len * _q_len {
                    buffer.resize(_d_len * _q_len, 0.0);
                }

                sgemm(
                    b'T',
                    b'N',
                    _d_len as i32,
                    _q_len as i32,
                    _dim as i32,
                    1.0,
                    _d,
                    _dim as i32,
                    _q,
                    _dim as i32,
                    0.0,
                    buffer.as_mut_slice(),
                    _d_len as i32,
                );

                let mut score = 0.0f32;
                for i in 0.._q_len {
                    let start = i * _d_len;
                    let query_sims = &buffer[start..start + _d_len];
                    score += max_avx2(query_sims);
                }

                score
            })
        }
    }


    pub fn maxsim_fused_doc_tiles(
        _q: &[f32],
        _d: &[f32],
        _q_len: usize,
        _d_len: usize,
        _dim: usize
    ) -> Vec<f32> {
        let n_docs = _d.len() / (_d_len * _dim);
        let _results = vec![0.0f32; n_docs];
        _results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pro_sgl_doc(_q: &[f32], _d: &[f32], _q_len: usize, _d_len: usize, _dim: usize) -> f32 {
        let mut score = 0.0f32;
        for qi in 0.._q_len {
            let mut _max = f32::NEG_INFINITY;
            for di in 0.._d_len {
                let mut _sum = 0.0f32;
                for d in 0.._dim {
                    _sum += _q[qi * _dim + d] * _d[di * _dim + d];
                }
                _max = _max.max(_sum);
            }
            score += _max;
        }

        score
    }

    fn assert_pro_sgl_doc_eq(_q: &[f32], _d: &[f32], _q_len: usize, _d_len: usize, _dim: usize) {
        let expected = pro_sgl_doc(_q, _d, _q_len, _d_len, _dim);
        let got = function::pro_sgl_doc(_q, _d, _q_len, _d_len, _dim);
        let diff = (got - expected).abs();
        assert!(diff < 1e-4 * expected.abs().max(1.0), "maxsim mismatch for input");
    }

    #[test]
    fn test_func_prosgldoc_singletokensingledoc() {
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let d = vec![1.0, 1.0, 1.0, 1.0];
        assert_pro_sgl_doc_eq(&q, &d, 1, 1, 4);
    }

    #[test]
    fn test_func_prosgldoc_multiplequerytokens() {
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let d = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        assert_pro_sgl_doc_eq(&q, &d, 2, 3, 4);
    }

    #[test]
    fn test_func_prosgldoc_orthogonalvectors() {
        let q = vec![
            1.0, 0.0, // q0
            0.0, 1.0, // q1
        ];
        let d = vec![
            1.0, 0.0, // d0
            0.0, 1.0, // d1
        ];
        assert_pro_sgl_doc_eq(&q, &d, 2, 2, 2);
    }

    #[test]
    fn test_func_prosgldoc_zeros() {
        let q = vec![0.0; 32];
        let d = vec![0.0; 48];
        assert_pro_sgl_doc_eq(&q, &d, 4, 6, 8);
    }

    #[test]
    fn test_func_prosgldoc_values() {
        let q = vec![-1.0, 2.0, -3.0, 4.0];
        let d = vec![1.0, -2.0, 3.0, -4.0];
        assert_pro_sgl_doc_eq(&q, &d, 1, 1, 4);
    }

    #[test]
    fn test_func_prosgldoc_variouslengths() {
        for q_len in 1..=10 {
            for d_len in 1..=10 {
                for dim in [1, 2, 4, 7, 8, 16, 31, 32, 33] {
                    let q: Vec<f32> = (0..q_len * dim).map(|i| (i as f32).sin()).collect();
                    let d: Vec<f32> = (0..d_len * dim).map(|i| (i as f32).cos()).collect();
                    assert_pro_sgl_doc_eq(&q, &d, q_len, d_len, dim);
                }
            }
        }
    }

    #[test]
    fn test_func_prosgldoc_comprehensivesizes() {
        for q_len in [1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 99] {
            for d_len in [1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 127, 128, 199] {
                for dim in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 64, 127] {
                    let q: Vec<f32> = (0..q_len * dim).map(|i| (i as f32).sin()).collect();
                    let d: Vec<f32> = (0..d_len * dim).map(|i| (i as f32).cos()).collect();
                    assert_pro_sgl_doc_eq(&q, &d, q_len, d_len, dim);
                }
            }
        }
    }

    #[test]
    fn test_func_prosgldoc_dimnotmultipleof8() {
        let q = vec![1.0, 2.0, 3.0];
        let d = vec![4.0, 5.0, 6.0];
        assert_pro_sgl_doc_eq(&q, &d, 1, 1, 3);
    }

    #[test]
    fn test_func_prosgldoc_longerthanquery() {
        let q = vec![1.0, 0.0];
        let d = vec![0.0, 1.0, 1.0, 0.0, 0.5, 0.5];
        assert_pro_sgl_doc_eq(&q, &d, 1, 3, 2);
    }

    #[test]
    fn test_func_prosgldoc_longerthandoc() {
        let q = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let d = vec![1.0, 0.0];
        assert_pro_sgl_doc_eq(&q, &d, 3, 1, 2);
    }
}
