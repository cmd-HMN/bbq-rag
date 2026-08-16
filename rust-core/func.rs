// Copyright 2024 cmd-HMN
//
// This file includes some or all code from the maxsim-cpu library.
// https://github.com/mixedbread-ai/maxsim-cpu/blob/main/src/lib.rs
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


//TODO
// REmove the unused functions
pub mod function {
    use crate::cpu::vec256::simd::max_avx2;
    use rayon::prelude::*;
    use std::cell::RefCell;

    use crate::blas::custom::sgemm as csgemm;

    #[cfg(all(target_arch = "x86_64", feature = "mkl"))]
    use crate::blas::mkl_blas::sgemm as msgemm;

    thread_local! {
        static BUFFER: RefCell<Vec<f32>> = RefCell::new(Vec::new());
        static BBUFFER: RefCell<Vec<f32>> = RefCell::new(Vec::with_capacity(1024 * 1024)); // 1MB
                                                                                           // batch
                                                                                           // buffer
    }

    pub fn generic_gpro_sgl_doc<F>(
        _q: &[f32],
        _d: &[f32],
        _q_len: usize,
        _d_len: usize,
        _dim: usize,
        sgemm: F,
    ) -> f32
    where
        F: Fn(u8, u8, i32, i32, i32, f32, &[f32], i32, &[f32], i32, f32, &mut [f32], i32),
    {
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

    pub fn pro_sgl_doc(_q: &[f32], _d: &[f32], _q_len: usize, _d_len: usize, _dim: usize) -> f32 {
        if _dim == 128 {
            unsafe { crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(_q, _d, _q_len, _d_len) }
        } else {
            unsafe { crate::cpu::vec256::simd::fused_dot_max_generic_avx2(_q, _d, _q_len, _d_len, _dim) }
        }
    }

    pub mod internal {
        use crate::func::function::{generic_gpro_sgl_doc, csgemm};
        #[cfg(all(target_arch = "x86_64", feature = "mkl"))]
        use crate::func::function::msgemm;

        pub fn pro_sgl_doc_csgemm(
            _q: &[f32],
            _d: &[f32],
            _q_len: usize,
            _d_len: usize,
            _dim: usize,
        ) -> f32 {
            generic_gpro_sgl_doc(
                _q,
                _d,
                _q_len,
                _d_len,
                _dim,
                |transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc| {
                    csgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
                },
            )
        }

        #[cfg(all(target_arch = "x86_64", feature = "mkl"))]
        pub fn pro_sgl_doc_msgemm(
            _q: &[f32],
            _d: &[f32],
            _q_len: usize,
            _d_len: usize,
            _dim: usize,
        ) -> f32 {
            generic_gpro_sgl_doc(
                _q,
                _d,
                _q_len,
                _d_len,
                _dim,
                |transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
                    msgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
                },
            )
        }
    }

    pub fn maxsim_fused_doc_tiles(
        _q: &[f32],
        _d: &[f32],
        _q_len: usize,
        _d_len: usize,
        _dim: usize,
    ) -> Vec<f32> {
        let n_docs = _d.len() / (_d_len * _dim);
        (0..n_docs)
            .into_par_iter()
            .map(|doc_idx| {
                let start = doc_idx * _d_len * _dim;
                let doc_data = &_d[start..start + _d_len * _dim];
                pro_sgl_doc(_q, doc_data, _q_len, _d_len, _dim)
            })
            .collect()
    }

    /// Zero-copy, high-performance variable-length MaxSim evaluation across document slices.
    pub fn maxsim_variable_length_slice(
        q: &[f32],
        d_flat: &[f32],
        doc_lengths: &[usize],
        q_len: usize,
        dim: usize,
    ) -> Vec<f32> {
        let n_docs = doc_lengths.len();
        if n_docs == 0 {
            return Vec::new();
        }

        let mut offsets = Vec::with_capacity(n_docs);
        let mut curr_offset = 0;
        for &doc_len in doc_lengths {
            offsets.push((curr_offset, doc_len));
            curr_offset += doc_len * dim;
        }

        if n_docs <= 24 {
            let mut results = Vec::with_capacity(n_docs);
            if dim == 128 {
                for (offset, doc_len) in offsets {
                    let doc_data = &d_flat[offset..offset + doc_len * 128];
                    let score = unsafe { crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(q, doc_data, q_len, doc_len) };
                    results.push(score);
                }
            } else {
                for (offset, doc_len) in offsets {
                    let doc_data = &d_flat[offset..offset + doc_len * dim];
                    let score = unsafe { crate::cpu::vec256::simd::fused_dot_max_generic_avx2(q, doc_data, q_len, doc_len, dim) };
                    results.push(score);
                }
            }
            results
        } else if dim == 128 {
            offsets
                .into_par_iter()
                .with_min_len(8)
                .map(|(offset, doc_len)| {
                    let doc_data = &d_flat[offset..offset + doc_len * 128];
                    unsafe { crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(q, doc_data, q_len, doc_len) }
                })
                .collect()
        } else {
            offsets
                .into_par_iter()
                .with_min_len(8)
                .map(|(offset, doc_len)| {
                    let doc_data = &d_flat[offset..offset + doc_len * dim];
                    unsafe { crate::cpu::vec256::simd::fused_dot_max_generic_avx2(q, doc_data, q_len, doc_len, dim) }
                })
                .collect()
        }
    }

    pub fn maxsim_variable_length(
        _q: &[f32],                      // [q_len * dim]
        _d: Vec<(usize, usize, &[f32])>, // [(doc_idx, doc_len, doc_data)]
        _q_len: usize,
        _dim: usize,
    ) -> Vec<f32> {
        let n_docs = _d.len();
        if n_docs == 0 {
            return Vec::new();
        }

        let mut results = vec![0.0f32; n_docs];
        let scores: Vec<(usize, f32)> = _d
            .into_par_iter()
            .map(|(doc_idx, doc_len, doc_data)| {
                let score = if _dim == 128 {
                    unsafe { crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(_q, doc_data, _q_len, doc_len) }
                } else {
                    unsafe { crate::cpu::vec256::simd::fused_dot_max_generic_avx2(_q, doc_data, _q_len, doc_len, _dim) }
                };
                (doc_idx, score)
            })
            .collect();

        for (doc_idx, score) in scores {
            results[doc_idx] = score;
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use crate::func::function::maxsim_variable_length;

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

    fn reference_maxsim_fused_doc_tiles(
        _q: &[f32],
        _d: &[f32],
        _q_len: usize,
        _d_len: usize,
        _dim: usize,
    ) -> Vec<f32> {
        let n_docs = _d.len() / (_d_len * _dim);
        let mut results = Vec::with_capacity(n_docs);
        for doc_idx in 0..n_docs {
            let doc_start = doc_idx * _d_len * _dim;
            let current_doc = &_d[doc_start..doc_start + _d_len * _dim];

            let mut score = 0.0f32;
            for qi in 0.._q_len {
                let mut max_val = f32::NEG_INFINITY;
                for di in 0.._d_len {
                    let mut sum = 0.0f32;
                    for d in 0.._dim {
                        sum += _q[qi * _dim + d] * current_doc[di * _dim + d];
                    }
                    max_val = max_val.max(sum);
                }
                score += max_val;
            }
            results.push(score);
        }

        results
    }

    fn generate_query(q_len: usize, dim: usize) -> Vec<f32> {
        vec![0.5; q_len * dim]
    }

    fn assert_pro_sgl_doc_eq(_q: &[f32], _d: &[f32], _q_len: usize, _d_len: usize, _dim: usize) {
        let expected = pro_sgl_doc(_q, _d, _q_len, _d_len, _dim);
        let got = function::pro_sgl_doc(_q, _d, _q_len, _d_len, _dim);
        let diff = (got - expected).abs();
        assert!(
            diff < 1e-4 * expected.abs().max(1.0),
            "maxsim mismatch for input"
        );
    }

    fn assert_maxsim_fused_doc_tiles_eq(
        _q: &[f32],
        _d: &[f32],
        _q_len: usize,
        _d_len: usize,
        _dim: usize,
    ) {
        let expected = reference_maxsim_fused_doc_tiles(_q, _d, _q_len, _d_len, _dim);

        let got = function::maxsim_fused_doc_tiles(_q, _d, _q_len, _d_len, _dim);

        assert_eq!(expected.len(), got.len(), "Document output count mismatch");

        for i in 0..expected.len() {
            let diff = (got[i] - expected[i]).abs();
            assert!(
                diff < 1e-4 * expected[i].abs().max(1.0),
                "MaxSim mismatch at doc {}! expected: {}, got: {}",
                i,
                expected[i],
                got[i]
            );
        }
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

    #[test]
    fn test_func_maxfusedtiles_doctilespseudorandom() {
        let q_len = 4;
        let d_len = 16;
        let dim = 32;
        let n_docs = 5;

        let q: Vec<f32> = (0..q_len * dim).map(|x| (x as f32 % 7.0) - 3.5).collect();

        let d: Vec<f32> = (0..n_docs * d_len * dim)
            .map(|x| (x as f32 % 11.0) - 5.5)
            .collect();

        assert_maxsim_fused_doc_tiles_eq(&q, &d, q_len, d_len, dim);
    }

    #[test]
    fn test_func_maxfusedtiles_doctileszeros() {
        let q_len = 2;
        let d_len = 8;
        let dim = 16;
        let n_docs = 3;

        let q = vec![0.0; q_len * dim];
        let d = vec![0.0; n_docs * d_len * dim];

        assert_maxsim_fused_doc_tiles_eq(&q, &d, q_len, d_len, dim);
    }

    #[test]
    fn test_func_maxfusedtiles_doctileslargebatch() {
        let q_len = 3;
        let d_len = 512;
        let dim = 16;
        let n_docs = 200;

        let q: Vec<f32> = (0..q_len * dim).map(|x| x as f32 % 3.0).collect();
        let d: Vec<f32> = (0..n_docs * d_len * dim).map(|x| x as f32 % 2.0).collect();

        assert_maxsim_fused_doc_tiles_eq(&q, &d, q_len, d_len, dim);
    }

    #[test]
    fn test_func_maxsimvariablelength_singleandpaddedbatches() {
        let dim = 128;
        let q_len = 10;
        let q = generate_query(q_len, dim);

        let doc0 = vec![1.0; 10 * dim]; // Length 10
        let doc1 = vec![1.0; 11 * dim]; // Length 11 (Within 20% of 10, will batch with doc0)
        let doc2 = vec![1.0; 30 * dim]; // Length 30 (Way larger, will be processed as a single doc)

        let d = vec![
            (0, 10, doc0.as_slice()),
            (1, 11, doc1.as_slice()),
            (2, 30, doc2.as_slice()),
        ];

        // Run the function
        let results = maxsim_variable_length(&q, d, q_len, dim);

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_func_maxsimvariablelength_perfectmatchlargebatch() {
        let dim = 128;
        let q_len = 10;
        let q = generate_query(q_len, dim);

        let doc_length = 15;
        let backing_data: Vec<Vec<f32>> = (0..35).map(|_| vec![1.0; doc_length * dim]).collect();

        let mut d = Vec::new();
        for (i, data) in backing_data.iter().enumerate() {
            d.push((i, doc_length, data.as_slice()));
        }

        let results = maxsim_variable_length(&q, d, q_len, dim);

        assert_eq!(results.len(), 35);
    }

    #[test]
    fn test_func_maxsimvariablelength_fastpathglobalbatching() {
        let dim = 128;
        let q_len = 10;
        let q = generate_query(q_len, dim);

        let min_len = 20;
        let max_len = 23; // 23 / 20 = 1.15 (which is <= 1.2)

        let mut backing_data = Vec::new();
        for i in 0..55 {
            let len = if i % 2 == 0 { min_len } else { max_len };
            backing_data.push(vec![1.0; len * dim]);
        }

        let mut d = Vec::new();
        for (i, data) in backing_data.iter().enumerate() {
            let len = if i % 2 == 0 { min_len } else { max_len };
            d.push((i, len, data.as_slice()));
        }

        let results = maxsim_variable_length(&q, d, q_len, dim);

        assert_eq!(results.len(), 55);
    }

    #[test]
    fn test_func_maxsimvariablelength_emptydocumentlist() {
        let dim = 128;
        let q_len = 10;
        let q = generate_query(q_len, dim);
        let d: Vec<(usize, usize, &[f32])> = vec![];

        let results = maxsim_variable_length(&q, d, q_len, dim);

        assert_eq!(results.len(), 0);
    }
}
