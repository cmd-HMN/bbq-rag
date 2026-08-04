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
        #[cfg(all(target_arch = "x86_64", feature = "mkl"))]
        {
            return generic_gpro_sgl_doc(
                _q,
                _d,
                _q_len,
                _d_len,
                _dim,
                |transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
                    msgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
                },
            );
        }

        #[cfg(not(all(target_arch = "x86_64", feature = "mkl")))]
        {
            return generic_gpro_sgl_doc(
                _q,
                _d,
                _q_len,
                _d_len,
                _dim,
                |transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc| {
                    unsafe { csgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }
                },
            );
        }
    }

    pub mod internal {
        use crate::func::function::{generic_gpro_sgl_doc, csgemm, msgemm};

        pub fn pro_sgl_doc_csgemm(
            _q: &[f32],
            _d: &[f32],
            _q_len: usize,
            _d_len: usize,
            _dim: usize,
        ) -> f32{
            generic_gpro_sgl_doc(
                _q,
                _d,
                _q_len,
                _d_len,
                _dim,
                |transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc| unsafe {
                    csgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
                },
                )
        }

        pub fn pro_sgl_doc_msgemm(
            _q: &[f32],
            _d: &[f32],
            _q_len: usize,
            _d_len: usize,
            _dim: usize,
        ) -> f32{
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
    #[inline(always)]
    fn pro_bth(
        _q: &[f32],
        _d: &[(usize, usize, &[f32])],
        _q_len: usize,
        _dim: usize,
        _bth_idx: &[usize], // [batch_start, batch_end] -> bathc size
        _max_len: usize,
        _results: &mut [f32],
    ) {
        let bth_size = _bth_idx.len();

        let bth_result = BBUFFER.with(|buffer| {
            let mut buff = buffer.borrow_mut();

            let req = bth_size * _max_len * _dim;
            buff.resize(req, 0.0);

            for (batch_idx, &sorted_idx) in _bth_idx.iter().enumerate() {
                let (_, doc_len, doc_data) = &_d[sorted_idx];
                let src_size = doc_len * _dim;
                let dst_offset = batch_idx * _max_len * _dim;

                // Copy actual data
                buff[dst_offset..dst_offset + src_size].copy_from_slice(&doc_data[..src_size]);

                // Clear only the padding area
                if *doc_len < _max_len {
                    let padding_start = dst_offset + src_size;
                    let padding_end = dst_offset + _max_len * _dim;
                    buff[padding_start..padding_end].fill(0.0);
                }
            }

            maxsim_fused_doc_tiles(&_q, &buff[..req], _q_len, _max_len, _dim)
        });

        for (batch_idx, &sorted_idx) in _bth_idx.iter().enumerate() {
            _results[sorted_idx] = bth_result[batch_idx];
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
        let mut _results = vec![0.0f32; n_docs];

        let _doc_tile_size = match _d_len {
            512 => 128,
            1024 => 64,
            2048 => 32,
            4096 => 16,
            _ => 32,
        };

        for _doc_tile_start in (0..n_docs).step_by(_doc_tile_size) {
            let _doc_tile_end = (_doc_tile_start + _doc_tile_size).min(n_docs);
            let _tile_docs = _doc_tile_end - _doc_tile_start;
            let _tile_tokens = _tile_docs * _d_len;

            let mut tile_sims = vec![0.0f32; _q_len * _tile_tokens];
            let tile_d_start = _doc_tile_start * _d_len * _dim;
            let tile_d_end = _doc_tile_end * _d_len * _dim;
            let tile_d = &_d[tile_d_start..tile_d_end];

            unsafe {
                csgemm(
                    b'T',
                    b'N',
                    _tile_tokens as i32,
                    _q_len as i32,
                    _dim as i32,
                    1.0,
                    tile_d,
                    _dim as i32,
                    _q,
                    _dim as i32,
                    0.0,
                    &mut tile_sims,
                    _tile_tokens as i32,
                );
            }

            let tile_results: Vec<f32> = (0.._tile_docs)
                .into_par_iter()
                .map(|tile_doc_idx| {
                    let doc_start = tile_doc_idx * _d_len;
                    let mut score = 0.0f32;

                    for qi in 0.._q_len {
                        let base_idx = doc_start + qi * _tile_tokens;
                        let doc_sims = &tile_sims[base_idx..base_idx + _d_len];
                        let max_val = max_avx2(doc_sims);
                        score += max_val;
                    }

                    score
                })
                .collect();

            for (i, &score) in tile_results.iter().enumerate() {
                _results[_doc_tile_start + i] = score;
            }
        }

        _results
    }

    pub fn maxsim_variable_length(
        _q: &[f32],                      // [q_len * dim]
        _d: Vec<(usize, usize, &[f32])>, // [(doc_idx, doc_len, doc_data)]
        _q_len: usize,
        _dim: usize,
    ) -> Vec<f32> {
        let n_docs = _d.len();
        let mut results = vec![0.0f32; n_docs];

        // Fast path: if all documents have similar lengths, process in one batch
        let (min_len, max_len) = _d
            .iter()
            .map(|(_, len, _)| *len)
            .fold((usize::MAX, 0), |(min, max), len| {
                (min.min(len), max.max(len))
            });

        if max_len as f32 / min_len as f32 <= 1.2 && n_docs >= 50 {
            let all_indices: Vec<usize> = (0..n_docs).collect();
            pro_bth(_q, &_d, _q_len, _dim, &all_indices, max_len, &mut results);
            return results;
        }

        // Sort documents by length for better batching
        let mut sorted_indices: Vec<usize> = (0..n_docs).collect();
        sorted_indices.sort_by_key(|&i| _d[i].1);

        // Process in larger batches with adaptive sizing
        let target_batch_size = 128; // Larger batches for better GEMM efficiency
        let mut i = 0;

        while i < n_docs {
            // Find batch end - include docs within 20% length difference
            let base_len = _d[sorted_indices[i]].1;
            let max_acceptable_len = (base_len as f32 * 1.2) as usize;

            let mut batch_end = i + 1;
            while batch_end < n_docs && batch_end < i + target_batch_size {
                if _d[sorted_indices[batch_end]].1 > max_acceptable_len {
                    break;
                }
                batch_end += 1;
            }

            let batch_size = batch_end - i;
            let current_indices = &sorted_indices[i..batch_end];

            if batch_size == 1 {
                // Single document
                let idx = sorted_indices[i];
                let (doc_idx, doc_len, doc_data) = &_d[idx];
                results[*doc_idx] = pro_sgl_doc(_q, doc_data, _q_len, *doc_len, _dim);
            } else {
                // Large batch - worth the overhead of batched processing
                // Check if all documents in batch have exactly the same length
                let first_len = _d[sorted_indices[i]].1;
                let all_same_length = sorted_indices[i..batch_end]
                    .iter()
                    .all(|&idx| _d[idx].1 == first_len);

                if all_same_length {
                    pro_bth(
                        _q,
                        &_d,
                        _q_len,
                        _dim,
                        current_indices,
                        first_len,
                        &mut results,
                    );
                } else {
                    let chunk_max_len = current_indices.iter().map(|&idx| _d[idx].1).max().unwrap();
                    pro_bth(
                        _q,
                        &_d,
                        _q_len,
                        _dim,
                        current_indices,
                        chunk_max_len,
                        &mut results,
                    );
                }
            }
            i = batch_end;
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
