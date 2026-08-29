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

//These are only blas function that somehow mimic the maxsim-cpu functionality
//And only use in the benchmark
//So no point to unit testing
//
//
//Implemntation might be unoptimized as what i am seeing potential downside
//Like thread_local! and few others

use super::{csgemm};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
use super::{msgemm};

pub mod function {
    use crate::cpu::vec256::simd::max_avx2;
    use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};
    use std::cell::RefCell;

    use super::{csgemm};

    #[cfg(all(target_arch = "x86_64", feature = "dev"))]
    use super::{msgemm};

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
            unsafe { crate::cpu::vec256::simd::fused_dot_max_dim128_avx2_f32(_q, _d, _q_len, _d_len) }
        } else {
            unsafe { crate::cpu::vec256::simd::fused_dot_max_generic_avx2_f32(_q, _d, _q_len, _d_len, _dim) }
        }
    }

    pub mod internal {
        use super::{csgemm, generic_gpro_sgl_doc};

        #[cfg(all(target_arch = "x86_64", feature = "dev"))]
        use super::{msgemm};

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

        #[cfg(all(target_arch = "x86_64", feature = "dev"))]
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

        if n_docs <= 64 {
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
                .with_min_len(64)
                .map(|(offset, doc_len)| {
                    let doc_data = &d_flat[offset..offset + doc_len * 128];
                    unsafe { crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(q, doc_data, q_len, doc_len) }
                })
                .collect()
        } else {
            offsets
                .into_par_iter()
                .with_min_len(64)
                .map(|(offset, doc_len)| {
                    let doc_data = &d_flat[offset..offset + doc_len * dim];
                    unsafe { crate::cpu::vec256::simd::fused_dot_max_generic_avx2(q, doc_data, q_len, doc_len, dim) }
                })
                .collect()
        }
    }

    pub fn maxsim_variable_length(
        _q: Vec<f32>,                      // [q_len * dim]
        _d: Vec<(usize, usize, Vec<f32>)>, // [(doc_idx, doc_len, doc_data)]
        _q_len: usize,
        _dim: usize,
    ) -> Vec<f32> {
        let n_docs = _d.len();
        if n_docs == 0 {
            return Vec::new();
        }

        let _q = &_q[.._q_len * _dim];

        _d.into_par_iter()
        .map(|(_doc_idx, doc_len, doc_data)| {
            let doc_data = &doc_data[..doc_len * _dim];
            if _dim == 128 {
                unsafe { 
                    crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(_q, doc_data, _q_len, doc_len) 
                }
            } else {
                unsafe { 
                    crate::cpu::vec256::simd::fused_dot_max_generic_avx2(_q, doc_data, _q_len, doc_len, _dim) 
                }
            }
        })
        .collect()
    }
}
