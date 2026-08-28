// /! All maxsim variants used in this project

pub mod function {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};
    use crate::cpu::{fused_dot_max_dim128_avx2, fused_dot_max_generic_avx2};
    
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
                    let score = unsafe { fused_dot_max_dim128_avx2(q, doc_data, q_len, doc_len) };
                    results.push(score);
                }
            } else {
                for (offset, doc_len) in offsets {
                    let doc_data = &d_flat[offset..offset + doc_len * dim];
                    let score = unsafe { fused_dot_max_generic_avx2(q, doc_data, q_len, doc_len, dim) };
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
                    unsafe { fused_dot_max_dim128_avx2(q, doc_data, q_len, doc_len) }
                })
                .collect()
        } else {
            offsets
                .into_par_iter()
                .with_min_len(64)
                .map(|(offset, doc_len)| {
                    let doc_data = &d_flat[offset..offset + doc_len * dim];
                    unsafe { fused_dot_max_generic_avx2(q, doc_data, q_len, doc_len, dim) }
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
                    fused_dot_max_dim128_avx2(_q, doc_data, _q_len, doc_len) 
                }
            } else {
                unsafe { 
                    fused_dot_max_generic_avx2(_q, doc_data, _q_len, doc_len, _dim) 
                }
            }
        })
        .collect()
    }
}
