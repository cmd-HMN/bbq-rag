// /! All maxsim variants used in this project

pub mod function {
    use crate::cpu::{dotmax128_f32avx2 as fused_dot_max_dim128_avx2, dotmaxg_f32avx2 as fused_dot_max_generic_avx2};
    use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
    use pyo3::PyResult;
    use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    pub fn omaxsim_variable_length_slice(
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

        if n_docs <= 4 {
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
                    let score =
                        unsafe { fused_dot_max_generic_avx2(q, doc_data, q_len, doc_len, dim) };
                    results.push(score);
                }
            }
            results
        } else if dim == 128 {
            offsets
                .into_par_iter()
                .map(|(offset, doc_len)| {
                    let doc_data = &d_flat[offset..offset + doc_len * 128];
                    unsafe { fused_dot_max_dim128_avx2(q, doc_data, q_len, doc_len) }
                })
                .collect()
        } else {
            offsets
                .into_par_iter()
                .map(|(offset, doc_len)| {
                    let doc_data = &d_flat[offset..offset + doc_len * dim];
                    unsafe { fused_dot_max_generic_avx2(q, doc_data, q_len, doc_len, dim) }
                })
                .collect()
        }
    }

    pub fn omaxsim_variable_length(
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
                    unsafe { fused_dot_max_dim128_avx2(_q, doc_data, _q_len, doc_len) }
                } else {
                    unsafe { fused_dot_max_generic_avx2(_q, doc_data, _q_len, doc_len, _dim) }
                }
            })
            .collect()
    }

    pub fn omaxsim<'py>(
        q: PyReadonlyArrayDyn<'py, f32>,
        d: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Vec<f32>> {
        let q_shape = q.shape();
        let d_shape = d.shape();

        let (q_len, dim) = match q_shape.len() {
            2 => (q_shape[0], q_shape[1]),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Query array must be 2D of shape (q_len, dim)",
                ));
            }
        };

        let q_slice = q.as_slice()?;
        let d_slice = d.as_slice()?;

        match d_shape.len() {
            2 => {
                let doc_tokens = d_shape[0];
                let score = if dim == 128 {
                    unsafe {
                        fused_dot_max_dim128_avx2(
                            q_slice, d_slice, q_len, doc_tokens,
                        )
                    }
                } else {
                    unsafe {
                        fused_dot_max_generic_avx2(
                            q_slice, d_slice, q_len, doc_tokens, dim,
                        )
                    }
                };
                Ok(vec![score])
            }
            3 => {
                let num_pages = d_shape[0];
                let tokens_per_page = d_shape[1];
                let page_stride = tokens_per_page * dim;

                let scores: Vec<f32> = if num_pages <= 24 {
                    let mut res = Vec::with_capacity(num_pages);
                    for page_idx in 0..num_pages {
                        let offset = page_idx * page_stride;
                        let page_data = &d_slice[offset..offset + page_stride];
                        let score = if dim == 128 {
                            unsafe {
                                fused_dot_max_dim128_avx2(
                                    q_slice,
                                    page_data,
                                    q_len,
                                    tokens_per_page,
                                )
                            }
                        } else {
                            unsafe {
                                fused_dot_max_generic_avx2(
                                    q_slice,
                                    page_data,
                                    q_len,
                                    tokens_per_page,
                                    dim,
                                )
                            }
                        };
                        res.push(score);
                    }
                    res
                } else {
                    (0..num_pages)
                        .into_par_iter()
                        .with_min_len(8)
                        .map(|page_idx| {
                            let offset = page_idx * page_stride;
                            let page_data = &d_slice[offset..offset + page_stride];
                            if dim == 128 {
                                unsafe {
                                    fused_dot_max_dim128_avx2(
                                        q_slice,
                                        page_data,
                                        q_len,
                                        tokens_per_page,
                                    )
                                }
                            } else {
                                unsafe {
                                    fused_dot_max_generic_avx2(
                                        q_slice,
                                        page_data,
                                        q_len,
                                        tokens_per_page,
                                        dim,
                                    )
                                }
                            }
                        })
                        .collect()
                };

                Ok(scores)
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "Document array must be 2D (tokens, dim) or 3D (num_pages, tokens_per_page, dim)",
            )),
        }
    }

    pub unsafe fn omaxsim_ptr(
        q_ptr: usize,
        d_ptr: usize,
        q_len: usize,
        doc_tokens: usize,
        dim: usize,
    ) -> f32 {
        let q_slice = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, q_len * dim) };
        let d_slice = unsafe { std::slice::from_raw_parts(d_ptr as *const f32, doc_tokens * dim) };
        if dim == 128 {
            unsafe { fused_dot_max_dim128_avx2(q_slice, d_slice, q_len, doc_tokens) }
        } else {
            unsafe { fused_dot_max_generic_avx2(q_slice, d_slice, q_len, doc_tokens, dim) }
        }
    }

    pub unsafe fn omaxsim_3d_ptr(
        q_ptr: usize,
        d_ptr: usize,
        q_len: usize,
        num_pages: usize,
        tokens_per_page: usize,
        dim: usize,
    ) -> Vec<f32> {
        let q_slice = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, q_len * dim) };
        let d_slice = unsafe {
            std::slice::from_raw_parts(d_ptr as *const f32, num_pages * tokens_per_page * dim)
        };
        let page_stride = tokens_per_page * dim;

        (0..num_pages)
            .into_par_iter()
            .map(|page_idx| {
                let offset = page_idx * page_stride;
                let page_data = &d_slice[offset..offset + page_stride];
                if dim == 128 {
                    unsafe {
                        fused_dot_max_dim128_avx2(
                            q_slice,
                            page_data,
                            q_len,
                            tokens_per_page,
                        )
                    }
                } else {
                    unsafe {
                        fused_dot_max_generic_avx2(
                            q_slice,
                            page_data,
                            q_len,
                            tokens_per_page,
                            dim,
                        )
                    }
                }
            })
            .collect()
    }

    pub fn omaxsim_vrlen<'py>(
        q: PyReadonlyArray1<'py, f32>,
        d: PyReadonlyArray1<'py, f32>,
        doc_lengths: Vec<usize>,
        q_len: usize,
        dim: usize,
    ) -> Vec<f32> {
        let q_slice = q.as_slice().expect("q must be contiguous");
        let d_slice = d.as_slice().expect("d must be contiguous");

        omaxsim_variable_length_slice(q_slice, d_slice, &doc_lengths, q_len, dim)
    }
}
