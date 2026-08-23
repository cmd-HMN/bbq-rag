pub mod blas;
pub mod cpu;
pub mod func;
pub mod quantization;

use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rayon::prelude::*;


//TODO 
// Add support to 1D array
/// MaxSim scoring supporting 2D and 3D NumPy arrays with zero Python overhead.
#[pyfunction]
#[gen_stub_pyfunction]
fn maxsim<'py>(
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
            ))
        }
    };

    let q_slice = q.as_slice()?;
    let d_slice = d.as_slice()?;

    match d_shape.len() {
        2 => {
            let doc_tokens = d_shape[0];
            let score = if dim == 128 {
                unsafe {
                    crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(
                        q_slice, d_slice, q_len, doc_tokens,
                    )
                }
            } else {
                unsafe {
                    crate::cpu::vec256::simd::fused_dot_max_generic_avx2(
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
                            crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(
                                q_slice,
                                page_data,
                                q_len,
                                tokens_per_page,
                            )
                        }
                    } else {
                        unsafe {
                            crate::cpu::vec256::simd::fused_dot_max_generic_avx2(
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
                                crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(
                                    q_slice,
                                    page_data,
                                    q_len,
                                    tokens_per_page,
                                )
                            }
                        } else {
                            unsafe {
                                crate::cpu::vec256::simd::fused_dot_max_generic_avx2(
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

/// Ultra-fast pointer-based MaxSim for 2D PyTorch tensors (`tensor.data_ptr()`).
#[pyfunction]
#[gen_stub_pyfunction]
unsafe fn maxsim_ptr(
    q_ptr: usize,
    d_ptr: usize,
    q_len: usize,
    doc_tokens: usize,
    dim: usize,
) -> f32 {
    let q_slice = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, q_len * dim) };
    let d_slice = unsafe { std::slice::from_raw_parts(d_ptr as *const f32, doc_tokens * dim) };
    if dim == 128 {
        unsafe {
            crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(q_slice, d_slice, q_len, doc_tokens)
        }
    } else {
        unsafe {
            crate::cpu::vec256::simd::fused_dot_max_generic_avx2(
                q_slice, d_slice, q_len, doc_tokens, dim,
            )
        }
    }
}

/// Ultra-fast pointer-based MaxSim for 3D PyTorch tensors (`tensor.data_ptr()`).
#[pyfunction]
#[gen_stub_pyfunction]
unsafe fn maxsim_3d_ptr(
    q_ptr: usize,
    d_ptr: usize,
    q_len: usize,
    num_pages: usize,
    tokens_per_page: usize,
    dim: usize,
) -> Vec<f32> {
    let q_slice = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, q_len * dim) };
    let d_slice =
        unsafe { std::slice::from_raw_parts(d_ptr as *const f32, num_pages * tokens_per_page * dim) };
    let page_stride = tokens_per_page * dim;

    (0..num_pages)
        .into_par_iter()
        .map(|page_idx| {
            let offset = page_idx * page_stride;
            let page_data = &d_slice[offset..offset + page_stride];
            if dim == 128 {
                unsafe {
                    crate::cpu::vec256::simd::fused_dot_max_dim128_avx2(
                        q_slice,
                        page_data,
                        q_len,
                        tokens_per_page,
                    )
                }
            } else {
                unsafe {
                    crate::cpu::vec256::simd::fused_dot_max_generic_avx2(
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

#[pyfunction]
#[gen_stub_pyfunction]
fn maxsim_vrlen<'py>(
    q: PyReadonlyArray1<'py, f32>,
    d: PyReadonlyArray1<'py, f32>,
    doc_lengths: Vec<usize>,
    q_len: usize,
    dim: usize,
) -> Vec<f32> {
    let q_slice = q.as_slice().expect("q must be contiguous");
    let d_slice = d.as_slice().expect("d must be contiguous");

    func::function::maxsim_variable_length_slice(q_slice, d_slice, &doc_lengths, q_len, dim)
}

#[pymodule]
fn maxsimd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(maxsim, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_3d_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_vrlen, m)?)?;
    Ok(())
}

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
