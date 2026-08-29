///! Contains the python bindings for the pfunc module
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::ops::{omaxsim, omaxsim_3d_ptr, omaxsim_ptr, omaxsim_vrlen};
use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn};

#[pyfunction]
#[gen_stub_pyfunction]
unsafe fn maxsim_ptr(
    q_ptr: usize,
    d_ptr: usize,
    q_len: usize,
    doc_tokens: usize,
    dim: usize,
) -> f32 {
    unsafe { omaxsim_ptr(q_ptr, d_ptr, q_len, doc_tokens, dim) }
}

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
    unsafe { omaxsim_3d_ptr(q_ptr, d_ptr, q_len, num_pages, tokens_per_page, dim) }
}

#[pyfunction]
#[gen_stub_pyfunction]
unsafe fn maxsim_vrlen<'py>(
    q: PyReadonlyArray1<'py, f32>,
    d: PyReadonlyArray1<'py, f32>,
    doc_lengths: Vec<usize>,
    q_len: usize,
    dim: usize,
) -> Vec<f32> {
    omaxsim_vrlen(q, d, doc_lengths, q_len, dim)
}

#[pyfunction]
#[gen_stub_pyfunction]
unsafe fn maxsim<'py>(
    q: PyReadonlyArrayDyn<'py, f32>,
    d: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Vec<f32>> {
    omaxsim(q, d)
}


#[pymodule]
fn maxsimd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_function(wrap_pyfunction!(maxsim, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_3d_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_vrlen, m)?)?;
    Ok(())
}
