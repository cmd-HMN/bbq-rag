///! Contains the python bindings for the pfunc module
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::ops::{DocLayout};
use crate::quantization::QTYPE;

#[pyfunction]
#[gen_stub_pyfunction]
#[pyo3(signature = (
    q_ptr,
    d_ptr,
    q_len,
    dim,
    layout_type,
    doc_tokens=0,
    batch_docs=0,
    batch_tokens=0,
    doc_lengths=None,
    q_scale_ptr=0,
    d_scale_ptr=0,
    dtype=0,
    jobs=-1
))]
unsafe fn maxsim(
    q_ptr: usize,
    d_ptr: usize,
    q_len: usize,
    dim: usize,
    layout_type: u8,
    doc_tokens: usize,
    batch_docs: usize,
    batch_tokens: usize,
    doc_lengths: Option<Vec<usize>>,
    q_scale_ptr: usize,
    d_scale_ptr: usize,
    dtype: u8,
    jobs: i32,
) -> PyResult<Vec<f32>> {
    let qtype = match dtype {
        0 => QTYPE::Float32,
        1 => QTYPE::Int8,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dtype must be 0 (Float32) or 1 (Int8)",
            ));
        }
    };

    let flat_lens;
    let layout = match layout_type {
        0 => DocLayout::Single { doc_tokens },
        1 => DocLayout::Batch {
            docs: batch_docs,
            tokens: batch_tokens,
        },
        2 => {
            flat_lens = doc_lengths.unwrap_or_default();
            DocLayout::Flat { d_len: &flat_lens }
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "layout_type must be 0 (Single), 1 (Batch), or 2 (Flat)",
            ));
        }
    };

    Ok(unsafe {
        // caller maxsim funciton
        crate::ops::maxsim(
            q_ptr,
            d_ptr,
            q_len,
            q_scale_ptr,
            d_scale_ptr,
            dim,
            layout,
            qtype,
            jobs,
        )
    })
}


#[pyfunction]
#[gen_stub_pyfunction]
#[pyo3(signature = (ptr, tokens, dim=128, out_ptr=0, scale_ptr=0, jobs=-1))]
unsafe fn qi8(
    ptr: usize,
    tokens: usize,
    dim: usize,
    out_ptr: usize,
    scale_ptr: usize,
    jobs: i32,
) -> PyResult<(Vec<i8>, Vec<f32>)> {
    unsafe {
        crate::ops::qi8(ptr, tokens, dim, out_ptr, scale_ptr, jobs)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }
}

#[pymodule]
fn maxsimd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_function(wrap_pyfunction!(maxsim, m)?)?;
    m.add_function(wrap_pyfunction!(qi8, m)?)?;

    Ok(())
}
