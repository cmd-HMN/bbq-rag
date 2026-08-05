pub mod blas;
pub mod cpu;
pub mod func;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

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

    let mut d_structured = Vec::with_capacity(doc_lengths.len());
    let mut offset = 0;

    for (doc_idx, &doc_len) in doc_lengths.iter().enumerate() {
        let chunk_size = doc_len * dim;
        let doc_data = &d_slice[offset..(offset + chunk_size)];

        d_structured.push((doc_idx, doc_len, doc_data));

        offset += chunk_size;
    }

    func::function::maxsim_variable_length(q_slice, d_structured, q_len, dim)
}

#[pymodule]
fn maxsimd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(maxsim_vrlen, m)?)?;
    Ok(())
}

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
