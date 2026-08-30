#[cfg(feature = "dev")]
pub mod blas;

pub mod cpu;
pub mod pfunc;
pub mod ops;
pub mod quantization;

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
