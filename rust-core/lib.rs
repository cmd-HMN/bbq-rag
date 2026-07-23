// use pyo3::prelude::*;
pub mod utils;
pub mod cpu;
pub mod func;
pub mod blas;

use pyo3_stub_gen::{derive::*};
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
