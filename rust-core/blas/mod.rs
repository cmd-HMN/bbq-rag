pub mod sgemm;
pub mod bfunc;

pub use sgemm::custom::sgemm as csgemm;

// will removed in the next commit
pub use bfunc::function::{maxsim_variable_length_slice, pro_sgl_doc, maxsim_fused_doc_tiles, maxsim_variable_length};
pub use bfunc::function::internal::{pro_sgl_doc_csgemm};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use bfunc::function::internal::{pro_sgl_doc_msgemm};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use sgemm::mkl_blas::sgemm as msgemm;
