pub mod sgemm;
pub mod func;

pub use sgemm::custom::sgemm as csgemm;

// will removed in the next commit
pub use func::function::maxsim_variable_length_slice;
pub use func::function::internal::{pro_sgl_doc_csgemm};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use func::function::internal::{pro_sgl_doc_msgemm};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use sgemm::mkl_blas::sgemm as msgemm;
