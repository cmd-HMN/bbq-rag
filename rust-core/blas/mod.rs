pub mod sgemm;
pub mod bfunc;

pub use sgemm::custom::sgemm as csgemm;

// will removed in the next commit
#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use bfunc::bfunction::{pro_sgl_doc, maxsim_fused_doc_tiles};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use bfunc::bfunction::internal::{pro_sgl_doc_csgemm};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use bfunc::bfunction::internal::{pro_sgl_doc_msgemm};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use sgemm::mkl_blas::sgemm as msgemm;
