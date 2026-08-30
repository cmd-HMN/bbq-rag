pub mod sgemm;
pub mod bfunc;
pub mod vec256;

pub use sgemm::custom::sgemm as csgemm;

pub use vec256::bvec256::max_avx2;

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use bfunc::bfunction::{pro_sgl_doc, maxsim_fused_doc_tiles};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use bfunc::bfunction::internal::{pro_sgl_doc_csgemm};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use bfunc::bfunction::internal::{pro_sgl_doc_msgemm};

#[cfg(all(target_arch = "x86_64", feature = "dev"))]
pub use sgemm::mkl_blas::sgemm as msgemm;
