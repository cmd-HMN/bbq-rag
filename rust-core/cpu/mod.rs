/// This will act a router
pub mod vec256;

pub use vec256::simd::{dotmax128_f32avx2, dotmaxg_f32avx2, ref_maxsimd128, ref_maxsim, max_avx2};
