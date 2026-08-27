/// This will act a router
pub mod vec256;

pub use vec256::simd::{fused_dot_max_dim128_avx2, fused_dot_max_generic_avx2, naive_maxsim_dim128, reference_maxsim, max_avx2};
