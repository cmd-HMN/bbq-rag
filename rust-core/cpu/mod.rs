/// This will act a router
pub mod vec256;

pub use vec256::simd::{
    dotmax128_f32, 
    dotmaxg_f32, 
    dotmaxtg_f32,
    dotmaxg_i8,
    ref_maxsimd128_f32, 
    ref_maxsim_f32,
    ref_maxsim_i8
};
