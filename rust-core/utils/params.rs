use std::sync::OnceLock;

/// This for Sgemm block function with type nn or nt
type Sgemm_block = unsafe fn(
    u8, u8, u32, u32, u32, 
    f32, &[f32], u32, 
    &[f32], u32, 
    f32, &mut [f32], u32
);
