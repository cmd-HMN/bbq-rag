/// Contain all the function to be used in the project

mod function {
    use crate::cpu::vec256::simd;
    //
    // /// Dot product
    // /// Simd based dot product for the f32 type
    // /// # Type Parameters
    // ///
    // /// * `T` - The size of the vector - can be 2, 4, 6 or 8
    // ///
    // /// # Arguments
    // ///
    // /// * `a` - A slice of f32
    // /// * `b` - A slice of f32
    // /// # Returns
    // /// A f32 dot product 
    // /// 
    // /// # Panics
    // /// Panics if the size of the vector is not 2, 4, 6 or 8
    // pub fn dot_f32<const T: usize>(a: &[f32], b: &[f32]) -> f32 {
    //     match T {
    //         2 => simd::dot_f32_2acc(a, b),
    //         4 => simd::dot_f32_4acc(a, b),
    //         6 => simd::dot_f32_6acc(a, b),
    //         8 => simd::dot_f32_8acc(a, b),
    //         _ => panic!("Unsupported vector size, Only support 2, 4, 6 or 8")
    //     }
    // }
    //
    // /// Norm Squared
    // /// # Type Parameters
    // /// 
    // /// * `T` - The size of the vector - can be 2, 4, 6 or 8
    // /// 
    // /// # Arguments
    // /// 
    // /// * `a` - A slice of f32
    // /// # Returns
    // /// A f32 norm squared
    // ///
    // /// # Panics
    // /// Panics if the size of the vector is not 2, 4, 6 or 8
    // ///
    // /// # Calls
    // /// * `dot_f32`
    // pub fn norm_sq_f32<const T: usize>(a: &[f32]) -> f32 {
    //     dot_f32::<T>(a, a)
    // }
    //

}
