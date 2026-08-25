//! Quantization
//! This file contain quantization logic only for 128 dim vectors
//! Using llama.cpp block apporch

use super::blocks::*;

/// qf32_to_qi8
///
/// # Arguments
/// 
/// * `values` - Vector of f32 values
///
/// # Returns
/// 
/// * `QData`
///
/// Inspired by FBGEMM
#[cfg(target_feature = "avx2")]
#[inline(always)]
fn qf32_to_qi8(values: Vec<f32>, dst: &mut Vec<i8>) -> QData {
    const LEN: u8 = 8;
    print!("qf32_to_qi8");
    println!("LEN {}", LEN);
    QData::default()
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_logic() {
        let value: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst: Vec<i8> = vec![];
        
        qf32_to_qi8(value, &mut dst);
        //
        // // Tells the compiler to crash the test if dst is empty
        // assert!(!dst.is_empty()); 
    }
}
