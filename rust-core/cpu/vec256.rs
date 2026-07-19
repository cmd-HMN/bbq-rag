// Max simd vectorization
// - avx2 (Intel x86_64)

mod simd {
    use core::f32;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    pub fn load_vec256(data: &Vec<f32>, offset: usize) -> __m256 {
        unsafe { _mm256_loadu_ps(data.as_ptr().add(offset)) }
    }

    #[inline]
    pub fn horizontal_sum(a: __m256) -> f32 {
        unsafe {
            let _high = _mm256_extractf128_ps(a, 1);
            let _low = _mm256_castps256_ps128(a);
            let sum128 = _mm_add_ps(_low, _high);

            let sh1 = _mm_shuffle_ps(sum128, sum128, 0b01_00_11_10);
            let sum64 = _mm_add_ps(sum128, sh1);

            let sh2 = _mm_shuffle_ps(sum64, sum64, 0b10_11_00_01);
            let sum32 = _mm_add_ps(sum64, sh2);
            _mm_cvtss_f32(sum32)
        }
    }

    // f32
    pub fn dot_f32_2acc(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        let _len = a.len();
        let mut i = 0;
        unsafe {
            let mut _acc0 = _mm256_setzero_ps();
            let mut _acc1 = _mm256_setzero_ps();

            while i + 16 <= _len {
                // only using 2 _acc register
                let _a0 = load_vec256(a, i);
                let _b0 = load_vec256(b, i);

                let _a1 = load_vec256(a, i + 8);
                let _b1 = load_vec256(b, i + 8);

                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(_a0, _b0));
                _acc1 = _mm256_add_ps(_acc1, _mm256_mul_ps(_a1, _b1));
                i += 16;
            }

            // left overs
            while i + 8 < _len {
                _acc0 = _mm256_add_ps(_acc0, _mm256_mul_ps(load_vec256(a, i), load_vec256(b, i)));
                i += 8;
            }

            let _acc = _mm256_add_ps(_acc0, _acc1);
            let _vector = horizontal_sum(_acc);

            let mut _scaler = 0.0;
            while i < _len {
                _scaler += a[i] * b[i];
                i += 1;
            }
            // return
            _scaler + _vector
        }
    }
}

mod scalar {
    pub fn dot_f32(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(c, d)| c * d).sum()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f32_matches_scalar() {
        let _lenghts = [
            1, 7, 8, 9, 15, 16, 17, 24, 100, 1005
        ];
        for len in _lenghts {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 13.0 % 7.0) - 3.5).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 14.0 % 8.0) - 1.5).collect();

            let expected = scalar::dot_f32(&a, &b);
            let actual = simd::dot_f32_2acc(&a, &b);

            assert!((expected - actual).abs() < 1e-2);
        }
    }

    #[test]
    fn test_horizontal_sum() {
        if is_x86_feature_detected!("avx2") {
            use std::arch::x86_64::_mm256_loadu_ps;
            unsafe {
                let data: [f32; 8] = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
                let vec = _mm256_loadu_ps(data.as_ptr());
                
                let expected: f32 = data.iter().sum();
                let actual = simd::horizontal_sum(vec);

                assert!((expected - actual).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_load_vec256() {
        if is_x86_feature_detected!("avx2") {
            use std::arch::x86_64::_mm256_storeu_ps;
            unsafe {
                let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
                let vec = simd::load_vec256(&data, 0);

                let mut out = [0.0; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), vec);

                for i in 0..8 {
                    assert_eq!(data[i], out[i]);
                }
            }
        }
    }
}
