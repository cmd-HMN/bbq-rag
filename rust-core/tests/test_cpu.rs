/// FOR CPU
use maxsimd::cpu::{
    fused_dot_max_dim128_avx2, fused_dot_max_generic_avx2, naive_maxsim_dim128, reference_maxsim,
    max_avx2,
};

fn generate_data(len: usize, dim: usize) -> Vec<f32> {
    use rand::{Rng, thread_rng};
    let mut rng = thread_rng();
    (0..(len * dim)).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn assert_approx_eq(a: f32, b: f32) {
    assert!(
        (a - b).abs() < 1e-4,
        "The values are not equal a: {}, b: {}",
        a,
        b
    );
}

fn test_fused_dot_max_dim128_correctnes() {
    let q_len = 10;
    let d_len = 25;
    let dim = 128;
    let q: Vec<f32> = (0..q_len * dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let d: Vec<f32> = (0..d_len * dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.1)
        .collect();

    let expected = reference_maxsim(&q, &d, q_len, d_len, dim);

    let got_128 = unsafe { fused_dot_max_dim128_avx2(&q, &d, q_len, d_len) };
    let got_generic = unsafe { fused_dot_max_generic_avx2(&q, &d, q_len, d_len, dim) };

    assert!(
        (got_128 - expected).abs() < 1e-4,
        "got_128: {}, expected: {}",
        got_128,
        expected
    );
    assert!(
        (got_generic - expected).abs() < 1e-4,
        "got_generic: {}, expected: {}",
        got_generic,
        expected
    );
}

fn test_simd_correctness_standard_batch() {
    let q_len = 32;
    let d_len = 100;
    let q = generate_data(q_len, 128);
    let d = generate_data(d_len, 128);

    let naive_score = naive_maxsim_dim128(&q, &d, q_len, d_len);
    let simd_score = unsafe { fused_dot_max_dim128_avx2(&q, &d, q_len, d_len) };
    assert_approx_eq(simd_score, naive_score);
}

fn test_simd_correctness_leftovers() {
    let q_len = 5;
    let d_len = 10;
    let q = generate_data(q_len, 128);
    let d = generate_data(d_len, 128);

    let naive_score = naive_maxsim_dim128(&q, &d, q_len, d_len);
    let simd_score = unsafe { fused_dot_max_dim128_avx2(&q, &d, q_len, d_len) };
    assert_approx_eq(simd_score, naive_score);
}

fn test_simd_correctness_empty() {
    let q: Vec<f32> = vec![];
    let d: Vec<f32> = vec![];
    let naive_score = naive_maxsim_dim128(&q, &d, 0, 0);
    let simd_score = unsafe { fused_dot_max_dim128_avx2(&q, &d, 0, 0) };
    assert_eq!(simd_score, 0.0);
    assert_eq!(naive_score, 0.0);
}

fn test_fused_dot_max_arbitrary_dim() {
    for dim in [1, 7, 8, 15, 16, 64, 130] {
        let q_len = 5;
        let d_len = 12;
        let q: Vec<f32> = (0..q_len * dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let d: Vec<f32> = (0..d_len * dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();

        let expected = reference_maxsim(&q, &d, q_len, d_len, dim);
        let got = unsafe { fused_dot_max_generic_avx2(&q, &d, q_len, d_len, dim) };
        assert!(
            (got - expected).abs() < 1e-4,
            "dim: {}, got: {}, expected: {}",
            dim,
            got,
            expected
        );
    }
}

fn test_max_avx2_slice_boundaries() {
    for len in 0..=40 {
        let v: Vec<f32> = (0..len).map(|i| (i as f32) * 1.5 - 10.0).collect();
        let naive = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let simd = max_avx2(&v);
        assert_eq!(simd, naive, "Failed on length {}", len);
    }
}

fn test_max_avx2_nans_and_infinities() {
    assert_eq!(max_avx2(&[f32::NEG_INFINITY, 1.0, f32::INFINITY]), f32::INFINITY);
    assert_eq!(max_avx2(&[f32::NEG_INFINITY; 50]), f32::NEG_INFINITY);

    let mixed = [f32::NAN, 1.0, 2.0];
    let expected = mixed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(max_avx2(&mixed), expected);
}

fn test_max_avx2_all_nans(){
    assert!(max_avx2(&[f32::NAN; 128]).is_nan());
}

test_me! {
    group cpu {
        correctness:         test_fused_dot_max_dim128_correctnes,
        standard_batch:      test_simd_correctness_standard_batch,
        leftovers:           test_simd_correctness_leftovers,
        empty:               test_simd_correctness_empty,
        arbitrary_dim:       test_fused_dot_max_arbitrary_dim,
        max_slice_lengths:   test_max_avx2_slice_boundaries,
        max_nans_infinities: test_max_avx2_nans_and_infinities,
        max_all_nans:        test_max_avx2_all_nans
    }
}
