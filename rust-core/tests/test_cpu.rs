/// FOR CPU
use maxsimd::cpu::{
  dotmax128_f32, dotmaxg_f32, dotmaxtg_f32, dotmaxg_i8, ref_maxsim_f32, ref_maxsimd128_f32, ref_maxsim_i8
};

use rand::random_range;

fn generate_data(len: usize, dim: usize) -> Vec<f32> {
    (0..(len * dim)).map(|_| random_range(-1.0..=1.0)).collect()
}

fn assert_approx_eq(name: &str, a: f32, b: f32) {
    assert!(
        (a - b).abs() < 1e-4,
        "{} --> The values are not equal a: {}, b: {}",
        name,
        a,
        b
    );
}

fn test_dot_max_128_correctnes() {
    let q_len = 10;
    let d_len = 25;
    let dim = 128;
    let q: Vec<f32> = (0..q_len * dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let d: Vec<f32> = (0..d_len * dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.1)
        .collect();

    let expected = ref_maxsim_f32(&q, &d, q_len, d_len, dim);

    let got_128 = unsafe { dotmax128_f32(&q, &d, q_len, d_len) };
    let got_generic = unsafe { dotmaxg_f32(&q, &d, q_len, d_len, dim) };
    let got_tg = unsafe { dotmaxtg_f32(&q, &d, q_len, d_len, 128) };

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

    assert!(
        (got_tg - expected).abs() < 1e-4,
        "got_tg: {}, expected: {}",
        got_tg,
        expected
    );
}

fn test_simd_correctness_standard_batch() {
    let q_len = 32;
    let d_len = 100;
    let q = generate_data(q_len, 128);
    let d = generate_data(d_len, 128);

    let naive_score = ref_maxsimd128_f32(&q, &d, q_len, d_len);
    let ss_128 = unsafe {dotmax128_f32(&q, &d, q_len, d_len) };
    let ss_t128 = unsafe { dotmaxtg_f32(&q, &d, q_len, d_len, 128) };
    let ss_g128 = unsafe { dotmaxg_f32(&q, &d, q_len, d_len, 128) };


    assert_approx_eq("ss_128", ss_128, naive_score);
    assert_approx_eq("ss_t128", ss_t128, naive_score);
    assert_approx_eq("ss_g128", ss_g128, naive_score);
}

fn test_simd_correctness_leftovers() {
    let q_len = 5;
    let d_len = 10;
    let q = generate_data(q_len, 128);
    let d = generate_data(d_len, 128);

    let naive_score = ref_maxsimd128_f32(&q, &d, q_len, d_len);
    let ss_128 = unsafe { dotmax128_f32(&q, &d, q_len, d_len) };
    let ss_t128 = unsafe { dotmaxtg_f32(&q, &d, q_len, d_len, 128) };
    let ss_g128 = unsafe { dotmaxg_f32(&q, &d, q_len, d_len, 128) };

    assert_approx_eq("ss_128", ss_128, naive_score);
    assert_approx_eq("ss_g128", ss_g128, naive_score);
    assert_approx_eq("ss_t128", ss_t128, naive_score);
}

fn test_simd_correctness_empty() {
    let q: Vec<f32> = vec![];
    let d: Vec<f32> = vec![];
    let naive_score = ref_maxsimd128_f32(&q, &d, 0, 0);
    let ss_128 = unsafe { dotmax128_f32(&q, &d, 0, 0) };
    let ss_t128 = unsafe { dotmaxtg_f32(&q, &d, 0, 0, 128) };
    let ss_g128 = unsafe { dotmaxg_f32(&q, &d, 0, 0, 128) };

    assert_eq!(ss_128, 0.0);
    assert_eq!(naive_score, 0.0);
    assert_eq!(ss_t128, 0.0);
    assert_eq!(ss_g128, 0.0);
}

fn test_dot_max_arbitrary_dim() {
    for dim in [1, 7, 8, 15, 16, 64, 130] {
        let q_len = 5;
        let d_len = 12;
        let q: Vec<f32> = (0..q_len * dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let d: Vec<f32> = (0..d_len * dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();

        let expected = ref_maxsim_f32(&q, &d, q_len, d_len, dim);
        let ss_g128 = unsafe { dotmaxg_f32(&q, &d, q_len, d_len, dim) };
        let ss_t128 = unsafe { dotmaxtg_f32(&q, &d, q_len, d_len, dim) };


        assert!(
            (ss_g128 - expected).abs() < 1e-4,
            "dim: {}, got: {}, expected: {}",
            dim,
            ss_g128,
            expected
        );
        assert!(
            (ss_t128 - expected).abs() < 1e-4,
            "dim: {}, ss_t128: {}, expected: {}",
            dim,
            ss_t128,
            expected
        );
    }
}

fn test_i8_simd_standard_batch() {
    let q_len = 8;
    let d_len = 16;
    let dim = 128;
    let num_blocks = dim / 32;

    let q: Vec<i8> = (0..q_len * dim).map(|i| ((i % 255) as i16 - 128) as i8).collect();
    let d: Vec<i8> = (0..d_len * dim).map(|i| (((i * 7) % 255) as i16 - 128) as i8).collect();

    let qs: Vec<f32> = (0..q_len * num_blocks).map(|i| 0.01 + ((i % 5) as f32) * 0.002).collect();
    let ds: Vec<f32> = (0..d_len * num_blocks).map(|i| 0.01 + ((i % 7) as f32) * 0.003).collect();

    let expected = ref_maxsim_i8(&q, &d, &qs, &ds, q_len, d_len, dim);
    let got = unsafe { dotmaxg_i8(&q, &d, &qs, &ds, q_len, d_len) };

    assert_approx_eq("i8_standard_batch", got, expected);
}

fn test_i8_simd_leftovers() {
    let q_len = 7;
    let d_len = 11;
    let dim = 128;
    let num_blocks = dim / 32;

    let q: Vec<i8> = (0..q_len * dim).map(|i| ((i % 251) as i16 - 120) as i8).collect();
    let d: Vec<i8> = (0..d_len * dim).map(|i| (((i * 13) % 251) as i16 - 120) as i8).collect();

    let qs: Vec<f32> = (0..q_len * num_blocks).map(|i| 0.005 + ((i % 4) as f32) * 0.001).collect();
    let ds: Vec<f32> = (0..d_len * num_blocks).map(|i| 0.005 + ((i % 6) as f32) * 0.001).collect();

    let expected = ref_maxsim_i8(&q, &d, &qs, &ds, q_len, d_len, dim);
    let got = unsafe { dotmaxg_i8(&q, &d, &qs, &ds, q_len, d_len) };

    assert_approx_eq("i8_leftovers", got, expected);
}

fn test_i8_simd_empty() {
    let q: Vec<i8> = vec![];
    let d: Vec<i8> = vec![];
    let qs: Vec<f32> = vec![];
    let ds: Vec<f32> = vec![];

    let got = unsafe { dotmaxg_i8(&q, &d, &qs, &ds, 0, 0) };
    assert_eq!(got, 0.0);
}

test_me! {
    group cpu {
        f32_correctness:        test_dot_max_128_correctnes,
        f32_standard_batch:     test_simd_correctness_standard_batch,
        f32_leftovers:          test_simd_correctness_leftovers,
        f32_empty:              test_simd_correctness_empty,
        f32_arbitrary_dim:      test_dot_max_arbitrary_dim,
        i8_standard:            test_i8_simd_standard_batch,
        i8_leftovers:           test_i8_simd_leftovers,
        i8_empty:               test_i8_simd_empty,
    }
}
