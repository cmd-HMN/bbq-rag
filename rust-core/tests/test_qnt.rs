// tests for quantization

use maxsimd::quantization::{QParmas, qf32_i8_d128, sq128x32_sq8};

// helper templete for qnt function
fn ht_q128x32_i8(value: &[f32; 128]) {
    let mut sdst = [0i8; QParmas::BASE_DIMS];

    sq128x32_sq8(value, &mut sdst);

    let qblock = qf32_i8_d128(value);
    let qdst: [i8; QParmas::BASE_DIMS] = qblock.to_array().unwrap();

    for i in 0..QParmas::BASE_DIMS {
        assert_eq!(
            sdst[i], qdst[i],
            "Destination Mismatch {} {}",
            sdst[i], qdst[i]
        );
    }
}

fn test_qf32_to_qi8_d128_correctness() {
    let input = [0.0f32; 128];
    ht_q128x32_i8(&input);
}

fn test_qf32_to_qi8_d128_rad() {
    use rand::{Rng, thread_rng};
    let mut rng = thread_rng();
    let mut input = [0.0f32; 128];
    for _ in 0..100 {
        for i in 0..128 {
            input[i] = rng.gen_range(-1.0..1.0);
        }
        ht_q128x32_i8(&input);
    }
}

fn test_qf32_to_qi8_d128_type_mismatch_error() {
    let input = [1.0f32; QParmas::BASE_DIMS];
    let qblock = qf32_i8_d128(&input);

    // SHOULD PANIC
    // Asking to a big heart
    let _: [f32; QParmas::BASE_DIMS] = qblock.to_array().unwrap();

    // // println!("res: {}", res.is_err());
    // // Damn stone heart
    // assert!(
    //     res.is_err(),
    //     "Must return Err when requesting Float32 from an Int8 QBlock!"
    // );
    // assert_eq!(
    //     res.unwrap_err(),
    //     "You are initializing with wrong datatype, its not Float32"
    // );
}

fn test_qf32_to_qi8_d128_independent_block_scales() {
    let mut input = [0.0f32; QParmas::BASE_DIMS];

    for i in 0..32 {
        input[i] = 100.0;
    }
    for i in 32..64 {
        input[i] = 0.01;
    }
    for i in 64..96 {
        input[i] = 0.0;
    }
    for i in 96..128 {
        input[i] = -0.01;
    }

    ht_q128x32_i8(&input);
}

fn test_qf32_to_qi8_d128_sequence() {
    let mut input = [0.0f32; QParmas::BASE_DIMS];
    for i in 0..QParmas::BASE_DIMS {
        input[i] = (i as f32) * 0.5;
    }
    ht_q128x32_i8(&input);
}

// check the largest values are clamp properly
fn test_qf32_to_qi8_d128_saturation(){
    let mut input = [0.0f32; QParmas::BASE_DIMS];
    use rand::{Rng, thread_rng};
    let mut rng = thread_rng();

    for i in 0..QParmas::BASE_DIMS {
        input[i] = rng.gen_range(-4242.0..424200.0);
    }
    ht_q128x32_i8(&input);
}

fn test_qf32_to_qi8_d128_type_near_zero(){
    let mut input = [0.0f32; QParmas::BASE_DIMS];
    use rand::{Rng, thread_rng};
    let mut rng = thread_rng();

    for i in 0..QParmas::BASE_DIMS {
        input[i] = rng.gen_range(1e-15..1e-10);
    }
    ht_q128x32_i8(&input);
}

fn test_qf32_to_qi8_d128_edge_of_hell(){
    let mut input = [0.0f32; QParmas::BASE_DIMS];
    use rand::{Rng, thread_rng};
    let mut rng = thread_rng();

    for i in 0..QParmas::BASE_DIMS {
        input[i] = rng.gen_range(-127.0..127.0);
    }
    ht_q128x32_i8(&input);
}

// q here is for quantization
test_me! {
    group quantization {
        qcorrectness: test_qf32_to_qi8_d128_correctness,
        qrandom_fuzz: test_qf32_to_qi8_d128_rad,
        qscales:      test_qf32_to_qi8_d128_independent_block_scales,
        qsequence:    test_qf32_to_qi8_d128_sequence,
        #[should_panic]
        qmismatch:    test_qf32_to_qi8_d128_type_mismatch_error,
        qsaturation:  test_qf32_to_qi8_d128_saturation,
        qnear_zero:   test_qf32_to_qi8_d128_type_near_zero,
        qedge_of_hell: test_qf32_to_qi8_d128_edge_of_hell
    }
}
