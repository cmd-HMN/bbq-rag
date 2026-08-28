///! THis if for blas testing
/// Will only run these test if the tag is present so that is not the testing
/// Earlier used for testing the blas functions
/// To run these test to cargo tests --features dev
use maxsimd::blas::{pro_sgl_doc as psd, maxsim_fused_doc_tiles};

fn pro_sgl_doc(_q: &[f32], _d: &[f32], _q_len: usize, _d_len: usize, _dim: usize) -> f32 {
    let mut score = 0.0f32;
    for qi in 0.._q_len {
        let mut _max = f32::NEG_INFINITY;
        for di in 0.._d_len {
            let mut _sum = 0.0f32;
            for d in 0.._dim {
                _sum += _q[qi * _dim + d] * _d[di * _dim + d];
            }
            _max = _max.max(_sum);
        }
        score += _max;
    }

    score
}

fn reference_maxsim_fused_doc_tiles(
    _q: &[f32],
    _d: &[f32],
    _q_len: usize,
    _d_len: usize,
    _dim: usize,
) -> Vec<f32> {
    let n_docs = _d.len() / (_d_len * _dim);
    let mut results = Vec::with_capacity(n_docs);
    for doc_idx in 0..n_docs {
        let doc_start = doc_idx * _d_len * _dim;
        let current_doc = &_d[doc_start..doc_start + _d_len * _dim];

        let mut score = 0.0f32;
        for qi in 0.._q_len {
            let mut max_val = f32::NEG_INFINITY;
            for di in 0.._d_len {
                let mut sum = 0.0f32;
                for d in 0.._dim {
                    sum += _q[qi * _dim + d] * current_doc[di * _dim + d];
                }
                max_val = max_val.max(sum);
            }
            score += max_val;
        }
        results.push(score);
    }

    results
}

// fn generate_query(q_len: usize, dim: usize) -> Vec<f32> {
//     vec![0.5; q_len * dim]
// }

fn assert_pro_sgl_doc_eq(_q: &[f32], _d: &[f32], _q_len: usize, _d_len: usize, _dim: usize) {
    let expected = pro_sgl_doc(_q, _d, _q_len, _d_len, _dim);
    let got = psd(_q, _d, _q_len, _d_len, _dim);
    let diff = (got - expected).abs();
    assert!(
        diff < 1e-4 * expected.abs().max(1.0),
        "maxsim mismatch for input"
    );
}

fn assert_maxsim_fused_doc_tiles_eq(
    _q: &[f32],
    _d: &[f32],
    _q_len: usize,
    _d_len: usize,
    _dim: usize,
) {
    let expected = reference_maxsim_fused_doc_tiles(_q, _d, _q_len, _d_len, _dim);

    let got = maxsim_fused_doc_tiles(_q, _d, _q_len, _d_len, _dim);

    assert_eq!(expected.len(), got.len(), "Document output count mismatch");

    for i in 0..expected.len() {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff < 1e-4 * expected[i].abs().max(1.0),
            "MaxSim mismatch at doc {}! expected: {}, got: {}",
            i,
            expected[i],
            got[i]
        );
    }
}

fn test_func_prosgldoc_singletokensingledoc() {
    let q = vec![1.0, 2.0, 3.0, 4.0];
    let d = vec![1.0, 1.0, 1.0, 1.0];
    assert_pro_sgl_doc_eq(&q, &d, 1, 1, 4);
}

fn test_func_prosgldoc_multiplequerytokens() {
    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let d = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
    assert_pro_sgl_doc_eq(&q, &d, 2, 3, 4);
}

fn test_func_prosgldoc_orthogonalvectors() {
    let q = vec![
        1.0, 0.0, // q0
        0.0, 1.0, // q1
    ];
    let d = vec![
        1.0, 0.0, // d0
        0.0, 1.0, // d1
    ];
    assert_pro_sgl_doc_eq(&q, &d, 2, 2, 2);
}

fn test_func_prosgldoc_zeros() {
    let q = vec![0.0; 32];
    let d = vec![0.0; 48];
    assert_pro_sgl_doc_eq(&q, &d, 4, 6, 8);
}

fn test_func_prosgldoc_values() {
    let q = vec![-1.0, 2.0, -3.0, 4.0];
    let d = vec![1.0, -2.0, 3.0, -4.0];
    assert_pro_sgl_doc_eq(&q, &d, 1, 1, 4);
}

fn test_func_prosgldoc_variouslengths() {
    for q_len in 1..=10 {
        for d_len in 1..=10 {
            for dim in [1, 2, 4, 7, 8, 16, 31, 32, 33] {
                let q: Vec<f32> = (0..q_len * dim).map(|i| (i as f32).sin()).collect();
                let d: Vec<f32> = (0..d_len * dim).map(|i| (i as f32).cos()).collect();
                assert_pro_sgl_doc_eq(&q, &d, q_len, d_len, dim);
            }
        }
    }
}

fn test_func_prosgldoc_comprehensivesizes() {
    for q_len in [1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 99] {
        for d_len in [1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 127, 128, 199] {
            for dim in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 64, 127] {
                let q: Vec<f32> = (0..q_len * dim).map(|i| (i as f32).sin()).collect();
                let d: Vec<f32> = (0..d_len * dim).map(|i| (i as f32).cos()).collect();
                assert_pro_sgl_doc_eq(&q, &d, q_len, d_len, dim);
            }
        }
    }
}

fn test_func_prosgldoc_dimnotmultipleof8() {
    let q = vec![1.0, 2.0, 3.0];
    let d = vec![4.0, 5.0, 6.0];
    assert_pro_sgl_doc_eq(&q, &d, 1, 1, 3);
}

fn test_func_prosgldoc_longerthanquery() {
    let q = vec![1.0, 0.0];
    let d = vec![0.0, 1.0, 1.0, 0.0, 0.5, 0.5];
    assert_pro_sgl_doc_eq(&q, &d, 1, 3, 2);
}

fn test_func_prosgldoc_longerthandoc() {
    let q = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
    let d = vec![1.0, 0.0];
    assert_pro_sgl_doc_eq(&q, &d, 3, 1, 2);
}

fn test_func_maxfusedtiles_doctilespseudorandom() {
    let q_len = 4;
    let d_len = 16;
    let dim = 32;
    let n_docs = 5;

    let q: Vec<f32> = (0..q_len * dim).map(|x| (x as f32 % 7.0) - 3.5).collect();

    let d: Vec<f32> = (0..n_docs * d_len * dim)
        .map(|x| (x as f32 % 11.0) - 5.5)
        .collect();

    assert_maxsim_fused_doc_tiles_eq(&q, &d, q_len, d_len, dim);
}

fn test_func_maxfusedtiles_doctileszeros() {
    let q_len = 2;
    let d_len = 8;
    let dim = 16;
    let n_docs = 3;

    let q = vec![0.0; q_len * dim];
    let d = vec![0.0; n_docs * d_len * dim];

    assert_maxsim_fused_doc_tiles_eq(&q, &d, q_len, d_len, dim);
}

fn test_func_maxfusedtiles_doctileslargebatch() {
    let q_len = 3;
    let d_len = 512;
    let dim = 16;
    let n_docs = 200;

    let q: Vec<f32> = (0..q_len * dim).map(|x| x as f32 % 3.0).collect();
    let d: Vec<f32> = (0..n_docs * d_len * dim).map(|x| x as f32 % 2.0).collect();

    assert_maxsim_fused_doc_tiles_eq(&q, &d, q_len, d_len, dim);
}

test_me! {
    group blas {
        sgl_single_token: test_func_prosgldoc_singletokensingledoc,
        sgl_multiple_tokens: test_func_prosgldoc_multiplequerytokens,
        sgl_orthogonal_vectors: test_func_prosgldoc_orthogonalvectors,
        sgl_zeros: test_func_prosgldoc_zeros,
        sgl_values: test_func_prosgldoc_values,
        sgl_various_lengths: test_func_prosgldoc_variouslengths,
        sgl_comprehensive_sizes: test_func_prosgldoc_comprehensivesizes,
        sgl_dim_not_multiple_of_8: test_func_prosgldoc_dimnotmultipleof8,
        sgl_longer_than_query: test_func_prosgldoc_longerthanquery,
        sgl_longer_than_doc: test_func_prosgldoc_longerthandoc,
        doctiles_pseudorandom: test_func_maxfusedtiles_doctilespseudorandom,
        doctiles_zeros: test_func_maxfusedtiles_doctileszeros,
        doctiles_large_batch: test_func_maxfusedtiles_doctileslargebatch,
    }
}
