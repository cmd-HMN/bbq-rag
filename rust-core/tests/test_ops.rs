///! This file contains tests for the ops module

use maxsimd::ops::{maxsim, DocLayout};
use maxsimd::quantization::QTYPE;

fn generate_query(q_len: usize, dim: usize) -> Vec<f32> {
    vec![0.5; q_len * dim]
}

fn test_func_maxsimvariablelength_singleandpaddedbatches() {
    let dim = 128;
    let q_len = 10;
    let q = generate_query(q_len, dim);

    let doc0 = vec![1.0; 10 * dim]; // Length 10
    let doc1 = vec![1.0; 11 * dim]; // Length 11
    let doc2 = vec![1.0; 30 * dim]; // Length 30

    let mut d_flat = Vec::new();
    d_flat.extend_from_slice(&doc0);
    d_flat.extend_from_slice(&doc1);
    d_flat.extend_from_slice(&doc2);
    let doc_lengths = [10, 11, 30];

    let results = unsafe {
        maxsim(
            q.as_ptr() as usize,
            d_flat.as_ptr() as usize,
            q_len,
            0,
            0,
            dim,
            DocLayout::Flat {
                d_len: &doc_lengths,
            },
            QTYPE::Float32,
            -1,
        )
    };

    assert_eq!(results.len(), 3);
}

fn test_func_maxsimvariablelength_perfectmatchlargebatch() {
    let dim = 128;
    let q_len = 10;
    let q = generate_query(q_len, dim);

    let doc_length = 15;
    let n_docs = 35;
    let d_flat = vec![1.0; n_docs * doc_length * dim];
    let doc_lengths = vec![doc_length; n_docs];

    let results = unsafe {
        maxsim(
            q.as_ptr() as usize,
            d_flat.as_ptr() as usize,
            q_len,
            0,
            0,
            dim,
            DocLayout::Flat {
                d_len: &doc_lengths,
            },
            QTYPE::Float32,
            -1,
        )
    };

    assert_eq!(results.len(), 35);
}

fn test_func_maxsimvariablelength_fastpathglobalbatching() {
    let dim = 128;
    let q_len = 10;
    let q = generate_query(q_len, dim);

    let min_len = 20;
    let max_len = 23;

    let mut doc_lengths = Vec::new();
    let mut total_tokens = 0;
    for i in 0..55 {
        let len = if i % 2 == 0 { min_len } else { max_len };
        doc_lengths.push(len);
        total_tokens += len;
    }
    let d_flat = vec![1.0; total_tokens * dim];

    let results = unsafe {
        maxsim(
            q.as_ptr() as usize,
            d_flat.as_ptr() as usize,
            q_len,
            0,
            0,
            dim,
            DocLayout::Flat {
                d_len: &doc_lengths,
            },
            QTYPE::Float32,
            -1,
        )
    };

    assert_eq!(results.len(), 55);
}

fn test_func_maxsimvariablelength_emptydocumentlist() {
    let dim = 128;
    let q_len = 10;
    let q = generate_query(q_len, dim);
    let d_flat: Vec<f32> = vec![];
    let doc_lengths: Vec<usize> = vec![];

    let results = unsafe {
        maxsim(
            q.as_ptr() as usize,
            d_flat.as_ptr() as usize,
            q_len,
            0,
            0,
            dim,
            DocLayout::Flat {
                d_len: &doc_lengths,
            },
            QTYPE::Float32,
            -1,
        )
    };

    assert_eq!(results.len(), 0);
}

fn test_func_maxsim_f32_all_layouts() {
    let dim = 128;
    let q_len = 4;
    let q = vec![0.5f32; q_len * dim];

    let d_single = vec![0.2f32; 8 * dim];
    let score_single = unsafe {
        maxsim(
            q.as_ptr() as usize,
            d_single.as_ptr() as usize,
            q_len,
            0,
            0,
            dim,
            DocLayout::Single { doc_tokens: 8 },
            QTYPE::Float32,
            1,
        )
    };
    assert_eq!(score_single.len(), 1);

    let d_batch = vec![0.2f32; 3 * 8 * dim];
    for jobs in [1, -1, 2] {
        let scores = unsafe {
            maxsim(
                q.as_ptr() as usize,
                d_batch.as_ptr() as usize,
                q_len,
                0,
                0,
                dim,
                DocLayout::Batch {
                    docs: 3,
                    tokens: 8,
                },
                QTYPE::Float32,
                jobs,
            )
        };
        assert_eq!(scores.len(), 3);
        for s in scores {
            assert!((s - score_single[0]).abs() < 1e-5);
        }
    }

    let doc_lengths = vec![8, 8, 8];
    for jobs in [1, -1, 2] {
        let scores = unsafe {
            maxsim(
                q.as_ptr() as usize,
                d_batch.as_ptr() as usize,
                q_len,
                0,
                0,
                dim,
                DocLayout::Flat {
                    d_len: &doc_lengths,
                },
                QTYPE::Float32,
                jobs,
            )
        };
        assert_eq!(scores.len(), 3);
        for s in scores {
            assert!((s - score_single[0]).abs() < 1e-5);
        }
    }
}

fn test_func_maxsim_i8_all_layouts() {
    let dim = 128;
    let q_len = 4;
    let q = vec![10i8; q_len * dim];
    let num_blocks = dim / 32;
    let q_scales = vec![1.0f32; q_len * num_blocks];

    // Single doc
    let d_single = vec![5i8; 8 * dim];
    let d_single_scales = vec![1.0f32; 8 * num_blocks];

    let score_single = unsafe {
        maxsim(
            q.as_ptr() as usize,
            d_single.as_ptr() as usize,
            q_len,
            q_scales.as_ptr() as usize,
            d_single_scales.as_ptr() as usize,
            dim,
            DocLayout::Single { doc_tokens: 8 },
            QTYPE::Int8,
            1,
        )
    };
    assert_eq!(score_single.len(), 1);

    let d_batch = vec![5i8; 3 * 8 * dim];
    let d_batch_scales = vec![1.0f32; 3 * 8 * num_blocks];
    for jobs in [1, -1, 2] {
        let scores = unsafe {
            maxsim(
                q.as_ptr() as usize,
                d_batch.as_ptr() as usize,
                q_len,
                q_scales.as_ptr() as usize,
                d_batch_scales.as_ptr() as usize,
                dim,
                DocLayout::Batch {
                    docs: 3,
                    tokens: 8,
                },
                QTYPE::Int8,
                jobs,
            )
        };
        assert_eq!(scores.len(), 3);
        for s in scores {
            assert!((s - score_single[0]).abs() < 1e-5);
        }
    }

    let doc_lengths = vec![8, 8, 8];
    for jobs in [1, -1, 2] {
        let scores = unsafe {
            maxsim(
                q.as_ptr() as usize,
                d_batch.as_ptr() as usize,
                q_len,
                q_scales.as_ptr() as usize,
                d_batch_scales.as_ptr() as usize,
                dim,
                DocLayout::Flat {
                    d_len: &doc_lengths,
                },
                QTYPE::Int8,
                jobs,
            )
        };
        assert_eq!(scores.len(), 3);
        for s in scores {
            assert!((s - score_single[0]).abs() < 1e-5);
        }
    }
}

test_me!{
    group ops {
        maxsim_variable_length_single_and_padded_batches: test_func_maxsimvariablelength_singleandpaddedbatches,
        maxsim_variable_length_perfect_match_large_batch: test_func_maxsimvariablelength_perfectmatchlargebatch,
        maxsim_variable_length_fast_path_global_batching: test_func_maxsimvariablelength_fastpathglobalbatching,
        maxsim_variable_length_empty_document_list: test_func_maxsimvariablelength_emptydocumentlist,
        maxsim_f32_all_layouts: test_func_maxsim_f32_all_layouts,
        maxsim_i8_all_layouts: test_func_maxsim_i8_all_layouts
    }
}
