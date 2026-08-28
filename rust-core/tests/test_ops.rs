///! This file contains tests for the ops module

use maxsimd::ops::{omaxsim_variable_length, omaxsim_variable_length_slice};

fn generate_query(q_len: usize, dim: usize) -> Vec<f32> {
    vec![0.5; q_len * dim]
}

fn test_func_maxsimvariablelength_singleandpaddedbatches() {
    let dim = 128;
    let q_len = 10;
    let q = generate_query(q_len, dim);

    let doc0 = vec![1.0; 10 * dim]; // Length 10
    let doc1 = vec![1.0; 11 * dim]; // Length 11 (Within 20% of 10, will batch with doc0)
    let doc2 = vec![1.0; 30 * dim]; // Length 30 (Way larger, will be processed as a single doc)

    let d = vec![(0, 10, doc0), (1, 11, doc1), (2, 30, doc2)];

    // Run the function
    let results = omaxsim_variable_length(q.clone(), d, q_len, dim);

    assert_eq!(results.len(), 3);
}

fn test_func_maxsimvariablelength_perfectmatchlargebatch() {
    let dim = 128;
    let q_len = 10;
    let q = generate_query(q_len, dim);

    let doc_length = 15;
    let backing_data: Vec<Vec<f32>> = (0..35).map(|_| vec![1.0; doc_length * dim]).collect();

    let mut d = Vec::new();
    for (i, data) in backing_data.into_iter().enumerate(){
        d.push((i, doc_length, data));
    }

    let results = omaxsim_variable_length(q, d, q_len, dim);

    assert_eq!(results.len(), 35);
}

fn test_func_maxsimvariablelength_fastpathglobalbatching() {
    let dim = 128;
    let q_len = 10;
    let q = generate_query(q_len, dim);

    let min_len = 20;
    let max_len = 23; // 23 / 20 = 1.15 (which is <= 1.2)

    let mut backing_data = Vec::new();
    for i in 0..55 {
        let len = if i % 2 == 0 { min_len } else { max_len };
        backing_data.push(vec![1.0; len * dim]);
    }

    let mut d = Vec::new();
    for (i, data) in backing_data.into_iter().enumerate()  {
        let len = if i % 2 == 0 { min_len } else { max_len };
        d.push((i, len, data));
    }

    let results = omaxsim_variable_length(q, d, q_len, dim);

    assert_eq!(results.len(), 55);
}

fn test_func_maxsimvariablelength_emptydocumentlist() {
    let dim = 128;
    let q_len = 10;
    let q = generate_query(q_len, dim);
    let d: Vec<(usize, usize, Vec<f32>)> = vec![];

    let results = omaxsim_variable_length(q, d, q_len, dim);

    assert_eq!(results.len(), 0);
}

test_me!{
    group ops {
        maxsim_variable_length_single_and_padded_batches: test_func_maxsimvariablelength_singleandpaddedbatches,
        maxsim_variable_length_perfect_match_large_batch: test_func_maxsimvariablelength_perfectmatchlargebatch,
        maxsim_variable_length_fast_path_global_batching: test_func_maxsimvariablelength_fastpathglobalbatching,
        maxsim_variable_length_empty_document_list: test_func_maxsimvariablelength_emptydocumentlist
    }
}
