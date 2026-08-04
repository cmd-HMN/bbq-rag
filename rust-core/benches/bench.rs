use criterion::{black_box, criterion_group, criterion_main, Criterion};
use maxsimd::func::function::internal::{pro_sgl_doc_csgemm, pro_sgl_doc_msgemm};

fn bench_pro_sgl_doc(c: &mut Criterion) {
    let mut group = c.benchmark_group("SGEMM Backends (Single Doc)");

    let dim = 128;       
    let q_len = 32;      
    let d_len = 256;     

    let q_data: Vec<f32> = vec![0.5; q_len * dim];
    let d_data: Vec<f32> = vec![0.2; d_len * dim];

    group.bench_function("Custom SGEMM", |b| {
        b.iter(|| {
            pro_sgl_doc_csgemm(
                black_box(&q_data),
                black_box(&d_data),
                black_box(q_len),
                black_box(d_len),
                black_box(dim),
            )
        })
    });

    #[cfg(feature = "mkl")]
    group.bench_function("MKL SGEMM", |b| {
        b.iter(|| {
            pro_sgl_doc_msgemm(
                black_box(&q_data),
                black_box(&d_data),
                black_box(q_len),
                black_box(d_len),
                black_box(dim),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_pro_sgl_doc);
criterion_main!(benches);
