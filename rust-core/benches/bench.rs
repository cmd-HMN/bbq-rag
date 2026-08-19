use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use maxsimd::cpu::vec256::simd::fused_dot_max_dim128_avx2;
use maxsimd::func::function::internal::pro_sgl_doc_msgemm;

macro_rules! single_func_benchmark {
    // with throughput
     ($grp: expr, throughput => $thput: expr, $name: expr, $func: expr $(, $args:expr)*) => {{
        $grp.throughput($thput);
        $grp.bench_function($name, |b| {
            b.iter(|| {
                black_box($func($(black_box($args)),*))
            })
        });
    }};

    // without throughput
    ($grp: expr, $name: expr, $func: expr $(, $args:expr)*) => {{
        $grp.bench_function($name, |b| {
            b.iter(|| {
                black_box($func($(black_box($args)),*))
            })
        });
    }};
}

fn bench_pro_sgl_doc(c: &mut Criterion) {
    let mut group = c.benchmark_group("Level 1 Functions");

    let dim = 128;
    let q_len = 32;
    let d_len = 256;

    let q_data: Vec<f32> = vec![0.5; q_len * dim];
    let d_data: Vec<f32> = vec![0.2; d_len * dim];

    unsafe {
        single_func_benchmark!(
            group,
            throughput => Throughput::Elements(dim as u64 * q_len as u64 * d_len as u64),
            "fused_dot_max_dim128_avx2",
            fused_dot_max_dim128_avx2,
            &q_data,
            &d_data,
            q_len,
            d_len
        );
    }
    single_func_benchmark!(
        group,
        throughput => Throughput::Elements(dim as u64 * q_len as u64 * d_len as u64),
        "pro_sgl_doc_msgemm",
        pro_sgl_doc_msgemm,
        &q_data,
        &d_data,
        q_len,
        d_len,
        dim
    );
    group.finish();
}

criterion_group!(benches, bench_pro_sgl_doc);
criterion_main!(benches);
