use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use pprof::criterion::{Output, PProfProfiler};
use maxsimd::cpu::vec256::simd::fused_dot_max_dim128_avx2;
use maxsimd::func::function::internal::pro_sgl_doc_msgemm;
use rand::Rng;

macro_rules! single_func_benchmark {
    (
        $grp: expr,
        throughput => $thput: expr,
        $name: expr,
        $func: expr,
        $q_len: expr,
        $d_len: expr,
        $dim: expr
        $(, $args:expr)*
    ) => {{
        $grp.throughput($thput);
        $grp.bench_function($name, |b| {
            b.iter_batched(
                || {
                    let mut rng = rand::thread_rng();
                    let q_data: Vec<f32> = (0..($q_len * $dim))
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect();
                    let d_data: Vec<f32> = (0..($d_len * $dim))
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect();
                    (q_data, d_data)
                },
                |(q_data, d_data)| {
                    black_box($func(
                        black_box(&q_data),
                        black_box(&d_data),
                        black_box($q_len),
                        black_box($d_len)
                        $(, black_box($args))*
                    ))
                },
                BatchSize::LargeInput,
            )
        });
    }};

    (
        $grp: expr,
        $name: expr,
        $func: expr,
        $q_len: expr,
        $d_len: expr,
        $dim: expr
        $(, $args:expr)*
    ) => {{
        $grp.bench_function($name, |b| {
            b.iter_batched(
                || {
                    let mut rng = rand::thread_rng();
                    let q_data: Vec<f32> = (0..($q_len * $dim))
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect();
                    let d_data: Vec<f32> = (0..($d_len * $dim))
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect();
                    (q_data, d_data)
                },
                // Measure Phase
                |(q_data, d_data)| {
                    black_box($func(
                        black_box(&q_data),
                        black_box(&d_data),
                        black_box($q_len),
                        black_box($d_len)
                        $(, black_box($args))*
                    ))
                },
                BatchSize::LargeInput,
            )
        });
    }};
}

fn bench_level_1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Level 1 Functions");

    // keeping same dim as colbert
    let dim = 128;

    //keeping this const for now
    let q_len = 32;

    let doc_lens = [
        128, // default
        256, 1024, 4096, 8192, // exterme testing
    ];

    for d_len in doc_lens {
        unsafe {
            single_func_benchmark!(
                group,
                throughput => Throughput::Elements(dim as u64 * q_len as u64 * d_len as u64),
                BenchmarkId::new("fused_dot_max_dim128_avx2", d_len),
                fused_dot_max_dim128_avx2,
                q_len,
                d_len,
                dim
            );
        }

        // choosing this as sole base line
        single_func_benchmark!(
            group,
            throughput => Throughput::Elements(dim as u64 * q_len as u64 * d_len as u64),
            BenchmarkId::new("pro_sgl_doc_msgemm", d_len),
            pro_sgl_doc_msgemm,
            q_len,
            d_len,
            // this if for calculating the doc len
            dim,
            dim // the function one dim (DONT GEt CONFUSE)
        );
    }
    group.finish();
}

criterion_group!(
    name=benches;
    // 100sec 
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_level_1
);

criterion_main!(benches);
