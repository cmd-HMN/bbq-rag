use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use maxsimd::cpu::vec256::simd::{fused_dot_max_dim128_avx2, fused_dot_max_generic_avx2};
use maxsimd::func::function::internal::pro_sgl_doc_msgemm;
use maxsimd::func::function::maxsim_variable_length;
use maxsimd::quantization::quantize::qnt::{qf32_i8_d128, sq128x32_sq8};

use pprof::criterion::{Output, PProfProfiler};
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
        let mut rng = rand::thread_rng();
        let q_data: Vec<f32> = (0..($q_len * $dim))
            .map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_data: Vec<f32> = (0..($d_len * $dim))
            .map(|_| rng.gen_range(-1.0..1.0)).collect();
        $grp.bench_function($name, |b| {
            b.iter(
                || {
                black_box($func(
                black_box(&q_data),
                black_box(&d_data),
                black_box($q_len),
                black_box($d_len)
                $(, black_box($args))*
            ))
                },
            )
        });
    }};


    //TODO
    //Will change this one
    // I think time has came for this TODO
    (
        $grp: expr,
        $name: expr,
        $func: expr,
        $q_len: expr,
        $d_len: expr,
        $dim: expr
        $(, $args:expr)*
    ) => {{
        let mut rng = rand::thread_rng();
        let q_data: Vec<f32> = (0..($q_len * $dim))
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let d_data: Vec<f32> = (0..($d_len * $dim))
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        $grp.bench_function($name, |b| {
            b.iter(
                || {
                    black_box($func(
                        black_box(&q_data),
                        black_box(&d_data),
                        black_box($q_len),
                        black_box($d_len)
                        $(, black_box($args))*
                    ))
                },
            )
        });
    }};
    
    // for other function with some expr 
    (
        $grp: expr,
        throughput => $thput: expr,
        $func: expr,
        $(args: expr)*
    ) => {{
        $grp.throughput($thput);
        $grp.bench_function($func, |b| {
            b.iter(
                || {
                    black_box($func(
                        $(black_box(args))*
                    ))
                },
            )
        });
    }} 
}

fn bench_cpu_level_1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Level 1 Functions (Docs)");

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
        };

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

    let mut d_group = c.benchmark_group("Level 1 Functions (Dims)");
    // this senrio would no come as will be using colbert style
    let dim_lengths = [64, 128, 384, 768, 1536];

    // keeping this in small range
    let d_len = 256;

    for dim in dim_lengths {
        unsafe {
            single_func_benchmark!(
                d_group,
                throughput => Throughput::Elements(dim as u64 * q_len as u64 * d_len as u64),
                BenchmarkId::new("fused_dot_max_generic_avx2", dim),
                fused_dot_max_generic_avx2,
                q_len,
                d_len,
                dim,
                dim
            );
        }

        single_func_benchmark!(
            d_group,
            throughput => Throughput::Elements(dim as u64 * q_len as u64 * d_len as u64),
            BenchmarkId::new("pro_sgl_doc_msgemm", dim),
            pro_sgl_doc_msgemm,
            q_len,
            d_len,
            // this if for calculating the doc len
            dim,
            dim // the function one dim (DONT GEt CONFUSE)
        );
    }

    d_group.finish();
}

// these are for the higher function the uses level 1
fn bench_cpu_level_2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Level 2 Functions");

    // keeping same dim as colbert
    let dim = 128;

    let q_len = 32;
    let d_len = 256;

    // for maxsim_variable_length -> base case
    group.throughput(Throughput::Elements(
        dim as u64 * q_len as u64 * d_len as u64,
    ));

    let mut rng = rand::thread_rng();
    let q_data: Vec<f32> = (0..(q_len * dim))
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let d_data: Vec<f32> = (0..(d_len * dim))
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // [(doc_idx, doc_len, doc_data)]
    let dd = vec![(0_usize, d_len, d_data)];

    group.bench_function("maxsim_variable_length", |b| {
        b.iter(|| {
            black_box(maxsim_variable_length(
                black_box(q_data.clone()),
                black_box(dd.clone()),
                black_box(q_len),
                black_box(dim),
            ))
        });
    });

    group.finish();
}

fn bench_quant(c: &mut Criterion) {}

criterion_group!(
    name=benches;
    // 100sec
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_cpu_level_1, bench_cpu_level_2
);

criterion_main!(benches);
