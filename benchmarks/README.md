# MaxSim Benchmark Directory (4-Way Comparison)

This directory contains benchmarking utilities to measure and validate performance across 4 MaxSim implementations:

1. **`maxsim-cpu`**: Official [`maxsim-cpu`](https://pypi.org/project/maxsim-cpu/) PyPI library package (`maxsim_cpu.maxsim_scores_variable`).
2. **`PyTorch`**: PyTorch matrix multiplication (batched padded & unpadded sequential loop).
3. **`NumPy`**: Pure NumPy reference baseline.
4. **`maxsimd`**: Your custom SIMD AVX2/MKL Rust extension (`maxsimd.maxsim_vrlen`).

## Running the Benchmark

From the root directory:

```bash
python3 benchmarks/benchmark_maxsim.py
```

## Generated Artifacts

- [`BENCHMARK_RESULTS.md`](file:///home/death_note/Projects/bbq_rag/benchmarks/BENCHMARK_RESULTS.md): Contains execution latency (ms), throughput (docs/sec), speedup factors, and accuracy validation across all 4 implementations.
