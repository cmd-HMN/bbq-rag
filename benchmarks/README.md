# MaxSim Benchmark Directory

This directory contains benchmarking utilities to measure and validate performance across MaxSim implementations and Rust FFI endpoints:

### Benchmarked Endpoints & Baselines:
1. **`maxsimd.maxsim_vrlen`**: Custom SIMD AVX2/FMA + Rayon for variable-length (ragged) flat document buffers.
2. **`maxsimd.maxsim_ptr`**: Direct PyTorch `tensor.data_ptr()` raw pointer passing for 2D tensors with zero PyO3 object overhead.
3. **`maxsimd.maxsim_3d_ptr`**: Direct PyTorch `tensor.data_ptr()` raw pointer passing for 3D tensors `(batch, tokens, dim)` + Rayon multithreading.
4. **`maxsimd.maxsim`**: Direct 3D NumPy array ingestion + Rayon multithreading.
5. **`maxsim-cpu`**: Official [`maxsim-cpu`](https://pypi.org/project/maxsim-cpu/) PyPI library package (`maxsim_scores_variable` and `maxsim_scores`).
6. **`PyTorch`**: PyTorch einsum & loop baselines.
7. **`NumPy`**: Pure NumPy reference baseline.

## Running the Benchmark

From the root directory:

```bash
python3 benchmarks/benchmark_maxsim.py
```

## Generated Artifacts

- [`assets/maxsim_benchmark_comparison.png`](../assets/maxsim_benchmark_comparison.png): 2x2 multi-panel high-resolution latency and throughput scaling graphs across both variable-length and uniform 3D document datasets.

