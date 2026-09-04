# MaxSim Benchmark Directory

This directory contains benchmarking utilities to measure and validate performance across MaxSim implementations:

### Benchmarked Endpoints & Baselines:
1. **`maxsimd.maxsim`**: Unified SIMD AVX2/FMA + Rayon multithreaded engine supporting 2D, 3D uniform batches, and ragged flat document buffers with zero-copy PyTorch/NumPy buffer ingestion.
2. **`maxsim-cpu`**: Official [`maxsim-cpu`](https://pypi.org/project/maxsim-cpu/) PyPI library package (`maxsim_scores_variable` and `maxsim_scores`).
3. **`PyTorch`**: PyTorch einsum & sequential loop baselines.
4. **`NumPy`**: Pure NumPy reference baseline.

## Running the Benchmark

From the root directory:

```bash
# Run with default all cores (jobs=-1) across full document range
python3 benchmarks/benchmark_maxsim.py

# Specify concurrency and custom document counts
python3 benchmarks/benchmark_maxsim.py --jobs 4 --docs 20 50 100 500
```

## Generated Artifacts

- [`assets/maxsim_benchmark_comparison.png`](../assets/maxsim_benchmark_comparison.png): 2x2 multi-panel high-resolution latency and throughput scaling graphs across both variable-length and uniform 3D document datasets.

