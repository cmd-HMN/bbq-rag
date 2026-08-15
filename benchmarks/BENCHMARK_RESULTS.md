# MaxSim Benchmark Report (4 Implementations)

Comparing:
1. **`maxsim-cpu`**: Official PyPI `maxsim-cpu` package
2. **`PyTorch`**: PyTorch Batched & Unpadded Loop
3. **`NumPy`**: Pure NumPy reference
4. **`maxsimd`**: Custom Rust AVX2/MKL SIMD implementation

### Small Workload (20 Docs, Q=10, L=50-150)
- **Correctness Check**: PASS
- **maxsimd Throughput**: 26636.6 docs/sec

| Implementation | Latency (ms) | Speed vs maxsimd |
|---|---|---|
| **4. maxsimd (My Implementation)** | **0.751 ± 0.107** | **1.00x (Baseline)** |
| 1. maxsim-cpu (PyPI) | 0.117 ± 0.013 | 6.39x faster |
| 2. PyTorch (Batched) | 0.674 ± 0.049 | 1.11x faster |
| 2. PyTorch (Loop) | 0.688 ± 0.060 | 1.09x faster |
| 3. NumPy (Reference) | 0.599 ± 0.119 | 1.25x faster |

### Medium Workload (100 Docs, Q=16, L=100-300)
- **Correctness Check**: PASS
- **maxsimd Throughput**: 19920.1 docs/sec

| Implementation | Latency (ms) | Speed vs maxsimd |
|---|---|---|
| **4. maxsimd (My Implementation)** | **5.020 ± 0.591** | **1.00x (Baseline)** |
| 1. maxsim-cpu (PyPI) | 1.426 ± 0.191 | 3.52x faster |
| 2. PyTorch (Batched) | 7.556 ± 0.944 | 1.51x slower |
| 2. PyTorch (Loop) | 4.655 ± 0.766 | 1.08x faster |
| 3. NumPy (Reference) | 4.862 ± 0.373 | 1.03x faster |

### Large Workload (500 Docs, Q=32, L=100-500)
- **Correctness Check**: PASS
- **maxsimd Throughput**: 8589.1 docs/sec

| Implementation | Latency (ms) | Speed vs maxsimd |
|---|---|---|
| **4. maxsimd (My Implementation)** | **58.213 ± 5.689** | **1.00x (Baseline)** |
| 1. maxsim-cpu (PyPI) | 10.724 ± 2.157 | 5.43x faster |
| 2. PyTorch (Batched) | 48.857 ± 4.508 | 1.19x faster |
| 2. PyTorch (Loop) | 21.187 ± 1.368 | 2.75x faster |
| 3. NumPy (Reference) | 26.380 ± 3.928 | 2.21x faster |

### High Variance Workload (100 Docs, Q=16, L=20-1000)
- **Correctness Check**: PASS
- **maxsimd Throughput**: 7404.5 docs/sec

| Implementation | Latency (ms) | Speed vs maxsimd |
|---|---|---|
| **4. maxsimd (My Implementation)** | **13.505 ± 2.147** | **1.00x (Baseline)** |
| 1. maxsim-cpu (PyPI) | 2.335 ± 0.070 | 5.78x faster |
| 2. PyTorch (Batched) | 14.922 ± 3.702 | 1.10x slower |
| 2. PyTorch (Loop) | 4.367 ± 0.026 | 3.09x faster |
| 3. NumPy (Reference) | 6.444 ± 1.251 | 2.10x faster |
