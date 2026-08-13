# MaxSim Benchmark Report (4 Implementations)

Comparing:
1. **`maxsim-cpu`**: Official PyPI `maxsim-cpu` package
2. **`PyTorch`**: PyTorch Batched & Unpadded Loop
3. **`NumPy`**: Pure NumPy reference
4. **`maxsimd`**: Custom Rust AVX2/MKL SIMD implementation

### Small Workload (20 Docs, Q=10, L=50-150)
- **Correctness Check**: PASS
- **maxsimd Throughput**: 24995.2 docs/sec

| Implementation | Latency (ms) | Speed vs maxsimd |
|---|---|---|
| **4. maxsimd (My Implementation)** | **0.800 ± 0.108** | **1.00x (Baseline)** |
| 1. maxsim-cpu (PyPI) | 0.109 ± 0.021 | 7.34x faster |
| 2. PyTorch (Batched) | 0.786 ± 0.160 | 1.02x faster |
| 2. PyTorch (Loop) | 0.776 ± 0.046 | 1.03x faster |
| 3. NumPy (Reference) | 0.615 ± 0.105 | 1.30x faster |

### Medium Workload (100 Docs, Q=16, L=100-300)
- **Correctness Check**: PASS
- **maxsimd Throughput**: 13527.2 docs/sec

| Implementation | Latency (ms) | Speed vs maxsimd |
|---|---|---|
| **4. maxsimd (My Implementation)** | **7.393 ± 1.637** | **1.00x (Baseline)** |
| 1. maxsim-cpu (PyPI) | 1.294 ± 0.167 | 5.71x faster |
| 2. PyTorch (Batched) | 8.595 ± 0.547 | 1.16x slower |
| 2. PyTorch (Loop) | 4.221 ± 0.407 | 1.75x faster |
| 3. NumPy (Reference) | 5.239 ± 0.546 | 1.41x faster |

### Large Workload (500 Docs, Q=32, L=100-500)
- **Correctness Check**: PASS
- **maxsimd Throughput**: 6963.6 docs/sec

| Implementation | Latency (ms) | Speed vs maxsimd |
|---|---|---|
| **4. maxsimd (My Implementation)** | **71.802 ± 8.442** | **1.00x (Baseline)** |
| 1. maxsim-cpu (PyPI) | 12.629 ± 2.718 | 5.69x faster |
| 2. PyTorch (Batched) | 65.999 ± 10.699 | 1.09x faster |
| 2. PyTorch (Loop) | 26.770 ± 1.700 | 2.68x faster |
| 3. NumPy (Reference) | 35.154 ± 4.808 | 2.04x faster |

### High Variance Workload (100 Docs, Q=16, L=20-1000)
- **Correctness Check**: PASS
- **maxsimd Throughput**: 7125.5 docs/sec

| Implementation | Latency (ms) | Speed vs maxsimd |
|---|---|---|
| **4. maxsimd (My Implementation)** | **14.034 ± 1.287** | **1.00x (Baseline)** |
| 1. maxsim-cpu (PyPI) | 4.019 ± 2.699 | 3.49x faster |
| 2. PyTorch (Batched) | 12.721 ± 0.463 | 1.10x faster |
| 2. PyTorch (Loop) | 4.436 ± 0.175 | 3.16x faster |
| 3. NumPy (Reference) | 5.921 ± 0.680 | 2.37x faster |
