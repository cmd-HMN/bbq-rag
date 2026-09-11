"""
Contains common imports and functions
"""

import os
import argparse
import numpy as np

import time
from typing import Tuple, TYPE_CHECKING

try: 
    import maxsim_cpu
except ImportError:
    maxsim_cpu = None

if TYPE_CHECKING:
    import torch
    import maxsimd

## ading lazy imports so it doesn't hurt performance
def __getattr__(name):
    if name == "torch":
        import torch
        globals()[name] = torch
        return torch

    elif name == "maxsimd":
        import maxsimd
        globals()[name] = maxsimd
        return maxsimd

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def configure_global_threads(jobs: int) -> None:
    """Synchronize thread count across PyTorch, maxsim-cpu (OpenMP), and maxsimd."""
    threads = (os.cpu_count() or 4) if jobs == -1 else max(1, jobs)
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass


    if maxsim_cpu is not None:
        try:
            import glob
            import ctypes

            # to change the thread for the omp library as maxsim_cpu uses libxsmm
            pkg_dir = os.path.dirname(maxsim_cpu.__file__)
            parent_dir = os.path.dirname(pkg_dir)
            for lib in glob.glob(
                os.path.join(parent_dir, "maxsim_cpu.libs", "libgomp*.so*")
            ):
                ctypes.CDLL(lib).omp_set_num_threads(threads)
        except Exception:
            pass


def mark_me(
    func,
    args,
    warmup_runs: int = 3,
    benchmark_runs: int = 15,
) -> Tuple[float, float]:

    # warm up the function
    for _ in range(warmup_runs):
        func(*args)

    times_ms = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    return mean_ms, std_ms

__all__ = [
    "np",
    "torch",
    "time",
    "maxsim_cpu",
    "maxsimd",
    "configure_global_threads",
    "mark_me",
    "argparse",
]
