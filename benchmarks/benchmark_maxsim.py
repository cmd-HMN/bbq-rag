"""
Benchmark Suite comparing 4 MaxSim implementations:
 1. maxsim-cpu (Official maxsim-cpu PyPI package)
 2. PyTorch (Unpadded loop & Padded batched)
 3. NumPy (Pure NumPy reference)
 4. maxsimd (Custom SIMD Rust extension)
"""

import time
import os
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import maxsim_cpu
import maxsimd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def maxsim_cpu_vrlen(
    q_mat: np.ndarray,
    doc_mats: List[np.ndarray],
) -> List[float]:
    """Official maxsim-cpu PyPI library function maxsim_scores_variable."""
    scores = maxsim_cpu.maxsim_scores_variable(q_mat, doc_mats)
    return scores.tolist() if isinstance(scores, np.ndarray) else list(scores)


def numpy_maxsim_vrlen(
    q_mat: np.ndarray,
    doc_mats: List[np.ndarray],
) -> List[float]:
    """Pure NumPy reference implementation of variable length MaxSim."""
    scores = []
    for d_doc in doc_mats:
        sim_matrix = np.dot(q_mat, d_doc.T)
        doc_score = np.sum(np.max(sim_matrix, axis=1))
        scores.append(float(doc_score))
    return scores


def torch_loop_maxsim_vrlen(
    q_tensor: torch.Tensor,
    doc_tensors: List[torch.Tensor],
) -> List[float]:
    """PyTorch native implementation looping through documents one by one."""
    scores = []
    with torch.no_grad():
        for doc in doc_tensors:
            sim_matrix = torch.matmul(q_tensor, doc.T)
            doc_score = torch.sum(torch.max(sim_matrix, dim=1).values).item()
            scores.append(doc_score)
    return scores


def torch_batched_maxsim_vrlen(
    q_tensor: torch.Tensor,
    doc_tensors: List[torch.Tensor],
) -> List[float]:
    """PyTorch padded batched matrix multiplication with mask."""
    max_len = max(doc.shape[0] for doc in doc_tensors)
    batch_size = len(doc_tensors)
    dim = q_tensor.shape[1]

    padded_docs = torch.zeros(batch_size, max_len, dim, dtype=q_tensor.dtype, device=q_tensor.device)
    mask = torch.full((batch_size, max_len), float('-inf'), dtype=q_tensor.dtype, device=q_tensor.device)

    for i, doc in enumerate(doc_tensors):
        l = doc.shape[0]
        padded_docs[i, :l] = doc
        mask[i, :l] = 0.0

    with torch.no_grad():
        sim_matrix = torch.einsum("qd, btd -> bqt", q_tensor, padded_docs)
        masked_sims = sim_matrix + mask.unsqueeze(1)
        doc_scores = torch.sum(torch.max(masked_sims, dim=-1).values, dim=-1)
        return doc_scores.cpu().tolist()


def maxsimd_vrlen(
    q_flat: np.ndarray,
    d_flat: np.ndarray,
    doc_lengths: List[int],
    q_len: int,
    dim: int,
) -> List[float]:
    """Custom maxsimd Rust extension implementation."""
    return maxsimd.maxsim_vrlen(q_flat, d_flat, doc_lengths, q_len, dim)


def generate_benchmark_data(
    num_docs: int,
    q_len: int,
    min_doc_len: int,
    max_doc_len: int,
    dim: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, List[int], torch.Tensor, List[torch.Tensor]]:
    """Generates synthetic benchmark data for query and documents."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    doc_lengths = np.random.randint(min_doc_len, max_doc_len + 1, size=num_docs).tolist()

    q_mat = np.random.randn(q_len, dim).astype(np.float32)
    q_mat = q_mat / np.linalg.norm(q_mat, axis=-1, keepdims=True)
    q_flat = q_mat.reshape(-1)

    doc_mats = []
    d_flat_list = []
    doc_tensors = []

    for l in doc_lengths:
        d_mat = np.random.randn(l, dim).astype(np.float32)
        d_mat = d_mat / np.linalg.norm(d_mat, axis=-1, keepdims=True)
        doc_mats.append(d_mat)
        d_flat_list.append(d_mat.reshape(-1))
        doc_tensors.append(torch.from_numpy(d_mat))

    d_flat = np.concatenate(d_flat_list, axis=0)
    q_tensor = torch.from_numpy(q_mat)

    return q_mat, doc_mats, q_flat, d_flat, doc_lengths, q_tensor, doc_tensors


def verify_correctness(
    q_mat: np.ndarray,
    doc_mats: List[np.ndarray],
    q_flat: np.ndarray,
    d_flat: np.ndarray,
    doc_lengths: List[int],
    q_len: int,
    dim: int,
    q_tensor: torch.Tensor,
    doc_tensors: List[torch.Tensor],
) -> bool:
    """Verifies that all 4 implementations yield matching scores."""
    s_maxsim_cpu = maxsim_cpu_vrlen(q_mat, doc_mats)
    s_maxsimd = maxsimd_vrlen(q_flat, d_flat, doc_lengths, q_len, dim)
    s_numpy = numpy_maxsim_vrlen(q_mat, doc_mats)
    s_torch = torch_loop_maxsim_vrlen(q_tensor, doc_tensors)

    diff_maxsimd_cpu = np.max(np.abs(np.array(s_maxsimd) - np.array(s_maxsim_cpu)))
    diff_maxsimd_numpy = np.max(np.abs(np.array(s_maxsimd) - np.array(s_numpy)))
    diff_maxsimd_torch = np.max(np.abs(np.array(s_maxsimd) - np.array(s_torch)))

    return diff_maxsimd_cpu < 1e-3 and diff_maxsimd_numpy < 1e-3 and diff_maxsimd_torch < 1e-3


def benchmark_execution(
    func,
    args,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
) -> Tuple[float, float]:
    """Runs warmup and measures average execution time and standard deviation in milliseconds."""
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


def run_benchmark_suite():
    console = Console()
    console.print(Panel("[bold green]MaxSim Performance Benchmark: 4 Implementations[/bold green]\n1. maxsim-cpu (PyPI)\n2. PyTorch\n3. NumPy\n4. maxsimd (Custom SIMD)", expand=False))

    workloads = [
        {"name": "Small Workload (20 Docs, Q=10, L=50-150)", "num_docs": 20, "q_len": 10, "min_len": 50, "max_len": 150, "dim": 128},
        {"name": "Medium Workload (100 Docs, Q=16, L=100-300)", "num_docs": 100, "q_len": 16, "min_len": 100, "max_len": 300, "dim": 128},
        {"name": "Large Workload (500 Docs, Q=32, L=100-500)", "num_docs": 500, "q_len": 32, "min_len": 100, "max_len": 500, "dim": 128},
        {"name": "High Variance Workload (100 Docs, Q=16, L=20-1000)", "num_docs": 100, "q_len": 16, "min_len": 20, "max_len": 1000, "dim": 128},
    ]

    all_results = []
    markdown_lines = [
        "# MaxSim Benchmark Report (4 Implementations)",
        "",
        "Comparing:",
        "1. **`maxsim-cpu`**: Official PyPI `maxsim-cpu` package",
        "2. **`PyTorch`**: PyTorch Batched & Unpadded Loop",
        "3. **`NumPy`**: Pure NumPy reference",
        "4. **`maxsimd`**: Custom Rust AVX2/MKL SIMD implementation",
        "",
    ]

    for wl in workloads:
        q_mat, doc_mats, q_flat, d_flat, doc_lengths, q_tensor, doc_tensors = generate_benchmark_data(
            num_docs=wl["num_docs"],
            q_len=wl["q_len"],
            min_doc_len=wl["min_len"],
            max_doc_len=wl["max_len"],
            dim=wl["dim"],
        )

        valid = verify_correctness(
            q_mat, doc_mats, q_flat, d_flat, doc_lengths, wl["q_len"], wl["dim"], q_tensor, doc_tensors
        )

        # 1. maxsim-cpu
        maxsim_cpu_mean, maxsim_cpu_std = benchmark_execution(
            maxsim_cpu_vrlen, (q_mat, doc_mats)
        )
        # 2. PyTorch Loop & Batched
        torch_loop_mean, torch_loop_std = benchmark_execution(
            torch_loop_maxsim_vrlen, (q_tensor, doc_tensors)
        )
        torch_batch_mean, torch_batch_std = benchmark_execution(
            torch_batched_maxsim_vrlen, (q_tensor, doc_tensors)
        )
        # 3. NumPy
        numpy_mean, numpy_std = benchmark_execution(
            numpy_maxsim_vrlen, (q_mat, doc_mats)
        )
        # 4. maxsimd
        maxsimd_mean, maxsimd_std = benchmark_execution(
            maxsimd_vrlen, (q_flat, d_flat, doc_lengths, wl["q_len"], wl["dim"])
        )

        def format_speed_comparison(impl_mean: float, baseline_mean: float) -> str:
            if impl_mean <= 0 or baseline_mean <= 0:
                return "N/A"
            if impl_mean < baseline_mean:
                ratio = baseline_mean / impl_mean
                return f"{ratio:.2f}x faster"
            elif impl_mean > baseline_mean:
                ratio = impl_mean / baseline_mean
                return f"{ratio:.2f}x slower"
            else:
                return "1.00x (Baseline)"

        throughput_docs_sec = wl["num_docs"] / (maxsimd_mean / 1000.0)

        table = Table(title=f"Results: {wl['name']}")
        table.add_column("Implementation", style="cyan")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Relative Speed (vs maxsimd)", justify="right", style="green")

        table.add_row("4. maxsimd (My Implementation)", f"{maxsimd_mean:.3f} ms", "1.00x (Baseline)")
        table.add_row("1. maxsim-cpu (PyPI)", f"{maxsim_cpu_mean:.3f} ms", format_speed_comparison(maxsim_cpu_mean, maxsimd_mean))
        table.add_row("2. PyTorch (Batched)", f"{torch_batch_mean:.3f} ms", format_speed_comparison(torch_batch_mean, maxsimd_mean))
        table.add_row("2. PyTorch (Loop)", f"{torch_loop_mean:.3f} ms", format_speed_comparison(torch_loop_mean, maxsimd_mean))
        table.add_row("3. NumPy (Reference)", f"{numpy_mean:.3f} ms", format_speed_comparison(numpy_mean, maxsimd_mean))

        console.print(table)
        console.print(f"[bold]Throughput (maxsimd):[/bold] {throughput_docs_sec:.1f} docs/sec | [bold]Correctness Check:[/bold] [green]{'PASS' if valid else 'FAIL'}[/green]\n")

        markdown_lines.append(f"### {wl['name']}")
        markdown_lines.append(f"- **Correctness Check**: {'PASS' if valid else 'FAIL'}")
        markdown_lines.append(f"- **maxsimd Throughput**: {throughput_docs_sec:.1f} docs/sec")
        markdown_lines.append("")
        markdown_lines.append("| Implementation | Latency (ms) | Speed vs maxsimd |")
        markdown_lines.append("|---|---|---|")
        markdown_lines.append(f"| **4. maxsimd (My Implementation)** | **{maxsimd_mean:.3f} ± {maxsimd_std:.3f}** | **1.00x (Baseline)** |")
        markdown_lines.append(f"| 1. maxsim-cpu (PyPI) | {maxsim_cpu_mean:.3f} ± {maxsim_cpu_std:.3f} | {format_speed_comparison(maxsim_cpu_mean, maxsimd_mean)} |")
        markdown_lines.append(f"| 2. PyTorch (Batched) | {torch_batch_mean:.3f} ± {torch_batch_std:.3f} | {format_speed_comparison(torch_batch_mean, maxsimd_mean)} |")
        markdown_lines.append(f"| 2. PyTorch (Loop) | {torch_loop_mean:.3f} ± {torch_loop_std:.3f} | {format_speed_comparison(torch_loop_mean, maxsimd_mean)} |")
        markdown_lines.append(f"| 3. NumPy (Reference) | {numpy_mean:.3f} ± {numpy_std:.3f} | {format_speed_comparison(numpy_mean, maxsimd_mean)} |")
        markdown_lines.append("")

    report_path = os.path.join(os.path.dirname(__file__), "BENCHMARK_RESULTS.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))

    console.print(f"[bold green]Saved 4-way benchmark report to {report_path}[/bold green]")


if __name__ == "__main__":
    run_benchmark_suite()
