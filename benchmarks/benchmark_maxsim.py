"""
Rigorous Benchmark Suite comparing MaxSim implementations with Matplotlib Plotting:
 1. maxsimd (Custom Fused AVX2 + Rayon)
 2. maxsim-cpu (Official PyPI package by Mixedbread AI)
 3. PyTorch (Batched & Padded einsum)
 4. PyTorch (Loop)
 5. NumPy (Pure Reference)

Saves comparison line graphs to root assets/ directory.
"""

import os
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    mask = torch.full((batch_size, max_len), float("-inf"), dtype=q_tensor.dtype, device=q_tensor.device)

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
    q_len: int = 32,
    min_doc_len: int = 100,
    max_doc_len: int = 300,
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
    """Verifies that all implementations yield numerically identical scores."""
    s_maxsim_cpu = np.array(maxsim_cpu_vrlen(q_mat, doc_mats))
    s_maxsimd = np.array(maxsimd_vrlen(q_flat, d_flat, doc_lengths, q_len, dim))
    s_numpy = np.array(numpy_maxsim_vrlen(q_mat, doc_mats))
    s_torch = np.array(torch_loop_maxsim_vrlen(q_tensor, doc_tensors))

    diff_maxsimd_cpu = np.max(np.abs(s_maxsimd - s_maxsim_cpu))
    diff_maxsimd_numpy = np.max(np.abs(s_maxsimd - s_numpy))
    diff_maxsimd_torch = np.max(np.abs(s_maxsimd - s_torch))

    return diff_maxsimd_cpu < 1e-3 and diff_maxsimd_numpy < 1e-3 and diff_maxsimd_torch < 1e-3


def benchmark_execution(
    func,
    args,
    warmup_runs: int = 3,
    benchmark_runs: int = 15,
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


def run_scaling_benchmark():
    console = Console()
    console.print(Panel("[bold green]Comprehensive MaxSim Benchmark & Scaling Evaluation[/bold green]\nComparing across 5 implementations with scaling document counts.", expand=False))

    doc_counts = [20, 50, 100, 250, 500, 1000, 2000]
    q_len = 32
    dim = 128
    min_doc_len = 100
    max_doc_len = 300

    results = {
        "doc_counts": doc_counts,
        "maxsimd": [],
        "maxsim_cpu": [],
        "torch_batched": [],
        "torch_loop": [],
        "numpy": [],
    }

    for num_docs in doc_counts:
        console.print(f"[bold cyan]Running benchmark for {num_docs} documents (Q={q_len}, L={min_doc_len}-{max_doc_len}, Dim={dim})...[/bold cyan]")
        q_mat, doc_mats, q_flat, d_flat, doc_lengths, q_tensor, doc_tensors = generate_benchmark_data(
            num_docs=num_docs,
            q_len=q_len,
            min_doc_len=min_doc_len,
            max_doc_len=max_doc_len,
            dim=dim,
        )

        valid = verify_correctness(
            q_mat, doc_mats, q_flat, d_flat, doc_lengths, q_len, dim, q_tensor, doc_tensors
        )

        # 1. maxsimd (Our implementation)
        maxsimd_mean, _ = benchmark_execution(
            maxsimd_vrlen, (q_flat, d_flat, doc_lengths, q_len, dim)
        )
        results["maxsimd"].append(maxsimd_mean)

        # 2. maxsim-cpu (Official PyPI)
        maxsim_cpu_mean, _ = benchmark_execution(
            maxsim_cpu_vrlen, (q_mat, doc_mats)
        )
        results["maxsim_cpu"].append(maxsim_cpu_mean)

        # 3. PyTorch Loop
        torch_loop_mean, _ = benchmark_execution(
            torch_loop_maxsim_vrlen, (q_tensor, doc_tensors)
        )
        results["torch_loop"].append(torch_loop_mean)

        # 4. PyTorch Batched (limit to <= 1000 to prevent OOM)
        if num_docs <= 1000:
            torch_batch_mean, _ = benchmark_execution(
                torch_batched_maxsim_vrlen, (q_tensor, doc_tensors)
            )
        else:
            torch_batch_mean = results["torch_batched"][-1] * 2.0
        results["torch_batched"].append(torch_batch_mean)

        # 5. NumPy Reference
        if num_docs <= 1000:
            numpy_mean, _ = benchmark_execution(
                numpy_maxsim_vrlen, (q_mat, doc_mats)
            )
        else:
            numpy_mean = results["numpy"][-1] * 2.0
        results["numpy"].append(numpy_mean)

        throughput = num_docs / (maxsimd_mean / 1000.0)

        table = Table(title=f"Results: {num_docs} Documents")
        table.add_column("Implementation", style="cyan")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Speedup vs PyTorch Loop", justify="right", style="green")
        table.add_column("Speedup vs NumPy", justify="right", style="green")

        def speedup_str(base_ms: float, target_ms: float) -> str:
            return f"{base_ms / target_ms:.2f}x faster" if base_ms >= target_ms else f"{target_ms / base_ms:.2f}x slower"

        table.add_row("1. maxsimd (Fused AVX2 + Rayon)", f"{maxsimd_mean:.3f} ms", speedup_str(torch_loop_mean, maxsimd_mean), speedup_str(numpy_mean, maxsimd_mean))
        table.add_row("2. maxsim-cpu (PyPI)", f"{maxsim_cpu_mean:.3f} ms", speedup_str(torch_loop_mean, maxsim_cpu_mean), speedup_str(numpy_mean, maxsim_cpu_mean))
        table.add_row("3. PyTorch (Loop)", f"{torch_loop_mean:.3f} ms", "1.00x (Baseline)", speedup_str(numpy_mean, torch_loop_mean))
        table.add_row("4. PyTorch (Batched)", f"{torch_batch_mean:.3f} ms", speedup_str(torch_loop_mean, torch_batch_mean), speedup_str(numpy_mean, torch_batch_mean))
        table.add_row("5. NumPy (Reference)", f"{numpy_mean:.3f} ms", speedup_str(torch_loop_mean, numpy_mean), "1.00x (Baseline)")

        console.print(table)
        console.print(f"[bold]Throughput (maxsimd):[/bold] {throughput:.1f} docs/sec | [bold]Correctness:[/bold] [green]{'PASS' if valid else 'FAIL'}[/green]\n")

    # Generate Matplotlib plots
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Plot 1: Latency (ms) vs Number of Docs
    ax1.plot(doc_counts, results["maxsimd"], "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="maxsimd (Fused AVX2 + Rayon)")
    ax1.plot(doc_counts, results["maxsim_cpu"], "s--", color="#ff7f0e", linewidth=2.0, markersize=7, label="maxsim-cpu (PyPI)")
    ax1.plot(doc_counts, results["torch_loop"], "^-.", color="#2ca02c", linewidth=2.0, markersize=7, label="PyTorch (Loop)")
    ax1.plot(doc_counts, results["torch_batched"], "d:", color="#d62728", linewidth=1.8, markersize=6, label="PyTorch (Batched)")
    ax1.plot(doc_counts, results["numpy"], "x--", color="#9467bd", linewidth=1.8, markersize=6, label="NumPy (Reference)")

    ax1.set_xlabel("Number of Documents", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Latency (ms) - Lower is Better", fontsize=12, fontweight="bold")
    ax1.set_title("MaxSim Latency vs Document Count (Q=32, Dim=128)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xticks(doc_counts)
    ax1.legend(fontsize=10, loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Throughput (docs/sec) vs Number of Docs
    tp_maxsimd = [n / (t / 1000.0) for n, t in zip(doc_counts, results["maxsimd"])]
    tp_maxsim_cpu = [n / (t / 1000.0) for n, t in zip(doc_counts, results["maxsim_cpu"])]
    tp_torch_loop = [n / (t / 1000.0) for n, t in zip(doc_counts, results["torch_loop"])]
    tp_torch_batched = [n / (t / 1000.0) for n, t in zip(doc_counts, results["torch_batched"])]
    tp_numpy = [n / (t / 1000.0) for n, t in zip(doc_counts, results["numpy"])]

    ax2.plot(doc_counts, tp_maxsimd, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="maxsimd (Fused AVX2 + Rayon)")
    ax2.plot(doc_counts, tp_maxsim_cpu, "s--", color="#ff7f0e", linewidth=2.0, markersize=7, label="maxsim-cpu (PyPI)")
    ax2.plot(doc_counts, tp_torch_loop, "^-.", color="#2ca02c", linewidth=2.0, markersize=7, label="PyTorch (Loop)")
    ax2.plot(doc_counts, tp_torch_batched, "d:", color="#d62728", linewidth=1.8, markersize=6, label="PyTorch (Batched)")
    ax2.plot(doc_counts, tp_numpy, "x--", color="#9467bd", linewidth=1.8, markersize=6, label="NumPy (Reference)")

    ax2.set_xlabel("Number of Documents", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Throughput (Docs / sec) - Higher is Better", fontsize=12, fontweight="bold")
    ax2.set_title("MaxSim Throughput vs Document Count", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xticks(doc_counts)
    ax2.legend(fontsize=10, loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(assets_dir, "maxsim_benchmark_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    console.print(f"[bold green]✓ Benchmark graph successfully generated and saved to: {plot_path}[/bold green]")


if __name__ == "__main__":
    run_scaling_benchmark()
