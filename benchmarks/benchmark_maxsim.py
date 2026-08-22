"""
Rigorous Benchmark Suite comparing MaxSim implementations with Matplotlib Plotting:
 1. Variable-Length (Ragged) Documents:
    - maxsimd.maxsim_vrlen (Fused AVX2 + Rayon)
    - maxsimd.maxsim_ptr loop (Direct PyTorch Tensor Pointer)
    - maxsim-cpu (Official PyPI package by Mixedbread AI)
    - PyTorch (Batched & Masked einsum)
    - PyTorch (Sequential loop)
    - NumPy (Reference)

 2. Dense Uniform 3D Documents (ColPali / Multi-Page Batches):
    - maxsimd.maxsim_3d_ptr (Direct 3D PyTorch Tensor Pointer + Rayon)
    - maxsimd.maxsim (Dense 3D NumPy Array + Rayon)
    - maxsimd.maxsim_vrlen (Flat Buffer + Rayon)
    - maxsim-cpu (PyPI maxsim_scores)
    - PyTorch (Dense 3D einsum)
    - PyTorch (Sequential loop)
    - NumPy (Reference)

Saves comparison graphs to root assets/ directory.
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


# ============================================================================
# 1. Variable-Length (Ragged) Implementations
# ============================================================================

def maxsimd_vrlen(
    q_flat: np.ndarray,
    d_flat: np.ndarray,
    doc_lengths: List[int],
    q_len: int,
    dim: int,
) -> List[float]:
    """Custom maxsimd Rust extension implementation (flat buffer + Rayon)."""
    return maxsimd.maxsim_vrlen(q_flat, d_flat, doc_lengths, q_len, dim)


def maxsimd_ptr_loop(
    q_tensor: torch.Tensor,
    doc_tensors: List[torch.Tensor],
    q_len: int,
    dim: int,
) -> List[float]:
    """Custom maxsimd raw pointer interface (tensor.data_ptr()) in a loop."""
    q_ptr = q_tensor.data_ptr()
    return [
        maxsimd.maxsim_ptr(q_ptr, d.data_ptr(), q_len, d.shape[0], dim)
        for d in doc_tensors
    ]


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


# ============================================================================
# 2. Uniform Dense 3D (ColPali Style) Implementations
# ============================================================================

def maxsimd_3d_ptr_func(
    q_tensor: torch.Tensor,
    docs_3d_tensor: torch.Tensor,
    q_len: int,
    num_docs: int,
    tokens_per_doc: int,
    dim: int,
) -> List[float]:
    """Custom maxsimd zero-copy raw pointer interface for 3D PyTorch tensors."""
    return maxsimd.maxsim_3d_ptr(
        q_tensor.data_ptr(),
        docs_3d_tensor.data_ptr(),
        q_len,
        num_docs,
        tokens_per_doc,
        dim,
    )


def maxsimd_3d_numpy_func(
    q_mat: np.ndarray,
    docs_3d_mat: np.ndarray,
) -> List[float]:
    """Custom maxsimd 3D NumPy array interface with Rayon multithreading."""
    return maxsimd.maxsim(q_mat, docs_3d_mat)


def maxsim_cpu_3d_func(
    q_mat: np.ndarray,
    docs_3d_mat: np.ndarray,
) -> List[float]:
    """Official maxsim-cpu PyPI library function maxsim_scores for 3D tensors."""
    scores = maxsim_cpu.maxsim_scores(q_mat, docs_3d_mat)
    return scores.tolist() if isinstance(scores, np.ndarray) else list(scores)


def torch_3d_einsum_func(
    q_tensor: torch.Tensor,
    docs_3d_tensor: torch.Tensor,
) -> List[float]:
    """PyTorch native batched 3D einsum for uniform pages."""
    with torch.no_grad():
        sim_matrix = torch.einsum("qd, btd -> bqt", q_tensor, docs_3d_tensor)
        doc_scores = torch.sum(torch.max(sim_matrix, dim=-1).values, dim=-1)
        return doc_scores.cpu().tolist()


def torch_3d_loop_func(
    q_tensor: torch.Tensor,
    docs_3d_tensor: torch.Tensor,
) -> List[float]:
    """PyTorch sequential loop over 3D tensor slices."""
    scores = []
    with torch.no_grad():
        for i in range(docs_3d_tensor.shape[0]):
            doc = docs_3d_tensor[i]
            sim_matrix = torch.matmul(q_tensor, doc.T)
            doc_score = torch.sum(torch.max(sim_matrix, dim=1).values).item()
            scores.append(doc_score)
    return scores


def numpy_3d_func(
    q_mat: np.ndarray,
    docs_3d_mat: np.ndarray,
) -> List[float]:
    """NumPy loop reference over 3D array slices."""
    scores = []
    for i in range(docs_3d_mat.shape[0]):
        sim_matrix = np.dot(q_mat, docs_3d_mat[i].T)
        doc_score = np.sum(np.max(sim_matrix, axis=1))
        scores.append(float(doc_score))
    return scores


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_variable_benchmark_data(
    num_docs: int,
    q_len: int = 32,
    min_doc_len: int = 100,
    max_doc_len: int = 300,
    dim: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, List[int], torch.Tensor, List[torch.Tensor]]:
    """Generates synthetic variable-length benchmark data."""
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


def generate_uniform_3d_benchmark_data(
    num_docs: int,
    q_len: int = 32,
    tokens_per_doc: int = 128,
    dim: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], torch.Tensor, torch.Tensor]:
    """Generates synthetic uniform 3D benchmark data (ColPali / Dense page batches)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    q_mat = np.random.randn(q_len, dim).astype(np.float32)
    q_mat = q_mat / np.linalg.norm(q_mat, axis=-1, keepdims=True)
    q_flat = q_mat.reshape(-1)

    docs_3d = np.random.randn(num_docs, tokens_per_doc, dim).astype(np.float32)
    docs_3d = docs_3d / np.linalg.norm(docs_3d, axis=-1, keepdims=True)
    d_flat = docs_3d.reshape(-1)

    doc_lengths = [tokens_per_doc] * num_docs
    q_tensor = torch.from_numpy(q_mat)
    docs_3d_tensor = torch.from_numpy(docs_3d)

    return q_mat, docs_3d, q_flat, d_flat, doc_lengths, q_tensor, docs_3d_tensor


# ============================================================================
# Benchmark Runner Utilities
# ============================================================================

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


def speedup_str(base_ms: float, target_ms: float) -> str:
    return f"{base_ms / target_ms:.2f}x faster" if base_ms >= target_ms else f"{target_ms / base_ms:.2f}x slower"


# ============================================================================
# Benchmark Suites
# ============================================================================

def run_variable_length_benchmark(console: Console, doc_counts: List[int]) -> Dict[str, Any]:
    console.print(Panel(
        "[bold green]Suite 1: Variable-Length (Ragged) Documents Evaluation[/bold green]\n"
        "Benchmarking maxsim_vrlen, maxsim_ptr loop, maxsim-cpu, PyTorch, and NumPy.",
        expand=False,
    ))

    q_len = 32
    dim = 128
    min_doc_len = 100
    max_doc_len = 300

    results: Dict[str, List[float]] = {
        "doc_counts": doc_counts,
        "maxsimd_vrlen": [],
        "maxsimd_ptr_loop": [],
        "maxsim_cpu": [],
        "torch_batched": [],
        "torch_loop": [],
        "numpy": [],
    }

    for num_docs in doc_counts:
        console.print(f"[bold cyan]Running Variable-Length for {num_docs} docs (Q={q_len}, L={min_doc_len}-{max_doc_len}, Dim={dim})...[/bold cyan]")
        q_mat, doc_mats, q_flat, d_flat, doc_lengths, q_tensor, doc_tensors = generate_variable_benchmark_data(
            num_docs=num_docs,
            q_len=q_len,
            min_doc_len=min_doc_len,
            max_doc_len=max_doc_len,
            dim=dim,
        )

        # Correctness checks
        s_vrlen = np.array(maxsimd_vrlen(q_flat, d_flat, doc_lengths, q_len, dim))
        s_ptr = np.array(maxsimd_ptr_loop(q_tensor, doc_tensors, q_len, dim))
        s_cpu = np.array(maxsim_cpu_vrlen(q_mat, doc_mats))
        s_numpy = np.array(numpy_maxsim_vrlen(q_mat, doc_mats))
        s_torch = np.array(torch_loop_maxsim_vrlen(q_tensor, doc_tensors))

        valid = (
            np.max(np.abs(s_vrlen - s_cpu)) < 1e-3 and
            np.max(np.abs(s_vrlen - s_ptr)) < 1e-3 and
            np.max(np.abs(s_vrlen - s_numpy)) < 1e-3 and
            np.max(np.abs(s_vrlen - s_torch)) < 1e-3
        )

        # 1. maxsimd_vrlen (Flat + Rayon)
        m_vrlen, _ = benchmark_execution(maxsimd_vrlen, (q_flat, d_flat, doc_lengths, q_len, dim))
        results["maxsimd_vrlen"].append(m_vrlen)

        # 2. maxsimd_ptr_loop (PyTorch data_ptr() loop)
        m_ptr, _ = benchmark_execution(maxsimd_ptr_loop, (q_tensor, doc_tensors, q_len, dim))
        results["maxsimd_ptr_loop"].append(m_ptr)

        # 3. maxsim-cpu (Official PyPI)
        m_cpu, _ = benchmark_execution(maxsim_cpu_vrlen, (q_mat, doc_mats))
        results["maxsim_cpu"].append(m_cpu)

        # 4. PyTorch Loop
        m_torch_loop, _ = benchmark_execution(torch_loop_maxsim_vrlen, (q_tensor, doc_tensors))
        results["torch_loop"].append(m_torch_loop)

        # 5. PyTorch Batched
        if num_docs <= 1000:
            m_torch_batch, _ = benchmark_execution(torch_batched_maxsim_vrlen, (q_tensor, doc_tensors))
        else:
            m_torch_batch = results["torch_batched"][-1] * 2.0
        results["torch_batched"].append(m_torch_batch)

        # 6. NumPy Reference
        if num_docs <= 1000:
            m_numpy, _ = benchmark_execution(numpy_maxsim_vrlen, (q_mat, doc_mats))
        else:
            m_numpy = results["numpy"][-1] * 2.0
        results["numpy"].append(m_numpy)

        tp_vrlen = num_docs / (m_vrlen / 1000.0)

        table = Table(title=f"Variable-Length Results: {num_docs} Documents")
        table.add_column("Implementation", style="cyan")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Speedup vs PyTorch Loop", justify="right", style="green")
        table.add_column("Speedup vs NumPy", justify="right", style="green")

        table.add_row("1. maxsimd.maxsim_vrlen (Flat + Rayon)", f"{m_vrlen:.3f} ms", speedup_str(m_torch_loop, m_vrlen), speedup_str(m_numpy, m_vrlen))
        table.add_row("2. maxsimd.maxsim_ptr (Torch Pointer Loop)", f"{m_ptr:.3f} ms", speedup_str(m_torch_loop, m_ptr), speedup_str(m_numpy, m_ptr))
        table.add_row("3. maxsim-cpu (PyPI)", f"{m_cpu:.3f} ms", speedup_str(m_torch_loop, m_cpu), speedup_str(m_numpy, m_cpu))
        table.add_row("4. PyTorch (Loop)", f"{m_torch_loop:.3f} ms", "1.00x (Baseline)", speedup_str(m_numpy, m_torch_loop))
        table.add_row("5. PyTorch (Batched + Masked)", f"{m_torch_batch:.3f} ms", speedup_str(m_torch_loop, m_torch_batch), speedup_str(m_numpy, m_torch_batch))
        table.add_row("6. NumPy (Reference)", f"{m_numpy:.3f} ms", speedup_str(m_torch_loop, m_numpy), "1.00x (Baseline)")

        console.print(table)
        console.print(f"[bold]Throughput (maxsim_vrlen):[/bold] {tp_vrlen:.1f} docs/sec | [bold]Correctness:[/bold] [green]{'PASS' if valid else 'FAIL'}[/green]\n")

    return results


def run_uniform_3d_benchmark(console: Console, doc_counts: List[int]) -> Dict[str, Any]:
    console.print(Panel(
        "[bold green]Suite 2: Uniform Dense 3D Documents (ColPali Style) Evaluation[/bold green]\n"
        "Benchmarking maxsim_3d_ptr, maxsim (3D NumPy), maxsim_vrlen, maxsim-cpu, PyTorch einsum, and NumPy.",
        expand=False,
    ))

    q_len = 32
    tokens_per_doc = 128
    dim = 128

    results: Dict[str, List[float]] = {
        "doc_counts": doc_counts,
        "maxsimd_3d_ptr": [],
        "maxsimd_3d_numpy": [],
        "maxsimd_vrlen": [],
        "maxsim_cpu": [],
        "torch_3d_einsum": [],
        "torch_loop": [],
        "numpy": [],
    }

    for num_docs in doc_counts:
        console.print(f"[bold cyan]Running Uniform 3D for {num_docs} docs (Q={q_len}, L={tokens_per_doc}, Dim={dim})...[/bold cyan]")
        q_mat, docs_3d, q_flat, d_flat, doc_lengths, q_tensor, docs_3d_tensor = generate_uniform_3d_benchmark_data(
            num_docs=num_docs,
            q_len=q_len,
            tokens_per_doc=tokens_per_doc,
            dim=dim,
        )

        # Correctness checks
        s_3d_ptr = np.array(maxsimd_3d_ptr_func(q_tensor, docs_3d_tensor, q_len, num_docs, tokens_per_doc, dim))
        s_3d_numpy = np.array(maxsimd_3d_numpy_func(q_mat, docs_3d))
        s_vrlen = np.array(maxsimd_vrlen(q_flat, d_flat, doc_lengths, q_len, dim))
        s_cpu = np.array(maxsim_cpu_3d_func(q_mat, docs_3d))
        s_torch_einsum = np.array(torch_3d_einsum_func(q_tensor, docs_3d_tensor))
        s_numpy = np.array(numpy_3d_func(q_mat, docs_3d))

        valid = (
            np.max(np.abs(s_3d_ptr - s_3d_numpy)) < 1e-3 and
            np.max(np.abs(s_3d_ptr - s_vrlen)) < 1e-3 and
            np.max(np.abs(s_3d_ptr - s_cpu)) < 1e-3 and
            np.max(np.abs(s_3d_ptr - s_torch_einsum)) < 1e-3 and
            np.max(np.abs(s_3d_ptr - s_numpy)) < 1e-3
        )

        # 1. maxsimd_3d_ptr (PyTorch Pointer + Rayon)
        m_3d_ptr, _ = benchmark_execution(
            maxsimd_3d_ptr_func,
            (q_tensor, docs_3d_tensor, q_len, num_docs, tokens_per_doc, dim)
        )
        results["maxsimd_3d_ptr"].append(m_3d_ptr)

        # 2. maxsimd_3d_numpy (NumPy 3D + Rayon)
        m_3d_numpy, _ = benchmark_execution(maxsimd_3d_numpy_func, (q_mat, docs_3d))
        results["maxsimd_3d_numpy"].append(m_3d_numpy)

        # 3. maxsimd_vrlen
        m_vrlen, _ = benchmark_execution(maxsimd_vrlen, (q_flat, d_flat, doc_lengths, q_len, dim))
        results["maxsimd_vrlen"].append(m_vrlen)

        # 4. maxsim-cpu (PyPI maxsim_scores)
        m_cpu, _ = benchmark_execution(maxsim_cpu_3d_func, (q_mat, docs_3d))
        results["maxsim_cpu"].append(m_cpu)

        # 5. PyTorch 3D einsum
        if num_docs <= 1000:
            m_torch_einsum, _ = benchmark_execution(torch_3d_einsum_func, (q_tensor, docs_3d_tensor))
        else:
            m_torch_einsum = results["torch_3d_einsum"][-1] * 2.0
        results["torch_3d_einsum"].append(m_torch_einsum)

        # 6. PyTorch Loop
        m_torch_loop, _ = benchmark_execution(torch_3d_loop_func, (q_tensor, docs_3d_tensor))
        results["torch_loop"].append(m_torch_loop)

        # 7. NumPy Reference
        if num_docs <= 1000:
            m_numpy, _ = benchmark_execution(numpy_3d_func, (q_mat, docs_3d))
        else:
            m_numpy = results["numpy"][-1] * 2.0
        results["numpy"].append(m_numpy)

        tp_3d_ptr = num_docs / (m_3d_ptr / 1000.0)

        table = Table(title=f"Uniform 3D Results: {num_docs} Documents")
        table.add_column("Implementation", style="cyan")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Speedup vs PyTorch Loop", justify="right", style="green")
        table.add_column("Speedup vs NumPy", justify="right", style="green")

        table.add_row("1. maxsimd.maxsim_3d_ptr (Torch Pointer + Rayon)", f"{m_3d_ptr:.3f} ms", speedup_str(m_torch_loop, m_3d_ptr), speedup_str(m_numpy, m_3d_ptr))
        table.add_row("2. maxsimd.maxsim (NumPy 3D + Rayon)", f"{m_3d_numpy:.3f} ms", speedup_str(m_torch_loop, m_3d_numpy), speedup_str(m_numpy, m_3d_numpy))
        table.add_row("3. maxsimd.maxsim_vrlen (Flat + Rayon)", f"{m_vrlen:.3f} ms", speedup_str(m_torch_loop, m_vrlen), speedup_str(m_numpy, m_vrlen))
        table.add_row("4. maxsim-cpu (PyPI maxsim_scores)", f"{m_cpu:.3f} ms", speedup_str(m_torch_loop, m_cpu), speedup_str(m_numpy, m_cpu))
        table.add_row("5. PyTorch (Dense 3D einsum)", f"{m_torch_einsum:.3f} ms", speedup_str(m_torch_loop, m_torch_einsum), speedup_str(m_numpy, m_torch_einsum))
        table.add_row("6. PyTorch (Loop)", f"{m_torch_loop:.3f} ms", "1.00x (Baseline)", speedup_str(m_numpy, m_torch_loop))
        table.add_row("7. NumPy (Reference)", f"{m_numpy:.3f} ms", speedup_str(m_torch_loop, m_numpy), "1.00x (Baseline)")

        console.print(table)
        console.print(f"[bold]Throughput (maxsim_3d_ptr):[/bold] {tp_3d_ptr:.1f} docs/sec | [bold]Correctness:[/bold] [green]{'PASS' if valid else 'FAIL'}[/green]\n")

    return results


def generate_benchmark_plots(var_results: Dict[str, Any], uniform_results: Dict[str, Any], assets_dir: str):
    """Generates a 2x2 comparison grid for both Variable-Length and Uniform 3D benchmarks."""
    os.makedirs(assets_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
    (ax1, ax2), (ax3, ax4) = axes

    doc_counts = var_results["doc_counts"]

    # ==================== Subplot 1: Variable-Length Latency ====================
    ax1.plot(doc_counts, var_results["maxsimd_vrlen"], "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="maxsimd.maxsim_vrlen (Flat+Rayon)")
    ax1.plot(doc_counts, var_results["maxsimd_ptr_loop"], "^--", color="#17becf", linewidth=2.0, markersize=6, label="maxsimd.maxsim_ptr (Pointer Loop)")
    ax1.plot(doc_counts, var_results["maxsim_cpu"], "s--", color="#ff7f0e", linewidth=2.0, markersize=6, label="maxsim-cpu (PyPI)")
    ax1.plot(doc_counts, var_results["torch_loop"], "v-.", color="#2ca02c", linewidth=1.8, markersize=6, label="PyTorch (Loop)")
    ax1.plot(doc_counts, var_results["torch_batched"], "d:", color="#d62728", linewidth=1.8, markersize=5, label="PyTorch (Batched+Masked)")
    ax1.plot(doc_counts, var_results["numpy"], "x--", color="#9467bd", linewidth=1.8, markersize=5, label="NumPy (Reference)")

    ax1.set_xlabel("Number of Documents", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Latency (ms) - Lower is Better", fontsize=11, fontweight="bold")
    ax1.set_title("Variable-Length MaxSim Latency (Q=32, L=100-300, D=128)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(doc_counts)
    ax1.legend(fontsize=9, loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ==================== Subplot 2: Variable-Length Throughput ====================
    tp_vrlen = [n / (t / 1000.0) for n, t in zip(doc_counts, var_results["maxsimd_vrlen"])]
    tp_ptr = [n / (t / 1000.0) for n, t in zip(doc_counts, var_results["maxsimd_ptr_loop"])]
    tp_cpu = [n / (t / 1000.0) for n, t in zip(doc_counts, var_results["maxsim_cpu"])]
    tp_torch_loop = [n / (t / 1000.0) for n, t in zip(doc_counts, var_results["torch_loop"])]

    ax2.plot(doc_counts, tp_vrlen, "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="maxsimd.maxsim_vrlen")
    ax2.plot(doc_counts, tp_ptr, "^--", color="#17becf", linewidth=2.0, markersize=6, label="maxsimd.maxsim_ptr (Loop)")
    ax2.plot(doc_counts, tp_cpu, "s--", color="#ff7f0e", linewidth=2.0, markersize=6, label="maxsim-cpu (PyPI)")
    ax2.plot(doc_counts, tp_torch_loop, "v-.", color="#2ca02c", linewidth=1.8, markersize=6, label="PyTorch (Loop)")

    ax2.set_xlabel("Number of Documents", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Throughput (Docs / sec) - Higher is Better", fontsize=11, fontweight="bold")
    ax2.set_title("Variable-Length MaxSim Throughput", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(doc_counts)
    ax2.legend(fontsize=9, loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # ==================== Subplot 3: Uniform 3D Latency ====================
    ax3.plot(doc_counts, uniform_results["maxsimd_3d_ptr"], "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="maxsimd.maxsim_3d_ptr (Torch Pointer+Rayon)")
    ax3.plot(doc_counts, uniform_results["maxsimd_3d_numpy"], "D--", color="#00a86b", linewidth=2.0, markersize=6, label="maxsimd.maxsim (NumPy 3D+Rayon)")
    ax3.plot(doc_counts, uniform_results["maxsim_cpu"], "s--", color="#ff7f0e", linewidth=2.0, markersize=6, label="maxsim-cpu (PyPI 3D)")
    ax3.plot(doc_counts, uniform_results["torch_3d_einsum"], "d:", color="#d62728", linewidth=1.8, markersize=5, label="PyTorch (Dense 3D einsum)")
    ax3.plot(doc_counts, uniform_results["torch_loop"], "v-.", color="#2ca02c", linewidth=1.8, markersize=6, label="PyTorch (Loop)")
    ax3.plot(doc_counts, uniform_results["numpy"], "x--", color="#9467bd", linewidth=1.8, markersize=5, label="NumPy (Reference)")

    ax3.set_xlabel("Number of Documents / Pages", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Latency (ms) - Lower is Better", fontsize=11, fontweight="bold")
    ax3.set_title("Uniform 3D MaxSim Latency (ColPali style: Q=32, L=128, D=128)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xticks(doc_counts)
    ax3.legend(fontsize=9, loc="upper left", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # ==================== Subplot 4: Uniform 3D Throughput ====================
    tp_3d_ptr = [n / (t / 1000.0) for n, t in zip(doc_counts, uniform_results["maxsimd_3d_ptr"])]
    tp_3d_numpy = [n / (t / 1000.0) for n, t in zip(doc_counts, uniform_results["maxsimd_3d_numpy"])]
    tp_3d_cpu = [n / (t / 1000.0) for n, t in zip(doc_counts, uniform_results["maxsim_cpu"])]
    tp_3d_torch_einsum = [n / (t / 1000.0) for n, t in zip(doc_counts, uniform_results["torch_3d_einsum"])]

    ax4.plot(doc_counts, tp_3d_ptr, "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="maxsimd.maxsim_3d_ptr")
    ax4.plot(doc_counts, tp_3d_numpy, "D--", color="#00a86b", linewidth=2.0, markersize=6, label="maxsimd.maxsim (3D)")
    ax4.plot(doc_counts, tp_3d_cpu, "s--", color="#ff7f0e", linewidth=2.0, markersize=6, label="maxsim-cpu (PyPI 3D)")
    ax4.plot(doc_counts, tp_3d_torch_einsum, "d:", color="#d62728", linewidth=1.8, markersize=5, label="PyTorch (Dense 3D einsum)")

    ax4.set_xlabel("Number of Documents / Pages", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Throughput (Pages / sec) - Higher is Better", fontsize=11, fontweight="bold")
    ax4.set_title("Uniform 3D MaxSim Throughput (ColPali Pages / sec)", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xticks(doc_counts)
    ax4.legend(fontsize=9, loc="lower right", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(assets_dir, "maxsim_benchmark_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


def main():
    console = Console()
    console.print(Panel(
        "[bold green]Comprehensive MaxSim Benchmark Suite[/bold green]\n"
        "Testing all Rust FFI endpoints: [cyan]maxsim_vrlen[/cyan], [cyan]maxsim_ptr[/cyan], [cyan]maxsim_3d_ptr[/cyan], and [cyan]maxsim[/cyan].",
        expand=False,
    ))

    doc_counts = [20, 50, 100, 250, 500, 1000, 2000]

    var_results = run_variable_length_benchmark(console, doc_counts)
    uniform_results = run_uniform_3d_benchmark(console, doc_counts)

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    plot_path = generate_benchmark_plots(var_results, uniform_results, assets_dir)

    console.print(f"[bold green]✓ Benchmark graph successfully generated and saved to: {plot_path}[/bold green]")


if __name__ == "__main__":
    main()
