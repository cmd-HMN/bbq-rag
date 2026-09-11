import re
from typing import List

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from rich.console import Console

from benchmarks.common import mark_me


def single_run(
    c: Console,
    q_len: int = 32,
    tokens_per_doc: int = 128,
    num_docs: int = 1000,
    dim: int = 128,
    seed: int = 42,
    jobs: int = -1,
):
    tag = f"[[bold red]docs: {num_docs}[/bold red]]"
    col_width = min(42, max(20, c.width - 20))

    with c.status("", spinner="bouncingBall") as status:
        # update status
        def set_status(msg: str, **kwargs):
            visible_len = len(re.sub(r"\[.*?\]", "", msg))
            spacing = " " * max(1, col_width - visible_len)
            status.update(f"{msg}{spacing}{tag}", **kwargs)

        set_status("Importing dependencies")
        from benchmarks.maxsimd.functions import (
            bm_maxsim,
            bm_maxsim_cpu,
            bm_numpy,
            bm_torch_simple,
            bm_torch_einsum,
        )
        from benchmarks.common import np, torch

        set_status("[bold cyan]Initializing environment and data...[/bold cyan]")

        np.random.seed(seed)
        torch.manual_seed(seed)

        q_mat = np.random.randn(q_len, dim).astype(np.float32)
        q_mat = q_mat / np.linalg.norm(q_mat, axis=-1, keepdims=True)

        docs_3d = np.random.randn(num_docs, tokens_per_doc, dim).astype(np.float32)
        docs_3d = docs_3d / np.linalg.norm(docs_3d, axis=-1, keepdims=True)

        q_tensor = torch.from_numpy(q_mat)
        docs_3d_tensor = torch.from_numpy(docs_3d)

        set_status("[bold cyan]Running benchmarks...[/bold cyan]")

        bm = [
            ("MaxSim", bm_maxsim, (q_tensor, docs_3d_tensor, jobs)),
            ("MaxSim-CPU", bm_maxsim_cpu, (q_mat, docs_3d)),
            ("NumPy", bm_numpy, (q_mat, docs_3d)),
            ("PyTorch Simple", bm_torch_simple, (q_tensor, docs_3d_tensor)),
            ("PyTorch Einsum", bm_torch_einsum, (q_tensor, docs_3d_tensor)),
        ]

        results = []

        for name, b, args in bm:
            set_status(
                f"[bold green]Running {name}[/bold green]", spinner="bouncingBall"
            )
            mean_ms, std_ms = mark_me(b, args)
            results.append((name, mean_ms, std_ms))

    return results


def greport(
    c: Console,
    r: dict,
    docs: List[int],
    q_len: int = 32,
    dim: int = 128,
    jobs: int = -1,
    asset_dir=None,
):

    series_data = {}
    for d in docs:
        for name, mean_ms, std_ms in r[d]:
            if name not in series_data:
                series_data[name] = {"means": [], "stds": []}
            series_data[name]["means"].append(mean_ms)
            series_data[name]["stds"].append(std_ms)

    from benchmarks.report import Report

    report = Report(suite="maxsimd")

    # evaluating using throughput
    throughput_data = {}
    for name, metrics in series_data.items():
        tp_means = [
            d / (m / 1000.0) if m > 0 else 0.0 for d, m in zip(docs, metrics["means"])
        ]
        tp_stds = [
            tp * (s / m) if m > 0 else 0.0
            for tp, m, s in zip(tp_means, metrics["means"], metrics["stds"])
        ]
        throughput_data[name] = {"means": tp_means, "stds": tp_stds}

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    for name, metrics in throughput_data.items():
        ax.plot(
            docs,
            metrics["means"],
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=name,
        )
    ax.set_title(
        "Throughput vs Documents (Higher is Better)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Document Count", fontsize=10, fontweight="bold")
    ax.set_ylabel("Throughput (Docs / sec)", fontsize=10, fontweight="bold")
    ax.set_xticks(docs)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    img_path = report.save_fig(fig, "b1", asset_dir=asset_dir)
    c.print(
        f"\n[bold green]Image saved to:[/bold green] [underline]{img_path}[/underline]"
    )

    page = report.create_html("maxsimd_benchmark", title="MaxSimd Benchmark Report")

    page.add_heading("MaxSimd Performance Benchmark", level=1, align="center")
    page.add_text(
        f"Benchmarking MaxSimd with (jobs={jobs}, dim={dim}, q_len={q_len}).",
        align="center",
        muted=True,
    )
    plotly_fig = go.Figure(
        data=[
            go.Scatter(
                x=list(docs),
                y=metrics["means"],
                mode="lines+markers",
                name=name,
                line=dict(width=2.5),
                marker=dict(size=7),
            )
            for name, metrics in throughput_data.items()
        ],
        layout=go.Layout(
            title="Throughput vs Documents (Higher is Better)",
            xaxis_title="Document Count",
            yaxis_title="Throughput (Docs / sec)",
            template="plotly_white",
        ),
    )
    page.add_plotly(plotly_fig)

    urls = {
        "MaxSim": "https://github.com/cmd-HMN/bbq-rag",
        "MaxSim-CPU": "https://pypi.org/project/maxsim-cpu/",
        "NumPy": "https://numpy.org",
        "PyTorch Simple": "https://pytorch.org",
        "PyTorch Einsum": "https://pytorch.org",
    }
    table_headers = ["Function"] + [f"{d} Docs" for d in docs]
    table_rows = []
    for name, metrics in series_data.items():
        url = urls.get(name)
        name_html = (
            f"<a href='{url}' target='_blank'><strong>{name}</strong></a>"
            if url
            else f"<strong>{name}</strong>"
        )
        row = [name_html]
        for i, (m, s) in enumerate(zip(metrics["means"], metrics["stds"])):
            tp = throughput_data[name]["means"][i]
            row.append(
                f"<div><strong>{m:.3f}</strong> <span class='ci-bound'>±{s:.3f} ms</span></div>"
                f"<div style='color: #1F78B4; font-size: 12px;'>{tp:,.0f} docs/s</div>"
            )
        table_rows.append(row)

    page.add_heading("Benchmark Measurements (Mean ± Std & Throughput):", level=4)
    page.add_table(headers=table_headers, rows=table_rows)

    html_path = page.save()
    c.print(
        f"[bold green]HTML report saved to:[/bold green] [underline]{html_path}[/underline]"
    )


def run(
    q_len: int = 32,
    # dim
    tokens_per_doc: int = 128,
    docs: List[int] = [
        20,
        50,
        100,
        250,
        500,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
    ],
    dim: int = 128,
    seed: int = 42,
    jobs: int = -1,
    report: bool = True,
    asset_dir=None,
):
    r = {}

    c = Console()

    for d in docs:
        r[d] = single_run(
            c,
            q_len=q_len,
            tokens_per_doc=tokens_per_doc,
            num_docs=d,
            dim=dim,
            seed=seed,
            jobs=jobs,
        )

    if report:
        greport(c, r, docs, q_len=q_len, dim=dim, jobs=jobs, asset_dir=asset_dir)
    else:
        for d in docs:
            c.print(f"[bold red]Results for {d} documents:[/bold red]")
            for name, mean_ms, std_ms in r[d]:
                c.print(
                    f"{name}: [bold green]{mean_ms:.2f}ms[/bold green] ± [bold red]{std_ms:.2f}ms[/bold red]"
                )
