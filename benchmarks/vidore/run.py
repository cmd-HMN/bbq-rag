import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from rich.console import Console

from benchmarks.common import np, time, torch
from benchmarks.maxsimd.functions import bm_maxsim, bm_maxsim_cpu, bm_torch_einsum
from benchmarks.report import Report
from benchmarks.vidore.data import VidoreData
from benchmarks.vidore.emb import Emb
from benchmarks.vidore.eval import Evaluator

os.environ.update(
    {
        "LIBXSMM_VERBOSE": "0",
        "TQDM_DISABLE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }
)
warnings.filterwarnings("ignore")

try:
    import transformers.utils.logging as tf_log

    tf_log.set_verbosity_error()
    tf_log.disable_progress_bar()
except ImportError:
    pass

try:
    import datasets.utils.logging as ds_log

    ds_log.disable_progress_bar()
except ImportError:
    pass

ENGINE_COLORS = {
    "MaxSimd (Rust)": "#2563eb",
    "MaxSim-CPU": "#f59e0b",
    "PyTorch Einsum": "#8b5cf6",
}
ENGINES = list(ENGINE_COLORS.keys())


def prepare_data(
    dataset_key: str,
    embedder: Emb,
    force: bool = False,
    data_dir: str = "data/vidore",
    emb_dir: str = "data/embeddings",
    status_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    d = VidoreData(data=data_dir)
    embs = embedder(
        d.get(dataset_key),
        key=dataset_key,
        odir=emb_dir,
        force=force,
        status_cb=status_cb,
    )
    q_np, c_np = embs["queries_emb"], embs["corpus_emb"]
    return {
        "queries_np": q_np,
        "corpus_np": c_np,
        "queries_tensor": torch.from_numpy(q_np).float(),
        "corpus_tensor": torch.from_numpy(c_np).float(),
        "queries": embs["queries"][: len(q_np)],
        "doc_ids": embs["doc_ids"][: len(c_np)],
    }


def run_engine(
    score_fn: Callable,
    q_data: Any,
    d_data: Any,
    num_queries: int,
) -> Tuple[np.ndarray, List[float]]:
    old_err, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        score_fn(q_data[0], d_data)
    finally:
        os.dup2(old_err, 2)
        os.close(devnull)
        os.close(old_err)

    scores, latencies = [], []
    for i in range(num_queries):
        t0 = time.perf_counter()
        scores.append(score_fn(q_data[i], d_data))
        latencies.append(time.perf_counter() - t0)
    return np.array(scores), latencies


def evaluate_dataset(
    data: Dict[str, Any],
    set_status_fn: Callable[[str], None],
) -> Dict[str, Any]:
    evaluator = Evaluator(queries=data["queries"], doc_ids=data["doc_ids"])
    engines = [
        (ENGINES[0], bm_maxsim, data["queries_tensor"], data["corpus_tensor"]),
        (ENGINES[1], bm_maxsim_cpu, data["queries_np"], data["corpus_np"]),
        (ENGINES[2], bm_torch_einsum, data["queries_tensor"], data["corpus_tensor"]),
    ]
    results = {}
    set_status_fn("Evaluating engines")
    for name, fn, q_in, d_in in engines:
        scores, latencies = run_engine(fn, q_in, d_in, len(data["queries"]))
        results[name] = evaluator.evaluate(scores, latencies_sec=latencies)

    baseline_ranks = results[ENGINES[2]]["rankings"]
    for res in results.values():
        res["parity"] = evaluator.parity_with(res["rankings"], baseline_ranks)
    return results


def _plotly_bar(
    title: str,
    y_title: str,
    series: Dict[str, List[float]],
    datasets: List[str],
    y_range: Optional[List[float]] = None,
) -> go.Figure:
    return go.Figure(
        data=[
            go.Bar(name=eng, x=datasets, y=vals, marker_color=ENGINE_COLORS[eng])
            for eng, vals in series.items()
        ],
        layout=go.Layout(
            title=title,
            barmode="group",
            xaxis_title="Dataset",
            yaxis_title=y_title,
            yaxis_range=y_range,
            template="plotly_white",
        ),
    )


def _make_bar_fig(
    categories: List[str],
    series: Dict[str, List[float]],
    title: str = "",
    y_label: str = "",
    ylim: Optional[Tuple[float, float]] = None,
) -> Any:
    num_cats = len(categories)
    fig, ax = plt.subplots(figsize=(max(10.0, num_cats * 1.3), 5.5), dpi=300)
    x = np.arange(num_cats)
    width = 0.8 / max(len(series), 1)

    for idx, (eng, vals) in enumerate(series.items()):
        offset = (idx - (len(series) - 1) / 2.0) * width
        rects = ax.bar(
            x + offset,
            vals,
            width,
            label=eng,
            color=ENGINE_COLORS.get(eng),
            edgecolor="white",
            linewidth=0.8,
        )
        if num_cats <= 6:
            ax.bar_label(rects, fmt="%.1f", padding=3, fontsize=7.5)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Dataset", fontsize=10, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    rot = 20 if any(len(str(c)) > 7 for c in categories) or num_cats > 4 else 0
    ax.set_xticklabels(
        categories,
        rotation=rot,
        ha="right" if rot > 0 else "center",
        fontsize=9.5,
        fontweight="medium",
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        fontsize=8.5,
        loc="upper right",
    )
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    return fig


def build_suite_report(
    datasets: List[str],
    all_results: Dict[str, Dict[str, Any]],
    asset_dir=None,
) -> Tuple[Path, List[Path]]:
    report = Report(suite="vidore")

    # Extract series metrics per engine across all datasets
    qps_data = {
        eng: [all_results[ds][eng]["latency"]["qps"] for ds in datasets]
        for eng in ENGINES
    }
    ndcg_data = {
        eng: [all_results[ds][eng]["ndcg_5"] for ds in datasets] for eng in ENGINES
    }
    lat_data = {
        eng: [all_results[ds][eng]["latency"]["mean_ms"] for ds in datasets]
        for eng in ENGINES
    }

    # Save PNG benchmark charts (b2 saved to asset_dir if specified)
    img_qps = report.save_fig(
        _make_bar_fig(
            datasets,
            qps_data,
            title="Throughput (QPS) across All ViDoRe Datasets (Higher is Better)",
            y_label="Queries Per Second (QPS)",
        ),
        "b2",
        asset_dir=asset_dir,
    )
    img_ndcg = report.save_fig(
        _make_bar_fig(
            datasets,
            ndcg_data,
            title="Retrieval Quality (nDCG@5) across All ViDoRe Datasets",
            y_label="nDCG@5 (%)",
            ylim=(0, 105),
        ),
        "b2_ndcg5",
        asset_dir=asset_dir,
    )
    img_lat = report.save_fig(
        _make_bar_fig(
            datasets,
            lat_data,
            title="Mean Query Latency (ms) across All ViDoRe Datasets (Lower is Better)",
            y_label="Mean Latency (ms)",
        ),
        "b2_latency",
        asset_dir=asset_dir,
    )

    page = report.create_html(
        "vidore_suite_benchmark",
        title="ViDoRe Full Suite Benchmark Report",
    )
    page.add_heading("ViDoRe Benchmark", level=1, align="center")
    page.add_text(
        f"Evaluated across {len(datasets)} official ViDoRe evaluation datasets.",
        align="center",
        muted=True,
    )

    page.add_heading("QPS across Datasets", level=2)
    page.add_plotly(
        _plotly_bar(
            "ViDoRe Datasets (Higher is Better)",
            "Queries Per Second (QPS)",
            qps_data,
            datasets,
        )
    )

    page.add_heading("nDCG@5 across Datasets", level=2)
    page.add_plotly(
        _plotly_bar(
            "Retrieval Quality (nDCG@5) across All Datasets",
            "nDCG@5 (%)",
            ndcg_data,
            datasets,
            y_range=[0, 105],
        )
    )

    # Summary table averaged across all datasets
    base_lat = float(
        np.mean([all_results[ds][ENGINES[2]]["latency"]["mean_ms"] for ds in datasets])
    )
    summary_rows = []
    for eng in ENGINES:
        res = [all_results[ds][eng] for ds in datasets]
        m_lat = float(np.mean([r["latency"]["mean_ms"] for r in res]))
        m_qps = float(np.mean([r["latency"]["qps"] for r in res]))
        speedup = f"{base_lat / m_lat:.2f}x" if m_lat > 0 else "1.00x"
        summary_rows.append(
            [
                f"<strong>{eng}</strong>",
                f"{m_lat:.2f} ms",
                f"{m_qps:.1f}",
                f"<strong>{speedup}</strong>",
                f"{np.mean([r['recall_1'] for r in res]):.1f}%",
                f"{np.mean([r['recall_5'] for r in res]):.1f}%",
                f"{np.mean([r['ndcg_5'] for r in res]):.1f}%",
                f"{np.mean([r['parity'] for r in res]):.1f}%",
            ]
        )

    page.add_heading(
        "Overall Benchmark Summary (Averaged across all datasets)", level=2
    )
    page.add_table(
        [
            "Engine",
            "Avg Latency",
            "Avg QPS",
            "Overall Speedup",
            "Mean Recall@1",
            "Mean Recall@5",
            "Mean nDCG@5",
            "PyTorch Parity",
        ],
        summary_rows,
    )

    # Detailed table per dataset
    detail_rows = [
        [
            f"<strong>{ds}</strong>",
            eng,
            f"{all_results[ds][eng]['latency']['mean_ms']:.2f} ms",
            f"{all_results[ds][eng]['latency']['qps']:.1f}",
            f"{all_results[ds][eng]['recall_1']:.1f}%",
            f"{all_results[ds][eng]['recall_5']:.1f}%",
            f"{all_results[ds][eng]['ndcg_5']:.1f}%",
            f"{all_results[ds][eng]['parity']:.1f}%",
        ]
        for ds in datasets
        for eng in ENGINES
    ]
    page.add_heading("Per-Dataset Detailed Breakdown", level=2)
    page.add_table(
        [
            "Dataset",
            "Engine",
            "Latency",
            "QPS",
            "Recall@1",
            "Recall@5",
            "nDCG@5",
            "Parity",
        ],
        detail_rows,
    )

    return page.save(), [img_qps, img_ndcg, img_lat]


def run(
    datasets: Optional[List[str]] = None,
    force: bool = False,
    report: bool = True,
    data_dir: str = "data/vidore",
    emb_dir: str = "data/embeddings",
    asset_dir=None,
):
    c = Console()
    d = VidoreData(data=data_dir)
    target_datasets = datasets or list(d.catalog.keys())
    total_datasets = len(target_datasets)
    col_width = min(46, max(20, c.width - 20))

    embedder = Emb()
    all_results = {}
    html_path, img_paths = None, []

    with c.status("", spinner="bouncingBall") as status:

        def set_status(msg: str, ds_idx: int, ds_key: str, **kwargs):
            tag = f"[[bold red]{ds_idx + 1}/{total_datasets} {ds_key}[/bold red]]"
            visible_len = len(re.sub(r"\[.*?\]", "", msg))
            spacing = " " * max(1, col_width - visible_len)
            status.update(f"{msg}{spacing}{tag}", **kwargs)

        for idx, ds_key in enumerate(target_datasets):

            def status_cb(msg: str):
                set_status(msg, idx, ds_key)

            set_status("Loading data & embeddings", idx, ds_key)
            data = prepare_data(
                dataset_key=ds_key,
                embedder=embedder,
                force=force,
                data_dir=data_dir,
                emb_dir=emb_dir,
                status_cb=status_cb,
            )
            all_results[ds_key] = evaluate_dataset(data=data, set_status_fn=status_cb)

        if report:
            status.update(
                "[bold cyan]Building full suite HTML report and charts...[/bold cyan]"
            )
            html_path, img_paths = build_suite_report(
                target_datasets, all_results, asset_dir=asset_dir
            )

    if report:
        for img_p in img_paths:
            c.print(
                f"[bold green]Image saved to:[/bold green] [underline]{img_p}[/underline]"
            )
        if html_path:
            c.print(
                f"[bold green]Report saved to:[/bold green] [underline]{html_path}[/underline]"
            )

    return all_results


if __name__ == "__main__":
    run()
