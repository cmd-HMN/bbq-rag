from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import jinja2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
TEMPLATE_PATH = ASSETS_DIR / "template.html"
DEFAULT_BENCH_DIR = REPO_ROOT / "bench"

DEFAULT_PALETTE = [
    "#2563eb",  # Blue
    "#f59e0b",  # Amber
    "#10b981",  # Emerald
    "#ef4444",  # Red
    "#8b5cf6",  # Purple
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
]


class HtmlPage:
    """Criterion-style HTML Page builder."""

    def __init__(self, name: str, output_path: Path, title: Optional[str] = None):
        self.name = name
        self.output_path = output_path
        self.title = title or name.replace("_", " ").title()
        self.elements: List[Dict[str, Any]] = []

    def _normalize_align(self, align: str) -> str:
        a = align.lower().strip()
        return "center" if a in ("middle", "center") else a

    def add_heading(self, text: str, level: int = 2, align: str = "left") -> "HtmlPage":
        self.elements.append(
            {
                "type": "heading",
                "text": text,
                "level": min(max(1, level), 6),
                "align": self._normalize_align(align),
            }
        )
        return self

    def add_text(
        self, text: str, align: str = "left", muted: bool = False
    ) -> "HtmlPage":
        self.elements.append(
            {
                "type": "text",
                "text": text,
                "align": self._normalize_align(align),
                "muted": muted,
            }
        )
        return self

    def add_image(
        self,
        src: str,
        caption: str = "",
        align: str = "center",
        width: Optional[str] = None,
    ) -> "HtmlPage":
        self.elements.append(
            {
                "type": "image",
                "src": src,
                "caption": caption,
                "align": self._normalize_align(align),
                "width": width,
            }
        )
        return self

    def add_plotly(
        self,
        fig: Any,
        caption: str = "",
        include_plotlyjs: str = "cdn",
    ) -> "HtmlPage":
        if hasattr(fig, "to_html"):
            html_str = fig.to_html(
                full_html=False,
                include_plotlyjs=include_plotlyjs,
                config={"responsive": True, "displayModeBar": True},
            )
        else:
            html_str = str(fig)
        self.elements.append({"type": "plotly", "html": html_str, "caption": caption})
        return self

    def add_table(self, headers: List[str], rows: List[List[Any]]) -> "HtmlPage":
        self.elements.append(
            {
                "type": "table",
                "headers": headers,
                "rows": rows,
            }
        )
        return self

    def add_metrics_table(
        self,
        x_values: Sequence[Union[int, float]],
        series_data: Dict[str, Dict[str, List[float]]],
        throughput_data: Optional[Dict[str, Dict[str, List[float]]]] = None,
        function_urls: Optional[Dict[str, str]] = None,
        x_unit: str = "Docs",
        title: str = "Detailed Measurements (Mean ± Std & Throughput):",
    ) -> "HtmlPage":
        if title:
            self.elements.append(
                {"type": "heading", "text": title, "level": 4, "align": "left"}
            )

        urls = function_urls or {}
        headers = ["Function"] + [f"{x} {x_unit}" for x in x_values]
        rows = []

        for name, metrics in series_data.items():
            means = metrics["means"]
            stds = metrics.get("stds", [0.0] * len(means))
            tps = (
                throughput_data[name]["means"]
                if throughput_data and name in throughput_data
                else None
            )

            url = urls.get(name)
            name_html = (
                f"<a href='{url}' target='_blank'><strong>{name}</strong></a>"
                if url
                else f"<strong>{name}</strong>"
            )

            row = [name_html]
            for i, (m, s) in enumerate(zip(means, stds)):
                cell = f"<div><strong>{m:.3f}</strong> <span class='ci-bound'>±{s:.3f} ms</span></div>"
                if tps is not None and i < len(tps):
                    cell += f"<div style='color: #1F78B4; font-size: 12px;'>{tps[i]:,.0f} docs/s</div>"
                row.append(cell)
            rows.append(row)

        return self.add_table(headers=headers, rows=rows)

    def save(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
        template = jinja2.Template(template_text)
        html = template.render(title=self.title, elements=self.elements)
        self.output_path.write_text(html, encoding="utf-8")
        return self.output_path


class Report:
    """Manages suite bench folder (bench/<suite>/) and file creation."""

    def __init__(
        self, suite: str = "maxsimd", bench_dir: Union[str, Path] = DEFAULT_BENCH_DIR
    ):
        self.suite = suite
        self.bench_dir = Path(bench_dir)
        self.suite_dir = self.bench_dir / self.suite
        self.images_dir = self.suite_dir / "images"

        self.suite_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def save_image(
        self,
        name: str,
        x_values: Sequence[Union[int, float]],
        series_data: Dict[str, Dict[str, List[float]]],
        title: str = "Throughput vs Documents",
        x_label: str = "Document Count",
        y_label: str = "Throughput (Docs / sec)",
    ) -> Path:
        filename = name if name.endswith(".png") else f"{name}.png"
        png_path = self.images_dir / filename

        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "default"
        )
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

        for idx, (s_name, metrics) in enumerate(series_data.items()):
            color = metrics.get("color") or DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
            means = metrics["means"]
            stds = metrics.get("stds", [0.0] * len(means))

            ax.plot(
                x_values,
                means,
                marker="o",
                linewidth=2.2,
                markersize=5,
                label=s_name,
                color=color,
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(x_label, fontsize=10, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=10, fontweight="bold")
        ax.set_xticks(x_values)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(str(png_path), dpi=300)
        plt.close(fig)

        return png_path

    def create_plotly_figure(
        self,
        x_values: Sequence[Union[int, float]],
        series_data: Dict[str, Dict[str, List[float]]],
        title: str = "Throughput vs Documents",
        x_label: str = "Document Count",
        y_label: str = "Throughput (Docs / sec)",
    ) -> Any:
        import plotly.graph_objects as go

        fig = go.Figure()
        for idx, (name, metrics) in enumerate(series_data.items()):
            color = metrics.get("color") or DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=list(x_values),
                    y=metrics["means"],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=7),
                    hovertemplate=f"<b>{name}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y:,.0f}}<extra></extra>",
                )
            )
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, family="Helvetica Neue, Arial, sans-serif"),
            ),
            xaxis=dict(
                title=dict(text=x_label, font=dict(weight="bold")),
                showgrid=True,
                gridcolor="#f0f0f0",
                tickmode="array",
                tickvals=list(x_values),
            ),
            yaxis=dict(
                title=dict(text=y_label, font=dict(weight="bold")),
                showgrid=True,
                gridcolor="#f0f0f0",
            ),
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#e2e8f0",
                borderwidth=1,
            ),
            margin=dict(l=50, r=30, t=60, b=50),
            font=dict(family="Helvetica Neue, Arial, sans-serif"),
        )
        return fig

    def create_html(self, name: str, title: Optional[str] = None) -> HtmlPage:
        filename = name if name.endswith(".html") else f"{name}.html"
        page_path = self.suite_dir / filename
        return HtmlPage(name=name, output_path=page_path, title=title)
