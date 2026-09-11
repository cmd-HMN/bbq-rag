from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jinja2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
TEMPLATE_PATH = ASSETS_DIR / "template.html"
DEFAULT_BENCH_DIR = REPO_ROOT / "bench"


class HtmlPage:
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

    def add_plotly(
        self,
        fig: Any,
        caption: str = "",
        include_plotlyjs: str = "cdn",
    ) -> "HtmlPage":
        html_str = (
            fig.to_html(
                full_html=False,
                include_plotlyjs=include_plotlyjs,
                config={"responsive": True, "displayModeBar": True},
            )
            if hasattr(fig, "to_html")
            else str(fig)
        )
        self.elements.append({"type": "plotly", "html": html_str, "caption": caption})
        return self

    def add_table(self, headers: List[str], rows: List[List[Any]]) -> "HtmlPage":
        self.elements.append({"type": "table", "headers": headers, "rows": rows})
        return self

    def save(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
        template = jinja2.Template(template_text)
        html = template.render(title=self.title, elements=self.elements)
        self.output_path.write_text(html, encoding="utf-8")
        return self.output_path


class Report:
    """Manages suite bench folder (bench/<suite>/) and artifact creation."""

    def __init__(
        self, suite: str = "maxsimd", bench_dir: Union[str, Path] = DEFAULT_BENCH_DIR
    ):
        self.suite = suite
        self.bench_dir = Path(bench_dir)
        self.suite_dir = self.bench_dir / self.suite
        self.images_dir = self.suite_dir / "images"

        self.suite_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(
        self,
        fig: Any,
        name: str,
        asset_dir: Optional[Union[str, Path, bool]] = None,
        to_assets: bool = False,
    ) -> Path:
        """Save a matplotlib figure to the suite images directory (and optionally asset folder)."""
        filename = name if name.endswith(".png") else f"{name}.png"
        png_path = self.images_dir / filename
        fig.tight_layout()
        fig.savefig(str(png_path), dpi=300)
        plt.close(fig)

        target = (
            asset_dir if asset_dir is not None else (ASSETS_DIR if to_assets else None)
        )
        if target and str(target).lower() not in ("false", "0", "no", "none"):
            out_dir = (
                ASSETS_DIR if target is True or target == "assets" else Path(target)
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / filename).write_bytes(png_path.read_bytes())
        return png_path

    save_img = save_fig

    def create_html(self, name: str, title: Optional[str] = None) -> HtmlPage:
        filename = name if name.endswith(".html") else f"{name}.html"
        return HtmlPage(name=name, output_path=self.suite_dir / filename, title=title)
