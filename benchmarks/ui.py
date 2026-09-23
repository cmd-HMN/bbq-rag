import os
from typing import Optional
from bbq.src.terminal.tui import print_bbq as _bbq_print_bbq

ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)


def print_bbq(
    logo_path: Optional[str] = None,
    name: str = "BB",
    tag: str = "BENCHMARK",
    without_logo: bool = False,
) -> None:
    path = logo_path or os.path.join(ASSETS_DIR, "logo.txt")
    _bbq_print_bbq(
        logo_path=path,
        name=name,
        tag=tag,
        without_logo=without_logo,
    )


__all__ = [
    "print_bbq",
]
