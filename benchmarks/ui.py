import os
from warnings import warn
import re

ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)
CYAN = "\033[1;36m"
RESET = "\033[0m"
DIM = "\033[90m"
BOLD = "\033[1m"


def print_bbq(
    logo_path: str = os.path.join(ASSETS_DIR, "logo.txt"), name: str = "BBQ-RAG"
) -> None:
    path = logo_path or os.path.join(ASSETS_DIR, "logo.txt")

    if not os.path.exists(path):
        print(f"\n{'=' * 55}\n          BENCHMARK [{name.upper()}]\n{'=' * 55}\n")
        warn(f"Logo file not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = [l_.rstrip("\r\n") for l_ in f if l_.strip()]

    ansi = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)
    vlen = max(len(ansi(_l)) for _l in data)

    mid = len(data) // 2

    print()
    for i, _l in enumerate(data):
        vslen = len(ansi(_l))

        spacing = " " * (vlen - vslen + 4)

        if i == mid - 1:
            print(
                f"{_l}{spacing}{DIM}│{RESET}  {BOLD}BENCHMARK{RESET} {CYAN}[{name}]{RESET}"
            )
        elif i == mid:
            divider = "─" * (len(name) + 14)
            print(f"{_l}{spacing}{DIM}│{RESET}  {DIM}{divider}{RESET}")
        else:
            print(_l)

    print()


__all__ = [
    "print_bbq",
]
