import os
import re
import logging
from typing import Optional, Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.markdown import Markdown
from rich.table import Table

from bbq.src.config import Config

CYAN = "\033[1;36m"
RESET = "\033[0m"
DIM = "\033[90m"
BOLD = "\033[1m"


def create_rich_console_logging_handler() -> RichHandler:
    return RichHandler(
        console=Console(),
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )


def render_server_status_rich_panel(config: Config) -> Panel:
    table = Table(show_header=False, expand=True, box=None)
    table.add_column("Key", style="bold green", width=22)
    table.add_column("Value", style="bold white")

    table.add_row("Base Model ID:", config.base_model_id)
    table.add_row("Adapter Model ID:", config.lora_adapter_id or "None")
    table.add_row("Device Preference:", config.device)
    table.add_row("Watch Directory:", config.watch_folder_path)
    table.add_row("Embeddings Directory:", config.embeddings_output_path)
    table.add_row("SQLite DB Path:", config.sqlite_db_path)

    return Panel(
        table,
        title="[bold yellow]BBQ RAG Persistent Document-Indexing Server[/bold yellow]",
        border_style="cyan",
    )


def configure_rich_logging_for_server() -> None:
    root_logger = logging.getLogger("bbq")
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    rich_handler = create_rich_console_logging_handler()
    root_logger.addHandler(rich_handler)


def _extract_server_label(server: Optional[Any]) -> Optional[str]:
    if server is None:
        return None
    s = str(server).strip()
    if not s:
        return None
    if s.isdigit():
        return s
    try:
        from urllib.parse import urlparse

        parsed = urlparse(s if "://" in s else f"//{s}")
        if parsed.port:
            return str(parsed.port)
        if parsed.hostname:
            return parsed.hostname
    except Exception:
        pass
    return s


def print_bbq(
    logo_path: Optional[str] = None,
    name: Optional[str] = None,
    tag: str = "CLIENT",
    without_logo: bool = False,
    server: Optional[Any] = None,
    top_k: Optional[int] = None,
) -> None:
    """
    Prints the stylized ASCII BBQ logo banner with tag, server, and top-k annotations.
    Format: BBQ [CLIENT] [8000] [top-10]
    If without_logo is True, suppression is respected and nothing is printed.
    """
    if without_logo:
        return

    path = logo_path
    if not path or not os.path.exists(path):
        for candidate in [
            os.path.join(os.getcwd(), "assets", "logo.txt"),
            os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                ),
                "assets",
                "logo.txt",
            ),
            os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "assets",
                "logo.txt",
            ),
        ]:
            if os.path.exists(candidate):
                path = candidate
                break

    server_label = _extract_server_label(server)
    plain_tokens = ["BBQ", f"[{tag.upper()}]"]
    styled_tokens = [f"{BOLD}BBQ{RESET}", f"{CYAN}[{tag.upper()}]{RESET}"]

    if server_label:
        plain_tokens.append(f"[{server_label}]")
        styled_tokens.append(f"{CYAN}[{server_label}]{RESET}")

    if top_k is not None:
        plain_tokens.append(f"[top-{top_k}]")
        styled_tokens.append(f"{CYAN}[top-{top_k}]{RESET}")

    if name and name.upper() not in ("BBQ", "BBQ-RAG", "CLIENT", "SERVER", tag.upper()):
        plain_tokens.append(f"[{name}]")
        styled_tokens.append(f"{CYAN}[{name}]{RESET}")

    plain_header = " ".join(plain_tokens)
    styled_header = " ".join(styled_tokens)

    if not path or not os.path.exists(path):
        print(f"\n{'=' * 55}\n          {plain_header}\n{'=' * 55}\n")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = [line.rstrip("\r\n") for line in f if line.strip()]
    except Exception:
        return

    ansi = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)
    vlen = max((len(ansi(line)) for line in data), default=0)
    mid = len(data) // 2

    divider_len = len(plain_header)
    divider = "─" * divider_len

    print()
    for i, line in enumerate(data):
        vslen = len(ansi(line))
        spacing = " " * (vlen - vslen + 4)
        if i == mid - 1:
            print(f"{line}{spacing}{DIM}│{RESET}  {styled_header}")
        elif i == mid:
            print(f"{line}{spacing}{DIM}│{RESET}  {DIM}{divider}{RESET}")
        else:
            print(line)
    print()


def render_query_results_rich(
    results: list[dict],
    console: Optional[Console] = None,
) -> None:
    """
    Renders top retrieved PDF document pages using a clean list showing page number and document name.
    """
    console = console or Console()
    console.print(
        Rule(
            title=f"[bold yellow]Top {len(results)} Matching PDF Pages[/bold yellow]",
            style="yellow",
        )
    )
    for i, res in enumerate(results, 1):
        book_name = res.get("filename") or os.path.basename(res.get("file_path", ""))
        page_num = res.get("page_number", "?")
        total_pages = res.get("total_pages")
        if total_pages is not None and total_pages != "?":
            page_info = f"Page [bold yellow]{page_num}[/bold yellow] of {total_pages}"
        else:
            page_info = f"Page [bold yellow]{page_num}[/bold yellow]"

        console.print(
            f" [bold magenta]{i:2d}.[/bold magenta] {page_info} — [bold white]{book_name}[/bold white]"
        )
    console.print()


def open_document_page(file_path: str, page_number: int = 1) -> bool:
    """
    Opens the PDF document at the specified page using evince, zathura, okular, or xdg-open.
    """
    import shutil
    import subprocess

    if not file_path:
        return False

    target_path = file_path
    if not os.path.exists(target_path):
        candidate = os.path.join(os.getcwd(), file_path)
        if os.path.exists(candidate):
            target_path = candidate

    target_path = os.path.abspath(target_path)
    if not os.path.exists(target_path):
        return False

    # 1. Evince (supports -p PAGE)
    if shutil.which("evince"):
        try:
            subprocess.Popen(
                ["evince", "-p", str(page_number), target_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            pass

    # 2. Zathura (supports -P PAGE)
    if shutil.which("zathura"):
        try:
            subprocess.Popen(
                ["zathura", "-P", str(page_number), target_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            pass

    # 3. Okular (supports -p PAGE)
    if shutil.which("okular"):
        try:
            subprocess.Popen(
                ["okular", "-p", str(page_number), target_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            pass

    # 4. Fallback to xdg-open
    if shutil.which("xdg-open"):
        try:
            subprocess.Popen(
                ["xdg-open", target_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            pass

    return False


def _read_terminal_key() -> str:
    """Reads a single keypress or ANSI escape sequence from stdin in raw mode."""
    import sys
    import os
    import select

    if not sys.stdin.isatty():
        return "ESC"

    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        raw = os.read(fd, 32)
        if not raw:
            return "ESC"

        if raw == b"\x1b":
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                raw += os.read(fd, 16)

        # Arrow keys: support both CSI (\x1b[) and SS3 (\x1bO) sequences across all terminals
        if raw in (b"\x1b[A", b"\x1bOA", b"\x1b[1;2A", b"\x1b[1;5A"):
            return "UP"
        elif raw in (b"\x1b[B", b"\x1bOB", b"\x1b[1;2B", b"\x1b[1;5B"):
            return "DOWN"
        elif raw in (b"\x1b[C", b"\x1bOC"):
            return "RIGHT"
        elif raw in (b"\x1b[D", b"\x1bOD"):
            return "LEFT"
        elif raw == b"\x1b":
            return "ESC"
        elif raw in (b"\r", b"\n"):
            return "ENTER"
        elif raw in (b"\x03", b"\x04"):  # Ctrl+C or Ctrl+D
            return "CTRL_C"
        elif raw in (b"k", b"K"):
            return "UP"
        elif raw in (b"j", b"J"):
            return "DOWN"
        elif raw in (b"q", b"Q"):
            return "ESC"
        return "ESC"
    except Exception:
        return "ESC"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)




def interactive_page_picker(
    results: list[dict],
    console: Optional[Console] = None,
) -> None:
    """
    Launches an interactive arrow-key picker (Option C) allowing the user
    to navigate retrieved pages, press Enter to open in PDF viewer, and Esc/q to proceed.
    """
    import sys

    console = console or Console()

    if not results or not sys.stdin.isatty():
        render_query_results_rich(results, console=console)
        return

    from rich.live import Live
    from rich.console import Group
    from rich.text import Text

    selected_index = 0
    total_options = len(results) + 1
    status_msg: Optional[str] = None

    def build_picker_renderable() -> Group:
        items = []
        items.append(
            Rule(
                title=f"[bold yellow]Top {len(results)} Matching PDF Pages[/bold yellow]",
                style="yellow",
            )
        )
        items.append(
            Text.from_markup(
                "[dim cyan]  Use ↑/↓ or j/k to navigate • Enter to open page in viewer • Esc/q to proceed[/dim cyan]\n"
            )
        )

        for i, res in enumerate(results):
            book_name = res.get("filename") or os.path.basename(
                res.get("file_path", "")
            )
            page_num = res.get("page_number", "?")
            total_pages = res.get("total_pages")
            if total_pages is not None and total_pages != "?":
                page_str = f"Page {page_num} of {total_pages}"
            else:
                page_str = f"Page {page_num}"

            if i == selected_index:
                line = Text.from_markup(
                    f" [bold cyan]▶ {i + 1:2d}.[/bold cyan] [bold yellow]{page_str}[/bold yellow] — [bold white]{book_name}[/bold white]"
                )
            else:
                line = Text.from_markup(
                    f"   [dim]{i + 1:2d}.[/dim] {page_str} — [dim]{book_name}[/dim]"
                )
            items.append(line)

        done_idx = len(results)
        if selected_index == done_idx:
            items.append(
                Text.from_markup(
                    "\n [bold green]▶ ✔ Done / Proceed to query prompt (Esc/q)[/bold green]"
                )
            )
        else:
            items.append(
                Text.from_markup(
                    "\n   [dim]✔ Done / Proceed to query prompt (Esc/q)[/dim]"
                )
            )

        if status_msg:
            items.append(Text.from_markup(f"\n  {status_msg}"))

        return Group(*items)

    try:
        with Live(
            build_picker_renderable(),
            console=console,
            auto_refresh=False,
            transient=True,
        ) as live:
            while True:
                live.update(build_picker_renderable())
                live.refresh()

                key = _read_terminal_key()

                if key == "UP":
                    selected_index = (selected_index - 1) % total_options
                elif key == "DOWN":
                    selected_index = (selected_index + 1) % total_options
                elif key == "ENTER":
                    if selected_index == len(results):
                        break
                    res = results[selected_index]
                    fpath = res.get("file_path", "")
                    pnum = res.get("page_number", 1)
                    fname = res.get("filename") or os.path.basename(fpath)
                    opened = open_document_page(fpath, page_number=pnum)
                    if opened:
                        status_msg = f"[bold green]✓ Opened Page {pnum} of {fname} in PDF viewer[/bold green]"
                    else:
                        status_msg = f"[bold red]✗ Could not open {fname} (viewer not found)[/bold red]"
                elif key in ("ESC", "CTRL_C", "CTRL_D"):
                    break
    except Exception:
        pass

    # Finally print clean static list into terminal scrollback history
    render_query_results_rich(results, console=console)


def render_llm_answer_rich(
    answer: str,
    engine: str = "gemini",
    console: Optional[Console] = None,
) -> None:
    """
    Renders LLM multimodal answer in a rich Panel using Markdown formatting.
    """
    console = console or Console()
    console.print(
        Panel(
            Markdown(answer),
            title=f"[bold green]GEMINI MULTIMODAL ANSWER ({engine})[/bold green]",
            title_align="left",
            border_style="green",
            padding=(1, 2),
        )
    )


def print_interactive_help(console: Optional[Console] = None) -> None:
    """
    Displays the help instructions for the interactive query prompt loop.
    """
    console = console or Console()
    help_content = (
        "[bold white]• Enter any query[/bold white]       Search across indexed PDF documents\n"
        "[bold white]• help[/bold white] or [bold white]?[/bold white]               Display this interactive commands help\n"
        "[bold white]• clear[/bold white] or [bold white]cls[/bold white]            Clear the terminal screen\n"
        "[bold white]• q[/bold white], [bold white]quit[/bold white], or [bold white]exit[/bold white]     Quit and exit interactive prompt\n"
        "[bold white]• Ctrl+C[/bold white]                    Cancel query or terminate session"
    )
    console.print(
        Panel(
            help_content,
            title=r"[bold yellow]bbq\[query] Interactive Commands[/bold yellow]",
            title_align="left",
            border_style="yellow",
            padding=(0, 1),
        )
    )
