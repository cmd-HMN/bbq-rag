import logging
from typing import Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler

from bbq.src.utils.config import ModelConfigWrapper


def create_rich_console_logging_handler() -> RichHandler:
    return RichHandler(
        console=Console(),
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )


def render_server_status_rich_panel(config: ModelConfigWrapper) -> Panel:
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
