import importlib
from typing import Any

from bbq.src.common.base import (
    BaseModelLoader,
    BaseProcessor,
    BaseEngineWrapper,
)
from bbq.src.utils.pdf_utils import (
    compute_file_sha256_hash,
    extract_pdf_pages_to_pil_images,
    extract_single_pdf_page_image,
)

_LAZY_IMPORTS = {
    "EngineModelLoader": "bbq.src.utils.model_loader",
    "EngineWrapper": "bbq.src.utils.model_loader",
    "initialize_engine_from_yaml_config": "bbq.src.utils.model_loader",
    "ProcessedFilesTracker": "bbq.src.utils.tracker",
    "PDFWatchHandler": "bbq.src.utils.watcher",
    "start_pdf_folder_watcher": "bbq.src.utils.watcher",
    "create_rich_console_logging_handler": "bbq.src.utils.tui",
    "render_server_status_rich_panel": "bbq.src.utils.tui",
    "configure_rich_logging_for_server": "bbq.src.utils.tui",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseModelLoader",
    "BaseProcessor",
    "BaseEngineWrapper",
    "compute_file_sha256_hash",
    "extract_pdf_pages_to_pil_images",
    "extract_single_pdf_page_image",
]
