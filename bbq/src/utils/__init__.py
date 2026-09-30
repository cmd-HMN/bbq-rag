import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from bbq.src.common.base import BaseModelLoader, BaseProcessor, BaseEngineWrapper

    from bbq.src.utils.model_loader import initialize_engine

from bbq.src.utils.futils import get_system_cache_dir
from bbq.src.utils.pdf_utils import (
    compute_file_sha256_hash,
    extract_pdf_pages_to_pil_images,
    extract_single_pdf_page_image,
)

_LAZY_IMPORTS = {
    "BaseModelLoader": "bbq.src.common.base",
    "BaseProcessor": "bbq.src.common.base",
    "BaseEngineWrapper": "bbq.src.common.base",
    "EngineModelLoader": "bbq.src.utils.model_loader",
    "EngineWrapper": "bbq.src.utils.model_loader",
    "initialize_engine": "bbq.src.utils.model_loader",
    "initialize_engine_from_yaml_config": "bbq.src.utils.model_loader",
    "model_loader": "bbq.src.utils.model_loader",
    "watcher": "bbq.src.utils.watcher",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        target_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(target_path)
        if target_path.endswith(f".{name}"):
            return module
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseModelLoader",
    "BaseProcessor",
    "BaseEngineWrapper",
    "compute_file_sha256_hash",
    "extract_pdf_pages_to_pil_images",
    "extract_single_pdf_page_image",
    "get_system_cache_dir",
    "initialize_engine",
]
