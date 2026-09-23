from typing import TYPE_CHECKING, Any
from bbq.src.common.version import __version__, _read_project_version

if TYPE_CHECKING:
    from bbq.src.common.base import (
        BaseModel,
        BaseModelLoader,
        BaseProcessor,
        BaseEngineWrapper,
    )
    from bbq.src.common.errors import (
        BaseBBQEngineException,
        ConfigFNFWarning,
        ConfigParseError,
        BaseModelConfigLoadError,
        BaseModelInstantiateError,
        LoRAAdapterLoadError,
        ProcessorLoadError,
    )

__all__ = [
    "__version__",
    "_read_project_version",
    "BaseModel",
    "BaseModelLoader",
    "BaseProcessor",
    "BaseEngineWrapper",
    "BaseBBQEngineException",
    "ConfigFNFWarning",
    "ConfigParseError",
    "BaseModelConfigLoadError",
    "BaseModelInstantiateError",
    "LoRAAdapterLoadError",
    "ProcessorLoadError",
]


def __getattr__(name: str) -> Any:
    if name in ("BaseModel", "BaseModelLoader", "BaseProcessor", "BaseEngineWrapper"):
        import bbq.src.common.base as base_mod

        return getattr(base_mod, name)
    elif name in (
        "BaseBBQEngineException",
        "ConfigFNFWarning",
        "ConfigParseError",
        "BaseModelConfigLoadError",
        "BaseModelInstantiateError",
        "LoRAAdapterLoadError",
        "ProcessorLoadError",
    ):
        import bbq.src.common.errors as err_mod

        return getattr(err_mod, name)
    elif name in ("__version__", "_read_project_version"):
        import bbq.src.common.version as ver_mod

        return getattr(ver_mod, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
