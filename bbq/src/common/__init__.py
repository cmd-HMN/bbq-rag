from typing import Any

__all__ = [
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
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
