from typing import List, Tuple, Type, Union, Any
from bbq.src.common.base import BaseModel, BaseProcessor
from bbq.src.models.idefics3.imodel import CIdeficsModel
from bbq.src.models.idefics3.iprocess import CIdeficsProcessor


class ModelRegistry:
    """
    Registry for model and processor class pairs.
    Matches base_model_id via model_cls.supports_model(base_model_id).
    """

    _entries: List[Tuple[Type[BaseModel], Type[BaseProcessor]]] = []

    @classmethod
    def register(
        cls,
        model_class: Type[BaseModel],
        processor_class: Type[BaseProcessor],
    ) -> None:
        """
        Registers a model and processor class pair.
        """
        pair = (model_class, processor_class)
        if pair not in cls._entries:
            cls._entries.append(pair)

    @classmethod
    def get_for_model(
        cls, base_model_id: str
    ) -> Tuple[Union[Type[BaseModel], Any], Type[BaseProcessor]]:
        """
        Finds and returns the (model_class, processor_class) pair supporting base_model_id.
        """
        for model_cls, processor_cls in cls._entries:
            if hasattr(model_cls, "supports_model") and model_cls.supports_model(base_model_id):
                return model_cls, processor_cls

        if not cls._entries:
            raise RuntimeError("No models registered in ModelRegistry.")

        # Default fallback to first registered entry
        return cls._entries[0]


# Register built-in default models
ModelRegistry.register(CIdeficsModel, CIdeficsProcessor)
