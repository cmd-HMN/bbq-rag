from typing import Any, List, Optional, Tuple, Type, Union

from peft import PeftConfig, PeftModel
import torch
from transformers import Idefics3Config
from transformers import logging as tf_logging

from bbq.src.common.base import BaseEngineWrapper, BaseModel, BaseModelLoader, BaseProcessor
from bbq.src.models.idefics3.imodel import (
    CIdeficsModel,
    sanitize_invalid_model_pad_token_id,
)
from bbq.src.models.idefics3.iprocess import CIdeficsProcessor
from bbq.src.utils.config import (
    ModelConfigWrapper,
    determine_target_torch_device,
    load_configuration_from_yaml_file,
    resolve_torch_data_type,
)
from bbq.src.utils.errors import (
    BaseModelConfigLoadError,
    BaseModelInstantiateError,
    LoRAAdapterLoadError,
    ProcessorLoadError,
)

tf_logging.set_verbosity_warning()

class EngineModelLoader(BaseModelLoader):
    """
    Class for loading engine models
    """
    @classmethod
    def load_model_and_processor(
        cls, config_input: Any
    ) -> Tuple[Union[CIdeficsModel, PeftModel], CIdeficsProcessor]:
        return cls.load_model_with_lora_adapters(config_input)

    @staticmethod
    def load_model_with_lora_adapters(
        config_input: Union[str, ModelConfigWrapper],
    ) -> Tuple[Union[CIdeficsModel, PeftModel], CIdeficsProcessor]:
        """
        Loads a model with LoRA adapters

        Args:
            config_input (Union[str, ModelConfigWrapper]): Model configuration

        Returns:
            Tuple[Union[CIdeficsModel, PeftModel], CIdeficsProcessor]: Loaded model and processor
        """
        tf_logging.set_verbosity_error()

        if isinstance(config_input, str):
            config: ModelConfigWrapper = load_configuration_from_yaml_file(config_input)
        else:
            config = config_input

        target_device = determine_target_torch_device(config.device)
        resolved_dtype: torch.dtype = resolve_torch_data_type(config.torch_dtype, target_device)

        
        try:
            base_model_config: Idefics3Config = Idefics3Config.from_pretrained(config.base_model_id)
            base_model_config = sanitize_invalid_model_pad_token_id(base_model_config)
        except Exception as exception_instance:
            raise BaseModelConfigLoadError(
                f"Failed to load explicit base configuration from {config.base_model_id}: {exception_instance}"
            ) from exception_instance

        try:
            base_model: Any = CIdeficsModel.from_pretrained(
                config.base_model_id,
                torch_dtype=resolved_dtype,
            )
        except Exception as exception_instance:
            raise BaseModelInstantiateError(
                f"Failed to instantiate base CIdeficsModel from {config.base_model_id}: {exception_instance}"
            ) from exception_instance

        model_to_use: Union[Any, PeftModel] = base_model

        if config.lora_adapter_id:
            try:
                lora_adapter_config: PeftConfig = PeftConfig.from_pretrained(config.lora_adapter_id)
                model_to_use = PeftModel.from_pretrained(
                    base_model,
                    config.lora_adapter_id,
                    config=lora_adapter_config,
                )
            except Exception as exception_instance:
                raise LoRAAdapterLoadError(
                    f"Failed to load and apply LoRA adapter weights from {config.lora_adapter_id}: {exception_instance}"
                ) from exception_instance

        model_to_use = model_to_use.to(target_device).eval()

        try:
            processor: CIdeficsProcessor = CIdeficsProcessor.from_pretrained(config.base_model_id)
        except Exception:
            try:
                processor = CIdeficsProcessor.from_pretrained(config.lora_adapter_id)
            except Exception as exception_instance:
                raise ProcessorLoadError(
                    f"Failed to load processor from {config.base_model_id} or {config.lora_adapter_id}: {exception_instance}"
                ) from exception_instance

        return model_to_use, processor

class EngineWrapper(BaseEngineWrapper):

    """
    Class for wrapping engine models
    """
    def __init__(
        self,
        model: Any,
        processor: BaseProcessor,
        config: ModelConfigWrapper,
    ) -> None:

        self.model = model
        self.processor = processor
        self.config: ModelConfigWrapper = config

    def encode_multimodal_document_images(self, images: List[Any]) -> torch.Tensor:
        """
        Encodes a list of images into embeddings
        """
        if not images:
            raise ValueError("Input image list cannot be empty.")
        
        processed_inputs = self.processor.process_images(images)
        target_device = next(self.model.parameters()).device
        processed_inputs = {
            k: v.to(target_device) if isinstance(v, torch.Tensor) else v
            for k, v in processed_inputs.items()
        }
        
        with torch.inference_mode():
            image_embeddings: torch.Tensor = self.model(**processed_inputs)
        
        return image_embeddings

    def encode_query_text_inputs(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a list of text inputs into embeddings
        """
        if not texts:
            raise ValueError("Input text list cannot be empty.")
   
        processed_inputs = self.processor.process_texts(texts)
        target_device = next(self.model.parameters()).device
        processed_inputs = {
            k: v.to(target_device) if isinstance(v, torch.Tensor) else v
            for k, v in processed_inputs.items()
        }
        
        with torch.inference_mode():
            query_embeddings: torch.Tensor = self.model(**processed_inputs)
        
        return query_embeddings

def initialize_engine_from_yaml_config(
    config_filepath: str = "config.yaml",
    loader_class: Optional[Type[BaseModelLoader]] = None,
) -> EngineWrapper:
    """
    Initializes an engine from a YAML configuration
    """
    config: ModelConfigWrapper = load_configuration_from_yaml_file(config_filepath)

    target_loader = loader_class or EngineModelLoader
    
    model, processor = target_loader.load_model_and_processor(config)
    
    return EngineWrapper(model=model, processor=processor, config=config)
