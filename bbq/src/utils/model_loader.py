from typing import Any, List, Optional, Tuple, Type, Union

import torch
from peft import PeftConfig, PeftModel
from transformers import logging as tf_logging

from bbq.src.common.base import (
    BaseEngineWrapper,
    BaseModel,
    BaseModelLoader,
    BaseProcessor,
)
from bbq.src.common.errors import (
    BaseModelInstantiateError,
    LoRAAdapterLoadError,
    ProcessorLoadError,
)
from bbq.src.config import Config
from bbq.src.models.registry import ModelRegistry

import os
import logging
import warnings

# Disable progress bars and warnings during weight loading
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Silence bitsandbytes BNB_CUDA_VERSION override warning
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*BNB_CUDA_VERSION.*")
warnings.filterwarnings("ignore", module=".*bitsandbytes.*")

tf_logging.set_verbosity_error()
try:
    tf_logging.disable_progress_bar()
except Exception:
    pass


def determine_target_torch_device(device_preference: str = "auto") -> str:
    """Determine the target torch device based on preference and CUDA availability."""
    if device_preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_preference


def resolve_torch_data_type(
    dtype_name: str = "bfloat16", target_device: str = "cpu"
) -> torch.dtype:
    """Resolve the torch data type based on name and target device."""
    if target_device == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    elif dtype_name == "float16":
        return torch.float16
    return torch.float32


def _from_pretrained_fast(loader_cls_or_fn: Any, model_id_or_obj: Any, **kwargs: Any) -> Any:
    """Load model/adapter/processor from local huggingface cache first to avoid network latency."""
    try:
        return loader_cls_or_fn.from_pretrained(
            model_id_or_obj, local_files_only=True, **kwargs
        )
    except Exception:
        return loader_cls_or_fn.from_pretrained(
            model_id_or_obj, local_files_only=False, **kwargs
        )


class EngineModelLoader(BaseModelLoader):
    """
    Class for loading engine models
    """

    @classmethod
    def load_model_and_processor(
        cls, config_input: Any
    ) -> Tuple[Union[BaseModel, PeftModel], BaseProcessor]:
        return cls.load_model_with_lora_adapters(config_input)

    @staticmethod
    def load_model_with_lora_adapters(
        config_input: Union[str, Config],
    ) -> Tuple[Union[BaseModel, PeftModel], BaseProcessor]:
        """
        Loads a model with LoRA adapters

        Args:
            config_input (Union[str, ModelConfigWrapper]): Model configuration

        Returns:
            Tuple[Union[BaseModel, PeftModel], BaseProcessor]: Loaded model and processor
        """
        tf_logging.set_verbosity_error()

        if isinstance(config_input, str):
            config: Config = Config.from_yaml(config_input)
        else:
            config = config_input

        target_device = determine_target_torch_device(config.device)
        resolved_dtype: torch.dtype = resolve_torch_data_type(
            config.torch_dtype, target_device
        )

        model_cls, processor_cls = ModelRegistry.get_for_model(config.base_model_id)

        try:
            base_model: Any = _from_pretrained_fast(
                model_cls,
                config.base_model_id,
                torch_dtype=resolved_dtype,
            )
        except Exception as exception_instance:
            raise BaseModelInstantiateError(
                f"Failed to instantiate base {model_cls.__name__} from {config.base_model_id}: {exception_instance}"
            ) from exception_instance

        model_to_use: Union[Any, PeftModel] = base_model

        if config.lora_adapter_id:
            try:
                lora_adapter_config: PeftConfig = _from_pretrained_fast(
                    PeftConfig,
                    config.lora_adapter_id,
                )
                try:
                    model_to_use = PeftModel.from_pretrained(
                        base_model,
                        config.lora_adapter_id,
                        config=lora_adapter_config,
                        local_files_only=True,
                    )
                except Exception:
                    model_to_use = PeftModel.from_pretrained(
                        base_model,
                        config.lora_adapter_id,
                        config=lora_adapter_config,
                        local_files_only=False,
                    )
            except Exception as exception_instance:
                raise LoRAAdapterLoadError(
                    f"Failed to load and apply LoRA adapter weights from {config.lora_adapter_id}: {exception_instance}"
                ) from exception_instance

        model_to_use = model_to_use.to(target_device).eval()

        try:
            processor: BaseProcessor = _from_pretrained_fast(
                processor_cls,
                config.base_model_id,
            )
        except Exception:
            try:
                processor = _from_pretrained_fast(
                    processor_cls,
                    config.lora_adapter_id,
                )
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
        config: Config,
    ) -> None:

        self.model = model
        self.processor = processor
        self.config: Config = config

    def encode_multimodal_document_images(
        self, images: List[Any], batch_size: int = 4
    ) -> torch.Tensor:
        """
        Encodes a list of images into embeddings using sub-batching to prevent OOM errors on large documents.
        """
        if not images:
            raise ValueError("Input image list cannot be empty.")

        target_device = next(self.model.parameters()).device
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            processed_inputs = self.processor.process_images(batch_images)
            processed_inputs = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in processed_inputs.items()
            }

            with torch.inference_mode():
                batch_embeddings: torch.Tensor = self.model(**processed_inputs)
                all_embeddings.append(batch_embeddings.cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)

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


def initialize_engine(
    config: Config,
    loader_class: Optional[Type[BaseModelLoader]] = None,
) -> EngineWrapper:
    """
    Initializes an engine directly from an injected Config instance.
    """
    target_loader = loader_class or EngineModelLoader
    model, processor = target_loader.load_model_and_processor(config)
    return EngineWrapper(model=model, processor=processor, config=config)


def initialize_engine_from_yaml_config(
    config_filepath: str = "config.yaml",
    loader_class: Optional[Type[BaseModelLoader]] = None,
) -> EngineWrapper:
    """
    Initializes an engine from a YAML configuration file path.
    """
    config: Config = Config.from_yaml(config_filepath)
    return initialize_engine(config=config, loader_class=loader_class)
