import os
import warnings
import traceback
from typing import Dict, Any
import yaml
import torch
from bbq.src.utils.errors import ConfigFNFWarning, ConfigParseError

warnings.simplefilter("always", ConfigFNFWarning)

class ModelConfigWrapper:
    """
    Wrapper class for model configuration.

    Args:
        base_model_id (str): The base model ID.
        lora_adapter_id (str): The LoRA adapter ID.
        embedding_dim (int): The embedding dimension.
        device (str): The device to use.
        torch_dtype (str): The data type.
        mask_non_image_embeddings (bool): Whether to mask non-image embeddings.
        visual_prompt_command (str): The visual prompt command.
    """
    def __init__(
        self,
        base_model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        lora_adapter_id: str = "vidore/colSmol-256M",
        embedding_dim: int = 128,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        mask_non_image_embeddings: bool = False,
        visual_prompt_command: str = "What is written on the image.",
    ) -> None:
        self.base_model_id: str = base_model_id
        self.lora_adapter_id: str = lora_adapter_id
        self.embedding_dim: int = embedding_dim
        self.device: str = device
        self.torch_dtype: str = torch_dtype
        self.mask_non_image_embeddings: bool = mask_non_image_embeddings
        self.visual_prompt_command: str = visual_prompt_command

    def format_visual_prompt_prefix(self) -> str:
        """
        Returns the formatted visual prompt prefix.
        """
        if self.visual_prompt_command.startswith("<|im_start|>"):
            return self.visual_prompt_command
        return f"<|im_start|>User:<image>{self.visual_prompt_command}<end_of_utterance>\nAssistant:"

    def convert_to_dictionary_format(self) -> Dict[str, Any]:
        """
        Converts the configuration to a dictionary format.
        """
        return {
            "base_model_id": self.base_model_id,
            "lora_adapter_id": self.lora_adapter_id,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "mask_non_image_embeddings": self.mask_non_image_embeddings,
            "visual_prompt_command": self.visual_prompt_command,
            "visual_prompt_prefix": self.format_visual_prompt_prefix(),
        }

def load_configuration_from_yaml_file(
    config_filepath: str = "config.yaml",
) -> ModelConfigWrapper:
    """
    Load configuration from a YAML file.

    Args:
        config_filepath (str): The path to the YAML configuration file.

    Returns:
        ModelConfigWrapper: An instance of ModelConfigWrapper containing the loaded configuration.
    """

    # Check if the configuration file exists

    if not os.path.exists(config_filepath):
        warning_msg: str = (
            f"Configuration file not found at path: '{config_filepath}'. "
            "Proceeding with default settings."
        )
        traceback.print_stack(limit=3)
        warnings.warn(warning_msg, category=ConfigFNFWarning, stacklevel=2)
        return ModelConfigWrapper()

    try:
        with open(config_filepath, "r", encoding="utf-8") as file_stream:
            parsed_yaml_data: Dict[str, Any] = yaml.safe_load(file_stream) or {}
    except Exception as exception_instance:
        traceback.print_exc()
        raise ConfigParseError(
            f"Failed to parse YAML configuration file: {exception_instance}"
        ) from exception_instance

    base_model_id: str = str(
        parsed_yaml_data.get("base_model_id", "HuggingFaceTB/SmolVLM-256M-Instruct")
    )
    lora_adapter_id: str = str(
        parsed_yaml_data.get("lora_adapter_id", "vidore/colSmol-256M")
    )
    embedding_dim: int = int(parsed_yaml_data.get("embedding_dim", 128))
    device: str = str(parsed_yaml_data.get("device", "auto"))
    torch_dtype: str = str(parsed_yaml_data.get("torch_dtype", "bfloat16"))
    mask_non_image_embeddings: bool = bool(
        parsed_yaml_data.get("mask_non_image_embeddings", False)
    )
    visual_prompt_command: str = str(
        parsed_yaml_data.get("visual_prompt_command", "What is written on the image.")
    )

    if not base_model_id:
        raise ValueError("base_model_id cannot be empty in configuration.")

    return ModelConfigWrapper(
        base_model_id=base_model_id,
        lora_adapter_id=lora_adapter_id,
        embedding_dim=embedding_dim,
        device=device,
        torch_dtype=torch_dtype,
        mask_non_image_embeddings=mask_non_image_embeddings,
        visual_prompt_command=visual_prompt_command,
    )

def determine_target_torch_device(device_preference: str = "auto") -> str:
    """
    Determine the target torch device based on the provided device preference.

    Args:
        device_preference (str): The device preference, either "auto" or a specific device name.

    Returns:
        str: The target torch device.
    """
    if device_preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_preference

def resolve_torch_data_type(
    dtype_name: str = "bfloat16", target_device: str = "cpu"
) -> torch.dtype:
    """
    Resolve the torch data type based on the provided dtype name and target device.

    Args:
        dtype_name (str): The data type name, either "bfloat16" or "float16".
        target_device (str): The target device, either "cpu" or "cuda".

    Returns:
        torch.dtype: The resolved torch data type.
    """
    if target_device == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    elif dtype_name == "float16":
        return torch.float16
    return torch.float32
