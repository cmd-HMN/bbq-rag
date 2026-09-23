from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import yaml

from bbq.src.common.errors import ConfigFNFWarning, ConfigParseError
from bbq.src.utils.futils import get_system_cache_dir


@dataclass
class Config:
    """
    BBQ configuration class that encapsulates various settings for the application.
    """

    base_model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    lora_adapter_id: str = "vidore/colSmol-256M"
    embedding_dim: int = 128
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    mask_non_image_embeddings: bool = False
    visual_prompt_command: str = "Explaing the image."
    watch_folder_path: str = "data/watch"
    embeddings_output_path: Optional[str] = None
    sqlite_db_path: Optional[str] = None
    pdf_render_dpi: int = 150
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.6-flash"
    rag_top_k: int = 3
    quantization: Optional[str] = "f32"

    def __post_init__(self) -> None:
        if not self.base_model_id:
            raise ValueError("base_model_id cannot be empty in configuration.")
        if self.embedding_dim is not None:
            self.embedding_dim = int(self.embedding_dim)
        if self.pdf_render_dpi is not None:
            self.pdf_render_dpi = int(self.pdf_render_dpi)
        if self.rag_top_k is not None:
            self.rag_top_k = int(self.rag_top_k)

        if not self.embeddings_output_path:
            self.embeddings_output_path = get_system_cache_dir("embeddings")
        if not self.sqlite_db_path:
            self.sqlite_db_path = get_system_cache_dir("tracker.db")

        if self.quantization:
            self.quantization = str(self.quantization).strip().lower()

        if self.gemini_api_key:
            self.gemini_api_key = str(self.gemini_api_key).strip() or None
        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )

    def format_visual_prompt_prefix(self) -> str:
        """
        Returns the formatted visual prompt prefix.
        """
        if self.visual_prompt_command.startswith("<|im_start|>"):
            return self.visual_prompt_command
        return f"<|im_start|>User:<image>{self.visual_prompt_command}<end_of_utterance>\nAssistant:"

    def convert_to_dictionary_format(self) -> Dict[str, Any]:
        """
        Converts the configuration to a dictionary format without exposing secret API keys.
        """
        data = asdict(self)
        data.pop("gemini_api_key", None)
        data["gemini_api_key_configured"] = bool(
            self.gemini_api_key and self.gemini_api_key.strip()
        )
        data["visual_prompt_prefix"] = self.format_visual_prompt_prefix()
        return data

    @classmethod
    def from_yaml(cls, config_filepath: str = "config.yaml") -> Config:
        """
        Load configuration from a YAML file.
        """
        resolved_path = config_filepath
        if not os.path.exists(resolved_path):
            candidates = [
                os.path.join(os.getcwd(), config_filepath),
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        )
                    ),
                    config_filepath,
                ),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    resolved_path = candidate
                    break

        if not os.path.exists(resolved_path):
            warnings.warn(
                f"Configuration file not found at path: '{config_filepath}'. Proceeding with default settings.",
                category=ConfigFNFWarning,
                stacklevel=2,
            )
            return cls()

        config_filepath = resolved_path

        try:
            with open(config_filepath, "r", encoding="utf-8") as file_stream:
                parsed_yaml_data: Dict[str, Any] = yaml.safe_load(file_stream) or {}
        except Exception as exception_instance:
            raise ConfigParseError(
                f"Failed to parse YAML configuration file: {exception_instance}"
            ) from exception_instance

        valid_keys = {f.name for f in fields(cls)}
        filtered_kwargs = {
            k: v
            for k, v in parsed_yaml_data.items()
            if k in valid_keys and v is not None
        }
        return cls(**filtered_kwargs)
