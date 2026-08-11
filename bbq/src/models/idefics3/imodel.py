from typing import Optional, Any
import torch
from torch import nn
from transformers import Idefics3Model, Idefics3PreTrainedModel
from transformers import logging as tf_logging

tf_logging.set_verbosity_warning()

class CIdeficsModel(Idefics3PreTrainedModel):
    """
    CIdeficsModel

    Args:
        config: Model configuration

    Attributes:
        model (Idefics3Model): Idefics3 model
        dim (int): Dimension of the embeddings
        linear (nn.Linear): Linear layer
        mask_non_image_embeddings (bool): Whether to mask non-image embeddings
        main_input_name (str): Name of the main input

    Example:
        >>> config = load_configuration_from_yaml_file("config.yaml")
        >>> model = CIdeficsModel(config)
    """
    def __init__(self, config: Any) -> None:
        config = sanitize_invalid_model_pad_token_id(config)
        super(CIdeficsModel, self).__init__(config)
        self.model: Idefics3Model = Idefics3Model(config=config)

        self.dim: int = getattr(config, "embedding_dim", 128)
        self.linear: nn.Linear = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.mask_non_image_embeddings: bool = getattr(config, "mask_non_image_embeddings", False)
        self.main_input_name: str = "doc_input_ids"
        self.post_init()

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of the model.

        Runs Idefics3PreTrainedModel.forward() method, followed by a linear layer to project the
        text embeddings to a lower dimension. Normalizes the embeddings. After based on the configs
        add masking.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            torch.Tensor: Output tensor

        Example:
            >>> model = CIdeficsModel(config)
            >>> output = model(**inputs)

        """
        outputs = self.model(*args, **kwargs)

        last_hidden_states: torch.Tensor = outputs[0]
        projected_embeddings: torch.Tensor = self.linear(last_hidden_states)

        # normalizes the embedding, antigravity suggested that
        normalized_embeddings: torch.Tensor = projected_embeddings / projected_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            normalized_embeddings = normalized_embeddings * kwargs["attention_mask"].unsqueeze(-1)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            image_token_id: Optional[int] = getattr(self.config, "image_token_id", None)
            if image_token_id is not None:
                image_mask: torch.Tensor = (kwargs["input_ids"] == image_token_id).unsqueeze(-1)
                normalized_embeddings = normalized_embeddings * image_mask
        return normalized_embeddings


def sanitize_invalid_model_pad_token_id(config: Any) -> Any:
    """
    Sanitize the pad token id of the model configuration.

    Args:
        config: Model configuration

    Returns:
        Any: Sanitized model configuration
    """
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None and hasattr(config, "text_config") and config.text_config is not None:
        vocab_size = getattr(config.text_config, "vocab_size", None)

    if vocab_size is not None:

        if getattr(config, "pad_token_id", None) is not None and config.pad_token_id >= vocab_size:
            text_pad_id = getattr(getattr(config, "text_config", None), "pad_token_id", None)
            if text_pad_id is not None and text_pad_id < vocab_size:
                config.pad_token_id = text_pad_id
            else:
                config.pad_token_id = None
   
        if hasattr(config, "text_config") and config.text_config is not None:
            if getattr(config.text_config, "pad_token_id", None) is not None and config.text_config.pad_token_id >= vocab_size:
                config.text_config.pad_token_id = None
    
    return config
