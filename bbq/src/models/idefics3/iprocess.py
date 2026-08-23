from typing import ClassVar, List, Union, Optional, Any
import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor
from transformers import logging as tf_logging
from maxsimd import maxsim_3d_ptr
from bbq.src.common.base import BaseProcessor

tf_logging.set_verbosity_warning()


class CIdeficsProcessor(Idefics3Processor, BaseProcessor):
    """
    Processor for the CIdeficsModel

    Attributes:
        query_prefix (str): The prefix for the query.
        query_augmentation_token (str): The augmentation token for the query.
        visual_prompt_prefix (str): The prefix for the visual prompt.
    """

    query_prefix: ClassVar[str] = ""
    query_augmentation_token: ClassVar[str] = "<end_of_utterance>"
    visual_prompt_prefix: str = (
        "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
    )

    def __init__(
        self,
        image_processor: Any = None,
        tokenizer: Any = None,
        image_seq_len: int = 64,
        chat_template: Optional[str] = None,
        visual_prompt_command: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            image_processor (Any, optional): The image processor to use. Defaults to None.
            tokenizer (Any, optional): The tokenizer to use. Defaults to None.
            image_seq_len (int, optional): The length of the image sequence. Defaults to 64.
            chat_template (Optional[str], optional): The chat template to use. Defaults to None.
            visual_prompt_command (Optional[str], optional): The visual prompt command to use.
        """
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_len=image_seq_len,
            chat_template=chat_template,
            **kwargs,
        )
        if visual_prompt_command is not None:
            self.visual_prompt_prefix = format_prompt_command_with_template_tags(
                visual_prompt_command
            )
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"

    def set_visual_prompt_command(self, command_text: str) -> None:
        self.visual_prompt_prefix = format_prompt_command_with_template_tags(
            command_text
        )

    def process_images(
        self,
        images: List[Image.Image],
        prompt_command: Optional[str] = None,
        padding: str = "longest",
        **kwargs: Any,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Processes a lsit of Images

        Args:
            images (List[Image.Image]): The images to process.
            prompt_command (Optional[str], optional): The prompt command to use. Defaults to None.
            padding (str, optional): The padding to use. Defaults to "longest".

        Returns:
            Union[BatchFeature, BatchEncoding]: The processed images
        """
        formatted_rgb_images: List[Image.Image] = [
            image.convert("RGB") for image in images
        ]

        target_prefix: str = (
            format_prompt_command_with_template_tags(prompt_command)
            if prompt_command is not None
            else self.visual_prompt_prefix
        )
        return self(
            text=[target_prefix] * len(formatted_rgb_images),
            images=formatted_rgb_images,
            padding=padding,  # type: ignore[call-arg]
            return_tensors="pt",  # type: ignore[call-arg]
        )

    def process_texts(
        self, texts: List[str], **kwargs: Any
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Processes a list of query texts by adding 'Question: ' prefix and '<end_of_utterance>' suffix
        required by ColPali / ColSmol models to activate visual-text projection features.
        """
        formatted_texts = []
        for text in texts:
            t = text.strip()
            if not t.startswith("Question:"):
                t = f"Question: {t}"
            if not t.endswith("<end_of_utterance>"):
                t = f"{t}<end_of_utterance>"
            formatted_texts.append(t)

        return self(text=formatted_texts, return_tensors="pt", padding="longest")  # type: ignore[call-arg]

    def score(
        self,
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Maxsim score between qs and ps

        Uses maxsim_3d_ptr for zero-copy memory pointer MaxSim scoring.

        Args:
            qs (Union[torch.Tensor, List[torch.Tensor]]): The queries
            ps (Union[torch.Tensor, List[torch.Tensor]]): The passages
            batch_size (int, optional): The batch size. Defaults to 128.

        Returns:
            torch.Tensor: The scores
        """
        if isinstance(qs, torch.Tensor):
            if qs.ndim == 2:
                qs_list = [qs]
            elif qs.ndim == 3:
                qs_list = [qs[i] for i in range(qs.shape[0])]
            else:
                qs_list = [qs]
        else:
            qs_list = list(qs)

        if isinstance(ps, torch.Tensor):
            if ps.ndim == 2:
                ps_tensor = ps.unsqueeze(0).contiguous().float().cpu()
            else:
                ps_tensor = ps.contiguous().float().cpu()
        elif isinstance(ps, list):
            ps_tensor = torch.stack(ps).contiguous().float().cpu()
        else:
            raise ValueError(f"Unsupported passages type: {type(ps)}")

        num_docs, tokens_per_doc, dim = ps_tensor.shape
        scores_list: List[List[float]] = []

        for q in qs_list:
            q_cont = q.contiguous().float().cpu()
            q_len = q_cont.shape[0]

            scores: List[float] = maxsim_3d_ptr(
                q_cont.data_ptr(),
                ps_tensor.data_ptr(),
                q_len,
                num_docs,
                tokens_per_doc,
                dim,
            )
            scores_list.append(scores)

        return torch.tensor(scores_list, dtype=torch.float32)


def format_prompt_command_with_template_tags(command_text: str) -> str:
    """
    Args:
        command_text (str): The command text to format.

    Returns:
        str: The formatted command text.
    """
    if command_text.startswith("<|im_start|>"):
        return command_text
    return f"<|im_start|>User:<image>{command_text}<end_of_utterance>\nAssistant:"
