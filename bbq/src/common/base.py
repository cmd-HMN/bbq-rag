from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Mapping

class BaseModel(ABC):
    """
    Abstract base class for models in BBQ RAG.
    """
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass of the model.
        """
        pass

class BaseModelLoader(ABC):
    @classmethod
    @abstractmethod
    def load_model_and_processor(
        cls, config_input: Any
    ) -> Tuple[Any, Any]:
        pass

class BaseProcessor(ABC):
    @abstractmethod
    def process_images(
        self, images: List[Any], **kwargs: Any
    ) -> Mapping[str, Any]:
        pass

    @abstractmethod
    def process_texts(
        self, texts: List[str], **kwargs: Any
    ) -> Mapping[str, Any]:
        pass

    @abstractmethod
    def score(
        self, qs: Any, ps: Any, batch_size: int = 128
    ) -> Any:
        pass

class BaseEngineWrapper(ABC):
    @abstractmethod
    def encode_multimodal_document_images(
        self, images: List[Any]
    ) -> Any:
        pass

    @abstractmethod
    def encode_query_text_inputs(
        self, texts: List[str]
    ) -> Any:
        pass
