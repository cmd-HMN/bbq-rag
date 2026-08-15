class BaseBBQEngineException(Exception):
    """
    Base class for all exceptions raised by the BBQ Engine.
    """
    def __init__(self, message: str = "An exception occurred in BBQ Engine.") -> None:
        super().__init__(message)
        self.message: str = message

class ConfigFNFWarning(UserWarning):
    """
    Warning raised when a configuration file is not found.
    """
    def __init__(self, message: str = "Configuration file was not found.") -> None:
        super().__init__(message)
        self.message: str = message

class ConfigParseError(BaseBBQEngineException):
    """
    Exception raised when a configuration file cannot be parsed.
    """
    def __init__(self, message: str = "Failed to parse YAML configuration file.") -> None:
        super().__init__(message)

class BaseModelConfigLoadError(BaseBBQEngineException):
    """
    Exception raised when the base model configuration fails to load.
    """
    def __init__(self, message: str = "Failed to load explicit base configuration.") -> None:
        super().__init__(message)

class BaseModelInstantiateError(BaseBBQEngineException):
    """
    Exception raised when the base model fails to instantiate.
    """
    def __init__(self, message: str = "Failed to instantiate base model.") -> None:
        super().__init__(message)

class LoRAAdapterLoadError(BaseBBQEngineException):
    """
    Exception raised when LoRA adapter weights fail to load or apply.
    """
    def __init__(self, message: str = "Failed to load and apply LoRA adapter weights.") -> None:
        super().__init__(message)

class ProcessorLoadError(BaseBBQEngineException):
    """
    Exception raised when the model processor fails to load.
    """
    def __init__(self, message: str = "Failed to load processor.") -> None:
        super().__init__(message)

class ModelNotFound(UserWarning):
    """
    Warning raised when the model is not found. Especially in Model Registry
    """
    def __init__(self, message: str = "Failed to load model.") -> None:
        super().__init__(message)
        self.message: str = message

class PdfNotFoundError(BaseBBQEngineException):
    """
    Exception raised when a PDF file is not found.
    """
    def __init__(self, message: str = "PDF file was not found.") -> None:
        super().__init__(message)
