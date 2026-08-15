import os
import logging
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

from bbq.src.config import get_system_cache_dir

class BBQLogger:
    """
    Logger manager class that sets up dual logging:
    - Interactive Rich console output
    - Persistent file logging saved to disk
    """

    def __init__(
        self,
        log_file_path: Optional[str] = None,
        logger_name: str = "bbq",
        level: int = logging.INFO,
    ) -> None:
        self.logger_name = logger_name
        self.level = level

        if not log_file_path:
            log_dir = get_system_cache_dir("logs")
            os.makedirs(log_dir, exist_ok=True)
            self.log_file_path = os.path.join(log_dir, "server.log")
        else:
            self.log_file_path = log_file_path

        self.logger = self.configure_logger()

    def configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.level)
        logger.handlers.clear()

        # 1. Rich Console Handler
        rich_handler = RichHandler(
            console=Console(),
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        rich_handler.setLevel(self.level)
        logger.addHandler(rich_handler)

        # 2. Disk File Handler
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8")
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(self.level)
        logger.addHandler(file_handler)

        return logger

    def get_logger(self) -> logging.Logger:
        return self.logger


def configure_server_logging(
    log_file_path: Optional[str] = None,
) -> BBQLogger:
    """
    Configures and returns a BBQLogger instance for the server.
    """
    return BBQLogger(log_file_path=log_file_path)
