from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from bbq.src.utils.futils import get_system_cache_dir


class BBQConsoleLogFormatter(logging.Formatter):
    """
    Formats console log records in the format:
    [LEVEL] message [Date]
    without leading column spaces or omitted timestamp alignment.
    """

    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"
    DIM = "\033[90m"

    def format(self, record: logging.LogRecord) -> str:
        date_str = self.formatTime(record, self.datefmt or "%Y-%m-%d %H:%M:%S")
        msg = record.getMessage()

        if sys.stdout.isatty():
            color = self.LEVEL_COLORS.get(record.levelno, "")
            line = f"{color}[{record.levelname}]{self.RESET} {msg} {self.DIM}[{date_str}]{self.RESET}"
        else:
            line = f"[{record.levelname}] {msg} [{date_str}]"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                line = f"{line}\n{record.exc_text}"
        if record.stack_info:
            line = f"{line}\n{self.formatStack(record.stack_info)}"
        return line


class BBQFileLogFormatter(logging.Formatter):
    """
    Formats file log records in plain text format:
    [LEVEL] message [Date]
    """

    def format(self, record: logging.LogRecord) -> str:
        date_str = self.formatTime(record, self.datefmt or "%Y-%m-%d %H:%M:%S")
        msg = record.getMessage()
        line = f"[{record.levelname}] {msg} [{date_str}]"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                line = f"{line}\n{record.exc_text}"
        if record.stack_info:
            line = f"{line}\n{self.formatStack(record.stack_info)}"
        return line


class BBQLogger:
    """
    Logger manager class that sets up dual logging:
    - Clean console output formatted as '[LEVEL] message [Date]'
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

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            BBQConsoleLogFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        )
        console_handler.setLevel(self.level)
        logger.addHandler(console_handler)

        log_dir = os.path.dirname(self.log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8")
        file_handler.setFormatter(BBQFileLogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
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
