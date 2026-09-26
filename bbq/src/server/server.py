from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
import warnings
from typing import Any, Optional

from rich.console import Console

from bbq.src.config import Config
from bbq.src.server.app import create_bbq_fastapi_app, run_http_server_in_thread
from bbq.src.server.ingestion import (
    process_single_pdf_file_deletion,
    process_single_pdf_file_ingestion,
    scan_and_ingest_existing_pdf_folder,
)
from bbq.src.storage.sql import SqlliteDB
from bbq.src.terminal import (
    configure_server_logging,
    print_bbq,
)

from bbq.src.utils.watcher import start_pdf_folder_watcher

# Disable progress bars and warnings during weight loading
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Silence bitsandbytes BNB_CUDA_VERSION override warning
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*BNB_CUDA_VERSION.*")
warnings.filterwarnings("ignore", module=".*bitsandbytes.*")


logger = logging.getLogger("bbq.server")


class BBQServer:
    """
    BBQServer: Object-oriented server orchestrator for document indexing & retrieval.

    Features:
    - Clean ASCII logo banner and Rich status spinners matching client interface.
    - Threaded fast engine weight loading without progress bar clutter or library warnings.
    - Automatic folder watching and PDF sync.
    - Non-blocking daemon HTTP server thread.
    - Clean KeyboardInterrupt / signal handling.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        config_filepath: str = "config.yaml",
        host: str = "0.0.0.0",
        port: int = 8000,
        without_logo: bool = False,
    ) -> None:
        self.config_filepath = config_filepath
        self.host = host
        self.port = port
        self.without_logo = without_logo

        self.console = Console()
        self.bbq_logger = configure_server_logging()
        self.config: Config = (
            config if config is not None else Config.from_yaml(config_filepath)
        )
        self.tracker = SqlliteDB(db_filepath=self.config.sqlite_db_path)

        self.engine: Optional[Any] = None
        self.observer: Any = None
        self.server_thread: Optional[threading.Thread] = None
        self.loading_thread: Optional[threading.Thread] = None
        self._is_ready = False

    def print_logo(
        self,
        name: Optional[str] = None,
        tag: str = "SERVER",
        server: Optional[Any] = None,
    ) -> None:
        """Prints the BBQ banner unless without_logo is enabled."""
        if not self.without_logo:
            srv = server if server is not None else self.port
            print_bbq(name=name, tag=tag, server=srv)

    def load_engine_threaded(self) -> None:
        """
        Loads AI model and processor weights in a background thread while displaying
        a responsive Rich status spinner interface.
        Handles Ctrl+C (KeyboardInterrupt) cleanly.
        """
        logger.info(
            f"Loading engine and model (base_model_id={self.config.base_model_id})..."
        )

        def _loader():
            try:
                os.environ["TQDM_DISABLE"] = "1"
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
                logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
                warnings.filterwarnings("ignore", message=".*BNB_CUDA_VERSION.*")
                warnings.filterwarnings("ignore", module=".*bitsandbytes.*")

                try:
                    import transformers.utils.logging as tfl

                    tfl.disable_progress_bar()
                    tfl.set_verbosity_error()
                except Exception:
                    pass

                from bbq.src.utils.model_loader import initialize_engine

                self.engine = initialize_engine(self.config)
                self._is_ready = True
                logger.info("Engine model and processor loaded successfully.")
            except Exception:
                logger.exception("Failed to load engine weights")

        self.loading_thread = threading.Thread(target=_loader, daemon=True)
        self.loading_thread.start()

        try:
            with self.console.status(
                f"[bold cyan]Loading AI model and weights ({self.config.base_model_id}). Please wait...[/bold cyan]",
                spinner="dots",
            ):
                while self.loading_thread.is_alive():
                    self.loading_thread.join(timeout=0.2)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Server startup interrupted by user (Ctrl+C). Shutting down...")
            self.stop()
            sys.exit(0)

        if self.engine:
            self.console.print(
                f"[bold green]✔ Model and weights loaded successfully ({self.config.base_model_id})[/bold green]"
            )

    def start(self) -> None:
        """
        Starts the persistent document indexing server pipeline.
        """
        self.print_logo()
        logger.info(
            f"Starting persistent document-indexing server (Logs saved to: {self.bbq_logger.log_file_path})..."
        )

        reset_count: int = self.tracker.reset_in_progress_processing_to_pending()
        if reset_count > 0:
            logger.info(
                f"Recovered startup state: reset {reset_count} leftover processing record(s) to pending."
            )

        # Register Signal Handlers
        self._register_signal_handlers()

        # Start FastAPI HTTP Server Thread immediately so port is open and status can be queried
        fastapi_app = create_bbq_fastapi_app(
            engine=self.engine,
            tracker=self.tracker,
            get_engine_callback=lambda: self.engine,
            config=self.config,
        )
        self.server_thread = run_http_server_in_thread(
            fastapi_app, host=self.host, port=self.port
        )

        self.console.print(
            f"[bold green]✔ BBQ server running on http://{self.host}:{self.port} "
            f"(quantization: {self.config.quantization})[/bold green]\n"
            f"[dim]Watching '{self.config.watch_folder_path}' for document changes. Press Ctrl+C to stop.[/dim]\n"
        )

        # File system watcher callbacks
        def pdf_detected_event_callback(pdf_filepath: str) -> None:
            logger.info(f"Folder watcher detected new/modified PDF: {pdf_filepath}")
            if self.engine:
                process_single_pdf_file_ingestion(
                    pdf_filepath=pdf_filepath,
                    engine=self.engine,
                    tracker=self.tracker,
                    console=self.console,
                )
            else:
                logger.info(
                    f"Engine loading in progress; '{pdf_filepath}' will be ingested once engine is ready."
                )

        def pdf_deleted_event_callback(pdf_filepath: str) -> None:
            logger.info(f"Folder watcher detected deleted PDF: {pdf_filepath}")
            process_single_pdf_file_deletion(
                pdf_filepath=pdf_filepath,
                tracker=self.tracker,
            )

        # Start Folder Watcher immediately
        logger.info(
            f"Starting file system watcher on directory: {self.config.watch_folder_path}"
        )
        self.observer, _ = start_pdf_folder_watcher(
            watch_directory_path=self.config.watch_folder_path,
            callback_on_pdf_ready=pdf_detected_event_callback,
            callback_on_pdf_deleted=pdf_deleted_event_callback,
        )

        # Threaded Engine Weight Loading with Spinner
        self.load_engine_threaded()

        if not self.engine:
            logger.error("Server startup aborted: Engine failed to load.")
            return

        # Scan watch folder and ingest existing PDFs
        try:
            with self.console.status(
                f"[bold green]Scanning '{self.config.watch_folder_path}' for existing PDF documents...[/bold green]",
                spinner="dots",
            ):
                scan_and_ingest_existing_pdf_folder(
                    engine=self.engine, tracker=self.tracker, console=self.console
                )
            self.console.print(
                f"[bold green]✔ PDF scan complete for '{self.config.watch_folder_path}'[/bold green]"
            )
        except (KeyboardInterrupt, SystemExit):
            logger.info("PDF scan interrupted by user (Ctrl+C). Shutting down...")
            self.stop()
            sys.exit(0)

    def _register_signal_handlers(self) -> None:
        def signal_handler_callback(signal_number: int, frame_object: Any) -> None:
            logger.info(
                f"Received shutdown signal ({signal_number}). Initiating graceful server exit..."
            )
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler_callback)
        signal.signal(signal.SIGTERM, signal_handler_callback)

    def stop(self) -> None:
        """Stops observer and resets database processing state."""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=1.0)
        except Exception as exc:
            logger.error(f"Error stopping watcher observer: {exc}")

        try:
            reset_count: int = self.tracker.reset_in_progress_processing_to_pending()
            if reset_count > 0:
                logger.info(
                    f"Reset {reset_count} in-progress processing file record(s) back to pending state."
                )
        except Exception as exc:
            logger.debug(
                f"Non-critical error resetting db status during shutdown: {exc}"
            )

        logger.info("Persistent document indexing server shutdown complete.")

    def is_ready(self) -> bool:
        return self._is_ready


def start_document_indexing_server(
    config: Optional[Config] = None,
    config_filepath: str = "config.yaml",
    host: str = "0.0.0.0",
    port: int = 8000,
    without_logo: bool = False,
) -> None:
    """Main entrypoint for starting the document indexing server."""
    server = BBQServer(
        config=config,
        config_filepath=config_filepath,
        host=host,
        port=port,
        without_logo=without_logo,
    )
    try:
        server.start()
        while True:
            if server.observer:
                server.observer.join(timeout=1.0)
            else:
                time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Server stopped by user (Ctrl+C). Exiting...")
        server.stop()
        sys.exit(0)
