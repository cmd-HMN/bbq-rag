import os
import sys
import signal
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bbq.src.utils.config import (
    ModelConfigWrapper,
    load_configuration_from_yaml_file,
)
from bbq.src.utils.model_loader import (
    EngineWrapper,
    initialize_engine_from_yaml_config,
)
from bbq.src.utils.tracker import ProcessedFilesTracker
from bbq.src.utils.pdf_utils import (
    compute_file_sha256_hash,
    extract_pdf_pages_to_pil_images,
)
from bbq.src.utils.watcher import start_pdf_folder_watcher
from bbq.src.utils.tui import (
    render_server_status_rich_panel,
    configure_rich_logging_for_server,
)

logger = logging.getLogger("bbq.server")


def process_single_pdf_file_ingestion(
    pdf_filepath: str,
    engine: EngineWrapper,
    tracker: ProcessedFilesTracker,
) -> None:
    if not os.path.exists(pdf_filepath):
        logger.warning(f"File not found for processing: {pdf_filepath}")
        return

    try:
        file_hash: str = compute_file_sha256_hash(pdf_filepath)
        existing_record: Optional[Dict[str, Any]] = tracker.fetch_file_record_by_hash(file_hash)

        if existing_record and existing_record.get("status") == "done":
            logger.info(f"Skipping already processed PDF (hash={file_hash[:10]}): {pdf_filepath}")
            return

        logger.info(f"Starting PDF ingestion (hash={file_hash[:10]}): {pdf_filepath}")
        tracker.update_file_status_to_processing(file_hash=file_hash, file_path=pdf_filepath)

        extracted_page_images = extract_pdf_pages_to_pil_images(
            pdf_filepath=pdf_filepath,
            dpi=engine.config.pdf_render_dpi,
        )
        num_pages: int = len(extracted_page_images)

        if num_pages == 0:
            error_msg: str = "PDF contains zero pages."
            logger.error(f"Failed PDF ingestion for {pdf_filepath}: {error_msg}")
            tracker.update_file_status_to_failed(
                file_hash=file_hash,
                file_path=pdf_filepath,
                error_message=error_msg,
            )
            return

        logger.info(f"Extracted {num_pages} page images from {pdf_filepath}. Encoding embeddings...")
        embeddings_tensor: torch.Tensor = engine.encode_multimodal_document_images(extracted_page_images)
        embeddings_numpy: np.ndarray = embeddings_tensor.cpu().float().numpy()

        os.makedirs(engine.config.embeddings_output_path, exist_ok=True)
        embedding_filename: str = f"{file_hash}.npy"
        output_npy_filepath: str = os.path.join(engine.config.embeddings_output_path, embedding_filename)

        np.save(output_npy_filepath, embeddings_numpy)
        logger.info(f"Saved embedding matrix to {output_npy_filepath} with shape {embeddings_numpy.shape}")

        tracker.update_file_status_to_done(
            file_hash=file_hash,
            file_path=pdf_filepath,
            num_pages=num_pages,
            embedding_path=output_npy_filepath,
        )
        logger.info(f"Successfully finished PDF ingestion for {pdf_filepath}")

    except Exception as exception_instance:
        logger.exception(f"Unhandled exception during PDF ingestion for {pdf_filepath}: {exception_instance}")
        try:
            file_hash_fallback: str = compute_file_sha256_hash(pdf_filepath)
            tracker.update_file_status_to_failed(
                file_hash=file_hash_fallback,
                file_path=pdf_filepath,
                error_message=str(exception_instance),
            )
        except Exception:
            pass


def scan_and_ingest_existing_pdf_folder(
    engine: EngineWrapper,
    tracker: ProcessedFilesTracker,
) -> None:
    watch_folder: str = engine.config.watch_folder_path
    if not os.path.exists(watch_folder):
        os.makedirs(watch_folder, exist_ok=True)
        logger.info(f"Created watch directory at {watch_folder}")

    logger.info(f"Scanning existing files in watch directory: {watch_folder}")
    for root_dir, _, file_names in os.walk(watch_folder):
        for file_name in sorted(file_names):
            if file_name.lower().endswith(".pdf"):
                full_pdf_path: str = os.path.join(root_dir, file_name)
                process_single_pdf_file_ingestion(
                    pdf_filepath=full_pdf_path,
                    engine=engine,
                    tracker=tracker,
                )


def register_graceful_shutdown_signal_handlers(
    tracker: ProcessedFilesTracker,
    observer_instance: Any,
) -> None:
    def signal_handler_callback(signal_number: int, frame_object: Any) -> None:
        logger.info(f"Received shutdown signal ({signal_number}). Initiating graceful server exit...")
        try:
            if observer_instance:
                observer_instance.stop()
                observer_instance.join()
        except Exception as exception_instance:
            logger.error(f"Error stopping watcher observer: {exception_instance}")

        reset_count: int = tracker.reset_in_progress_processing_to_pending()
        logger.info(f"Reset {reset_count} in-progress processing file record(s) back to pending state.")
        logger.info("Persistent document indexing server shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler_callback)
    signal.signal(signal.SIGTERM, signal_handler_callback)


def execute_background_server_pipeline(
    config_filepath: str,
    config: ModelConfigWrapper,
    tracker: ProcessedFilesTracker,
) -> Any:
    reset_count: int = tracker.reset_in_progress_processing_to_pending()
    if reset_count > 0:
        logger.info(f"Recovered startup state: reset {reset_count} leftover processing record(s) to pending.")

    logger.info(f"Loading engine and model once for process lifetime (base_model_id={config.base_model_id})...")
    engine: EngineWrapper = initialize_engine_from_yaml_config(config_filepath)
    logger.info("Engine model and processor loaded successfully.")

    scan_and_ingest_existing_pdf_folder(engine=engine, tracker=tracker)

    def pdf_detected_event_callback(pdf_filepath: str) -> None:
        logger.info(f"Folder watcher detected new/modified PDF: {pdf_filepath}")
        process_single_pdf_file_ingestion(
            pdf_filepath=pdf_filepath,
            engine=engine,
            tracker=tracker,
        )

    logger.info(f"Starting file system watcher on directory: {config.watch_folder_path}")
    observer_instance, _ = start_pdf_folder_watcher(
        watch_directory_path=config.watch_folder_path,
        callback_on_pdf_ready=pdf_detected_event_callback,
    )

    register_graceful_shutdown_signal_handlers(
        tracker=tracker,
        observer_instance=observer_instance,
    )
    return observer_instance


def start_document_indexing_server(
    config_filepath: str = "config.yaml",
) -> None:
    configure_rich_logging_for_server()
    config: ModelConfigWrapper = load_configuration_from_yaml_file(config_filepath)

    console = Console()
    console.print(render_server_status_rich_panel(config))

    logger.info("Starting persistent document-indexing server (Rich Interface Mode)...")
    tracker = ProcessedFilesTracker(db_filepath=config.sqlite_db_path)

    observer_instance = execute_background_server_pipeline(
        config_filepath=config_filepath,
        config=config,
        tracker=tracker,
    )
    logger.info(f"Server is actively watching folder '{config.watch_folder_path}'. Press Ctrl+C to stop.")
    try:
        while True:
            observer_instance.join(timeout=1.0)
    except (KeyboardInterrupt, SystemExit):
        pass
