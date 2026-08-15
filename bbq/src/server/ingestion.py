import gc
import logging
import os
from typing import Any, Dict, Optional

from rich.console import Console

from bbq.src.storage.sql import SqlliteDB
from bbq.src.utils.pdf_utils import (
    compute_file_sha256_hash,
    extract_pdf_page_range_to_pil_images,
    get_pdf_total_pages,
)

logger = logging.getLogger("bbq.server")


def process_single_pdf_file_ingestion(
    pdf_filepath: str,
    engine: Any,
    tracker: SqlliteDB,
    console: Optional[Console] = None,
    page_chunk_size: int = 8,
) -> None:
    if not os.path.exists(pdf_filepath):
        logger.warning(f"File not found for processing: {pdf_filepath}")
        return

    import numpy as np
    import torch

    try:
        file_hash: str = compute_file_sha256_hash(pdf_filepath)
        existing_record: Optional[Dict[str, Any]] = tracker.fetch_file_record_by_hash(file_hash)

        if existing_record and existing_record.get("status") == "done":
            logger.info(f"Skipping already processed PDF (hash={file_hash[:10]}): {pdf_filepath}")
            return

        logger.info(f"Starting PDF ingestion (hash={file_hash[:10]}): {pdf_filepath}")
        tracker.update_file_status_to_processing(file_hash=file_hash, file_path=pdf_filepath)

        filename = os.path.basename(pdf_filepath)
        num_pages: int = get_pdf_total_pages(pdf_filepath)

        if num_pages == 0:
            error_msg: str = "PDF contains zero pages."
            logger.error(f"Failed PDF ingestion for {pdf_filepath}: {error_msg}")
            tracker.update_file_status_to_failed(
                file_hash=file_hash,
                file_path=pdf_filepath,
                error_message=error_msg,
            )
            return

        logger.info(
            f"PDF {filename} has {num_pages} total pages. Streaming ingestion in chunks of {page_chunk_size} pages..."
        )
        status_ctx = (
            console.status(f"[bold yellow]Processing {filename} (0/{num_pages} pages)...[/bold yellow]", spinner="dots")
            if console
            else None
        )

        all_page_embeddings = []

        try:
            if status_ctx:
                status_ctx.start()

            for page_start in range(0, num_pages, page_chunk_size):
                page_end = min(page_start + page_chunk_size, num_pages)
                if status_ctx:
                    status_ctx.update(
                        f"[bold yellow]Processing {filename} ({page_start + 1}-{page_end}/{num_pages} pages)...[/bold yellow]"
                    )

                chunk_images = extract_pdf_page_range_to_pil_images(
                    pdf_filepath=pdf_filepath,
                    start_page_idx=page_start,
                    end_page_idx=page_end,
                    dpi=engine.config.pdf_render_dpi,
                )

                chunk_tensor: torch.Tensor = engine.encode_multimodal_document_images(chunk_images, batch_size=4)
                all_page_embeddings.append(chunk_tensor.cpu().float().numpy())

                del chunk_images
                gc.collect()

            embeddings_numpy: np.ndarray = np.concatenate(all_page_embeddings, axis=0)

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
            logger.info(f"Successfully finished PDF ingestion for {pdf_filepath} ({num_pages} pages)")
        finally:
            if status_ctx:
                status_ctx.stop()

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


def process_single_pdf_file_deletion(
    pdf_filepath: str,
    tracker: SqlliteDB,
) -> bool:
    logger.info(f"Processing deletion for PDF file: {pdf_filepath}")
    deleted_record = tracker.remove_file_and_embeddings(pdf_filepath)
    if deleted_record:
        logger.info(
            f"Successfully removed document (hash={deleted_record['file_hash'][:10]}) "
            f"and embeddings from model/database: {pdf_filepath}"
        )
        return True
    else:
        logger.warning(f"No record found in database for deleted PDF file: {pdf_filepath}")
        return False


def sync_and_cleanup_deleted_files(
    watch_folder: str,
    tracker: SqlliteDB,
) -> None:
    records = tracker.fetch_all_records()
    for record in records:
        file_path = record.get("file_path")
        if file_path and not os.path.exists(file_path):
            logger.info(f"Detected removed file during startup sync: {file_path}")
            process_single_pdf_file_deletion(pdf_filepath=file_path, tracker=tracker)


def scan_and_ingest_existing_pdf_folder(
    engine: Any,
    tracker: SqlliteDB,
    console: Optional[Console] = None,
) -> None:
    watch_folder: str = engine.config.watch_folder_path
    if not os.path.exists(watch_folder):
        os.makedirs(watch_folder, exist_ok=True)
        logger.info(f"Created watch directory at {watch_folder}")

    sync_and_cleanup_deleted_files(watch_folder=watch_folder, tracker=tracker)

    logger.info(f"Scanning existing files in watch directory: {watch_folder}")
    for root_dir, _, file_names in os.walk(watch_folder):
        for file_name in sorted(file_names):
            if file_name.lower().endswith(".pdf"):
                full_pdf_path: str = os.path.join(root_dir, file_name)
                process_single_pdf_file_ingestion(
                    pdf_filepath=full_pdf_path,
                    engine=engine,
                    tracker=tracker,
                    console=console,
                )
