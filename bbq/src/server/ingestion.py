from __future__ import annotations

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
    batch_size: int = 4,
) -> None:
    if not os.path.exists(pdf_filepath):
        logger.warning(f"File not found for processing: {pdf_filepath}")
        return

    import numpy as np
    import torch

    file_hash: Optional[str] = None
    try:
        file_hash = compute_file_sha256_hash(pdf_filepath)
        existing_record: Optional[Dict[str, Any]] = tracker.fetch_file_record_by_hash(
            file_hash
        )

        if existing_record and existing_record.get("status") == "done":
            logger.info(
                f"Skipping already processed PDF (hash={file_hash[:10]}): {pdf_filepath}"
            )
            return

        logger.info(f"Starting PDF ingestion (hash={file_hash[:10]}): {pdf_filepath}")
        tracker.update_file_status_to_processing(
            file_hash=file_hash, file_path=pdf_filepath
        )

        filename = os.path.basename(pdf_filepath)
        num_pages: int = get_pdf_total_pages(pdf_filepath)

        if num_pages == 0:
            error_msg = "PDF contains zero pages."
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
            console.status(
                f"[bold yellow]Processing {filename} (0/{num_pages} pages)...[/bold yellow]",
                spinner="dots",
            )
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

                chunk_tensor: torch.Tensor = engine.encode_multimodal_document_images(
                    chunk_images, batch_size=batch_size
                )
                all_page_embeddings.append(chunk_tensor.cpu().float().numpy())

                del chunk_images
                gc.collect()

            embeddings_numpy: np.ndarray = np.concatenate(all_page_embeddings, axis=0)

            os.makedirs(engine.config.embeddings_output_path, exist_ok=True)

            # Quantization checking: only apply qi8 if configured and dim is 128
            quant_mode = (
                getattr(engine.config, "quantization", None) or "f32"
            ).strip().lower()

            is_quantized = False
            if quant_mode in ("qi8", "int8"):
                dim = embeddings_numpy.shape[-1]
                if dim == 128:
                    is_quantized = True
                else:
                    logger.warning(
                        f"Quantization '{quant_mode}' requested but vector dimension is {dim} "
                        f"(requires 128). Saving unquantized float32 embeddings."
                    )

            if is_quantized:
                from bbq.maxsimd.quantization import qi8

                val, scale = qi8(embeddings_numpy, dim=128)
                output_filepath = os.path.join(
                    engine.config.embeddings_output_path, f"{file_hash}.npz"
                )
                np.savez(output_filepath, values=val, scales=scale)
                logger.info(
                    f"Saved quantized ({quant_mode}) embedding matrix to {output_filepath} "
                    f"(values: {val.shape}, scales: {scale.shape})"
                )
            else:
                output_filepath = os.path.join(
                    engine.config.embeddings_output_path, f"{file_hash}.npy"
                )
                np.save(output_filepath, embeddings_numpy)
                logger.info(
                    f"Saved unquantized embedding matrix to {output_filepath} with shape {embeddings_numpy.shape}"
                )

            tracker.update_file_status_to_done(
                file_hash=file_hash,
                file_path=pdf_filepath,
                num_pages=num_pages,
                embedding_path=output_filepath,
            )
            logger.info(
                f"Successfully finished PDF ingestion for {pdf_filepath} ({num_pages} pages)"
            )
        finally:
            if status_ctx:
                status_ctx.stop()

    except Exception as exception_instance:
        logger.exception(
            f"Unhandled exception during PDF ingestion for {pdf_filepath}"
        )
        try:
            target_hash = file_hash or compute_file_sha256_hash(pdf_filepath)
            tracker.update_file_status_to_failed(
                file_hash=target_hash,
                file_path=pdf_filepath,
                error_message=str(exception_instance),
            )
        except Exception as fallback_err:
            logger.error(
                f"Failed to record ingestion error for {pdf_filepath}: {fallback_err}"
            )


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
        logger.warning(
            f"No record found in database for deleted PDF file: {pdf_filepath}"
        )
        return False


def sync_and_cleanup_deleted_files(
    watch_folder: Optional[str],
    tracker: SqlliteDB,
) -> None:
    records = tracker.fetch_all_records()
    for record in records:
        file_path = record.get("file_path")
        if not file_path:
            continue
        if watch_folder and not os.path.abspath(file_path).startswith(
            os.path.abspath(watch_folder)
        ):
            continue
        if not os.path.exists(file_path):
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
                full_pdf_path = os.path.join(root_dir, file_name)
                process_single_pdf_file_ingestion(
                    pdf_filepath=full_pdf_path,
                    engine=engine,
                    tracker=tracker,
                    console=console,
                )
