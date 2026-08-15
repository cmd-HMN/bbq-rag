import os
import io
import sys
import signal
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bbq.src.config import (
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
    extract_single_pdf_page_image,
)
from bbq.src.utils.watcher import start_pdf_folder_watcher
from bbq.src.terminal.tui import (
    render_server_status_rich_panel,
    configure_rich_logging_for_server,
)
from fastapi import FastAPI, HTTPException, Response
import uvicorn
import threading
from maxsimd import maxsim_vrlen

logger = logging.getLogger("bbq.server")


import gc
from bbq.src.utils.pdf_utils import (
    compute_file_sha256_hash,
    extract_pdf_pages_to_pil_images,
    extract_pdf_page_range_to_pil_images,
    extract_single_pdf_page_image,
    get_pdf_total_pages,
)


def process_single_pdf_file_ingestion(
    pdf_filepath: str,
    engine: EngineWrapper,
    tracker: ProcessedFilesTracker,
    console: Optional[Console] = None,
    page_chunk_size: int = 8,
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

        logger.info(f"PDF {filename} has {num_pages} total pages. Streaming ingestion in chunks of {page_chunk_size} pages...")
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
                    status_ctx.update(f"[bold yellow]Processing {filename} ({page_start+1}-{page_end}/{num_pages} pages)...[/bold yellow]")

                chunk_images = extract_pdf_page_range_to_pil_images(
                    pdf_filepath=pdf_filepath,
                    start_page_idx=page_start,
                    end_page_idx=page_end,
                    dpi=engine.config.pdf_render_dpi,
                )

                chunk_tensor: torch.Tensor = engine.encode_multimodal_document_images(
                    chunk_images, batch_size=4
                )
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
    tracker: ProcessedFilesTracker,
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
    tracker: ProcessedFilesTracker,
) -> None:
    records = tracker.fetch_all_records()
    for record in records:
        file_path = record.get("file_path")
        if file_path and not os.path.exists(file_path):
            logger.info(f"Detected removed file during startup sync: {file_path}")
            process_single_pdf_file_deletion(pdf_filepath=file_path, tracker=tracker)


def scan_and_ingest_existing_pdf_folder(
    engine: EngineWrapper,
    tracker: ProcessedFilesTracker,
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


def query_indexed_documents(
    query_text: str,
    engine: EngineWrapper,
    tracker: ProcessedFilesTracker,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Encodes the query text, loads indexed PDF page embeddings, calculates MaxSim scores,
    and returns top-k matching PDF pages.
    """
    if not query_text.strip():
        return []

    records = tracker.fetch_all_records()
    completed_records = [r for r in records if r.get("status") == "done" and r.get("embedding_path")]

    if not completed_records:
        logger.info("No completed PDF embeddings found in tracker database.")
        return []

    # Encode query: shape [1, q_len, dim]
    q_tensor: torch.Tensor = engine.encode_query_text_inputs([query_text])
    q_flat = q_tensor[0].cpu().float().numpy()
    q_len = q_flat.shape[0]
    dim = q_flat.shape[1]
    q_flat_1d = np.ascontiguousarray(q_flat.reshape(-1), dtype=np.float32)

    all_page_results = []

    for record in completed_records:
        emb_path = record["embedding_path"]
        if not os.path.exists(emb_path):
            continue

        doc_emb = np.load(emb_path)
        num_pages = record.get("num_pages", 1)

        if doc_emb.ndim == 3:
            num_pages = doc_emb.shape[0]
            tokens_per_page = doc_emb.shape[1]
            page_lengths = [tokens_per_page] * num_pages
            d_flat_1d = np.ascontiguousarray(doc_emb.reshape(-1), dtype=np.float32)

            scores = maxsim_vrlen(q_flat_1d, d_flat_1d, page_lengths, q_len, dim)
            for page_idx, score in enumerate(scores):
                all_page_results.append({
                    "score": float(score),
                    "file_path": record["file_path"],
                    "filename": os.path.basename(record["file_path"]),
                    "file_hash": record["file_hash"],
                    "page_number": page_idx + 1,
                    "total_pages": num_pages,
                })
        elif doc_emb.ndim == 2:
            tokens = doc_emb.shape[0]
            d_flat_1d = np.ascontiguousarray(doc_emb.reshape(-1), dtype=np.float32)
            scores = maxsim_vrlen(q_flat_1d, d_flat_1d, [tokens], q_len, dim)
            all_page_results.append({
                "score": float(scores[0]),
                "file_path": record["file_path"],
                "filename": os.path.basename(record["file_path"]),
                "file_hash": record["file_hash"],
                "page_number": 1,
                "total_pages": 1,
            })

    all_page_results.sort(key=lambda x: x["score"], reverse=True)
    return all_page_results[:top_k]


def create_colpali_fastapi_app(
    engine: EngineWrapper,
    tracker: ProcessedFilesTracker,
) -> FastAPI:
    """
    Creates the FastAPI web server for query retrieval, image rendering, and server management.
    """
    app = FastAPI(title="ColPali RAG Server", version="1.0.0")

    @app.get("/")
    def root():
        return {"status": "online", "message": "ColPali RAG Document Retrieval Server"}

    @app.get("/status")
    def get_status():
        records = tracker.fetch_all_records()
        done_count = sum(1 for r in records if r.get("status") == "done")
        return {
            "status": "online",
            "base_model_id": engine.config.base_model_id,
            "device": str(next(engine.model.parameters()).device),
            "watch_folder": engine.config.watch_folder_path,
            "total_documents": len(records),
            "indexed_documents": done_count,
        }

    @app.get("/documents")
    def get_documents():
        return {"documents": tracker.fetch_all_records()}

    @app.post("/query")
    def post_query(body: Dict[str, Any]):
        query_text = body.get("query", "")
        top_k = body.get("top_k", 5)
        if not query_text:
            raise HTTPException(status_code=400, detail="Field 'query' cannot be empty.")
        results = query_indexed_documents(
            query_text=query_text,
            engine=engine,
            tracker=tracker,
            top_k=top_k,
        )
        return {"query": query_text, "top_k": top_k, "results": results}

    @app.get("/query")
    def get_query(q: str, top_k: int = 5):
        if not q:
            raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty.")
        results = query_indexed_documents(
            query_text=q,
            engine=engine,
            tracker=tracker,
            top_k=top_k,
        )
        return {"query": q, "top_k": top_k, "results": results}

    @app.get("/page_image")
    def get_page_image(
        file_path: str,
        page_number: int = 1,
        dpi: int = 150,
    ):
        try:
            pil_img = extract_single_pdf_page_image(file_path, page_number=page_number, dpi=dpi)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app


def run_http_server_in_thread(
    app: FastAPI,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> threading.Thread:
    """Runs Uvicorn HTTP server in a daemon thread."""
    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    server_thread = threading.Thread(target=_run, daemon=True)
    server_thread.start()
    logger.info(f"ColPali query API server listening on http://{host}:{port}")
    return server_thread


def execute_background_server_pipeline(
    config_filepath: str,
    config: ModelConfigWrapper,
    tracker: ProcessedFilesTracker,
    console: Console,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> Tuple[Any, EngineWrapper]:
    reset_count: int = tracker.reset_in_progress_processing_to_pending()
    if reset_count > 0:
        logger.info(f"Recovered startup state: reset {reset_count} leftover processing record(s) to pending.")

    logger.info(f"Loading engine and model once for process lifetime (base_model_id={config.base_model_id})...")
    with console.status(f"[bold cyan]Loading AI model and weights ({config.base_model_id}). Please wait...[/bold cyan]", spinner="dots"):
        engine: EngineWrapper = initialize_engine_from_yaml_config(config_filepath)
    logger.info("Engine model and processor loaded successfully.")

    with console.status(f"[bold green]Scanning '{config.watch_folder_path}' for existing PDF documents...[/bold green]", spinner="dots"):
        scan_and_ingest_existing_pdf_folder(engine=engine, tracker=tracker, console=console)

    def pdf_detected_event_callback(pdf_filepath: str) -> None:
        logger.info(f"Folder watcher detected new/modified PDF: {pdf_filepath}")
        process_single_pdf_file_ingestion(
            pdf_filepath=pdf_filepath,
            engine=engine,
            tracker=tracker,
            console=console,
        )

    def pdf_deleted_event_callback(pdf_filepath: str) -> None:
        logger.info(f"Folder watcher detected deleted PDF: {pdf_filepath}")
        process_single_pdf_file_deletion(
            pdf_filepath=pdf_filepath,
            tracker=tracker,
        )

    logger.info(f"Starting file system watcher on directory: {config.watch_folder_path}")
    observer_instance, _ = start_pdf_folder_watcher(
        watch_directory_path=config.watch_folder_path,
        callback_on_pdf_ready=pdf_detected_event_callback,
        callback_on_pdf_deleted=pdf_deleted_event_callback,
    )

    fastapi_app = create_colpali_fastapi_app(engine=engine, tracker=tracker)
    run_http_server_in_thread(fastapi_app, host=host, port=port)

    register_graceful_shutdown_signal_handlers(
        tracker=tracker,
        observer_instance=observer_instance,
    )
    return observer_instance, engine


def start_document_indexing_server(
    config_filepath: str = "config.yaml",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    configure_rich_logging_for_server()
    config: ModelConfigWrapper = load_configuration_from_yaml_file(config_filepath)

    console = Console()
    console.print(render_server_status_rich_panel(config))

    logger.info("Starting persistent document-indexing server (Rich Interface Mode)...")
    tracker = ProcessedFilesTracker(db_filepath=config.sqlite_db_path)

    observer_instance, _ = execute_background_server_pipeline(
        config_filepath=config_filepath,
        config=config,
        tracker=tracker,
        console=console,
        host=host,
        port=port,
    )
    logger.info(f"Server is actively watching folder '{config.watch_folder_path}' and listening on port {port}. Press Ctrl+C to stop.")
    try:
        while True:
            observer_instance.join(timeout=1.0)
    except (KeyboardInterrupt, SystemExit):
        pass

