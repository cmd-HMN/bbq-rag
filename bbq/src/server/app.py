from __future__ import annotations

import io
import logging
import threading
import time
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response

from bbq.src.server.retrieval import query_indexed_documents
from bbq.src.storage.sql import SqlliteDB
from bbq.src.utils.pdf_utils import extract_single_pdf_page_image

logger = logging.getLogger("bbq.server")


def create_bbq_fastapi_app(
    engine: Optional[Any] = None,
    tracker: Optional[SqlliteDB] = None,
    get_engine_callback: Optional[Any] = None,
    config: Optional[Any] = None,
) -> FastAPI:
    """
    Creates the FastAPI web server for query retrieval, image rendering, and server status.
    """
    app = FastAPI(title="BBQ RAG Server", version="0.1.0")

    def _get_active_engine() -> Optional[Any]:
        if engine is not None:
            return engine
        if get_engine_callback is not None:
            return get_engine_callback()
        return None

    @app.get("/")
    def root(request: Request):
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Root endpoint '/' accessed by client [{client_host}]")
        return {"status": "online", "message": "BBQ RAG Document Retrieval Server"}

    @app.get("/status")
    def get_status(request: Request):
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Status check requested by client [{client_host}]")
        active_engine = _get_active_engine()
        records = tracker.fetch_all_records() if tracker else []
        done_count = sum(1 for r in records if r.get("status") == "done")

        engine_ready = active_engine is not None
        cfg = getattr(active_engine, "config", None) or config
        base_model_id = cfg.base_model_id if cfg else "Loading..."
        watch_folder = cfg.watch_folder_path if cfg else "data/watch"
        quantization_mode = (
            getattr(cfg, "quantization", None) or "f32"
        ).strip().lower()

        device = "Loading..."
        if (
            active_engine
            and hasattr(active_engine, "model")
            and active_engine.model is not None
        ):
            param = next(active_engine.model.parameters(), None)
            if param is not None:
                device = str(param.device)

        return {
            "status": "online" if engine_ready else "loading_engine",
            "engine_ready": engine_ready,
            "base_model_id": base_model_id,
            "device": device,
            "watch_folder": watch_folder,
            "quantization": quantization_mode,
            "total_documents": len(records),
            "indexed_documents": done_count,
        }

    @app.get("/documents")
    def get_documents(request: Request):
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Document list requested by client [{client_host}]")
        return {"documents": tracker.fetch_all_records() if tracker else []}

    def _handle_query(
        query_text: str, top_k: int, client_host: str
    ) -> Dict[str, Any]:
        query_text = (query_text or "").strip()
        if not query_text:
            logger.warning(f"Rejected empty query from client [{client_host}]")
            raise HTTPException(
                status_code=400, detail="Query cannot be empty."
            )

        active_engine = _get_active_engine()
        if not active_engine:
            logger.warning(
                f"Rejected query from client [{client_host}]: Engine is still loading weights."
            )
            raise HTTPException(
                status_code=503,
                detail="AI Model weights are currently initializing. Please try again in a few seconds.",
            )

        start_time = time.time()
        results = query_indexed_documents(
            query_text=query_text,
            engine=active_engine,
            tracker=tracker,
            top_k=top_k,
        )
        elapsed = time.time() - start_time
        logger.info(
            f"Successfully served query to [{client_host}] in {elapsed:.3f}s: returned {len(results)} match(es)"
        )
        return {"query": query_text, "top_k": top_k, "results": results}

    @app.post("/query")
    def post_query(body: Dict[str, Any], request: Request):
        client_host = request.client.host if request.client else "unknown"
        query_text = body.get("query", "")
        top_k = body.get("top_k", 5)
        logger.info(
            f"Received POST /query request from client [{client_host}]: '{query_text}' (top_k={top_k})"
        )
        return _handle_query(
            query_text=query_text, top_k=top_k, client_host=client_host
        )

    @app.get("/query")
    def get_query(q: str, request: Request, top_k: int = 5):
        client_host = request.client.host if request.client else "unknown"
        logger.info(
            f"Received GET /query request from client [{client_host}]: '{q}' (top_k={top_k})"
        )
        return _handle_query(query_text=q, top_k=top_k, client_host=client_host)

    @app.get("/page_image")
    def get_page_image(
        file_path: str,
        request: Request,
        page_number: int = 1,
        dpi: int = 150,
    ):
        client_host = request.client.host if request.client else "unknown"
        logger.info(
            f"Client [{client_host}] requested page image for '{file_path}' (page={page_number}, dpi={dpi})"
        )
        try:
            pil_img = extract_single_pdf_page_image(
                pdf_filepath=file_path, page_number=page_number, dpi=dpi
            )
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            logger.info(
                f"Successfully served page image to client [{client_host}]"
            )
            return Response(content=buf.getvalue(), media_type="image/png")
        except Exception as exc:
            logger.error(
                f"Failed to serve page image to client [{client_host}]: {exc}"
            )
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
    logger.info(f"BBQ query API server listening on http://{host}:{port}")
    return server_thread
