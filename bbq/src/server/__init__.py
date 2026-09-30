from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from bbq.src.server.app import (
        create_bbq_fastapi_app,
        run_http_server_in_thread,
    )
    from bbq.src.server.ingestion import (
        process_single_pdf_file_ingestion,
        process_single_pdf_file_deletion,
        scan_and_ingest_existing_pdf_folder,
        sync_and_cleanup_deleted_files,
    )
    from bbq.src.server.retrieval import query_indexed_documents
    from bbq.src.server.server import (
        BBQServer,
        start_document_indexing_server,
    )

__all__ = [
    "BBQServer",
    "start_document_indexing_server",
    "query_indexed_documents",
    "create_bbq_fastapi_app",
    "run_http_server_in_thread",
    "process_single_pdf_file_ingestion",
    "process_single_pdf_file_deletion",
    "scan_and_ingest_existing_pdf_folder",
    "sync_and_cleanup_deleted_files",
]

_EXPORTS = {
    "BBQServer": "bbq.src.server.server",
    "start_document_indexing_server": "bbq.src.server.server",
    "query_indexed_documents": "bbq.src.server.retrieval",
    "create_bbq_fastapi_app": "bbq.src.server.app",
    "run_http_server_in_thread": "bbq.src.server.app",
    "process_single_pdf_file_ingestion": "bbq.src.server.ingestion",
    "process_single_pdf_file_deletion": "bbq.src.server.ingestion",
    "scan_and_ingest_existing_pdf_folder": "bbq.src.server.ingestion",
    "sync_and_cleanup_deleted_files": "bbq.src.server.ingestion",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path:
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
