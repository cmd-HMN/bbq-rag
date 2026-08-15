from typing import Any

__all__ = [
    "BBQServer",
    "start_document_indexing_server",
    "execute_background_server_pipeline",
    "query_indexed_documents",
    "create_bbq_fastapi_app",
    "create_colpali_fastapi_app",
    "run_http_server_in_thread",
    "process_single_pdf_file_ingestion",
    "process_single_pdf_file_deletion",
    "scan_and_ingest_existing_pdf_folder",
    "sync_and_cleanup_deleted_files",
]


def __getattr__(name: str) -> Any:
    if name in ("BBQServer", "start_document_indexing_server", "execute_background_server_pipeline"):
        from bbq.src.server.server import (
            BBQServer,
            start_document_indexing_server,
            execute_background_server_pipeline,
        )
        mapping = {
            "BBQServer": BBQServer,
            "start_document_indexing_server": start_document_indexing_server,
            "execute_background_server_pipeline": execute_background_server_pipeline,
        }
        return mapping[name]
    elif name == "query_indexed_documents":
        from bbq.src.server.retrieval import query_indexed_documents
        return query_indexed_documents
    elif name in ("create_bbq_fastapi_app", "create_colpali_fastapi_app", "run_http_server_in_thread"):
        from bbq.src.server.app import create_bbq_fastapi_app, create_colpali_fastapi_app, run_http_server_in_thread
        mapping = {
            "create_bbq_fastapi_app": create_bbq_fastapi_app,
            "create_colpali_fastapi_app": create_colpali_fastapi_app,
            "run_http_server_in_thread": run_http_server_in_thread,
        }
        return mapping[name]
    elif name in (
        "process_single_pdf_file_ingestion",
        "process_single_pdf_file_deletion",
        "scan_and_ingest_existing_pdf_folder",
        "sync_and_cleanup_deleted_files",
    ):
        from bbq.src.server.ingestion import (
            process_single_pdf_file_ingestion,
            process_single_pdf_file_deletion,
            scan_and_ingest_existing_pdf_folder,
            sync_and_cleanup_deleted_files,
        )
        mapping = {
            "process_single_pdf_file_ingestion": process_single_pdf_file_ingestion,
            "process_single_pdf_file_deletion": process_single_pdf_file_deletion,
            "scan_and_ingest_existing_pdf_folder": scan_and_ingest_existing_pdf_folder,
            "sync_and_cleanup_deleted_files": sync_and_cleanup_deleted_files,
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
