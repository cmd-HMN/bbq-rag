import os
import threading
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import bbq.src.server as server_module
from bbq.src.config import Config
from bbq.src.server.app import create_bbq_fastapi_app, run_http_server_in_thread
from bbq.src.server.ingestion import (
    process_single_pdf_file_deletion,
    process_single_pdf_file_ingestion,
    scan_and_ingest_existing_pdf_folder,
    sync_and_cleanup_deleted_files,
)
from bbq.src.server.retrieval import query_indexed_documents
from bbq.src.server.server import BBQServer, start_document_indexing_server
from bbq.src.storage.sql import SqlliteDB


@pytest.fixture
def mock_config(tmp_path):
    """Provides a lightweight Config pointing to temporary directories."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    db_file = tmp_path / "bbq_test.db"

    cfg = Config()
    cfg.watch_folder_path = str(watch_dir)
    cfg.embeddings_output_path = str(emb_dir)
    cfg.sqlite_db_path = str(db_file)
    cfg.base_model_id = "test/mock-model"
    cfg.embedding_dim = 128
    cfg.rag_top_k = 3
    cfg.quantization = "f32"
    cfg.pdf_render_dpi = 72
    return cfg


@pytest.fixture
def mock_tracker(tmp_path):
    """Provides a fresh SqlliteDB instance using a temporary file."""
    db_file = str(tmp_path / "tracker_test.db")
    return SqlliteDB(db_filepath=db_file)


class DummyEngine:
    """Mock multimodal engine for testing retrieval and ingestion."""

    def __init__(self, config: Config, dim: int = 128):
        self.config = config
        self.dim = dim
        # Mock parameter to report device
        mock_param = torch.nn.Parameter(torch.zeros(1, device="cpu"))
        self.model = torch.nn.Module()
        self.model.register_parameter("dummy_weight", mock_param)

    def encode_query_text_inputs(self, queries: List[str]) -> torch.Tensor:
        # Return shape [len(queries), q_tokens=4, dim=128]
        return torch.randn(len(queries), 4, self.dim, dtype=torch.float32)

    def encode_multimodal_document_images(
        self, images: List[Any], batch_size: int = 4
    ) -> torch.Tensor:
        # Return shape [len(images), doc_tokens=8, dim=128]
        return torch.randn(len(images), 8, self.dim, dtype=torch.float32)


def test_server_module_dynamic_exports():
    """Verify lazy exports via bbq.src.server.__getattr__."""
    assert hasattr(server_module, "BBQServer")
    assert hasattr(server_module, "create_bbq_fastapi_app")
    assert hasattr(server_module, "query_indexed_documents")
    assert hasattr(server_module, "process_single_pdf_file_ingestion")
    assert hasattr(server_module, "process_single_pdf_file_deletion")
    assert hasattr(server_module, "scan_and_ingest_existing_pdf_folder")
    assert hasattr(server_module, "sync_and_cleanup_deleted_files")
    assert hasattr(server_module, "run_http_server_in_thread")
    assert hasattr(server_module, "start_document_indexing_server")

    with pytest.raises(AttributeError, match="has no attribute 'non_existent_symbol'"):
        _ = getattr(server_module, "non_existent_symbol")


def test_fastapi_root_endpoint():
    """Verify GET / returns expected online payload."""
    app = create_bbq_fastapi_app()
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "BBQ RAG" in data["message"]


def test_fastapi_status_engine_loading(mock_tracker, mock_config):
    """Verify GET /status when engine is not yet loaded."""
    app = create_bbq_fastapi_app(engine=None, tracker=mock_tracker, config=mock_config)
    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "loading_engine"
    assert data["engine_ready"] is False
    assert data["base_model_id"] == "test/mock-model"
    assert data["device"] == "Loading..."
    assert data["total_documents"] == 0
    assert data["indexed_documents"] == 0


def test_fastapi_status_engine_ready(mock_tracker, mock_config):
    """Verify GET /status when engine is active."""
    engine = DummyEngine(mock_config)
    # Add a mock record in tracker
    mock_tracker.update_file_status_to_processing("hash123", "/path/to/doc.pdf")
    mock_tracker.update_file_status_to_done(
        "hash123", "/path/to/doc.pdf", 5, "/path/to/emb.npy"
    )

    app = create_bbq_fastapi_app(
        engine=engine, tracker=mock_tracker, config=mock_config
    )
    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["engine_ready"] is True
    assert data["device"] == "cpu"
    assert data["quantization"] == "f32"
    assert data["total_documents"] == 1
    assert data["indexed_documents"] == 1


def test_fastapi_documents_endpoint(mock_tracker):
    """Verify GET /documents returns list of tracked documents."""
    mock_tracker.update_file_status_to_processing("hash_a", "/docA.pdf")
    app = create_bbq_fastapi_app(tracker=mock_tracker)
    client = TestClient(app)
    response = client.get("/documents")
    assert response.status_code == 200
    docs = response.json().get("documents", [])
    assert len(docs) == 1
    assert docs[0]["file_hash"] == "hash_a"


def test_fastapi_query_empty_rejects(mock_tracker, mock_config):
    """Verify query endpoints reject empty or whitespace queries with 400."""
    engine = DummyEngine(mock_config)
    app = create_bbq_fastapi_app(
        engine=engine, tracker=mock_tracker, config=mock_config
    )
    client = TestClient(app)

    # POST empty query
    res_post = client.post("/query", json={"query": "   ", "top_k": 5})
    assert res_post.status_code == 400
    assert "Query cannot be empty" in res_post.json()["detail"]

    # GET empty query
    res_get = client.get("/query?q=")
    assert res_get.status_code == 400
    assert "Query cannot be empty" in res_get.json()["detail"]


def test_fastapi_query_engine_not_ready(mock_tracker):
    """Verify query endpoints return 503 when engine is None."""
    app = create_bbq_fastapi_app(engine=None, tracker=mock_tracker)
    client = TestClient(app)

    res_post = client.post("/query", json={"query": "test query"})
    assert res_post.status_code == 503
    assert "initializing" in res_post.json()["detail"]

    res_get = client.get("/query?q=test")
    assert res_get.status_code == 503
    assert "initializing" in res_get.json()["detail"]


@patch("bbq.src.server.app.query_indexed_documents")
def test_fastapi_query_success(mock_query_func, mock_tracker, mock_config):
    """Verify POST and GET /query return structured retrieval results."""
    mock_query_func.return_value = [
        {"filename": "doc.pdf", "page_number": 1, "score": 25.5}
    ]
    engine = DummyEngine(mock_config)
    app = create_bbq_fastapi_app(
        engine=engine, tracker=mock_tracker, config=mock_config
    )
    client = TestClient(app)

    # POST /query
    res_post = client.post("/query", json={"query": "machine learning", "top_k": 3})
    assert res_post.status_code == 200
    post_data = res_post.json()
    assert post_data["query"] == "machine learning"
    assert post_data["top_k"] == 3
    assert len(post_data["results"]) == 1
    assert post_data["results"][0]["score"] == 25.5

    # GET /query
    res_get = client.get("/query?q=deep+learning&top_k=2")
    assert res_get.status_code == 200
    get_data = res_get.json()
    assert get_data["query"] == "deep learning"
    assert get_data["top_k"] == 2


@patch("bbq.src.server.app.extract_single_pdf_page_image")
def test_fastapi_page_image_endpoint(mock_extract_img):
    """Verify GET /page_image renders PNG bytes or returns 400 on error."""
    # Success case: returns 10x10 PIL image
    img = Image.new("RGB", (10, 10), color="blue")
    mock_extract_img.return_value = img

    app = create_bbq_fastapi_app()
    client = TestClient(app)
    res = client.get("/page_image?file_path=/fake/doc.pdf&page_number=1&dpi=72")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 0

    # Error case: exception raised
    mock_extract_img.side_effect = RuntimeError("PDF page does not exist")
    res_err = client.get("/page_image?file_path=/fake/doc.pdf&page_number=99")
    assert res_err.status_code == 400
    assert "PDF page does not exist" in res_err.json()["detail"]


@patch("uvicorn.run")
def test_run_http_server_in_thread(mock_uvicorn_run):
    """Verify run_http_server_in_thread spawns a daemon thread running uvicorn."""
    app = create_bbq_fastapi_app()
    thread = run_http_server_in_thread(app, host="127.0.0.1", port=9999)
    assert isinstance(thread, threading.Thread)
    assert thread.daemon is True
    thread.join(timeout=1.0)
    mock_uvicorn_run.assert_called_once_with(
        app, host="127.0.0.1", port=9999, log_level="warning"
    )


def test_query_indexed_documents_empty_inputs(mock_tracker, mock_config):
    """Verify empty query or empty database returns empty list."""
    engine = DummyEngine(mock_config)
    # Empty query
    assert query_indexed_documents("", engine, mock_tracker) == []
    assert query_indexed_documents("   ", engine, mock_tracker) == []

    # Non-empty query, but no indexed documents in tracker
    assert query_indexed_documents("hello", engine, mock_tracker) == []


def test_query_indexed_documents_float32(mock_tracker, mock_config, tmp_path):
    """Verify retrieval against float32 .npy embeddings."""
    engine = DummyEngine(mock_config, dim=128)

    # Create dummy float32 document embeddings: 2 pages, 8 tokens per page, 128 dim
    doc_emb = np.random.randn(2, 8, 128).astype(np.float32)
    emb_file = str(tmp_path / "doc1.npy")
    np.save(emb_file, doc_emb)

    mock_tracker.update_file_status_to_done(
        file_hash="hash1",
        file_path="/watch/doc1.pdf",
        num_pages=2,
        embedding_path=emb_file,
    )

    results = query_indexed_documents(
        query_text="query test",
        engine=engine,
        tracker=mock_tracker,
        top_k=5,
    )
    assert len(results) == 2
    assert results[0]["filename"] == "doc1.pdf"
    assert results[0]["file_hash"] == "hash1"
    assert results[0]["total_pages"] == 2
    assert "score" in results[0]
    assert results[0]["score"] >= results[1]["score"]


def test_query_indexed_documents_quantized_qi8(mock_tracker, mock_config, tmp_path):
    """Verify retrieval against quantized qi8 .npz embeddings."""
    mock_config.quantization = "qi8"
    engine = DummyEngine(mock_config, dim=128)

    from bbq.maxsimd.quantization import qi8

    # Create 3 pages of document embeddings, quantize to qi8
    doc_f32 = np.random.randn(3, 8, 128).astype(np.float32)
    val, scale = qi8(doc_f32, dim=128)

    emb_file = str(tmp_path / "quant_doc.npz")
    np.savez(emb_file, values=val, scales=scale)

    mock_tracker.update_file_status_to_done(
        file_hash="hash_quant",
        file_path="/watch/quant_doc.pdf",
        num_pages=3,
        embedding_path=emb_file,
    )

    results = query_indexed_documents(
        query_text="quantum computing",
        engine=engine,
        tracker=mock_tracker,
        top_k=2,
    )
    assert len(results) == 2
    assert results[0]["filename"] == "quant_doc.pdf"
    assert results[0]["score"] >= results[1]["score"]


def test_query_indexed_documents_mixed_modes(mock_tracker, mock_config, tmp_path):
    """Verify query handles cross-mode matching (float32 query against qi8 doc, and qi8 query against f32 doc)."""
    from bbq.maxsimd.quantization import qi8

    # Doc 1: quantized .npz
    doc1_f32 = np.random.randn(2, 6, 128).astype(np.float32)
    v1, s1 = qi8(doc1_f32, dim=128)
    file_npz = str(tmp_path / "doc1.npz")
    np.savez(file_npz, values=v1, scales=s1)
    mock_tracker.update_file_status_to_done("h1", "/watch/doc1.pdf", 2, file_npz)

    # Doc 2: unquantized .npy
    doc2_f32 = np.random.randn(2, 6, 128).astype(np.float32)
    file_npy = str(tmp_path / "doc2.npy")
    np.save(file_npy, doc2_f32)
    mock_tracker.update_file_status_to_done("h2", "/watch/doc2.pdf", 2, file_npy)

    # Case A: Engine is configured as f32 -> scores against both doc1 (.npz) and doc2 (.npy)
    mock_config.quantization = "f32"
    engine_f32 = DummyEngine(mock_config, dim=128)
    results_f32 = query_indexed_documents("hello", engine_f32, mock_tracker, top_k=10)
    assert len(results_f32) == 4

    # Case B: Engine is configured as qi8 -> scores against both doc1 (.npz) and doc2 (.npy)
    mock_config.quantization = "qi8"
    engine_qi8 = DummyEngine(mock_config, dim=128)
    results_qi8 = query_indexed_documents("hello", engine_qi8, mock_tracker, top_k=10)
    assert len(results_qi8) == 4


def test_query_indexed_documents_corrupted_file_handling(
    mock_tracker, mock_config, tmp_path
):
    """Verify retrieval gracefully catches corrupt or missing embedding files without failing."""
    engine = DummyEngine(mock_config, dim=128)

    # Create a corrupted file
    corrupt_file = str(tmp_path / "corrupt.npy")
    with open(corrupt_file, "wb") as f:
        f.write(b"not a valid numpy file")

    mock_tracker.update_file_status_to_done(
        "h_corrupt", "/watch/corrupt.pdf", 1, corrupt_file
    )
    mock_tracker.update_file_status_to_done(
        "h_missing", "/watch/missing.pdf", 1, "/non/existent/path.npy"
    )

    # Should not raise exception and return empty
    results = query_indexed_documents("test", engine, mock_tracker)
    assert results == []

def test_process_single_pdf_file_ingestion_not_found(mock_tracker, mock_config):
    """Verify process_single_pdf_file_ingestion returns early if file does not exist."""
    engine = DummyEngine(mock_config)
    process_single_pdf_file_ingestion("/nonexistent/file.pdf", engine, mock_tracker)
    assert len(mock_tracker.fetch_all_records()) == 0


def test_process_single_pdf_file_ingestion_skips_done(
    mock_tracker, mock_config, tmp_path
):
    """Verify process_single_pdf_file_ingestion skips already processed PDFs."""
    dummy_pdf = tmp_path / "already_done.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 mock content")

    from bbq.src.utils.pdf_utils import compute_file_sha256_hash

    pdf_hash = compute_file_sha256_hash(str(dummy_pdf))
    mock_tracker.update_file_status_to_done(
        pdf_hash, str(dummy_pdf), 1, "/fake/emb.npy"
    )

    engine = DummyEngine(mock_config)
    with patch("bbq.src.server.ingestion.get_pdf_total_pages") as mock_pages:
        process_single_pdf_file_ingestion(str(dummy_pdf), engine, mock_tracker)
        mock_pages.assert_not_called()


@patch("bbq.src.server.ingestion.get_pdf_total_pages", return_value=0)
def test_process_single_pdf_file_ingestion_zero_pages(
    mock_pages, mock_tracker, mock_config, tmp_path
):
    """Verify zero-page PDF marks status as failed."""
    dummy_pdf = tmp_path / "zero_pages.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 mock")

    engine = DummyEngine(mock_config)
    process_single_pdf_file_ingestion(str(dummy_pdf), engine, mock_tracker)

    records = mock_tracker.fetch_all_records()
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert "zero pages" in records[0]["error_message"]


@patch("bbq.src.server.ingestion.extract_pdf_page_range_to_pil_images")
@patch("bbq.src.server.ingestion.get_pdf_total_pages", return_value=2)
def test_process_single_pdf_file_ingestion_f32_success(
    mock_pages, mock_extract, mock_tracker, mock_config, tmp_path
):
    """Verify successful ingestion in float32 mode produces .npy embeddings."""
    mock_config.quantization = "f32"
    engine = DummyEngine(mock_config, dim=128)

    mock_extract.return_value = [Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]

    dummy_pdf = tmp_path / "doc.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 test document content")

    process_single_pdf_file_ingestion(str(dummy_pdf), engine, mock_tracker)

    record = mock_tracker.fetch_file_record_by_hash(
        mock_tracker.fetch_all_records()[0]["file_hash"]
    )
    assert record["status"] == "done"
    assert record["num_pages"] == 2
    assert record["embedding_path"].endswith(".npy")
    assert os.path.exists(record["embedding_path"])


@patch("bbq.src.server.ingestion.extract_pdf_page_range_to_pil_images")
@patch("bbq.src.server.ingestion.get_pdf_total_pages", return_value=1)
def test_process_single_pdf_file_ingestion_qi8_success(
    mock_pages, mock_extract, mock_tracker, mock_config, tmp_path
):
    """Verify successful ingestion in qi8 mode produces .npz embeddings."""
    mock_config.quantization = "qi8"
    engine = DummyEngine(mock_config, dim=128)

    mock_extract.return_value = [Image.new("RGB", (10, 10))]
    dummy_pdf = tmp_path / "doc_qi8.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 test qi8 document")

    process_single_pdf_file_ingestion(str(dummy_pdf), engine, mock_tracker)

    record = mock_tracker.fetch_all_records()[0]
    assert record["status"] == "done"
    assert record["embedding_path"].endswith(".npz")
    assert os.path.exists(record["embedding_path"])

    loaded = np.load(record["embedding_path"])
    assert "values" in loaded.files
    assert "scales" in loaded.files


def test_process_single_pdf_file_deletion(mock_tracker, tmp_path):
    """Verify deletion removes file and associated embeddings from tracker and filesystem."""
    emb_file = tmp_path / "test_emb.npy"
    emb_file.write_bytes(b"dummy embedding data")
    pdf_file = str(tmp_path / "sample.pdf")

    mock_tracker.update_file_status_to_done("hash_del", pdf_file, 1, str(emb_file))
    assert os.path.exists(str(emb_file))

    # Existing file deletion
    deleted = process_single_pdf_file_deletion(pdf_file, mock_tracker)
    assert deleted is True
    assert not os.path.exists(str(emb_file))
    assert len(mock_tracker.fetch_all_records()) == 0

    # Non-existent record deletion returns False
    assert process_single_pdf_file_deletion("/unknown.pdf", mock_tracker) is False


def test_sync_and_cleanup_deleted_files(mock_tracker, tmp_path):
    """Verify cleanup purges missing files inside watch folder but preserves existing ones."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)

    # File A exists
    file_a = watch_dir / "existing.pdf"
    file_a.write_bytes(b"pdf data")
    mock_tracker.update_file_status_to_done(
        "ha", str(file_a), 1, str(tmp_path / "a.npy")
    )

    # File B deleted from disk
    file_b = watch_dir / "removed.pdf"
    mock_tracker.update_file_status_to_done(
        "hb", str(file_b), 1, str(tmp_path / "b.npy")
    )

    # File C outside watch folder
    outside_file = tmp_path / "outside.pdf"
    mock_tracker.update_file_status_to_done(
        "hc", str(outside_file), 1, str(tmp_path / "c.npy")
    )

    sync_and_cleanup_deleted_files(str(watch_dir), mock_tracker)

    remaining_hashes = [r["file_hash"] for r in mock_tracker.fetch_all_records()]
    assert "ha" in remaining_hashes
    assert "hb" not in remaining_hashes
    assert "hc" in remaining_hashes


@patch("bbq.src.server.ingestion.process_single_pdf_file_ingestion")
def test_scan_and_ingest_existing_pdf_folder(
    mock_ingest, mock_tracker, mock_config, tmp_path
):
    """Verify scan_and_ingest_existing_pdf_folder processes all .pdf files."""
    watch_dir = tmp_path / "watch_scan"
    watch_dir.mkdir(parents=True, exist_ok=True)
    mock_config.watch_folder_path = str(watch_dir)

    (watch_dir / "file1.pdf").write_bytes(b"pdf1")
    (watch_dir / "file2.PDF").write_bytes(b"pdf2")
    (watch_dir / "file3.txt").write_bytes(b"not a pdf")

    engine = DummyEngine(mock_config)
    scan_and_ingest_existing_pdf_folder(engine, mock_tracker)

    assert mock_ingest.call_count == 2


def test_bbq_server_initialization(mock_config):
    """Verify BBQServer initializes attributes correctly."""
    server = BBQServer(
        config=mock_config, host="127.0.0.1", port=8001, without_logo=True
    )
    assert server.host == "127.0.0.1"
    assert server.port == 8001
    assert server.without_logo is True
    assert server.is_ready() is False
    assert server.engine is None


def test_bbq_server_print_logo(capsys, mock_config):
    """Verify print_logo honors without_logo flag."""
    server_no_logo = BBQServer(config=mock_config, without_logo=True)
    server_no_logo.print_logo()
    assert capsys.readouterr().out == ""

    server_with_logo = BBQServer(config=mock_config, without_logo=False)
    server_with_logo.print_logo()
    out = capsys.readouterr().out
    assert "SERVER" in out


@patch("bbq.src.utils.model_loader.initialize_engine")
def test_bbq_server_load_engine_threaded(mock_init_engine, mock_config):
    """Verify load_engine_threaded initializes engine and sets _is_ready."""
    mock_init_engine.return_value = DummyEngine(mock_config)
    server = BBQServer(config=mock_config, without_logo=True)

    server.load_engine_threaded()
    assert server.is_ready() is True
    assert server.engine is not None


@patch("bbq.src.server.server.start_pdf_folder_watcher")
@patch("bbq.src.server.server.run_http_server_in_thread")
@patch("bbq.src.server.server.scan_and_ingest_existing_pdf_folder")
@patch("bbq.src.utils.model_loader.initialize_engine")
def test_bbq_server_start_and_stop_lifecycle(
    mock_init_engine,
    mock_scan,
    mock_run_http,
    mock_watcher,
    mock_config,
):
    """Verify BBQServer.start runs the full server pipeline and stop() cleans up."""
    mock_init_engine.return_value = DummyEngine(mock_config)
    mock_obs = MagicMock()
    mock_watcher.return_value = (mock_obs, None)
    mock_run_http.return_value = MagicMock()

    server = BBQServer(config=mock_config, without_logo=True)
    with patch.object(server, "_register_signal_handlers"):
        server.start()

    assert server.is_ready() is True
    mock_scan.assert_called_once()
    mock_watcher.assert_called_once()
    mock_run_http.assert_called_once()

    server.stop()
    mock_obs.stop.assert_called_once()


@patch("bbq.src.server.server.BBQServer")
def test_start_document_indexing_server(mock_server_cls, mock_config):
    """Verify start_document_indexing_server instantiates and starts BBQServer."""
    mock_server_instance = MagicMock()
    # Let start() run and stop immediately by raising KeyboardInterrupt on join
    mock_server_instance.observer.join.side_effect = KeyboardInterrupt
    mock_server_cls.return_value = mock_server_instance

    with pytest.raises(SystemExit):
        start_document_indexing_server(config=mock_config, without_logo=True)

    mock_server_instance.start.assert_called_once()
    mock_server_instance.stop.assert_called_once()
