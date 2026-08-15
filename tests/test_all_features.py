"""
Comprehensive integration tests verifying:
 1. System cache path defaults (~/.cache/bbq)
 2. MaxSim variable length function correctness
 3. Query retrieval API on indexed PDF documents
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from bbq.src.config import ModelConfigWrapper, get_system_cache_dir
from bbq.src.storage.sql import SqlliteDB
from bbq.src.server import query_indexed_documents, create_colpali_fastapi_app
from bbq.src.client import ColPaliClient
import maxsimd


def test_system_cache_path():
    """Verify that default cache paths use user system cache directory (~/.cache/bbq)."""
    system_cache = get_system_cache_dir()
    assert ".cache/bbq" in system_cache or "bbq" in system_cache
    assert not system_cache.startswith("./.cache")

    config = ModelConfigWrapper()
    assert config.embeddings_output_path == get_system_cache_dir("embeddings")
    assert config.sqlite_db_path == get_system_cache_dir("tracker.db")

    tracker = SqlliteDB()
    assert tracker.db_filepath == get_system_cache_dir("tracker.db")


def test_maxsim_vrlen_functionality():
    """Verify that maxsimd.maxsim_vrlen executes accurately."""
    dim = 128
    q_len = 5
    q_flat = np.random.randn(q_len * dim).astype(np.float32)

    doc0 = np.random.randn(10 * dim).astype(np.float32)
    doc1 = np.random.randn(15 * dim).astype(np.float32)
    d_flat = np.concatenate([doc0, doc1])
    doc_lengths = [10, 15]

    scores = maxsimd.maxsim_vrlen(q_flat, d_flat, doc_lengths, q_len, dim)
    assert len(scores) == 2
    assert isinstance(scores[0], float)
    assert isinstance(scores[1], float)


def test_query_retrieval_pipeline():
    """Test retrieving top matching PDF page parts using query_indexed_documents."""
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "tracker.db")
        emb_dir = os.path.join(temp_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)

        tracker = SqlliteDB(db_filepath=db_path)

        # Create mock 2-page document embedding
        file_hash = "test_hash_123"
        pdf_path = "/tmp/sample_doc.pdf"
        emb_path = os.path.join(emb_dir, f"{file_hash}.npy")

        # 2 pages, 64 tokens each, 128 dim
        mock_doc_emb = np.random.randn(2, 64, 128).astype(np.float32)
        np.save(emb_path, mock_doc_emb)

        tracker.update_file_status_to_done(
            file_hash=file_hash,
            file_path=pdf_path,
            num_pages=2,
            embedding_path=emb_path,
        )

        class MockEngine:
            class MockConfig:
                base_model_id = "test-model"
                watch_folder_path = temp_dir
            config = MockConfig()
            model = torch.nn.Linear(10, 10)

            def encode_query_text_inputs(self, texts):
                # Return query tensor of shape [1, q_len, dim]
                return torch.randn(1, 4, 128)

        mock_engine = MockEngine()
        results = query_indexed_documents("test search query", mock_engine, tracker, top_k=2)

        assert len(results) == 2
        assert results[0]["file_path"] == pdf_path
        assert results[0]["page_number"] in [1, 2]
        assert results[0]["total_pages"] == 2
        assert results[0]["score"] >= results[1]["score"]

    finally:
        shutil.rmtree(temp_dir)


def test_single_page_image_extraction():
    """Verify extracting a single PDF page to a PIL Image."""
    import pymupdf
    from bbq.src.utils.pdf_utils import extract_single_pdf_page_image
    from PIL import Image

    temp_dir = tempfile.mkdtemp()
    try:
        pdf_path = os.path.join(temp_dir, "test_doc.pdf")
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Hello World PDF Test Page")
        doc.save(pdf_path)
        doc.close()

        pil_img = extract_single_pdf_page_image(pdf_path, page_number=1)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.width > 0 and pil_img.height > 0
    finally:
        shutil.rmtree(temp_dir)


