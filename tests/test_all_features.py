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
from bbq.src.server import query_indexed_documents, create_bbq_fastapi_app
from bbq.src.client import BBQClient
import maxsimd


def test_system_cache_path():
    """Verify that default cache paths use user system cache directory (~/.cache/bbq)."""
    system_cache = get_system_cache_dir()
    assert ".cache/bbq" in system_cache or "bbq" in system_cache
    assert not system_cache.startswith("./.cache")

    config = ModelConfigWrapper()
    assert config.embeddings_output_path == get_system_cache_dir("embeddings")
    assert config.sqlite_db_path == get_system_cache_dir("tracker.db")
    assert config.gemini_model == "gemini-1.5-flash"
    assert config.rag_top_k == 3

    tracker = SqlliteDB()
    assert tracker.db_filepath == get_system_cache_dir("tracker.db")


def test_config_gemini_settings_and_client():
    """Verify Gemini configuration in ModelConfigWrapper and client initialization."""
    from bbq.src.config import load_configuration_from_yaml_file

    config = ModelConfigWrapper(
        gemini_api_key="TEST_API_KEY_ABC",
        gemini_model="gemini-2.0-flash",
        rag_top_k=3,
    )
    assert config.gemini_api_key == "TEST_API_KEY_ABC"
    assert config.gemini_model == "gemini-2.0-flash"
    assert config.rag_top_k == 3

    dict_format = config.convert_to_dictionary_format()
    assert dict_format["gemini_api_key_configured"] is True
    assert dict_format["gemini_model"] == "gemini-2.0-flash"
    assert dict_format["rag_top_k"] == 3

    gemini_client = config.get_gemini_client()
    assert gemini_client.api_key == "TEST_API_KEY_ABC"
    assert gemini_client.model == "gemini-2.0-flash"
    assert gemini_client.is_available() is True



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


def test_maxsim_zero_copy_and_pointer():
    """Verify maxsim 2D, 3D, and raw pointer functions."""
    dim = 128
    q_len = 8
    q_mat = np.random.randn(q_len, dim).astype(np.float32)
    d_2d = np.random.randn(20, dim).astype(np.float32)
    d_3d = np.random.randn(3, 20, dim).astype(np.float32)

    # 2D and 3D direct calls
    s_2d = maxsimd.maxsim(q_mat, d_2d)
    s_3d = maxsimd.maxsim(q_mat, d_3d)
    assert len(s_2d) == 1
    assert len(s_3d) == 3

    # Pointer calls
    s_ptr = maxsimd.maxsim_ptr(q_mat.ctypes.data, d_2d.ctypes.data, q_len, 20, dim)
    s_3d_ptr = maxsimd.maxsim_3d_ptr(q_mat.ctypes.data, d_3d.ctypes.data, q_len, 3, 20, dim)
    assert np.isclose(s_2d[0], s_ptr)
    assert np.allclose(s_3d, s_3d_ptr)


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


def test_gemini_client_and_fallback():
    """Verify GeminiClient behavior with no key, invalid key, and image encoding."""
    from bbq.src.client.gemini import GeminiClient
    from PIL import Image

    # 1. No key
    client_no_key = GeminiClient(api_key="")
    assert not client_no_key.is_available()
    img = Image.new("RGB", (64, 64), color="red")
    answer = client_no_key.generate_answer("What is this?", [img])
    assert answer is None

    # 2. Image encoding part
    part = GeminiClient._image_to_base64_part(img)
    assert "inlineData" in part
    assert part["inlineData"]["mimeType"] == "image/jpeg"
    assert len(part["inlineData"]["data"]) > 0

    # 3. Invalid key should gracefully return None without raising unhandled exceptions
    client_invalid = GeminiClient(api_key="INVALID_TEST_KEY_12345", timeout=5)
    assert client_invalid.is_available()
    answer_invalid = client_invalid.generate_answer("Test question", [img])
    assert answer_invalid is None



