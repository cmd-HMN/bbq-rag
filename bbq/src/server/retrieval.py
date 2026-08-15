import logging
import os
from typing import Any, Dict, List

from bbq.src.storage.sql import SqlliteDB

logger = logging.getLogger("bbq.server")


def query_indexed_documents(
    query_text: str,
    engine: Any,
    tracker: SqlliteDB,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Encodes the query text, loads indexed PDF page embeddings, calculates MaxSim scores,
    and returns top-k matching PDF pages.
    """
    if not query_text.strip():
        return []

    import numpy as np
    import torch
    from maxsimd import maxsim_vrlen

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
                all_page_results.append(
                    {
                        "score": float(score),
                        "file_path": record["file_path"],
                        "filename": os.path.basename(record["file_path"]),
                        "file_hash": record["file_hash"],
                        "page_number": page_idx + 1,
                        "total_pages": num_pages,
                    }
                )
        elif doc_emb.ndim == 2:
            tokens = doc_emb.shape[0]
            d_flat_1d = np.ascontiguousarray(doc_emb.reshape(-1), dtype=np.float32)
            scores = maxsim_vrlen(q_flat_1d, d_flat_1d, [tokens], q_len, dim)
            all_page_results.append(
                {
                    "score": float(scores[0]),
                    "file_path": record["file_path"],
                    "filename": os.path.basename(record["file_path"]),
                    "file_hash": record["file_hash"],
                    "page_number": 1,
                    "total_pages": 1,
                }
            )

    all_page_results.sort(key=lambda x: x["score"], reverse=True)
    return all_page_results[:top_k]
