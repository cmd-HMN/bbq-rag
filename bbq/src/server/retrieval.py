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
    from maxsimd import maxsim

    records = tracker.fetch_all_records()
    completed_records = [r for r in records if r.get("status") == "done" and r.get("embedding_path")]

    if not completed_records:
        logger.info("No completed PDF embeddings found in tracker database.")
        return []

    # Encode query: shape [1, q_len, dim] -> [q_len, dim]
    q_tensor: torch.Tensor = engine.encode_query_text_inputs([query_text])
    q_mat = q_tensor[0].cpu().float().numpy()

    page_metadata = []
    page_embeddings_list = []

    for record in completed_records:
        emb_path = record["embedding_path"]
        if not os.path.exists(emb_path):
            continue

        doc_emb = np.load(emb_path)
        if doc_emb.ndim == 2:
            doc_emb = np.expand_dims(doc_emb, axis=0)

        num_pages = doc_emb.shape[0]
        for page_idx in range(num_pages):
            page_metadata.append(
                {
                    "file_path": record["file_path"],
                    "filename": os.path.basename(record["file_path"]),
                    "file_hash": record["file_hash"],
                    "page_number": page_idx + 1,
                    "total_pages": num_pages,
                }
            )
        page_embeddings_list.append(doc_emb)

    if not page_embeddings_list:
        return []

    # Batch all document pages into a single contiguous 3D array: (total_pages, tokens_per_page, dim)
    batched_docs_3d = (
        page_embeddings_list[0]
        if len(page_embeddings_list) == 1
        else np.concatenate(page_embeddings_list, axis=0)
    )

    # Multi-core Rayon parallelized MaxSim scoring across all document pages
    scores = maxsim(q_mat, batched_docs_3d)

    all_page_results = []
    for meta, score in zip(page_metadata, scores):
        meta_dict = meta.copy()
        meta_dict["score"] = float(score)
        all_page_results.append(meta_dict)

    all_page_results.sort(key=lambda x: x["score"], reverse=True)
    return all_page_results[:top_k]
