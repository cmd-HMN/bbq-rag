from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from bbq.src.storage.sql import SqlliteDB

logger = logging.getLogger("bbq.server")


def query_indexed_documents(
    query_text: str,
    engine: Any,
    tracker: SqlliteDB,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Encodes query text, loads indexed page embeddings, calculates MaxSim scores
    (supporting float32 and quantized INT8 qi8 modes via maxsimd), and returns top-k matching PDF pages.
    """
    query_text = (query_text or "").strip()
    if not query_text:
        return []

    import numpy as np
    import torch
    from maxsimd import maxsim

    records = tracker.fetch_all_records()
    completed_records = [
        r for r in records if r.get("status") == "done" and r.get("embedding_path")
    ]

    if not completed_records:
        logger.info("No completed PDF embeddings found in tracker database.")
        return []

    effective_top_k = (
        int(top_k)
        if top_k is not None
        else getattr(engine.config, "rag_top_k", 5)
    )

    # Encode query: shape [1, q_len, dim] -> [q_len, dim]
    q_tensor: torch.Tensor = engine.encode_query_text_inputs([query_text])
    q_mat = q_tensor[0].cpu().float().numpy()

    # Check configured quantization mode
    quant_mode = (
        getattr(engine.config, "quantization", None) or "f32"
    ).strip().lower()
    use_quant = quant_mode in ("qi8", "int8")

    q_val = None
    q_scale = None
    if use_quant:
        if q_mat.shape[-1] == 128:
            from bbq.maxsimd.quantization import qi8

            q_val, q_scale = qi8(q_mat, dim=128)
        else:
            logger.warning(
                f"Quantization '{quant_mode}' requested but query embedding dimension is "
                f"{q_mat.shape[-1]} (requires 128). Falling back to unquantized retrieval."
            )
            use_quant = False

    all_page_results: List[Dict[str, Any]] = []

    for record in completed_records:
        emb_path = record.get("embedding_path")
        if not emb_path or not os.path.exists(emb_path):
            continue

        try:
            loaded = np.load(emb_path)
        except Exception as load_err:
            logger.error(
                f"Failed to load embedding file '{emb_path}': {load_err}"
            )
            continue

        # Check if saved file is quantized (.npz with values and scales)
        if isinstance(loaded, np.lib.npyio.NpzFile) or (
            hasattr(loaded, "files") and "values" in loaded.files
        ):
            doc_val = loaded["values"]
            doc_scale = loaded["scales"]
            num_pages = doc_val.shape[0] if doc_val.ndim == 3 else 1

            if not use_quant:
                # Query was unquantized float32, quantize on-the-fly to score against quantized doc
                if q_mat.shape[-1] == 128:
                    from bbq.maxsimd.quantization import qi8

                    curr_q_val, curr_q_scale = qi8(q_mat, dim=128)
                    scores = maxsim(
                        curr_q_val,
                        doc_val,
                        q_scale=curr_q_scale,
                        d_scale=doc_scale,
                    )
                else:
                    logger.error(
                        f"Cannot score quantized document '{emb_path}' with non-128 dim query"
                    )
                    continue
            else:
                scores = maxsim(
                    q_val, doc_val, q_scale=q_scale, d_scale=doc_scale
                )
        else:
            doc_emb = loaded
            if doc_emb.ndim == 2:
                doc_emb = np.expand_dims(doc_emb, axis=0)
            num_pages = doc_emb.shape[0]

            if use_quant and doc_emb.shape[-1] == 128:
                # Document was unquantized float32, quantize on-the-fly to match query
                from bbq.maxsimd.quantization import qi8

                doc_val, doc_scale = qi8(doc_emb, dim=128)
                scores = maxsim(
                    q_val, doc_val, q_scale=q_scale, d_scale=doc_scale
                )
            else:
                scores = maxsim(q_mat, doc_emb)

        scores_list = [float(s) for s in scores]
        filename = os.path.basename(record.get("file_path", ""))
        for page_idx, score in enumerate(scores_list):
            all_page_results.append(
                {
                    "file_path": record["file_path"],
                    "filename": filename,
                    "file_hash": record["file_hash"],
                    "page_number": page_idx + 1,
                    "total_pages": num_pages,
                    "score": score,
                }
            )

    all_page_results.sort(key=lambda x: x["score"], reverse=True)
    return all_page_results[: max(0, effective_top_k)]
