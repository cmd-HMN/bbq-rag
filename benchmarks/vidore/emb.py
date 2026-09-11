import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional
from benchmarks.common import torch, np


class Emb:
    def __init__(self):
        self.engine = None

    def _get_engine(self):
        # this to remove the warnings
        if self.engine is None:
            old_stderr = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull, 2)
                from bbq.src.utils.model_loader import (
                    initialize_engine_from_yaml_config,
                )

                self.engine = initialize_engine_from_yaml_config("config.yaml")
            finally:
                os.dup2(old_stderr, 2)
                os.close(devnull)
                os.close(old_stderr)
        return self.engine

    def __call__(
        self,
        data_pack: Any,
        key: str,
        odir: str = "data/embeddings",
        force: bool = False,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        save_dir = Path(odir) / key
        save_dir.mkdir(parents=True, exist_ok=True)

        corpus_file = save_dir / "corpus.npy"
        queries_file = save_dir / "queries.npy"

        if not force and corpus_file.exists() and queries_file.exists():
            return {
                "corpus_emb": np.load(corpus_file),
                "queries_emb": np.load(queries_file),
                "queries": data_pack["queries"],
                "doc_ids": list(data_pack["images"].keys()),
            }

        valid_queries = [q for q in data_pack["queries"] if q.get("query")]
        query_texts = [q["query"] for q in valid_queries]

        queries_emb = self._emb_txt(query_texts, bsize=16, status_cb=status_cb)
        np.save(queries_file, queries_emb)

        corpus_emb = self._emb_img(data_pack["images"], bsize=4, status_cb=status_cb)
        np.save(corpus_file, corpus_emb)

        return {
            "corpus_emb": corpus_emb,
            "queries_emb": queries_emb,
            "queries": valid_queries,
            "doc_ids": list(data_pack["images"].keys()),
        }

    def _emb_txt(
        self,
        data,
        bsize: int = 16,
        status_cb: Optional[Callable[[str], None]] = None,
    ):
        engine = self._get_engine()
        res = []
        total = len(data)
        last_update = 0.0

        for i in range(0, total, bsize):
            now = time.perf_counter()
            if status_cb and (now - last_update >= 0.4 or i == 0 or i + bsize >= total):
                status_cb(f"Embedding queries ({min(i + bsize, total)}/{total})")
                last_update = now

            batch = data[i : i + bsize]
            with torch.inference_mode():
                res.append(engine.encode_query_text_inputs(batch).cpu().float().numpy())

        if not res:
            return np.empty((0, 0, 128))

        max_len = max(arr.shape[1] for arr in res)
        padded = [
            np.pad(arr, ((0, 0), (0, max_len - arr.shape[1]), (0, 0))) for arr in res
        ]
        return np.concatenate(padded, axis=0)

    def _emb_img(
        self,
        data,
        bsize: int = 4,
        status_cb: Optional[Callable[[str], None]] = None,
    ):
        engine = self._get_engine()
        total = len(data)
        res = []
        keys = list(data.keys()) if hasattr(data, "keys") else list(range(total))
        last_update = 0.0

        for i in range(0, total, bsize):
            now = time.perf_counter()
            if status_cb and (now - last_update >= 0.4 or i == 0 or i + bsize >= total):
                status_cb(f"Embedding documents ({min(i + bsize, total)}/{total})")
                last_update = now

            bkeys = keys[i : i + bsize]
            bimg = [data[k] for k in bkeys]

            with torch.inference_mode():
                res.append(
                    engine.encode_multimodal_document_images(bimg, batch_size=bsize)
                    .cpu()
                    .float()
                    .numpy()
                )

        if not res:
            return np.empty((0, 0, 128))

        max_len = max(arr.shape[1] for arr in res)
        padded = [
            np.pad(arr, ((0, 0), (0, max_len - arr.shape[1]), (0, 0))) for arr in res
        ]
        return np.concatenate(padded, axis=0)
