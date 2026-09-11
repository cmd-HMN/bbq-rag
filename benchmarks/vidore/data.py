"""
Handles data
"""

import json
from pathlib import Path
from typing import Dict, Optional

from benchmarks.vidore.img import Img


class VidoreData:
    def __init__(self, data: str = "data/vidore", cat: str = "assets/bench.json"):
        self.dpath = Path(data)
        self.cpath = Path(cat)
        self.catalog = self._load_cat()

    def __call__(self, key: Optional[str] = None, force: bool = False):
        if key is not None:
            return self.fetch(key, force=force)
        return self.fetch_all(force=force)

    def _load_cat(self) -> Dict:
        if not self.cpath.exists():
            raise FileNotFoundError(f"File not found: {self.cpath}")

        with open(self.cpath) as f:
            return json.load(f)

    def fetch(self, key: str, force: bool = False):
        if key not in self.catalog:
            raise KeyError(
                f"Dataset key '{key}' not found in catalog. Available: {list(self.catalog.keys())}"
            )

        info = self.catalog[key]
        hf_id = info["hf_id"]
        target = self.dpath / key

        if not force and target.exists():
            return target

        from datasets import load_dataset

        dataset = load_dataset(hf_id, split="test")
        dataset.save_to_disk(str(target))
        return target

    def fetch_all(self, force: bool = False):
        results: Dict[str, Path] = {}
        for key in self.catalog:
            results[key] = self.fetch(key, force=force)
        return results

    def get(self, key: Optional[str] = None, split: bool = True):
        if key is None:
            return {k: self.get(k, split=split) for k in self.catalog}

        if key not in self.catalog:
            raise KeyError(
                f"Dataset key '{key}' not found in catalog. Available: {list(self.catalog.keys())}"
            )

        target = self.dpath / key

        if not target.exists():
            raise FileNotFoundError(
                f"Dataset {key} not found. Use `fetch` to download it first."
            )

        from datasets import load_from_disk

        dataset = load_from_disk(str(target))

        if not split:
            return dataset

        filenames = dataset["image_filename"]
        query_texts = dataset["query"]

        doc_indices = {}
        queries = []

        for idx, (doc_id, query) in enumerate(zip(filenames, query_texts)):
            if doc_id not in doc_indices:
                doc_indices[doc_id] = idx

            queries.append(
                {
                    "id": idx,
                    "query": query,
                    "doc_id": doc_id,
                }
            )

        return {
            "queries": queries,
            "images": Img(dataset, doc_indices),
        }
