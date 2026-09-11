from typing import Any, Dict, List, Optional
import numpy as np

from benchmarks.vidore.score import (
    latency_stats,
    mrr_at_k,
    ndcg_at_k,
    parity,
    recall_at_k,
)


class Evaluator:
    """
    Evaluates engine scoring matrices against ground-truth targets.
    Returns pure results without printing.
    """

    def __init__(self, queries: List[Dict[str, Any]], doc_ids: List[str]):
        self.queries = queries
        self.doc_ids = doc_ids
        self.num_queries = len(queries)
        self.num_docs = len(doc_ids)

        # Map each query's target doc_id to its index in doc_ids
        self.doc2idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        self.targets = [self.doc2idx[q["doc_id"]] for q in queries]

    def evaluate(
        self,
        scores_matrix: np.ndarray,
        latencies_sec: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Takes a 2D score matrix of shape (num_queries, num_docs) and
        per-query latencies, returning pure metrics without printing.
        """
        all_rankings: List[List[int]] = []
        r1_list: List[float] = []
        r5_list: List[float] = []
        ndcg5_list: List[float] = []
        mrr10_list: List[float] = []

        for i in range(self.num_queries):
            tidx = self.targets[i]
            # Rank document indices descending by score
            ridx = np.argsort(scores_matrix[i])[::-1].tolist()
            all_rankings.append(ridx)

            r1_list.append(recall_at_k(ridx, tidx, k=1))
            r5_list.append(recall_at_k(ridx, tidx, k=5))
            ndcg5_list.append(ndcg_at_k(ridx, tidx, k=5))
            mrr10_list.append(mrr_at_k(ridx, tidx, k=10))

        lat_metrics = latency_stats(latencies_sec or [])

        return {
            "num_queries": self.num_queries,
            "num_docs": self.num_docs,
            "recall_1": float(np.mean(r1_list)) * 100.0 if r1_list else 0.0,
            "recall_5": float(np.mean(r5_list)) * 100.0 if r5_list else 0.0,
            "ndcg_5": float(np.mean(ndcg5_list)) * 100.0 if ndcg5_list else 0.0,
            "mrr_10": float(np.mean(mrr10_list)) * 100.0 if mrr10_list else 0.0,
            "latency": lat_metrics,
            "rankings": all_rankings,
        }

    def parity_with(
        self,
        rankings_a: List[List[int]],
        rankings_b: List[List[int]],
    ) -> float:
        """Calculates exact Top-1 parity between two ranking lists."""
        return parity(rankings_a, rankings_b)
