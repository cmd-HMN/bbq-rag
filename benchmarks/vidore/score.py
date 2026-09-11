import math
from typing import Dict, List
import numpy as np


def recall_at_k(ridx: List[int], tidx: int, k: int = 5) -> float:
    return 1.0 if tidx in ridx[:k] else 0.0


def ndcg_at_k(ridx: List[int], tidx: int, k: int = 5) -> float:
    try:
        rank = ridx[:k].index(tidx)
        return 1.0 / math.log2(rank + 2)
    except ValueError:
        return 0.0


def mrr_at_k(ridx: List[int], tidx: int, k: int = 10) -> float:
    try:
        rank = ridx[:k].index(tidx)
        return 1.0 / (rank + 1)
    except ValueError:
        return 0.0


def latency_stats(latencies_sec: List[float]) -> Dict[str, float]:
    if not latencies_sec:
        return {
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "qps": 0.0,
        }

    ms = np.array(latencies_sec) * 1000.0
    total_time = sum(latencies_sec)
    qps = len(latencies_sec) / total_time if total_time > 0 else 0.0

    return {
        "mean_ms": float(np.mean(ms)),
        "std_ms": float(np.std(ms)),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "qps": float(qps),
    }


def parity(rank_a: List[List[int]], rank_b: List[List[int]]) -> float:
    if not rank_a or not rank_b or len(rank_a) != len(rank_b):
        return 0.0
    matches = sum(1 for ra, rb in zip(rank_a, rank_b) if ra[0] == rb[0])
    return (matches / len(rank_a)) * 100.0
