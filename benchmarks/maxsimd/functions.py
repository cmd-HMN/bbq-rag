from benchmarks.common import np, torch, maxsimd, maxsim_cpu
from typing import List

def bm_maxsim(
    q_tensor: torch.Tensor,
    docs_3d_tensor: torch.Tensor,
    jobs: int = -1,
) -> List[float]:
    """Maxim function with f32 only"""

    return maxsimd.maxsim(
        q_tensor,
        docs_3d_tensor,
        jobs=jobs,
    )

# use 32 dim as it doesn't support variable dim
def bm_maxsim_cpu(
    q_mat: np.ndarray,
    docs_3d_mat: np.ndarray,
) -> List[float]:
    """Official maxsim-cpu PyPI library function maxsim_scores"""

    if q_mat.shape[0] <= 32:
        scores = maxsim_cpu.maxsim_scores(q_mat, docs_3d_mat)  # type: ignore
    else:
        scores = np.zeros(docs_3d_mat.shape[0], dtype=np.float32)
        for i in range(0, q_mat.shape[0], 32):
            q_chunk = np.ascontiguousarray(q_mat[i : i + 32])
            scores += maxsim_cpu.maxsim_scores(q_chunk, docs_3d_mat)  # type: ignore

    return scores.tolist() if isinstance(scores, np.ndarray) else list(scores)


# this one is used in the colpali, not the same as this one
def bm_torch_einsum(
    q_tensor: torch.Tensor,
    docs_3d_tensor: torch.Tensor,
) -> List[float]:
    """PyTorch native batched 3D einsum for uniform pages"""

    with torch.no_grad():
        sim_matrix = torch.einsum("qd, btd -> bqt", q_tensor, docs_3d_tensor)
        doc_scores = torch.sum(torch.max(sim_matrix, dim=-1).values, dim=-1)
        return doc_scores.cpu().tolist()


def bm_torch_simple(
    q_tensor: torch.Tensor,
    docs_3d_tensor: torch.Tensor,
) -> List[float]:
    """PyTorch simple loop."""
   
    scores = []
    with torch.no_grad():
        for i in range(docs_3d_tensor.shape[0]):
            doc = docs_3d_tensor[i]
            sim_matrix = torch.matmul(q_tensor, doc.T)
            doc_score = torch.sum(torch.max(sim_matrix, dim=1).values).item()
            scores.append(doc_score)
    return scores


def bm_numpy(
    q_mat: np.ndarray,
    docs_3d_mat: np.ndarray,
) -> List[float]:
    """NumPyy"""
 
    scores = []
    for i in range(docs_3d_mat.shape[0]):
        sim_matrix = np.dot(q_mat, docs_3d_mat[i].T)
        doc_score = np.sum(np.max(sim_matrix, axis=1))
        scores.append(float(doc_score))
    return scores
