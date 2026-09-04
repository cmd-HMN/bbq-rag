from typing import Union, Optional, Tuple, Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .maxsimd import qi8 as _raw_qi8

__all__ = ["qi8"]


def qi8(
    tensor_or_array: Any,
    dim: Optional[int] = None,
    jobs: int = -1,
) -> Tuple[Any, Any]:
    """
    Quantize Float32 embeddings to Int8 with per-block (32-dim) scaling factors.

    Supports PyTorch Tensors, NumPy NDArrays, or raw pointers.
    Zero-copy: writes directly into pre-allocated memory of identical library type.

    Args:
        tensor_or_array: Input f32 tensor/array with shape (..., dim).
        dim: Embedding dimension (inferred from last dimension if None, default 128).
        jobs: Concurrency (-1 for all cores, 1 for sequential, N for custom pool).

    Returns:
        (values, scale):
            - values: Int8 tensor/array of identical shape (..., dim).
            - scale: Float32 tensor/array of shape (..., dim // 32).
    """
    if tensor_or_array is None:
        raise ValueError("Input buffer cannot be None")

    if torch is not None and isinstance(tensor_or_array, torch.Tensor):
        if not tensor_or_array.is_contiguous():
            tensor_or_array = tensor_or_array.contiguous()
        if tensor_or_array.dtype != torch.float32:
            raise ValueError(
                f"Expected torch.float32 tensor, got {tensor_or_array.dtype}"
            )

        shape = tuple(tensor_or_array.shape)
        if dim is None:
            if len(shape) == 0:
                raise ValueError("Cannot quantize a scalar tensor")
            dim = shape[-1]

        if dim != 128:
            raise ValueError(f"Currently qi8 only supports dim=128 vectors, got {dim}")

        num_blocks = dim // 32
        total_elements = tensor_or_array.numel()
        tokens = total_elements // dim
        if len(shape) == 1 and tokens > 1:
            scale_shape = (tokens, num_blocks)
        else:
            scale_shape = shape[:-1] + (num_blocks,)

        out_data = torch.empty(shape, dtype=torch.int8, device=tensor_or_array.device)
        out_scales = torch.empty(
            scale_shape, dtype=torch.float32, device=tensor_or_array.device
        )

        _raw_qi8(
            tensor_or_array.data_ptr(),
            tokens,
            dim=dim,
            out_ptr=out_data.data_ptr(),
            scale_ptr=out_scales.data_ptr(),
            jobs=jobs,
        )
        return out_data, out_scales

    if isinstance(tensor_or_array, np.ndarray):
        if (
            not tensor_or_array.flags["C_CONTIGUOUS"]
            or tensor_or_array.dtype != np.float32
        ):
            tensor_or_array = np.ascontiguousarray(tensor_or_array, dtype=np.float32)

        shape = tuple(tensor_or_array.shape)
        if dim is None:
            if len(shape) == 0:
                raise ValueError("Cannot quantize a scalar array")
            dim = shape[-1]

        if dim != 128:
            raise ValueError(f"Currently qi8 only supports dim=128 vectors, got {dim}")

        num_blocks = dim // 32
        total_elements = tensor_or_array.size
        tokens = total_elements // dim
        if len(shape) == 1 and tokens > 1:
            scale_shape = (tokens, num_blocks)
        else:
            scale_shape = shape[:-1] + (num_blocks,)

        out_data = np.empty(shape, dtype=np.int8)
        out_scales = np.empty(scale_shape, dtype=np.float32)

        _raw_qi8(
            tensor_or_array.ctypes.data,
            tokens,
            dim=dim,
            out_ptr=out_data.ctypes.data,
            scale_ptr=out_scales.ctypes.data,
            jobs=jobs,
        )
        return out_data, out_scales

    if isinstance(tensor_or_array, int):
        if dim is None:
            dim = 128
        return _raw_qi8(tensor_or_array, 1, dim=dim, out_ptr=0, scale_ptr=0, jobs=jobs)

    raise TypeError(
        f"Unsupported input type: {type(tensor_or_array)}. Expected torch.Tensor or np.ndarray."
    )
