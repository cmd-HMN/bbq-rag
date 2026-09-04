use pyo3_stub_gen::Result;
use std::path::PathBuf;
use std::process::Command;

const INIT_PY_CONTENT: &str = r#"from typing import Union, Optional, Sequence, Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .maxsimd import (
    maxsim as _raw_maxsim,
    __version__,
)
from . import quantization

__all__ = [
    "maxsim",
    "quantization",
    "__version__",
]

def _extract_ptr_shape_dtype(obj):
    if obj is None:
        return 0, None, None
    if isinstance(obj, int):
        return obj, None, None

    if torch is not None and isinstance(obj, torch.Tensor):
        if not obj.is_contiguous():
            obj = obj.contiguous()
        dt = 0 if obj.dtype == torch.float32 else (1 if obj.dtype == torch.int8 else -1)
        return obj.data_ptr(), tuple(obj.shape), dt

    if isinstance(obj, np.ndarray):
        if not obj.flags["C_CONTIGUOUS"]:
            obj = np.ascontiguousarray(obj)
        dt = 0 if obj.dtype == np.float32 else (1 if obj.dtype == np.int8 else -1)
        return obj.ctypes.data, tuple(obj.shape), dt

    if hasattr(obj, "data_ptr"):
        s = getattr(obj, "shape", None)
        return obj.data_ptr(), tuple(s) if s is not None else None, 0

    if hasattr(obj, "ctypes"):
        s = getattr(obj, "shape", None)
        return obj.ctypes.data, tuple(s) if s is not None else None, 0

    raise TypeError(f"Unsupported buffer type: {type(obj)}. Expected torch.Tensor, np.ndarray, or raw pointer int.")


def maxsim(
    q: Any,
    d: Any,
    doc_lengths: Optional[Sequence[int]] = None,
    q_scale: Optional[Any] = None,
    d_scale: Optional[Any] = None,
    q_len: Optional[int] = None,
    dim: Optional[int] = None,
    jobs: int = 1,
    dtype: Optional[int] = None,
    **kwargs: Any,
) -> list[float]:
    """
    Unified MaxSim similarity engine with zero-copy pointer extraction.
    
    Accepts PyTorch Tensors, NumPy NDArrays, or raw memory pointers.
    Auto-detects document layout:
      - 2D (Single Doc): (q_len, dim) vs (d_len, dim)
      - 3D (Uniform Batch): (q_len, dim) vs (batch_docs, tokens, dim)
      - Ragged (Flat Buffer): (q_len, dim) vs (total_tokens, dim) with doc_lengths
      
    Args:
        q: Query tensor/array or memory pointer.
        d: Document(s) tensor/array or memory pointer.
        doc_lengths: Optional document token lengths for flat ragged layout.
        q_scale: Optional block quantization scale for query (INT8).
        d_scale: Optional block quantization scale for docs (INT8).
        q_len: Optional query token length (auto-inferred from tensor shape).
        dim: Optional embedding dimension (auto-inferred from tensor shape).
        jobs: Concurrency (1 = sequential caller thread, -1 = all CPU cores, N = custom pool).
        dtype: 0 for Float32, 1 for Int8 (auto-inferred from tensor dtype).
    """
    q_ptr, q_shape, q_dt = _extract_ptr_shape_dtype(q)
    d_ptr, d_shape, d_dt = _extract_ptr_shape_dtype(d)

    q_scale_ptr = _extract_ptr_shape_dtype(q_scale)[0] if q_scale is not None else 0
    d_scale_ptr = _extract_ptr_shape_dtype(d_scale)[0] if d_scale is not None else 0

    if q_len is None and q_shape is not None:
        q_len = q_shape[0]
    if dim is None:
        if q_shape is not None and len(q_shape) >= 2:
            dim = q_shape[1]
        elif d_shape is not None and len(d_shape) >= 2:
            dim = d_shape[-1]

    if q_len is None or dim is None:
        raise ValueError("Could not infer q_len or dim; please pass them explicitly.")

    if dtype is None:
        dtype = q_dt if q_dt is not None and q_dt != -1 else 0

    if doc_lengths is not None:
        return _raw_maxsim(
            q_ptr,
            d_ptr,
            q_len,
            dim,
            layout_type=2,
            doc_lengths=list(doc_lengths),
            q_scale_ptr=q_scale_ptr,
            d_scale_ptr=d_scale_ptr,
            dtype=dtype,
            jobs=jobs,
        )
    elif d_shape is not None and len(d_shape) == 3:
        return _raw_maxsim(
            q_ptr,
            d_ptr,
            q_len,
            dim,
            layout_type=1,
            batch_docs=d_shape[0],
            batch_tokens=d_shape[1],
            q_scale_ptr=q_scale_ptr,
            d_scale_ptr=d_scale_ptr,
            dtype=dtype,
            jobs=jobs,
        )
    else:
        batch_docs = kwargs.get("batch_docs", 0)
        batch_tokens = kwargs.get("batch_tokens", 0)
        doc_tokens = kwargs.get("doc_tokens", 0)
        layout_type = kwargs.get("layout_type", None)

        if layout_type is None:
            if batch_docs > 0 and batch_tokens > 0:
                layout_type = 1
            else:
                layout_type = 0
                if doc_tokens == 0 and d_shape is not None and len(d_shape) > 0:
                    doc_tokens = d_shape[0]

        return _raw_maxsim(
            q_ptr,
            d_ptr,
            q_len,
            dim,
            layout_type=layout_type,
            doc_tokens=doc_tokens,
            batch_docs=batch_docs,
            batch_tokens=batch_tokens,
            q_scale_ptr=q_scale_ptr,
            d_scale_ptr=d_scale_ptr,
            dtype=dtype,
            jobs=jobs,
        )
"#;

const QUANTIZATION_PY_CONTENT: &str = r#"from typing import Union, Optional, Tuple, Any
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
            raise ValueError(f"Expected torch.float32 tensor, got {tensor_or_array.dtype}")

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
        out_scales = torch.empty(scale_shape, dtype=torch.float32, device=tensor_or_array.device)

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
        if not tensor_or_array.flags["C_CONTIGUOUS"] or tensor_or_array.dtype != np.float32:
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

    raise TypeError(f"Unsupported input type: {type(tensor_or_array)}. Expected torch.Tensor or np.ndarray.")
"#;

const QUANTIZATION_PYI_CONTENT: &str = r#"import typing

def qi8(
    tensor_or_array: typing.Any,
    dim: typing.Optional[int] = None,
    jobs: int = -1,
) -> typing.Tuple[typing.Any, typing.Any]: ...
"#;

fn main() -> Result<()> {
    let stub = maxsimd::stub_info()?;
    stub.generate()?;

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let init_py = manifest.join("bbq/maxsimd/__init__.py");
    let init_pyi = manifest.join("bbq/maxsimd/__init__.pyi");
    let quant_py = manifest.join("bbq/maxsimd/quantization.py");
    let quant_pyi = manifest.join("bbq/maxsimd/quantization.pyi");

    if let Some(parent) = init_py.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&init_py, INIT_PY_CONTENT)?;
    std::fs::write(&quant_py, QUANTIZATION_PY_CONTENT)?;
    std::fs::write(&quant_pyi, QUANTIZATION_PYI_CONTENT)?;
    println!("Generated `maxsimd` stubs and python wrappers");

    if init_pyi.exists() {
        let mut pyi_content = std::fs::read_to_string(&init_pyi)?;
        if !pyi_content.contains("from . import quantization") {
            pyi_content = format!("from . import quantization\n{}", pyi_content);
        }
        if !pyi_content.contains("q: typing.Any") {
            pyi_content.push_str("\n@typing.overload\ndef maxsim(\n    q: typing.Any,\n    d: typing.Any,\n    doc_lengths: typing.Optional[typing.Sequence[builtins.int]] = None,\n    q_scale: typing.Optional[typing.Any] = None,\n    d_scale: typing.Optional[typing.Any] = None,\n    q_len: typing.Optional[builtins.int] = None,\n    dim: typing.Optional[builtins.int] = None,\n    jobs: builtins.int = 1,\n    dtype: typing.Optional[builtins.int] = None,\n    **kwargs: typing.Any,\n) -> builtins.list[builtins.float]: ...\n");
        }
        if !pyi_content.contains("__version__") {
            pyi_content = format!(
                "{}\n__version__ = '{}'\n",
                pyi_content,
                env!("CARGO_PKG_VERSION")
            );
        }
        std::fs::write(&init_pyi, pyi_content)?;
    }

    let _ = Command::new("ruff")
        .args([
            "format",
            init_py.to_str().unwrap(),
            init_pyi.to_str().unwrap(),
            quant_py.to_str().unwrap(),
            quant_pyi.to_str().unwrap(),
        ])
        .status();

    Ok(())
}
