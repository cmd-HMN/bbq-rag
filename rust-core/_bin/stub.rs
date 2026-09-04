use pyo3_stub_gen::Result;
use std::path::PathBuf;
use std::process::Command;

const INIT_PY_CONTENT: &str = r#"from typing import Union, Optional, Sequence, Any
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from .maxsimd import (
    maxsim as _raw_maxsim,
    __version__,
)

__all__ = [
    "maxsim",
    "__version__",
]

def _extract_ptr_shape_dtype(obj):
    if obj is None:
        return 0, None, None
    if isinstance(obj, int):
        return obj, None, None

    if _HAS_TORCH and isinstance(obj, torch.Tensor):
        if not obj.is_contiguous():
            obj = obj.contiguous()
        dt = 0 if obj.dtype == torch.float32 else (1 if obj.dtype == torch.int8 else -1)
        return obj.data_ptr(), obj.shape, dt

    if isinstance(obj, np.ndarray):
        if not obj.flags["C_CONTIGUOUS"]:
            obj = np.ascontiguousarray(obj)
        dt = 0 if obj.dtype == np.float32 else (1 if obj.dtype == np.int8 else -1)
        return obj.ctypes.data, obj.shape, dt

    if hasattr(obj, "data_ptr"):
        return obj.data_ptr(), getattr(obj, "shape", None), 0

    if hasattr(obj, "ctypes"):
        return obj.ctypes.data, getattr(obj, "shape", None), 0

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
                if doc_tokens == 0 and d_shape is not None:
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

fn main() -> Result<()> {
    let stub = maxsimd::stub_info()?;
    stub.generate()?;

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let init_py = manifest.join("bbq/maxsimd/__init__.py");
    let init_pyi = manifest.join("bbq/maxsimd/__init__.pyi");

    if let Some(parent) = init_py.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&init_py, INIT_PY_CONTENT)?;
    println!("Generated `maxsimd` stub and python wrapper in __init__.py");

    if init_pyi.exists() {
        let mut pyi_content = std::fs::read_to_string(&init_pyi)?;
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
        ])
        .status();

    Ok(())
}
