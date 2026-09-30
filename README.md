<div align="center">

# <img src="assets/logo.png" alt="BBQ-RAG Logo" width="160" style="vertical-align: middle; margin-right: 12px;"/> BBQ-RAG

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-0284c7)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-ea580c.svg)](https://www.rust-lang.org/)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%2B%20FMA-16a34a.svg)](#performance--benchmarks)
[![VLM](https://img.shields.io/badge/VLM-ColPali%20%2F%20colSmol-7c3aed.svg)](#quickstart--usage)
[![Scoring](https://img.shields.io/badge/Scoring-MaxSim%20Multi--Vector-0ea5e9.svg)](#performance--benchmarks)
[![FFI](https://img.shields.io/badge/FFI-PyO3%20Zero--Copy-e11d48.svg)](https://pyo3.rs/)
[![Throughput](https://img.shields.io/badge/throughput->%20150k%20pages/s-f97316.svg)](#performance--benchmarks)
[![Multimodal RAG](https://img.shields.io/badge/LLM-Google%20Gemini-4285f4.svg)](#quickstart--usage)
[![About](https://img.shields.io/badge/Lineage-ColPali%20%26%20maxsim--cpu-f59e0b.svg)](ABOUT.md)

</div>

BBQ-RAG is an ultra-fast, lightweight visual document retrieval and late-interaction (ColPali / MaxSim) search engine. By combining Vision-Language Model embeddings (ColPali, colSmol, SmolVLM) with an in-register fused AVX2/FMA SIMD compute engine in Rust (`maxsimd`), BBQ-RAG scores over **150,000 document pages per second** on standard multi-core CPUs with zero intermediate heap allocations.

It provides a complete end-to-end local RAG pipeline: an automated PDF directory watcher, a persistent background multi-vector embedding server, unified zero-copy Python bindings, Google Gemini multimodal answer generation with offline fallback, and an integrated benchmarking suite for throughput scaling and ViDoRe retrieval evaluation.

---

## Performance & Benchmarks

BBQ-RAG's fused AVX2/FMA Rust SIMD kernel evaluates over 150,000 document pages per second on standard multi-core CPUs with zero intermediate heap allocations. For detailed methodology, multi-baseline comparisons, and ViDoRe multimodal retrieval evaluations, see the [benchmark report](benchmarks/README.md).

![MaxSim SIMD Throughput Scaling](assets/b1.png)

## Installation

> [!IMPORTANT]
> **Hardware Support: x86_64 with FMA Only**  
> BBQ-RAG's native SIMD compute engine (`maxsimd`) is strictly optimized for **x86_64 (64-bit AMD / Intel)** CPUs with **FMA (Fused Multiply-Add)** and **AVX2** vector instruction sets.  
> - **Supported**: 64-bit x86 CPUs with FMA (Intel Haswell / Core 4th Gen+ and AMD Piledriver / Zen+).
> - **Unsupported**: ARM architectures (e.g., Apple Silicon M1/M2/M3/M4, AWS Graviton, Raspberry Pi) and legacy x86 CPUs lacking FMA support.
> 
> Running `pip install .` automatically triggers an architecture and CPU instruction check. If your machine does not meet the hardware requirements, the installation will halt and inform you that BBQ-RAG cannot be run on your machine.

Install BBQ-RAG and compile the native Rust SIMD extension in one step:

```bash
git clone https://github.com/cmd-HMN/bbq-rag.git
cd bbq-rag

pip install .
```

> **Requirements**: Python 3.10+, Rust 1.75+ (with `cargo`), and an x86_64 CPU supporting AVX2 and FMA instructions.

---

## Quickstart & Usage

### 1. Configure the Engine

Edit `config.yaml` to set your model IDs, watch directory, and Gemini preferences:

```yaml
# only inference supported
base_model_id: "HuggingFaceTB/SmolVLM-256M-Instruct"
# only support this model
lora_adapter_id: "vidore/colSmol-256M"

# change the embedding_dim if you want
embedding_dim: 128

# can be auto selected
device: "auto"

# data type
torch_dtype: "bfloat16"

# masking for images in the model
mask_non_image_embeddings: false

# prompt for vision model
visual_prompt_command: "Describe this image."

# watching folder
watch_folder_path: "data/watch"

# pdf image dpi rendered for embeddings & viewing
pdf_render_dpi: 150

# Output directory for saved retrieved page images
images_output_dir: "data/rr"

# Google Gemini multimodal RAG settings
# If gemini_api_key is omitted/empty, GEMINI_API_KEY / GOOGLE_API_KEY env vars are used.
gemini_api_key: ""
gemini_model: "gemini-3.6-flash"
rag_top_k: 5

# Quantization precision mode: "f32" (default) or "qi8" (quantized INT8)
quantization: "f32"
```

### 2. Start the Indexing Server

Launch the document indexing server. It will monitor `data/watch/` for new PDF files and automatically compute embeddings:

```bash
# Using the bbq CLI command
bbq server --config config.yaml

# Or via Python module
python -m bbq.src.main server --config config.yaml
```

### 3. Query Documents via CLI

Search indexed documents from the command line:

```bash
# Fetch top matching pages (prints ASCII BBQ banner on run)
bbq client "What was the operating margin in Q3?"
# Or: python -m bbq.src.main query "What was the operating margin in Q3?"

# Fetch without ASCII logo banner
bbq client --without-logo "What was the operating margin in Q3?"

# Save retrieved page images to disk (default: data/rr/) in background
bbq client "What is UX design?" --save-images
# Or using shorthand flags with custom output directory:
bbq client "What is UX design?" -i -o data/rr/

# Query with Gemini Multimodal LLM (displays live cooking spinner and multimodal answer)
export GEMINI_API_KEY="your-gemini-api-key"
bbq client "Summarize the revenue growth" --use-llm

# Continuous interactive query prompt loop (--infinite / -inf)
bbq client --infinite
# Or with image saving and LLM cooking enabled:
bbq client --infinite -i --use-llm

# Run directly via the client module
python -m bbq.src.client --infinite --use-llm
```

#### Interactive Document Page Opener
When querying in an interactive terminal, BBQ displays an interactive page picker:
- **`↑` / `↓`** or **`j` / `k`**: Navigate between top retrieved PDF document pages.
- **`Enter`**: Open the selected document at that exact page in your system PDF viewer (`evince`, `okular`, or `xdg-open`).
- **`Esc` / `q`** (or selecting **`Done`**): Proceed to display results and return to prompt.
- Pass **`--without-opener`** (or **`--no-opener`**) to bypass the interactive picker and immediately print matching pages.

In `--infinite` mode, an interactive prompt (`bbq[query] >> `) appears for continuous querying:
- Type any query to search.
- Type `help` or `?` to show the commands menu.
- Type `clear` or `cls` to clear the terminal screen.
- Type `q`, `quit`, or `exit` (or `Ctrl+C`) to quit.

If no Gemini API key is provided or the API is unavailable, the client automatically displays the matching document pages without crashing.

### 4. Python API Usage

#### Client Query, Image Saving & RAG

```python
from bbq.src.client import BBQClient
from bbq.src.config import Config

config = Config.from_yaml("config.yaml")
client = BBQClient(server_url="http://localhost:8000", config=config)

# 1. Standard retrieval query
results = client.query(query_text="Explain the cash flow breakdown in the report", top_k=5)

# 2. Save retrieved page images to disk (default configured dir: data/rr)
saved_paths = client.save_page_images(results)
print(f"Saved {len(saved_paths)} images to data/rr")

# Or save asynchronously in a background daemon thread:
# thread = client.save_page_images_threaded(
#     results,
#     on_progress=lambda curr, total, path: print(f"Saved {curr}/{total}: {path}"),
#     on_complete=lambda paths: print(f"All {len(paths)} images saved!")
# )

# 3. Multimodal answer generation with Gemini
response = client.query_and_answer(
    query_text="Explain the cash flow breakdown in the report",
    top_k=3,
    use_llm=True,
    save_images=True,
)

if response["answer"]:
    print("Gemini Multimodal Answer:\n", response["answer"])
else:
    print("Retrieved Document Pages (Fallback):")
    for source in response["sources"]:
        print(f"File: {source['file_path']} | Page: {source['page_number']} | Score: {source['score']:.4f}")
```

#### Direct Zero-Copy MaxSim in Python

`maxsimd.maxsim` is the single unified SIMD entry point. It directly ingests 2D document matrices, 3D uniform page batches, or ragged flat buffers from either PyTorch tensors or NumPy arrays with zero memory copies:

```python
import torch
import numpy as np
import maxsimd

# Query tensor: shape (32, 128)
query = torch.randn(32, 128, dtype=torch.float32)

# Multi-page documents: shape (100, 128, 128) - works seamlessly with NumPy or PyTorch
docs = torch.randn(100, 128, 128, dtype=torch.float32)

# Unified SIMD AVX2 + Rayon multi-threaded MaxSim call
scores = maxsimd.maxsim(query, docs, jobs=-1)
print("Top 5 Document Scores:", scores[:5])
```

---

## Testing & Verification

Run the full integration test suite and Rust unit tests:

```bash
# Run Python unit and regression tests
pytest

# Run tests in CI mode (headless, skips GUI viewer tests)
CI=true pytest

# Run Rust unit tests (BLAS and AVX2 kernels)
cargo test

# Run scaling and ViDoRe benchmark suite
python3 benchmarks
```

---

## Acknowledgments & Lineage

BBQ-RAG builds upon the foundational research and open-source contributions of the visual document retrieval community:

- **ColPali**: Concept inspired by the paper _"ColPali: Efficient Document Retrieval with Vision Language Models"_ ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) and the [`illuin-tech/colpali`](https://github.com/illuin-tech/colpali) codebase by Manuel Faysse et al.
- **maxsim-cpu**: Inspired by and benchmarked in reference to [`maxsim-cpu`](https://pypi.org/project/maxsim-cpu/) for CPU late-interaction scoring.

We express our sincere thanks to the original authors and maintainers for their pioneering contributions. All original rights, architectures, paper concepts, and model weights remain reserved to their respective authors and institutions.

For full project lineage, paper citations, and intellectual property notices, see [ABOUT.md](ABOUT.md).

---
