<div align="center">

# <img src="assets/logo.png" alt="BBQ-RAG Logo" width="160" style="vertical-align: middle; margin-right: 12px;"/> BBQ-RAG

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-0284c7)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-ea580c.svg)](https://www.rust-lang.org/)
[![CPU](<https://img.shields.io/badge/CPU-x86__64%20(AVX2%20%2B%20FMA)-16a34a.svg>)](#performance--benchmarks)
[![Model](https://img.shields.io/badge/Model-ColPali%20%2F%20SmolVLM-7c3aed.svg)](#key-features)
[![FFI](https://img.shields.io/badge/FFI-PyO3%200.29-e11d48.svg)](https://pyo3.rs/)
[![Repo Size](https://img.shields.io/badge/repo%20size-~3.2%20MB-0ea5e9.svg)](#)
[![Environment](https://img.shields.io/badge/environment-~2.5--3.0%20GB-8b5cf6.svg)](#)
[![Throughput](https://img.shields.io/badge/throughput->%20150k%20pages/s-f97316.svg)](#performance--benchmarks)
[![About](https://img.shields.io/badge/Lineage-ColPali%20%26%20maxsim--cpu-f59e0b.svg)](ABOUT.md)

</div>

BBQ-RAG is an ultra-fast, lightweight visual document retrieval and late-interaction (ColPali / MaxSim) search engine. By combining Vision-Language Model embeddings (ColPali, colSmol, SmolVLM) with an in-register fused AVX2/FMA SIMD compute engine in Rust (`maxsimd`), BBQ-RAG scores over **150,000 document pages per second** on standard multi-core CPUs with zero intermediate heap allocations.

It provides a complete end-to-end local RAG pipeline: an automated PDF directory watcher, a persistent background multi-vector embedding server, unified zero-copy Python bindings, Google Gemini multimodal answer generation with offline fallback, and an integrated benchmarking suite for throughput scaling and ViDoRe retrieval evaluation.

---

## Key Features

- **Fused AVX2/FMA SIMD MaxSim Engine**: 4-way query unrolled SIMD dot-product pipeline with in-register maximum tracking written in Rust (`maxsimd`), eliminating intermediate similarity matrix allocation and cutting memory load traffic by 27x.
- **Adaptive Multi-Threaded Parallelism**: Dynamic execution routing that runs small batches sequentially on the main thread to eliminate work-stealing overhead, and switches automatically to Rayon chunked work pools for large multi-page collections.
- **Unified Zero-Copy FFI**: A single unified Python entry point (`maxsimd.maxsim`) accepting contiguous NumPy arrays and PyTorch tensors directly with zero memory copying or overhead.
- **Vision-Language Indexing Server**: Automated folder monitoring (`data/watch/`), background PDF page rasterization at 150 DPI via PyMuPDF, and persistent SQLite metadata and multi-vector embedding storage.
- **Client-Side Gemini Multimodal RAG**: Seamless grounded answer synthesis with Google Gemini (`gemini-2.5-flash` / `gemini-2.0-flash`) over top retrieved visual pages, with automatic fallback to page viewing when offline or without an API key.
- **Production Benchmarking Suite**: Comprehensive micro- and macro-benchmark suites with CLI controls (`--suite maxsimd` for throughput scaling, `--suite vidore` evaluating retrieval quality, QPS, and numerical parity against PyTorch across 10 official ViDoRe datasets).

---

## Performance & Benchmarks

> For in-depth benchmark comparisons, multi-baseline analysis, and full ViDoRe multimodal retrieval evaluations, see [**MORE INFO ON THE BENCHMARK README**](benchmarks/README.md).

![MaxSim SIMD Throughput Scaling](assets/b1.png)

## Installation

### Prerequisites

- Python 3.10 or higher
- Rust 1.75 or higher (with `cargo`)
- x86_64 CPU supporting AVX2 and FMA instructions

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/cmd-HMN/bbq-rag.git
cd bbq-rag

python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Compile Rust Extension

Build the high-performance release binary using `maturin`:

```bash
maturin develop --release
```

Or build a redistributable wheel:

```bash
maturin build --release -o dist/
pip install dist/bbq_rag-*.whl
```

---

## Quickstart & Usage

### 1. Configure the Engine

Edit `config.yaml` to set your model IDs, watch directory, and Gemini preferences:

```yaml
base_model_id: "HuggingFaceTB/SmolVLM-256M-Instruct"
lora_adapter_id: "vidore/colSmol-256M"
embedding_dim: 128
device: "auto"
torch_dtype: "bfloat16"

watch_folder_path: "data/watch"
pdf_render_dpi: 150

# Optional Google Gemini multimodal RAG settings
gemini_api_key: ""
gemini_model: "gemini-3.6-flash"
rag_top_k: 3
```

### 2. Start the Indexing Server

Launch the document indexing server. It will monitor `data/watch/` for new PDF files and automatically compute embeddings:

```bash
python -m bbq.src.main server --config config.yaml
```

### 3. Query Documents via CLI

Search indexed documents from the command line:

```bash
# Query top 3 matching pages
python -m bbq.src.main query "What was the operating margin in Q3?" --top-k 3

# Query with Gemini Multimodal RAG (generates grounded answer from top 3 page images)
export GEMINI_API_KEY="your-gemini-api-key"
python -m bbq.src.main query "Summarize the revenue growth" --top-k 3
```

If no Gemini API key is provided or the API is unavailable, the client automatically displays the matching document pages without crashing.

### 4. Python API Usage

#### Client Query & RAG

```python
from bbq.src.client import BBQClient
from bbq.src.config import load_configuration_from_yaml_file

config = load_configuration_from_yaml_file("config.yaml")
client = BBQClient(server_url="http://localhost:8000", config=config)

response = client.query_and_answer(
    query_text="Explain the cash flow breakdown in the report",
    top_k=3,
)

if response["answer"]:
    print("Gemini Multimodal Answer:\n", response["answer"])
else:
    print("Retrieved Book Pages (Fallback):")
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
# Run Python integration and regression tests (7 test suites)
python3 -m pytest

# Run Rust unit tests (47 tests for BLAS and AVX2 kernels)
cargo test

# Run scaling and ViDoRe benchmark suite
python3 benchmarks
```

---

## Roadmap & TODO
- [ ] **Audit Boilerplate & Fix Errors for Scalability**: Review and harden all boilerplate code, fix edge-case runtime errors, eliminate redundant allocations, and optimize the codebase for production-grade throughput and scalability.
- [ ] **Eliminate Circular Dependencies**: Audit and refactor inter-module imports across `client`, `server`, `storage`, and `config`.
- [ ] **Origins & Attribution Reference**: See [ABOUT.md](ABOUT.md) for full project lineage, paper citations ([ColPali arXiv:2407.01449](https://arxiv.org/abs/2407.01449)), `maxsim-cpu` references, and reserved rights notices.

---

## Acknowledgments & Lineage

BBQ-RAG builds upon the foundational research and open-source contributions of the visual document retrieval community:

- **ColPali**: Concept inspired by the paper _"ColPali: Efficient Document Retrieval with Vision Language Models"_ ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) and the [`illuin-tech/colpali`](https://github.com/illuin-tech/colpali) codebase by Manuel Faysse et al.
- **maxsim-cpu**: Inspired by and benchmarked in reference to [`maxsim-cpu`](https://pypi.org/project/maxsim-cpu/) for CPU late-interaction scoring.

We express our sincere thanks to the original authors and maintainers for their pioneering contributions. All original rights, architectures, paper concepts, and model weights remain reserved to their respective authors and institutions.

For full project lineage, paper citations, and intellectual property notices, see [ABOUT.md](ABOUT.md).

---

## Disclaimer

This project, its documentation, and parts of its codebase and benchmarks were developed with the assistance of AI tools. As an evolving early-stage project, this README and documentation may contain preliminary assumptions or specifications that are actively being refined. Future commits will continuously audit, validate, and update these details to ensure ongoing accuracy, correctness, and benchmarking rigor. No warranties or guarantees of fitness for a particular purpose are provided.
