# Visual RAG Toolkit

[![PyPI version](https://badge.fury.io/py/visual-rag-toolkit.svg)](https://badge.fury.io/py/visual-rag-toolkit)
[![CI](https://github.com/Ara-Yeroyan/visual-rag-toolkit/actions/workflows/ci.yaml/badge.svg)](https://github.com/Ara-Yeroyan/visual-rag-toolkit/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

End-to-end visual document retrieval toolkit featuring **fast multi-stage retrieval** (prefetch with pooled vectors + exact MaxSim reranking).

This repo contains:
- a **Python package** (`visual_rag`)
- a **Streamlit demo app** (`demo/`)
- **benchmark & evaluation scripts** for ViDoRe v2 (`benchmarks/`)

## 🎯 Key Features

- **Modular**: PDF → images, embedding, Qdrant indexing, retrieval can be used independently.
- **Multi-stage retrieval**: two-stage and three-stage retrieval modes built for Qdrant named vectors.
- **Model-aware embedding**: ColSmol + ColPali support behind a single `VisualEmbedder` interface.
- **Token hygiene**: query special-token filtering by default for more stable MaxSim behavior.
- **Practical pipelines**: robust indexing, retries, optional Cloudinary image URLs, evaluation reporting.

## 📦 Installation

```bash
# Core package (minimal dependencies)
pip install visual-rag-toolkit

# With specific features
pip install visual-rag-toolkit[embedding]    # ColSmol/ColPali embedding support
pip install visual-rag-toolkit[pdf]          # PDF processing
pip install visual-rag-toolkit[qdrant]       # Vector database
pip install visual-rag-toolkit[cloudinary]   # Image CDN
pip install visual-rag-toolkit[ui]           # Streamlit demo dependencies

# All dependencies
pip install visual-rag-toolkit[all]
```

### System dependencies (PDF)

`pdf2image` requires Poppler.

- macOS: `brew install poppler`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y poppler-utils`

## 🚀 Quick Start

### Minimal: embed a query and run two-stage search (server-side)

```python
from qdrant_client import QdrantClient
from visual_rag import VisualEmbedder, TwoStageRetriever

client = QdrantClient(url="https://YOUR_QDRANT", api_key="YOUR_KEY")
collection_name = "your_collection"

# Embed query tokens
embedder = VisualEmbedder(model_name="vidore/colpali-v1.3")
q = embedder.embed_query("What is the budget allocation?")

# Fast path: all stages computed in Qdrant (prefetch + exact rerank)
retriever = TwoStageRetriever(client, collection_name)
results = retriever.search_server_side(
    query_embedding=q,
    top_k=10,
    prefetch_k=256,
    stage1_mode="tokens_vs_experimental",  # or: tokens_vs_tiles / pooled_query_vs_tiles / pooled_query_vs_global
)

for r in results[:3]:
    print(r["id"], r["score_final"])
```

### Process a PDF into images (no embedding, no vector DB)

```python
from pathlib import Path
from visual_rag import PDFProcessor

processor = PDFProcessor(dpi=140)
images, texts = processor.process_pdf(Path("report.pdf"))
print(len(images), "pages")
```

## 🔬 Multi-stage Retrieval (Two-stage / Three-stage)

Traditional ColBERT-style MaxSim scoring compares all query tokens vs all document tokens, which becomes expensive at scale.

**Our approach:**

```
Stage 1: Fast prefetch with tile-level pooled vectors
         ├── Pool each tile (64 patches) → num_tiles vectors
         ├── Use HNSW index for O(log N) retrieval  
         └── Retrieve top-K candidates (e.g., 200)

Stage 2: Exact MaxSim reranking on candidates
         ├── Load full multi-vector embeddings
         ├── Compute exact ColBERT MaxSim scores
         └── Return top-k results (e.g., 10)
```

Three-stage extends this with an additional “cheap prefetch” stage before stage 2.

## 📁 Package Structure

```
visual-rag-toolkit/
├── visual_rag/              # Import as: from visual_rag import ...
│   ├── embedding/           # VisualEmbedder, pooling functions
│   ├── indexing/            # PDFProcessor, QdrantIndexer, CloudinaryUploader
│   ├── retrieval/           # TwoStageRetriever
│   ├── visualization/       # Saliency maps
│   ├── cli/                 # Command-line: visual-rag process/search
│   └── config.py            # load_config, get, get_section
│
├── benchmarks/              # ViDoRe evaluation scripts
└── examples/                # Usage examples
```

## ⚙️ Configuration

Configure via environment variables or YAML:

```bash
# Qdrant credentials (preferred names used by the demo + scripts)
export SIGIR_QDRANT_URL="https://your-cluster.qdrant.io"
export SIGIR_QDRANT_KEY="your-api-key"

# Backwards-compatible fallbacks (also supported)
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-api-key"

export VISUALRAG_MODEL="vidore/colSmol-500M"

# Special token handling (default: filter them out)
export VISUALRAG_INCLUDE_SPECIAL_TOKENS=true  # Include special tokens
```

Or use a config file (`visual_rag.yaml`):

```yaml
model:
  name: "vidore/colSmol-500M"
  batch_size: 4
  
qdrant:
  url: "https://your-cluster.qdrant.io"
  collection: "my_documents"
  
search:
  strategy: "two_stage"  # or "multi_vector", "pooled"
  prefetch_k: 200
  top_k: 10
```

## 🖥️ Demo (Streamlit)

```bash
pip install "visual-rag-toolkit[ui,qdrant,embedding,pdf]"

# Option A: from Python
python -c "import visual_rag; visual_rag.demo()"

# Option B: CLI launcher
visual-rag-demo
```

## 📊 Benchmark Evaluation

Run ViDoRe benchmark evaluation:

```bash
# Example: evaluate a collection against ViDoRe BEIR datasets in Qdrant
python -m benchmarks.vidore_beir_qdrant.run_qdrant_beir \
  --datasets vidore/esg_reports_v2 vidore/biomedical_lectures_v2 \
  --collection YOUR_COLLECTION \
  --mode two_stage \
  --stage1-mode tokens_vs_experimental \
  --prefetch-k 256 \
  --top-k 100 \
  --evaluation-scope union
```

More commands (including multi-stage variants and cropping configs) live in:
- `benchmarks/vidore_tatdqa_test/COMMANDS.md`

## 🔧 Development

```bash
git clone https://github.com/Ara-Yeroyan/visual-rag-toolkit
cd visual-rag-toolkit
pip install -e ".[dev]"
pytest tests/ -v
```

## 📄 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{visual_rag_toolkit,
  title = {Visual RAG Toolkit: Scalable Visual Document Retrieval with Two-Stage Pooling},
  author = {Ara Yeroyan},
  year = {2026},
  url = {https://github.com/Ara-Yeroyan/visual-rag-toolkit}
}
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Qdrant](https://qdrant.tech/) - Vector database with multi-vector support
- [ColPali](https://github.com/illuin-tech/colpali) - Visual document retrieval models
- [ViDoRe](https://huggingface.co/spaces/vidore/vidore-leaderboard) - Benchmark dataset

