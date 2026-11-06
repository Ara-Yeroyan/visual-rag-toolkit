# Visual RAG Toolkit

[![PyPI](https://img.shields.io/pypi/v/visual-rag-toolkit)](https://pypi.org/project/visual-rag-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/visual-rag-toolkit)](https://pypi.org/project/visual-rag-toolkit/)
[![License](https://img.shields.io/pypi/l/visual-rag-toolkit)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/Yeroyan/visual-rag-toolkit)

End-to-end visual document retrieval toolkit featuring **fast multi-stage retrieval** (prefetch with pooled vectors + exact MaxSim reranking).

**[Try the Live Demo](https://huggingface.co/spaces/Yeroyan/visual-rag-toolkit)** - Upload PDFs, index to Qdrant, and query with visual retrieval.

**[Watch the Tutorial](https://drive.google.com/file/d/1SmpVJicsvyZ-awlwYtSkLw6BCSO7IJTw/view)** - Video walkthrough of the toolkit in action.

This repo contains:
- a **Python package** (`visual_rag`)
- a **Streamlit demo app** (`demo/`)
- **benchmark & evaluation scripts** for ViDoRe v2 (`benchmarks/`)

## 🎯 Key Features

- **Modular**: PDF → images, embedding, Qdrant indexing, retrieval can be used independently.
- **Multi-stage retrieval**: two-stage and three-stage retrieval modes built for Qdrant named vectors.
- **Model-aware embedding**: ColSmol, ColPali, and ColQwen2/2.5 support behind a single `VisualEmbedder` interface.
- **Configurable pooling**: adaptive mean-pooling cap for ColQwen2.5 (`--max-mean-pool-vectors`), and experimental pooling stored as Qdrant named vectors (`experimental_pooling` (ColQwen Gaussian alias), `experimental_pooling_gaussian`, `experimental_pooling_triangular`, `experimental_pooling_{k}` (ColPali), `experimental_pooling_2d` (ColSmol)).
- **Single-stage ablations**: direct search modes over experimental pooled vectors (tokens-vs-doc and pooled-query-vs-doc) for fast storage-reduction experiments.
- **Token hygiene**: query special-token filtering by default for more stable MaxSim behavior.
- **Practical pipelines**: robust indexing, retries, optional Cloudinary image URLs, evaluation reporting.

## 📦 Installation

```bash
# Core package (minimal dependencies)
pip install visual-rag-toolkit

# With specific features
pip install visual-rag-toolkit[ui]           # Streamlit demo dependencies
pip install visual-rag-toolkit[qdrant]       # Vector database
pip install visual-rag-toolkit[embedding]    # ColSmol/ColPali/ColQwen2(.5) embedding support
pip install visual-rag-toolkit[cloudinary]   # Image CDN

# All dependencies
pip install visual-rag-toolkit[all]
```

### System dependencies (PDF)

`pdf2image` requires Poppler.

- macOS: `brew install poppler`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y poppler-utils`

**ColQwen2.5 note:** `vidore/colqwen2.5-v0.2` requires `transformers>=4.45.0` and `colpali-engine>=0.3.7` (installing ColPali/ColQwen from source may be required for the latest processors).

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
    stage1_mode="tokens_vs_experimental_pooling",  # or: tokens_vs_standard_pooling / pooled_query_vs_standard_pooling / pooled_query_vs_global
)

for r in results[:3]:
    print(r["id"], r["score_final"])
```

### End-to-end: ingest PDFs (with cropping) → index in Qdrant

This is the "SDK-style" pipeline: PDF → images → optional crop → embed → store vectors + payload in Qdrant.

```python
import os
from pathlib import Path

import numpy as np
import torch

from visual_rag import VisualEmbedder
from visual_rag.indexing import ProcessingPipeline, QdrantIndexer

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_KEY = os.getenv("QDRANT_API_KEY", "")

collection = "my_visual_docs"

embedder = VisualEmbedder(
    model_name="vidore/colSmol-500M",
    torch_dtype=torch.float16,
    output_dtype=np.float16,
    batch_size=8,
)

indexer = QdrantIndexer(
    url=QDRANT_URL,
    api_key=QDRANT_KEY,
    collection_name=collection,
    prefer_grpc=True,
    vector_datatype="float16",
)

# Creates collection + required payload indexes (e.g., "filename" for skip_existing)
indexer.create_collection(force_recreate=False)

pipeline = ProcessingPipeline(
    embedder=embedder,
    indexer=indexer,
    embedding_strategy="all",  # store full tokens + pooled vectors in one pass
    crop_empty=True,
    crop_empty_percentage_to_remove=0.99,  # kept for traceability
    crop_empty_remove_page_number=True,
    crop_empty_preserve_border_px=1,
    crop_empty_uniform_rowcol_std_threshold=3.0,
)

pdfs = [Path("docs/a.pdf"), Path("docs/b.pdf")]
for pdf_path in pdfs:
    result = pipeline.process_pdf(
        pdf_path,
        skip_existing=True,  # Skip pages already in Qdrant (uses filename index)
        upload_to_cloudinary=False,
        upload_to_qdrant=True,
    )
    # Logs automatically shown:
    # [10:23:45] 📚 Processing PDF: a.pdf
    # [10:23:45] 🖼️ Converting PDF to images...
    # [10:23:46]    ✅ Converted 12 pages
    # [10:23:46] 📦 Processing pages 1-8/12
    # [10:23:46] 🤖 Generating embeddings for 8 pages...
    # [10:23:48] 📤 Uploading batch of 8 pages...
    # [10:23:48]    ✅ Uploaded 8 points to Qdrant
    # [10:23:48] 📦 Processing pages 9-12/12
    # [10:23:48] 🤖 Generating embeddings for 4 pages...
    # [10:23:50] 📤 Uploading batch of 4 pages...
    # [10:23:50]    ✅ Uploaded 4 points to Qdrant
    # [10:23:50] ✅ Completed a.pdf: 12 uploaded, 0 skipped, 0 failed
```

CLI equivalent:

```bash
export QDRANT_URL="https://YOUR_QDRANT"
export QDRANT_API_KEY="YOUR_KEY"

visual-rag process \
  --reports-dir ./docs \
  --collection my_visual_docs \
  --model vidore/colSmol-500M \
  --strategy all \
  --batch-size 8 \
  --qdrant-vector-dtype float16 \
  --prefer-grpc \
  --crop-empty \
  --crop-empty-remove-page-number
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

Three-stage extends this with an additional "cheap prefetch" stage before stage 2.

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
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-api-key"

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
  --stage1-mode tokens_vs_experimental_pooling \
  --prefetch-k 256 \
  --top-k 100 \
  --evaluation-scope union
```

More commands (including multi-stage variants and cropping configs) live in:
- `examples/COMMANDS.md`

## 🔧 Development

```bash
git clone https://github.com/Ara-Yeroyan/visual-rag-toolkit
cd visual-rag-toolkit
pip install -e ".[dev]"
pytest tests/ -v
```

## 📄 Citation

If you use this toolkit or its retrieval methods in your research, please cite the SIGIR 2026 paper:

```bibtex
@inproceedings{yeroyan2026visualrag,
  author = {Yeroyan, Ara},
  title = {Visual {RAG} at Scale: Tile-Level Spatial Pooling for Efficient Multi-Vector Document Retrieval},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series = {SIGIR '26},
  year = {2026},
  pages = {5245--5249},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3805712.3808383},
  url = {https://doi.org/10.1145/3805712.3808383}
}
```

To additionally acknowledge the software implementation, you may cite:

```bibtex
@software{yeroyan2026visualragtoolkit,
  author = {Yeroyan, Ara},
  title = {Visual RAG Toolkit},
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
