"""
Visual RAG Toolkit - End-to-end visual document retrieval with two-stage pooling.

A modular toolkit for building visual document retrieval systems:

Components:
-----------
- embedding: Visual and text embedding generation (ColPali, etc.)
- indexing: PDF processing, Qdrant indexing, Cloudinary uploads
- retrieval: Single-stage and two-stage retrieval with MaxSim
- visualization: Saliency maps and attention visualization
- cli: Command-line interface

Quick Start:
------------
>>> from visual_rag import VisualEmbedder, PDFProcessor, TwoStageRetriever
>>> 
>>> # Process PDFs
>>> processor = PDFProcessor(dpi=140)
>>> images, texts = processor.process_pdf("report.pdf")
>>> 
>>> # Generate embeddings
>>> embedder = VisualEmbedder()
>>> embeddings = embedder.embed_images(images)
>>> query_emb = embedder.embed_query("What is the budget?")
>>> 
>>> # Search with two-stage retrieval
>>> retriever = TwoStageRetriever(qdrant_client, "my_collection")
>>> results = retriever.search(query_emb, top_k=10)

Each component works independently - use only what you need.
"""

__version__ = "0.1.0"

# Import main classes at package level for convenience
# These are optional - if dependencies aren't installed, we catch the error

try:
    from visual_rag.embedding.visual_embedder import VisualEmbedder
except ImportError:
    VisualEmbedder = None

try:
    from visual_rag.indexing.pdf_processor import PDFProcessor
except ImportError:
    PDFProcessor = None

try:
    from visual_rag.indexing.qdrant_indexer import QdrantIndexer
except ImportError:
    QdrantIndexer = None

try:
    from visual_rag.indexing.cloudinary_uploader import CloudinaryUploader
except ImportError:
    CloudinaryUploader = None

try:
    from visual_rag.retrieval.two_stage import TwoStageRetriever
except ImportError:
    TwoStageRetriever = None

try:
    from visual_rag.retrieval.multi_vector import MultiVectorRetriever
except ImportError:
    MultiVectorRetriever = None

try:
    from visual_rag.qdrant_admin import QdrantAdmin
except ImportError:
    QdrantAdmin = None

try:
    from visual_rag.demo_runner import demo
except ImportError:
    demo = None

# Config utilities (always available)
from visual_rag.config import load_config, get, get_section

__all__ = [
    # Version
    "__version__",
    
    # Main classes
    "VisualEmbedder",
    "PDFProcessor",
    "QdrantIndexer",
    "CloudinaryUploader",
    "TwoStageRetriever",
    "MultiVectorRetriever",
    "QdrantAdmin",
    "demo",
    
    # Config utilities
    "load_config",
    "get",
    "get_section",
]
