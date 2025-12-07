"""
Indexing module - PDF processing, embedding storage, and CDN uploads.

Components:
- PDFProcessor: Convert PDFs to images and extract text
- QdrantIndexer: Upload embeddings to Qdrant vector database
- CloudinaryUploader: Upload images to Cloudinary CDN
- ProcessingPipeline: End-to-end PDF → Qdrant pipeline
"""

from visual_rag.indexing.pdf_processor import PDFProcessor
from visual_rag.indexing.qdrant_indexer import QdrantIndexer
from visual_rag.indexing.cloudinary_uploader import CloudinaryUploader
from visual_rag.indexing.pipeline import ProcessingPipeline

__all__ = [
    "PDFProcessor",
    "QdrantIndexer",
    "CloudinaryUploader",
    "ProcessingPipeline",
]
