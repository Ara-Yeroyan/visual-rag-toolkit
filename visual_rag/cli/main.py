#!/usr/bin/env python3
"""
Visual RAG Toolkit CLI

Provides command-line interface for:
- Processing PDFs (embedding, Cloudinary upload, Qdrant indexing)
- Searching documents
- Managing collections

Usage:
    # Process PDFs (like process_pdfs_saliency_v2.py)
    visual-rag process --reports-dir ./pdfs --metadata-file metadata.json
    
    # Search
    visual-rag search --query "budget allocation" --collection my_docs
    
    # Show collection info
    visual-rag info --collection my_docs
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def cmd_process(args):
    """
    Process PDFs: convert → embed → upload to Cloudinary → index in Qdrant.
    
    Equivalent to process_pdfs_saliency_v2.py
    """
    from visual_rag import VisualEmbedder, QdrantIndexer, CloudinaryUploader, load_config
    from visual_rag.indexing.pipeline import ProcessingPipeline
    
    # Load environment
    load_dotenv()
    
    # Load config
    config = {}
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
    
    # Get PDFs
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        logger.error(f"❌ Reports directory not found: {reports_dir}")
        sys.exit(1)
    
    pdf_paths = sorted(reports_dir.glob("*.pdf")) + sorted(reports_dir.glob("*.PDF"))
    if not pdf_paths:
        logger.error(f"❌ No PDF files found in: {reports_dir}")
        sys.exit(1)
    
    logger.info(f"📁 Found {len(pdf_paths)} PDF files")
    
    # Load metadata mapping
    metadata_mapping = {}
    if args.metadata_file:
        metadata_mapping = ProcessingPipeline.load_metadata_mapping(Path(args.metadata_file))
    
    # Dry run - just show summary
    if args.dry_run:
        logger.info("🏃 DRY RUN MODE")
        logger.info(f"   PDFs: {len(pdf_paths)}")
        logger.info(f"   Metadata entries: {len(metadata_mapping)}")
        logger.info(f"   Collection: {args.collection}")
        logger.info(f"   Cloudinary: {'ENABLED' if not args.no_cloudinary else 'DISABLED'}")
        
        for pdf in pdf_paths[:10]:
            has_meta = "✓" if pdf.stem.lower() in metadata_mapping else "✗"
            logger.info(f"   {has_meta} {pdf.name}")
        if len(pdf_paths) > 10:
            logger.info(f"   ... and {len(pdf_paths) - 10} more")
        return
    
    # Get settings
    model_name = args.model or config.get("model", {}).get("name", "vidore/colSmol-500M")
    collection_name = args.collection or config.get("qdrant", {}).get("collection_name", "visual_documents")
    
    # Initialize embedder
    logger.info(f"🤖 Initializing embedder: {model_name}")
    embedder = VisualEmbedder(model_name=model_name, batch_size=args.batch_size)
    
    # Initialize Qdrant indexer
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        logger.error("❌ QDRANT_URL environment variable not set")
        sys.exit(1)
    
    logger.info(f"🔌 Connecting to Qdrant: {qdrant_url}")
    indexer = QdrantIndexer(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
    )
    
    # Create collection if needed
    indexer.create_collection(force_recreate=args.force_recreate)
    indexer.create_payload_indexes()
    
    # Initialize Cloudinary uploader (optional)
    cloudinary_uploader = None
    if not args.no_cloudinary:
        try:
            project_name = config.get("project_name", "visual_docs")
            cloudinary_uploader = CloudinaryUploader(folder=project_name)
        except ValueError as e:
            logger.warning(f"⚠️ Cloudinary not configured: {e}")
            logger.warning("   Continuing without Cloudinary uploads")
    
    # Create pipeline
    pipeline = ProcessingPipeline(
        embedder=embedder,
        indexer=indexer,
        cloudinary_uploader=cloudinary_uploader,
        metadata_mapping=metadata_mapping,
        config=config,
        embedding_strategy=args.strategy,
    )
    
    # Process PDFs
    total_uploaded = 0
    total_skipped = 0
    total_failed = 0
    
    skip_existing = not args.no_skip_existing
    
    for pdf_idx, pdf_path in enumerate(pdf_paths, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 [{pdf_idx}/{len(pdf_paths)}] {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        result = pipeline.process_pdf(
            pdf_path,
            skip_existing=skip_existing,
            upload_to_cloudinary=(not args.no_cloudinary),
            upload_to_qdrant=True,
        )
        
        total_uploaded += result["uploaded"]
        total_skipped += result["skipped"]
        total_failed += result["failed"]
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"   Total PDFs: {len(pdf_paths)}")
    logger.info(f"   Uploaded: {total_uploaded}")
    logger.info(f"   Skipped: {total_skipped}")
    logger.info(f"   Failed: {total_failed}")
    
    info = indexer.get_collection_info()
    if info:
        logger.info(f"   Collection points: {info.get('points_count', 'N/A')}")


def cmd_search(args):
    """Search documents."""
    from visual_rag import VisualEmbedder
    from visual_rag.retrieval import TwoStageRetriever
    from qdrant_client import QdrantClient
    
    load_dotenv()
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        logger.error("❌ QDRANT_URL not set")
        sys.exit(1)
    
    # Initialize
    logger.info(f"🤖 Loading model: {args.model}")
    embedder = VisualEmbedder(model_name=args.model)
    
    logger.info(f"🔌 Connecting to Qdrant")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    retriever = TwoStageRetriever(client, args.collection)
    
    # Embed query
    logger.info(f"🔍 Query: {args.query}")
    query_embedding = embedder.embed_query(args.query)
    
    # Build filter
    filter_obj = None
    if args.year or args.source or args.district:
        filter_obj = retriever.build_filter(
            year=args.year,
            source=args.source,
            district=args.district,
        )
    
    # Search
    results = retriever.search(
        query_embedding=query_embedding.numpy(),
        top_k=args.top_k,
        prefetch_k=args.prefetch_k,
        filter_obj=filter_obj,
    )
    
    # Display results
    logger.info(f"\n📊 Results ({len(results)}):")
    for i, result in enumerate(results, 1):
        payload = result.get("payload", {})
        score = result.get("score_final", result.get("score_stage1", 0))
        
        filename = payload.get("filename", "N/A")
        page_num = payload.get("page_number", "N/A")
        year = payload.get("year", "N/A")
        source = payload.get("source", "N/A")
        
        logger.info(f"  {i}. {filename} p.{page_num}")
        logger.info(f"     Score: {score:.4f} | Year: {year} | Source: {source}")
        
        # Text snippet
        text = payload.get("text", "")
        if text and args.show_text:
            snippet = text[:200].replace("\n", " ")
            logger.info(f"     Text: {snippet}...")


def cmd_info(args):
    """Show collection info."""
    from qdrant_client import QdrantClient
    
    load_dotenv()
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        logger.error("❌ QDRANT_URL not set")
        sys.exit(1)
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    try:
        info = client.get_collection(args.collection)
        
        status = info.status
        if hasattr(status, "value"):
            status = status.value
        
        indexed_count = getattr(info, "indexed_vectors_count", 0) or 0
        if isinstance(indexed_count, dict):
            indexed_count = sum(indexed_count.values())
        
        logger.info(f"📊 Collection: {args.collection}")
        logger.info(f"   Status: {status}")
        logger.info(f"   Points: {info.points_count}")
        logger.info(f"   Indexed vectors: {indexed_count}")
        
        # Show vector config
        if hasattr(info, "config") and hasattr(info.config, "params"):
            vectors = getattr(info.config.params, "vectors", {})
            if vectors:
                logger.info(f"   Vectors: {list(vectors.keys())}")
        
    except Exception as e:
        logger.error(f"❌ Could not get collection info: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="visual-rag",
        description="Visual RAG Toolkit - Visual document retrieval with ColPali",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PDFs (like process_pdfs_saliency_v2.py)
  visual-rag process --reports-dir ./pdfs --metadata-file metadata.json
  
  # Process without Cloudinary
  visual-rag process --reports-dir ./pdfs --no-cloudinary
  
  # Search
  visual-rag search --query "budget allocation" --collection my_docs
  
  # Search with filters
  visual-rag search --query "budget" --year 2023 --source "Local Government"
  
  # Show collection info
  visual-rag info --collection my_docs
        """,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # =========================================================================
    # PROCESS command
    # =========================================================================
    process_parser = subparsers.add_parser(
        "process",
        help="Process PDFs: embed, upload to Cloudinary, index in Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    process_parser.add_argument(
        "--reports-dir", type=str, required=True,
        help="Directory containing PDF files"
    )
    process_parser.add_argument(
        "--metadata-file", type=str,
        help="JSON file with filename → metadata mapping (like filename_metadata.json)"
    )
    process_parser.add_argument(
        "--collection", type=str, default="visual_documents",
        help="Qdrant collection name"
    )
    process_parser.add_argument(
        "--model", type=str, default="vidore/colSmol-500M",
        help="Model name (vidore/colSmol-500M, vidore/colpali-v1.3, etc.)"
    )
    process_parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Embedding batch size"
    )
    process_parser.add_argument(
        "--config", type=str,
        help="Path to config.yaml file"
    )
    process_parser.add_argument(
        "--no-cloudinary", action="store_true",
        help="Skip Cloudinary uploads"
    )
    process_parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Process all pages even if they exist in Qdrant"
    )
    process_parser.add_argument(
        "--force-recreate", action="store_true",
        help="Delete and recreate collection"
    )
    process_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without doing it"
    )
    process_parser.add_argument(
        "--strategy", type=str, default="pooling",
        choices=["pooling", "standard", "all"],
        help="Embedding strategy: 'pooling' (NOVEL), 'standard' (BASELINE), "
             "'all' (embed once, store BOTH for comparison)"
    )
    process_parser.set_defaults(func=cmd_process)
    
    # =========================================================================
    # SEARCH command
    # =========================================================================
    search_parser = subparsers.add_parser(
        "search",
        help="Search documents",
    )
    search_parser.add_argument(
        "--query", type=str, required=True,
        help="Search query"
    )
    search_parser.add_argument(
        "--collection", type=str, default="visual_documents",
        help="Qdrant collection name"
    )
    search_parser.add_argument(
        "--model", type=str, default="vidore/colSmol-500M",
        help="Model name"
    )
    search_parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of results"
    )
    search_parser.add_argument(
        "--prefetch-k", type=int, default=200,
        help="Prefetch candidates for two-stage retrieval"
    )
    search_parser.add_argument(
        "--year", type=int,
        help="Filter by year"
    )
    search_parser.add_argument(
        "--source", type=str,
        help="Filter by source"
    )
    search_parser.add_argument(
        "--district", type=str,
        help="Filter by district"
    )
    search_parser.add_argument(
        "--show-text", action="store_true",
        help="Show text snippets in results"
    )
    search_parser.set_defaults(func=cmd_search)
    
    # =========================================================================
    # INFO command
    # =========================================================================
    info_parser = subparsers.add_parser(
        "info",
        help="Show collection info",
    )
    info_parser.add_argument(
        "--collection", type=str, default="visual_documents",
        help="Qdrant collection name"
    )
    info_parser.set_defaults(func=cmd_info)
    
    # Parse and execute
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
