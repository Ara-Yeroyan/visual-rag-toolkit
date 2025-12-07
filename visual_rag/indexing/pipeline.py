"""
Processing Pipeline - Complete PDF → Qdrant pipeline with saliency metadata.

This module combines all components for end-to-end processing:
- PDF → images conversion
- Image resizing for ColPali
- Embedding generation with token info
- Tile-level pooling
- Cloudinary upload (optional)
- Qdrant indexing with full metadata for saliency maps

The metadata stored includes everything needed for saliency visualization:
- Tile structure (num_tiles, tile_rows, tile_cols, patches_per_tile)
- Image dimensions (original and resized)
- Token info (num_visual_tokens, visual_token_indices)
"""

import gc
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """
    End-to-end pipeline for PDF processing and indexing.
    
    This pipeline:
    1. Converts PDFs to images
    2. Resizes for ColPali processing
    3. Generates embeddings with token info
    4. Computes pooling (strategy-dependent)
    5. Uploads images to Cloudinary (optional)
    6. Stores in Qdrant with full saliency metadata
    
    Args:
        embedder: VisualEmbedder instance
        indexer: QdrantIndexer instance (optional)
        cloudinary_uploader: CloudinaryUploader instance (optional)
        pdf_processor: PDFProcessor instance (optional, auto-created)
        metadata_mapping: Dict mapping filenames to extra metadata
        config: Configuration dict
        embedding_strategy: How to process embeddings before storing:
            - "pooling" (default): Extract visual tokens only, compute tile-level pooling
              This is our NOVEL contribution - preserves spatial structure while reducing size.
            - "standard": Push ALL tokens as-is (including special tokens, padding)
              This is the baseline approach for comparison.
    
    Example:
        >>> from visual_rag import VisualEmbedder, QdrantIndexer, CloudinaryUploader
        >>> from visual_rag.indexing.pipeline import ProcessingPipeline
        >>> 
        >>> # Our novel pooling strategy (default)
        >>> pipeline = ProcessingPipeline(
        ...     embedder=VisualEmbedder(),
        ...     indexer=QdrantIndexer(url, api_key, "my_collection"),
        ...     embedding_strategy="pooling",  # Visual tokens only + tile pooling
        ... )
        >>> 
        >>> # Standard baseline (all tokens, no filtering)
        >>> pipeline_baseline = ProcessingPipeline(
        ...     embedder=VisualEmbedder(),
        ...     indexer=QdrantIndexer(url, api_key, "my_collection_baseline"),
        ...     embedding_strategy="standard",  # All tokens as-is
        ... )
        >>> 
        >>> pipeline.process_pdf(Path("report.pdf"))
    """
    
    # Valid embedding strategies
    # - "pooling": Visual tokens only + tile-level pooling (NOVEL)
    # - "standard": All tokens + global mean (BASELINE)
    # - "all": Embed once, push BOTH representations (efficient comparison)
    STRATEGIES = ["pooling", "standard", "all"]
    
    def __init__(
        self,
        embedder=None,
        indexer=None,
        cloudinary_uploader=None,
        pdf_processor=None,
        metadata_mapping: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        embedding_strategy: str = "pooling",
    ):
        self.embedder = embedder
        self.indexer = indexer
        self.cloudinary_uploader = cloudinary_uploader
        self.metadata_mapping = metadata_mapping or {}
        self.config = config or {}
        
        # Validate and set embedding strategy
        if embedding_strategy not in self.STRATEGIES:
            raise ValueError(
                f"Invalid embedding_strategy: {embedding_strategy}. "
                f"Must be one of: {self.STRATEGIES}"
            )
        self.embedding_strategy = embedding_strategy
        
        logger.info(f"📊 Embedding strategy: {embedding_strategy}")
        if embedding_strategy == "pooling":
            logger.info("   → Visual tokens only + tile-level mean pooling (NOVEL)")
        else:
            logger.info("   → All tokens as-is (BASELINE)")
        
        # Create PDF processor if not provided
        if pdf_processor is None:
            from visual_rag.indexing.pdf_processor import PDFProcessor
            dpi = self.config.get("processing", {}).get("dpi", 140)
            pdf_processor = PDFProcessor(dpi=dpi)
        self.pdf_processor = pdf_processor
        
        # Config defaults
        self.embedding_batch_size = self.config.get("batching", {}).get("embedding_batch_size", 8)
        self.upload_batch_size = self.config.get("batching", {}).get("upload_batch_size", 8)
        self.delay_between_uploads = self.config.get("delays", {}).get("between_uploads", 0.5)
    
    def process_pdf(
        self,
        pdf_path: Path,
        skip_existing: bool = True,
        upload_to_cloudinary: bool = True,
        upload_to_qdrant: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single PDF end-to-end.
        
        Args:
            pdf_path: Path to PDF file
            skip_existing: Skip pages that already exist in Qdrant
            upload_to_cloudinary: Upload images to Cloudinary
            upload_to_qdrant: Upload embeddings to Qdrant
        
        Returns:
            Dict with processing results:
            {
                "filename": str,
                "total_pages": int,
                "uploaded": int,
                "skipped": int,
                "failed": int,
                "pages": [...],  # Page data with embeddings and metadata
            }
        """
        pdf_path = Path(pdf_path)
        logger.info(f"📚 Processing PDF: {pdf_path.name}")
        
        # Check existing pages
        existing_ids: Set[str] = set()
        if skip_existing and self.indexer:
            existing_ids = self.indexer.get_existing_ids(pdf_path.name)
            if existing_ids:
                logger.info(f"   Found {len(existing_ids)} existing pages")
        
        # Convert PDF to images
        logger.info(f"🖼️ Converting PDF to images...")
        images, texts = self.pdf_processor.process_pdf(pdf_path)
        total_pages = len(images)
        logger.info(f"   ✅ Converted {total_pages} pages")
        
        # Get extra metadata
        extra_metadata = self._get_extra_metadata(pdf_path.name)
        if extra_metadata:
            logger.info(f"   📋 Found extra metadata: {list(extra_metadata.keys())}")
        
        # Process in batches
        uploaded = 0
        skipped = 0
        failed = 0
        all_pages = []
        upload_queue = []
        
        for batch_start in range(0, total_pages, self.embedding_batch_size):
            batch_end = min(batch_start + self.embedding_batch_size, total_pages)
            batch_images = images[batch_start:batch_end]
            batch_texts = texts[batch_start:batch_end]
            
            logger.info(f"📦 Processing pages {batch_start + 1}-{batch_end}/{total_pages}")
            
            # Filter pages that need processing
            pages_to_process = []
            for i, (img, text) in enumerate(zip(batch_images, batch_texts)):
                page_num = batch_start + i + 1
                chunk_id = self.generate_chunk_id(pdf_path.name, page_num)
                
                if skip_existing and chunk_id in existing_ids:
                    skipped += 1
                    continue
                
                pages_to_process.append({
                    "index": i,
                    "page_num": page_num,
                    "chunk_id": chunk_id,
                    "image": img,
                    "text": text,
                })
            
            if not pages_to_process:
                logger.info("   All pages in batch exist, skipping...")
                continue
            
            # Generate embeddings with token info
            logger.info(f"🤖 Generating embeddings for {len(pages_to_process)} pages...")
            images_to_embed = [p["image"] for p in pages_to_process]
            
            embeddings, token_infos = self.embedder.embed_images(
                images_to_embed,
                batch_size=self.embedding_batch_size,
                return_token_info=True,
                show_progress=True,
            )
            
            # Process each page
            for idx, page_info in enumerate(pages_to_process):
                orig_img = page_info["image"]
                page_num = page_info["page_num"]
                chunk_id = page_info["chunk_id"]
                text = page_info["text"]
                embedding = embeddings[idx]
                token_info = token_infos[idx]
                
                try:
                    page_data = self._process_single_page(
                        pdf_path=pdf_path,
                        page_num=page_num,
                        chunk_id=chunk_id,
                        total_pages=total_pages,
                        orig_img=orig_img,
                        text=text,
                        embedding=embedding,
                        token_info=token_info,
                        extra_metadata=extra_metadata,
                        upload_to_cloudinary=upload_to_cloudinary,
                    )
                    
                    all_pages.append(page_data)
                    
                    if upload_to_qdrant and self.indexer:
                        upload_queue.append(page_data)
                        
                        # Upload in batches
                        if len(upload_queue) >= self.upload_batch_size:
                            count = self._upload_batch(upload_queue)
                            uploaded += count
                            upload_queue = []
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed page {page_num}: {e}")
                    failed += 1
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Upload remaining pages
        if upload_queue and upload_to_qdrant and self.indexer:
            count = self._upload_batch(upload_queue)
            uploaded += count
        
        logger.info(f"✅ Completed {pdf_path.name}: {uploaded} uploaded, {skipped} skipped, {failed} failed")
        
        return {
            "filename": pdf_path.name,
            "total_pages": total_pages,
            "uploaded": uploaded,
            "skipped": skipped,
            "failed": failed,
            "pages": all_pages,
        }
    
    def _process_single_page(
        self,
        pdf_path: Path,
        page_num: int,
        chunk_id: str,
        total_pages: int,
        orig_img,
        text: str,
        embedding: torch.Tensor,
        token_info: Dict[str, Any],
        extra_metadata: Dict[str, Any],
        upload_to_cloudinary: bool = True,
    ) -> Dict[str, Any]:
        """Process a single page with full metadata for saliency."""
        from visual_rag.embedding.pooling import tile_level_mean_pooling, global_mean_pooling
        
        # Resize image for ColPali
        resized_img, tile_rows, tile_cols = self.pdf_processor.resize_for_colpali(orig_img)
        
        # Use processor's tile info if available (more accurate)
        proc_n_rows = token_info.get("n_rows")
        proc_n_cols = token_info.get("n_cols")
        if proc_n_rows and proc_n_cols:
            tile_rows = proc_n_rows
            tile_cols = proc_n_cols
        
        # Convert embedding to numpy
        if isinstance(embedding, torch.Tensor):
            if embedding.dtype == torch.bfloat16:
                full_embedding = embedding.cpu().float().numpy()
            else:
                full_embedding = embedding.cpu().numpy()
        else:
            full_embedding = np.array(embedding)
        full_embedding = full_embedding.astype(np.float32)
        
        # Token info for metadata
        visual_indices = token_info["visual_token_indices"]
        num_visual_tokens = token_info["num_visual_tokens"]
        
        # Compute tile structure
        num_tiles = tile_rows * tile_cols + 1  # +1 for global tile
        patches_per_tile = 64
        
        # =========================================================================
        # STRATEGY: "pooling" (NOVEL) vs "standard" (BASELINE) vs "all" (BOTH)
        # =========================================================================
        
        # Always compute visual-only embedding (needed for pooling and saliency)
        visual_embedding = full_embedding[visual_indices]
        
        # Compute tile-level mean pooling: [num_visual_tokens, 128] → [num_tiles, 128]
        tile_pooled = tile_level_mean_pooling(visual_embedding, num_tiles, patches_per_tile)
        
        # Compute global mean (baseline pooling)
        global_pooled = global_mean_pooling(full_embedding)
        
        if self.embedding_strategy == "pooling":
            # NOVEL APPROACH: Visual tokens only + tile-level pooling
            embedding_for_initial = visual_embedding
            embedding_for_pooling = tile_pooled
            
        elif self.embedding_strategy == "standard":
            # BASELINE: All tokens + global mean
            embedding_for_initial = full_embedding
            embedding_for_pooling = global_pooled.reshape(1, -1)
            
        else:  # "all" - Push BOTH representations (efficient for comparison)
            # Embed once, store multiple vector representations
            # This allows comparing both strategies without re-embedding
            embedding_for_initial = visual_embedding  # Use visual for search
            embedding_for_pooling = tile_pooled       # Use tile-level for fast prefetch
            
            # ALSO store standard representations as additional vectors
            # These will be added to metadata for optional use
            pass  # Extra vectors handled in return dict below
        
        # Upload to Cloudinary
        original_url = None
        resized_url = None
        
        if upload_to_cloudinary and self.cloudinary_uploader:
            base_filename = f"{pdf_path.stem}_page_{page_num}"
            original_url, resized_url = self.cloudinary_uploader.upload_original_and_resized(
                orig_img, resized_img, base_filename
            )
        
        # Sanitize text
        safe_text = self._sanitize_text(text[:10000]) if text else ""
        
        # Build metadata (everything needed for saliency)
        metadata = {
            # Document info
            "filename": pdf_path.name,
            "page_number": page_num,
            "total_pages": total_pages,
            "has_text": bool(text and text.strip()),
            "text": safe_text,
            
            # Image URLs
            "page": resized_url or "",  # For display
            "original_url": original_url or "",
            "resized_url": resized_url or "",
            
            # Dimensions (needed for saliency overlay)
            "original_width": orig_img.width,
            "original_height": orig_img.height,
            "resized_width": resized_img.width,
            "resized_height": resized_img.height,
            
            # Tile structure (needed for saliency)
            "num_tiles": num_tiles,
            "tile_rows": tile_rows,
            "tile_cols": tile_cols,
            "patches_per_tile": patches_per_tile,
            
            # Token info (needed for saliency)
            "num_visual_tokens": num_visual_tokens,
            "visual_token_indices": visual_indices,
            "total_tokens": len(full_embedding),  # Total tokens in raw embedding
            
            # Strategy used (important for paper comparison)
            "embedding_strategy": self.embedding_strategy,
            
            # Extra metadata (year, district, etc.)
            **extra_metadata,
        }
        
        result = {
            "id": chunk_id,
            "visual_embedding": embedding_for_initial,    # "initial" vector in Qdrant
            "tile_pooled_embedding": embedding_for_pooling,  # "mean_pooling" vector in Qdrant
            "metadata": metadata,
            "image": orig_img,
            "resized_image": resized_img,
        }
        
        # For "all" strategy, include BOTH representations for comparison
        if self.embedding_strategy == "all":
            result["extra_vectors"] = {
                # Standard baseline vectors (for comparison)
                "full_embedding": full_embedding,           # All tokens [total, 128]
                "global_pooled": global_pooled,             # Global mean [128]
                # Pooling vectors (already in main result)
                "visual_embedding": visual_embedding,       # Visual only [visual, 128]  
                "tile_pooled": tile_pooled,                 # Tile-level [tiles, 128]
            }
        
        return result
    
    def _upload_batch(self, upload_queue: List[Dict[str, Any]]) -> int:
        """Upload batch to Qdrant."""
        if not upload_queue or not self.indexer:
            return 0
        
        logger.info(f"📤 Uploading batch of {len(upload_queue)} pages...")
        
        count = self.indexer.upload_batch(
            upload_queue,
            delay_between_batches=self.delay_between_uploads,
        )
        
        return count
    
    def _get_extra_metadata(self, filename: str) -> Dict[str, Any]:
        """Get extra metadata for a filename."""
        if not self.metadata_mapping:
            return {}
        
        # Normalize filename
        filename_clean = filename.replace(".pdf", "").replace(".PDF", "").strip().lower()
        
        # Try exact match
        if filename_clean in self.metadata_mapping:
            return self.metadata_mapping[filename_clean].copy()
        
        # Try fuzzy match
        from difflib import SequenceMatcher
        
        best_match = None
        best_score = 0.0
        
        for known_filename, metadata in self.metadata_mapping.items():
            score = SequenceMatcher(None, filename_clean, known_filename.lower()).ratio()
            if score > best_score and score > 0.75:
                best_score = score
                best_match = metadata
        
        if best_match:
            logger.debug(f"Fuzzy matched '{filename}' with score {best_score:.2f}")
            return best_match.copy()
        
        return {}
    
    def _sanitize_text(self, text: str) -> str:
        """Remove invalid Unicode characters."""
        if not text:
            return ""
        return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    
    @staticmethod
    def generate_chunk_id(filename: str, page_number: int) -> str:
        """Generate deterministic chunk ID."""
        content = f"{filename}:page:{page_number}"
        hash_obj = hashlib.sha256(content.encode())
        hex_str = hash_obj.hexdigest()[:32]
        return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
    
    @staticmethod
    def load_metadata_mapping(json_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Load metadata mapping from JSON file.
        
        Expected format:
        {
            "filenames": {
                "Report Name 2023": {"year": 2023, "source": "Local Government", ...},
                ...
            }
        }
        
        Or simple format:
        {
            "Report Name 2023": {"year": 2023, "source": "Local Government", ...},
            ...
        }
        """
        import json
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Check if nested under "filenames"
        if "filenames" in data and isinstance(data["filenames"], dict):
            mapping = data["filenames"]
        else:
            mapping = data
        
        # Normalize keys to lowercase
        normalized = {}
        for filename, metadata in mapping.items():
            key = filename.lower().strip().replace(".pdf", "")
            normalized[key] = metadata
        
        logger.info(f"📖 Loaded metadata for {len(normalized)} files")
        return normalized

