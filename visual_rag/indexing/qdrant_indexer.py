"""
Qdrant Indexer - Upload embeddings to Qdrant vector database.

Works INDEPENDENTLY of PDF processing and embedding generation.
Use it if you already have embeddings and just need to upload.

Features:
- Named vectors for multi-vector and pooled search
- Batch uploading with retry logic
- Skip-existing for incremental updates
- Configurable payload indexes
"""

import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)


class QdrantIndexer:
    """
    Upload visual embeddings to Qdrant.
    
    Works independently - just needs embeddings and metadata.
    
    Args:
        url: Qdrant server URL
        api_key: Qdrant API key
        collection_name: Name of the collection
        timeout: Request timeout in seconds
        prefer_grpc: Use gRPC protocol (faster but may have issues)
    
    Example:
        >>> indexer = QdrantIndexer(
        ...     url="https://your-cluster.qdrant.io:6333",
        ...     api_key="your-api-key",
        ...     collection_name="my_collection",
        ... )
        >>> 
        >>> # Create collection
        >>> indexer.create_collection()
        >>> 
        >>> # Upload points
        >>> indexer.upload_batch(points)
    """
    
    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        timeout: int = 60,
        prefer_grpc: bool = False,
    ):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "Qdrant client not installed. "
                "Install with: pip install visual-rag-toolkit[qdrant]"
            )
        
        self.collection_name = collection_name
        self.timeout = timeout
        
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
            prefer_grpc=prefer_grpc,
            check_compatibility=False,
        )
        
        logger.info(f"🔌 Connected to Qdrant: {url}")
        logger.info(f"   Collection: {collection_name}")
    
    def collection_exists(self) -> bool:
        """Check if collection exists."""
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)
    
    def create_collection(
        self,
        embedding_dim: int = 128,
        force_recreate: bool = False,
        enable_quantization: bool = False,
    ) -> bool:
        """
        Create collection with multi-vector support.
        
        Creates two named vectors:
        - initial: Full multi-vector embeddings (num_patches × dim)
        - mean_pooling: Tile-level pooled vectors (num_tiles × dim)
        
        Args:
            embedding_dim: Embedding dimension (128 for ColSmol)
            force_recreate: Delete and recreate if exists
            enable_quantization: Enable int8 quantization
        
        Returns:
            True if created, False if already existed
        """
        from qdrant_client.http import models
        from qdrant_client.http.models import (
            Distance,
            VectorParams,
            OptimizersConfigDiff,
            HnswConfigDiff,
            ScalarQuantizationConfig,
            ScalarType,
        )
        
        if self.collection_exists():
            if force_recreate:
                logger.info(f"🗑️ Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"✅ Collection already exists: {self.collection_name}")
                return False
        
        logger.info(f"📦 Creating collection: {self.collection_name}")
        
        # Multi-vector config for ColBERT-style MaxSim
        multivector_config = models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        )
        
        # HNSW config for pooled vectors
        hnsw_config = HnswConfigDiff(
            m=32,
            ef_construct=100,
            full_scan_threshold=10000,
            on_disk=True,
        )
        
        # Optional quantization
        quantization_config = None
        if enable_quantization:
            logger.info("   Quantization: ENABLED (int8)")
            quantization_config = ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        
        # Vector configs
        vectors_config = {
            "initial": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=True,
                multivector_config=multivector_config,
                quantization_config=quantization_config,
            ),
            "mean_pooling": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=False,  # Keep in RAM for fast prefetch
                multivector_config=multivector_config,
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
            ),
        }
        
        # Optimizer config for low-RAM clusters
        optimizer_config = OptimizersConfigDiff(
            indexing_threshold=20000,  # High threshold to avoid indexing (saves RAM)
            memmap_threshold=0,  # Use mmap immediately
            flush_interval_sec=5,  # Flush WAL frequently
        )
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            optimizers_config=optimizer_config,
        )
        
        logger.info(f"✅ Collection created: {self.collection_name}")
        return True
    
    def create_payload_indexes(
        self,
        fields: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Create payload indexes for filtering.
        
        Args:
            fields: List of {field, type} dicts
                   type can be: integer, keyword, bool, float, text
        """
        from qdrant_client.http import models
        
        type_mapping = {
            "integer": models.PayloadSchemaType.INTEGER,
            "keyword": models.PayloadSchemaType.KEYWORD,
            "bool": models.PayloadSchemaType.BOOL,
            "float": models.PayloadSchemaType.FLOAT,
            "text": models.PayloadSchemaType.TEXT,
        }
        
        # Default fields
        if fields is None:
            fields = [
                {"field": "filename", "type": "keyword"},
                {"field": "page_number", "type": "integer"},
                {"field": "year", "type": "integer"},
                {"field": "source", "type": "keyword"},
                {"field": "district", "type": "keyword"},
                {"field": "has_text", "type": "bool"},
            ]
        
        logger.info("📇 Creating payload indexes...")
        
        for field_config in fields:
            field_name = field_config["field"]
            field_type_str = field_config.get("type", "keyword")
            field_type = type_mapping.get(field_type_str, models.PayloadSchemaType.KEYWORD)
            
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.info(f"   ✅ {field_name} ({field_type_str})")
            except Exception as e:
                logger.debug(f"   Index {field_name} might already exist: {e}")
    
    def upload_batch(
        self,
        points: List[Dict[str, Any]],
        max_retries: int = 3,
        delay_between_batches: float = 0.5,
    ) -> int:
        """
        Upload a batch of points to Qdrant.
        
        Each point should have:
        - id: Unique point ID (string or UUID)
        - visual_embedding: Full embedding [num_patches, dim]
        - tile_pooled_embedding: Pooled embedding [num_tiles, dim]
        - metadata: Payload dict
        
        Args:
            points: List of point dicts
            max_retries: Retry attempts on failure
            delay_between_batches: Delay after upload
        
        Returns:
            Number of successfully uploaded points
        """
        from qdrant_client.http import models
        
        if not points:
            return 0
        
        # Build Qdrant points
        qdrant_points = []
        for point_data in points:
            point = models.PointStruct(
                id=point_data["id"],
                vector={
                    "initial": point_data["visual_embedding"].tolist(),
                    "mean_pooling": point_data["tile_pooled_embedding"].tolist(),
                },
                payload=point_data["metadata"],
            )
            qdrant_points.append(point)
        
        # Upload with retry
        for attempt in range(max_retries):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=qdrant_points,
                    wait=True,
                )
                
                if delay_between_batches > 0:
                    time.sleep(delay_between_batches)
                
                return len(qdrant_points)
                
            except Exception as e:
                logger.warning(
                    f"Upload attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"❌ Upload failed after {max_retries} attempts")
        return 0
    
    def check_exists(self, chunk_id: str) -> bool:
        """Check if a point already exists."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=False,
                with_vectors=False,
            )
            return len(result) > 0
        except Exception:
            return False
    
    def get_existing_ids(self, filename: str) -> Set[str]:
        """Get all point IDs for a specific file."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        existing_ids = set()
        offset = None
        
        while True:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                ),
                limit=100,
                offset=offset,
                with_payload=["page_number"],
                with_vectors=False,
            )
            
            points, next_offset = results
            
            for point in points:
                existing_ids.add(str(point.id))
            
            if next_offset is None or len(points) == 0:
                break
            offset = next_offset
        
        return existing_ids
    
    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            
            status = info.status
            if hasattr(status, "value"):
                status = status.value
            
            indexed_count = getattr(info, "indexed_vectors_count", 0) or 0
            if isinstance(indexed_count, dict):
                indexed_count = sum(indexed_count.values())
            
            return {
                "status": str(status),
                "points_count": getattr(info, "points_count", 0),
                "indexed_vectors_count": indexed_count,
            }
        except Exception as e:
            logger.warning(f"Could not get collection info: {e}")
            return None
    
    @staticmethod
    def generate_point_id(filename: str, page_number: int) -> str:
        """
        Generate deterministic point ID from filename and page.
        
        Returns a valid UUID string.
        """
        content = f"{filename}:page:{page_number}"
        hash_obj = hashlib.sha256(content.encode())
        hex_str = hash_obj.hexdigest()[:32]
        # Format as UUID
        return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


