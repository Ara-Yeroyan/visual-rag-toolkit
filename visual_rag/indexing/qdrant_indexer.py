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
from urllib.parse import urlparse
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
        vector_datatype: str = "float32",
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
        if vector_datatype not in ("float32", "float16"):
            raise ValueError("vector_datatype must be 'float32' or 'float16'")
        self.vector_datatype = vector_datatype
        self._np_vector_dtype = np.float16 if vector_datatype == "float16" else np.float32

        grpc_port = None
        if prefer_grpc:
            try:
                parsed = urlparse(url)
                port = parsed.port
                if port == 6333:
                    grpc_port = 6334
            except Exception:
                grpc_port = None
        
        def _make_client(use_grpc: bool):
            return QdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout,
                prefer_grpc=bool(use_grpc),
                grpc_port=grpc_port,
                check_compatibility=False,
            )

        self.client = _make_client(prefer_grpc)
        if prefer_grpc:
            try:
                _ = self.client.get_collections()
            except Exception as e:
                msg = str(e)
                if "StatusCode.PERMISSION_DENIED" in msg or "http2 header with status: 403" in msg:
                    self.client = _make_client(False)
                else:
                    raise
        
        logger.info(f"🔌 Connected to Qdrant: {url}")
        logger.info(f"   Collection: {collection_name}")
        logger.info(f"   Vector datatype: {self.vector_datatype}")
    
    def collection_exists(self) -> bool:
        """Check if collection exists."""
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)
    
    def create_collection(
        self,
        embedding_dim: int = 128,
        force_recreate: bool = False,
        enable_quantization: bool = False,
        indexing_threshold: int = 20000,
        full_scan_threshold: int = 0,
    ) -> bool:
        """
        Create collection with multi-vector support.
        
        Creates named vectors:
        - initial: Full multi-vector embeddings (num_patches × dim)
        - mean_pooling: Tile-level pooled vectors (num_tiles × dim)
        - experimental_pooling: Experimental multi-vector pooling (varies by model)
        - global_pooling: Single vector pooled representation (dim)
        
        Args:
            embedding_dim: Embedding dimension (128 for ColSmol)
            force_recreate: Delete and recreate if exists
            enable_quantization: Enable int8 quantization
            indexing_threshold: Qdrant optimizer indexing threshold (set 0 to always build ANN indexes)
        
        Returns:
            True if created, False if already existed
        """
        from qdrant_client.http import models
        from qdrant_client.http.models import Distance, VectorParams
        
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
        
        # Vector configs - simplified for compatibility
        datatype = models.Datatype.FLOAT16 if self.vector_datatype == "float16" else models.Datatype.FLOAT32
        vectors_config = {
            "initial": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=True,
                multivector_config=multivector_config,
                datatype=datatype,
            ),
            "mean_pooling": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=False,
                multivector_config=multivector_config,
                datatype=datatype,
            ),
            "experimental_pooling": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=False,
                multivector_config=multivector_config,
                datatype=datatype,
            ),
            "global_pooling": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=False,
                datatype=datatype,
            ),
        }
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
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
        
        if not fields:
            return
        
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
        wait: bool = True,
        stop_event=None,
    ) -> int:
        """
        Upload a batch of points to Qdrant.
        
        Each point should have:
        - id: Unique point ID (string or UUID)
        - visual_embedding: Full embedding [num_patches, dim]
        - tile_pooled_embedding: Pooled embedding [num_tiles, dim]
        - experimental_pooled_embedding: Experimental pooled embedding [*, dim]
        - global_pooled_embedding: Pooled embedding [dim]
        - metadata: Payload dict
        
        Args:
            points: List of point dicts
            max_retries: Retry attempts on failure
            delay_between_batches: Delay after upload
            wait: Wait for operation to complete on Qdrant server
            stop_event: Optional threading.Event used to cancel uploads early
        
        Returns:
            Number of successfully uploaded points
        """
        from qdrant_client.http import models
        
        if not points:
            return 0

        def _is_cancelled() -> bool:
            return stop_event is not None and getattr(stop_event, "is_set", lambda: False)()
        
        def _is_payload_too_large_error(e: Exception) -> bool:
            msg = str(e)
            if ("JSON payload" in msg and "larger than allowed" in msg) or ("Payload error:" in msg and "limit:" in msg):
                return True
            content = getattr(e, "content", None)
            if content is not None:
                try:
                    if isinstance(content, (bytes, bytearray)):
                        text = content.decode("utf-8", errors="ignore")
                    else:
                        text = str(content)
                except Exception:
                    text = ""
                if ("JSON payload" in text and "larger than allowed" in text) or ("Payload error" in text and "limit" in text):
                    return True
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    text = str(getattr(resp, "text", "") or "")
                except Exception:
                    text = ""
                if ("JSON payload" in text and "larger than allowed" in text) or ("Payload error" in text and "limit" in text):
                    return True
            return False

        def _to_list(val):
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

        def _build_qdrant_points(batch_points: List[Dict[str, Any]]) -> List[models.PointStruct]:
            qdrant_points: List[models.PointStruct] = []
            for p in batch_points:
                global_pooled = p.get("global_pooled_embedding")
                if global_pooled is None:
                    tile_pooled = np.array(p["tile_pooled_embedding"], dtype=np.float32)
                    global_pooled = tile_pooled.mean(axis=0)
                global_pooled = np.array(global_pooled, dtype=np.float32).reshape(-1)

                initial = np.array(p["visual_embedding"], dtype=np.float32).astype(self._np_vector_dtype, copy=False)
                mean_pooling = np.array(p["tile_pooled_embedding"], dtype=np.float32).astype(self._np_vector_dtype, copy=False)
                experimental_pooling = np.array(p["experimental_pooled_embedding"], dtype=np.float32).astype(
                    self._np_vector_dtype, copy=False
                )
                global_pooling = global_pooled.astype(self._np_vector_dtype, copy=False)

                qdrant_points.append(
                    models.PointStruct(
                        id=p["id"],
                        vector={
                            "initial": _to_list(initial),
                            "mean_pooling": _to_list(mean_pooling),
                            "experimental_pooling": _to_list(experimental_pooling),
                            "global_pooling": _to_list(global_pooling),
                        },
                        payload=p["metadata"],
                    )
                )
            return qdrant_points
        
        # Upload with retry
        for attempt in range(max_retries):
            try:
                if _is_cancelled():
                    return 0
                qdrant_points = _build_qdrant_points(points)
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=qdrant_points,
                    wait=wait,
                )

                if delay_between_batches > 0:
                    if _is_cancelled():
                        return 0
                    time.sleep(delay_between_batches)

                return len(points)

            except Exception as e:
                if _is_payload_too_large_error(e) and len(points) > 1:
                    mid = len(points) // 2
                    left = points[:mid]
                    right = points[mid:]
                    logger.warning(
                        f"Upload payload too large for {len(points)} points; splitting into {len(left)} + {len(right)}"
                    )
                    return self.upload_batch(
                        left,
                        max_retries=max_retries,
                        delay_between_batches=delay_between_batches,
                        wait=wait,
                        stop_event=stop_event,
                    ) + self.upload_batch(
                        right,
                        max_retries=max_retries,
                        delay_between_batches=delay_between_batches,
                        wait=wait,
                        stop_event=stop_event,
                    )

                logger.warning(f"Upload attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    if _is_cancelled():
                        return 0
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


