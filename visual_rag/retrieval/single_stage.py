"""
Single-Stage Retrieval for Visual Document Search.

Provides direct search without the two-stage complexity.
Use when:
- Collection is small (<10K documents)
- Latency is not critical
- Maximum accuracy is required
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SingleStageRetriever:
    """
    Single-stage visual document retrieval using native Qdrant search.
    
    Supports three strategies:
    1. multi_vector (SOTA): Native MaxSim on full embeddings
    2. pooled_tile: Search on tile-level pooled vectors
    3. pooled_global: Search on globally pooled vectors
    
    Args:
        qdrant_client: Connected Qdrant client
        collection_name: Name of the Qdrant collection
    
    Example:
        >>> retriever = SingleStageRetriever(client, "my_collection")
        >>> results = retriever.search(query, top_k=10)
    """
    
    def __init__(
        self,
        qdrant_client,
        collection_name: str,
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
    
    def search(
        self,
        query_embedding: Union[torch.Tensor, np.ndarray],
        top_k: int = 10,
        strategy: str = "multi_vector",
        filter_obj=None,
    ) -> List[Dict[str, Any]]:
        """
        Single-stage search with configurable strategy.
        
        Args:
            query_embedding: Query embeddings [num_tokens, dim]
            top_k: Number of results
            strategy: "multi_vector", "pooled_tile", or "pooled_global"
            filter_obj: Qdrant filter
        
        Returns:
            List of results with scores and metadata
        """
        query_np = self._to_numpy(query_embedding)
        
        if strategy == "multi_vector":
            # Native multi-vector MaxSim
            vector_name = "initial"
            query_vector = query_np.tolist()
            logger.info(f"🎯 Multi-vector search on '{vector_name}'")
            
        elif strategy == "pooled_tile":
            # Tile-level pooled
            vector_name = "mean_pooling"
            query_pooled = query_np.mean(axis=0)
            query_vector = query_pooled.tolist()
            logger.info(f"🔍 Tile-pooled search on '{vector_name}'")
            
        elif strategy == "pooled_global":
            # Global mean pooling (single vector)
            vector_name = "mean_pooling"
            query_pooled = query_np.mean(axis=0)
            query_vector = query_pooled.tolist()
            logger.info(f"🔍 Global-pooled search on '{vector_name}'")
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=vector_name,
            query_filter=filter_obj,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            timeout=120,
        ).points
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "score_final": r.score,
                "payload": r.payload,
            }
            for r in results
        ]
    
    def _to_numpy(self, embedding: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert embedding to numpy array."""
        if isinstance(embedding, torch.Tensor):
            if embedding.dtype == torch.bfloat16:
                return embedding.cpu().float().numpy()
            return embedding.cpu().numpy()
        return np.array(embedding, dtype=np.float32)


