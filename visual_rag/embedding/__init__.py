"""
Embedding module - Visual and text embedding generation.

Provides:
- VisualEmbedder: Backend-agnostic visual embedder (ColPali, etc.)
- Pooling utilities: tile-level, global, MaxSim scoring
"""

from visual_rag.embedding.visual_embedder import VisualEmbedder, ColPaliEmbedder
from visual_rag.embedding.pooling import (
    tile_level_mean_pooling,
    global_mean_pooling,
    compute_maxsim_score,
    compute_maxsim_batch,
)

__all__ = [
    # Main embedder
    "VisualEmbedder",
    "ColPaliEmbedder",  # Backward compatibility alias
    
    # Pooling functions
    "tile_level_mean_pooling",
    "global_mean_pooling",
    "compute_maxsim_score",
    "compute_maxsim_batch",
]
