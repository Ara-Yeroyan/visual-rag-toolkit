"""
Pooling strategies for multi-vector embeddings.

Provides:
- Tile-level mean pooling: Preserves spatial structure (num_tiles × dim)
- Global mean pooling: Single vector (1 × dim)
- MaxSim scoring for ColBERT-style late interaction
"""

import numpy as np
import torch
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def tile_level_mean_pooling(
    embedding: Union[torch.Tensor, np.ndarray],
    num_tiles: int,
    patches_per_tile: int = 64,
) -> np.ndarray:
    """
    Compute tile-level mean pooling for multi-vector embeddings.
    
    Instead of collapsing to 1×dim (global pooling), this preserves spatial
    structure by computing mean per tile → num_tiles × dim.
    
    This is our NOVEL contribution for scalable visual retrieval:
    - Faster than full MaxSim (fewer vectors to compare)
    - More accurate than global pooling (preserves spatial info)
    - Ideal for two-stage retrieval (prefetch with pooled, rerank with full)
    
    Args:
        embedding: Visual token embeddings [num_visual_tokens, dim]
        num_tiles: Number of tiles (including global tile)
        patches_per_tile: Patches per tile (64 for ColSmol)
    
    Returns:
        Tile-level pooled embeddings [num_tiles, dim]
    
    Example:
        >>> # Image with 4×3 tiles + 1 global = 13 tiles
        >>> # Each tile has 64 patches → 832 visual tokens
        >>> pooled = tile_level_mean_pooling(embedding, num_tiles=13)
        >>> print(pooled.shape)  # (13, 128)
    """
    # Convert to numpy
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy().astype(np.float32)
    else:
        emb_np = np.array(embedding, dtype=np.float32)
    
    num_visual_tokens = emb_np.shape[0]
    expected_tokens = num_tiles * patches_per_tile
    
    # Handle mismatch (can happen with global tile variations)
    if num_visual_tokens != expected_tokens:
        logger.debug(
            f"Token count mismatch: {num_visual_tokens} vs expected {expected_tokens}"
        )
        actual_tiles = num_visual_tokens // patches_per_tile
        if actual_tiles * patches_per_tile != num_visual_tokens:
            actual_tiles += 1  # Include partial tile
        num_tiles = actual_tiles
    
    # Compute mean per tile
    tile_embeddings = []
    for tile_idx in range(num_tiles):
        start_idx = tile_idx * patches_per_tile
        end_idx = min(start_idx + patches_per_tile, num_visual_tokens)
        
        if start_idx >= num_visual_tokens:
            break
        
        tile_patches = emb_np[start_idx:end_idx]
        tile_mean = tile_patches.mean(axis=0)
        tile_embeddings.append(tile_mean)
    
    return np.array(tile_embeddings, dtype=np.float32)


def global_mean_pooling(
    embedding: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """
    Compute global mean pooling → single vector.
    
    This is the simplest pooling but loses all spatial information.
    Use for fastest retrieval when accuracy can be sacrificed.
    
    Args:
        embedding: Multi-vector embeddings [num_tokens, dim]
    
    Returns:
        Pooled vector [dim]
    """
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy()
    else:
        emb_np = np.array(embedding)
    
    return emb_np.mean(axis=0).astype(np.float32)


def compute_maxsim_score(
    query_embedding: np.ndarray,
    doc_embedding: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Compute ColBERT-style MaxSim late interaction score.
    
    For each query token, finds max similarity with any document token,
    then sums across query tokens.
    
    This is the standard scoring for ColBERT/ColPali:
    score = Σ_q max_d (sim(q, d))
    
    Args:
        query_embedding: Query embeddings [num_query_tokens, dim]
        doc_embedding: Document embeddings [num_doc_tokens, dim]
        normalize: L2 normalize embeddings before scoring (recommended)
    
    Returns:
        MaxSim score (higher is better)
    
    Example:
        >>> query = embedder.embed_query("budget allocation")
        >>> doc = embeddings[0]  # From embed_images
        >>> score = compute_maxsim_score(query, doc)
    """
    if normalize:
        # L2 normalize
        query_norm = query_embedding / (
            np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8
        )
        doc_norm = doc_embedding / (
            np.linalg.norm(doc_embedding, axis=1, keepdims=True) + 1e-8
        )
    else:
        query_norm = query_embedding
        doc_norm = doc_embedding
    
    # Compute similarity matrix: [num_query, num_doc]
    similarity_matrix = np.dot(query_norm, doc_norm.T)
    
    # MaxSim: For each query token, take max similarity with any doc token
    max_similarities = similarity_matrix.max(axis=1)
    
    # Sum across query tokens
    score = float(max_similarities.sum())
    
    return score


def compute_maxsim_batch(
    query_embedding: np.ndarray,
    doc_embeddings: list,
    normalize: bool = True,
) -> list:
    """
    Compute MaxSim scores for multiple documents efficiently.
    
    Args:
        query_embedding: Query embeddings [num_query_tokens, dim]
        doc_embeddings: List of document embeddings
        normalize: L2 normalize embeddings
    
    Returns:
        List of MaxSim scores
    """
    # Pre-normalize query once
    if normalize:
        query_norm = query_embedding / (
            np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8
        )
    else:
        query_norm = query_embedding
    
    scores = []
    for doc_emb in doc_embeddings:
        if normalize:
            doc_norm = doc_emb / (
                np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8
            )
        else:
            doc_norm = doc_emb
        
        sim_matrix = np.dot(query_norm, doc_norm.T)
        max_sims = sim_matrix.max(axis=1)
        scores.append(float(max_sims.sum()))
    
    return scores


