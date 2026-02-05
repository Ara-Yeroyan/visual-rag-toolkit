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


def _infer_output_dtype(
    embedding: Union[torch.Tensor, np.ndarray],
    output_dtype: Optional[np.dtype] = None,
) -> np.dtype:
    """Infer output dtype: use provided, else match input (fp16→fp16, bf16→fp32, fp32→fp32)."""
    if output_dtype is not None:
        return output_dtype
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.float16:
            return np.float16
        return np.float32
    if isinstance(embedding, np.ndarray) and embedding.dtype == np.float16:
        return np.float16
    return np.float32


def tile_level_mean_pooling(
    embedding: Union[torch.Tensor, np.ndarray],
    num_tiles: int,
    patches_per_tile: int = 64,
    output_dtype: Optional[np.dtype] = None,
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
        output_dtype: Output dtype (default: infer from input, fp16→fp16, bf16→fp32)
    
    Returns:
        Tile-level pooled embeddings [num_tiles, dim]
    
    Example:
        >>> # Image with 4×3 tiles + 1 global = 13 tiles
        >>> # Each tile has 64 patches → 832 visual tokens
        >>> pooled = tile_level_mean_pooling(embedding, num_tiles=13)
        >>> print(pooled.shape)  # (13, 128)
    """
    out_dtype = _infer_output_dtype(embedding, output_dtype)
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy().astype(np.float32)
    else:
        emb_np = np.array(embedding, dtype=np.float32)
    
    num_visual_tokens = emb_np.shape[0]
    expected_tokens = num_tiles * patches_per_tile
    
    if num_visual_tokens != expected_tokens:
        logger.debug(
            f"Token count mismatch: {num_visual_tokens} vs expected {expected_tokens}"
        )
        actual_tiles = num_visual_tokens // patches_per_tile
        if actual_tiles * patches_per_tile != num_visual_tokens:
            actual_tiles += 1
        num_tiles = actual_tiles
    
    tile_embeddings = []
    for tile_idx in range(num_tiles):
        start_idx = tile_idx * patches_per_tile
        end_idx = min(start_idx + patches_per_tile, num_visual_tokens)
        
        if start_idx >= num_visual_tokens:
            break
        
        tile_patches = emb_np[start_idx:end_idx]
        tile_mean = tile_patches.mean(axis=0)
        tile_embeddings.append(tile_mean)
    
    return np.array(tile_embeddings, dtype=out_dtype)


def colpali_row_mean_pooling(
    embedding: Union[torch.Tensor, np.ndarray],
    grid_size: int = 32,
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    out_dtype = _infer_output_dtype(embedding, output_dtype)
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy().astype(np.float32)
    else:
        emb_np = np.array(embedding, dtype=np.float32)

    num_tokens, dim = emb_np.shape
    expected = int(grid_size) * int(grid_size)
    if num_tokens != expected:
        raise ValueError(f"Expected {expected} visual tokens for grid_size={grid_size}, got {num_tokens}")

    grid = emb_np.reshape(int(grid_size), int(grid_size), int(dim))
    pooled = grid.mean(axis=1)
    return pooled.astype(out_dtype)


def colsmol_experimental_pooling(
    embedding: Union[torch.Tensor, np.ndarray],
    num_tiles: int,
    patches_per_tile: int = 64,
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    out_dtype = _infer_output_dtype(embedding, output_dtype)
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy().astype(np.float32)
    else:
        emb_np = np.array(embedding, dtype=np.float32)

    num_visual_tokens, dim = emb_np.shape
    if num_tiles <= 0:
        raise ValueError("num_tiles must be > 0")
    if patches_per_tile <= 0:
        raise ValueError("patches_per_tile must be > 0")

    last_tile_start = (int(num_tiles) - 1) * int(patches_per_tile)
    if last_tile_start >= num_visual_tokens:
        actual_tiles = int(num_visual_tokens) // int(patches_per_tile)
        if actual_tiles * int(patches_per_tile) != int(num_visual_tokens):
            actual_tiles += 1
        if actual_tiles <= 0:
            raise ValueError(
                f"Not enough tokens for num_tiles={num_tiles}, patches_per_tile={patches_per_tile}: got {num_visual_tokens}"
            )
        num_tiles = actual_tiles
        last_tile_start = (int(num_tiles) - 1) * int(patches_per_tile)

    prefix = emb_np[:last_tile_start]
    last_tile = emb_np[last_tile_start : min(last_tile_start + int(patches_per_tile), num_visual_tokens)]

    if prefix.size:
        prefix_tiles = prefix.reshape(-1, int(patches_per_tile), int(dim))
        prefix_means = prefix_tiles.mean(axis=1)
    else:
        prefix_means = np.zeros((0, int(dim)), dtype=out_dtype)

    return np.concatenate([prefix_means.astype(out_dtype), last_tile.astype(out_dtype)], axis=0)


def colpali_experimental_pooling_from_rows(
    row_vectors: Union[torch.Tensor, np.ndarray],
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Experimental "convolution-style" pooling with window size 3.
    
    For N input rows, produces N + 2 output vectors:
    - Position 0: row[0] alone (1 row)
    - Position 1: mean(rows[0:2]) (2 rows)
    - Position 2: mean(rows[0:3]) (3 rows)
    - Positions 3 to N-1: sliding window of 3 (rows[i-2:i+1])
    - Position N: mean(rows[N-2:N]) (last 2 rows)
    - Position N+1: row[N-1] alone (last row)
    
    For N=32 rows: produces 34 vectors.
    """
    out_dtype = _infer_output_dtype(row_vectors, output_dtype)
    if isinstance(row_vectors, torch.Tensor):
        if row_vectors.dtype == torch.bfloat16:
            rows = row_vectors.cpu().float().numpy()
        else:
            rows = row_vectors.cpu().numpy().astype(np.float32)
    else:
        rows = np.array(row_vectors, dtype=np.float32)

    n, dim = rows.shape
    if n < 1:
        raise ValueError("row_vectors must be non-empty")
    if n == 1:
        return rows.astype(out_dtype)
    if n == 2:
        return np.stack([rows[0], rows[:2].mean(axis=0), rows[1]], axis=0).astype(out_dtype)
    if n == 3:
        return np.stack([
            rows[0],
            rows[:2].mean(axis=0),
            rows[:3].mean(axis=0),
            rows[1:3].mean(axis=0),
            rows[2],
        ], axis=0).astype(out_dtype)

    out = np.zeros((n + 2, dim), dtype=np.float32)
    out[0] = rows[0]
    out[1] = rows[:2].mean(axis=0)
    out[2] = rows[:3].mean(axis=0)
    for i in range(3, n):
        out[i] = rows[i - 2 : i + 1].mean(axis=0)
    out[n] = rows[n - 2 : n].mean(axis=0)
    out[n + 1] = rows[n - 1]
    return out.astype(out_dtype)


def global_mean_pooling(
    embedding: Union[torch.Tensor, np.ndarray],
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Compute global mean pooling → single vector.
    
    This is the simplest pooling but loses all spatial information.
    Use for fastest retrieval when accuracy can be sacrificed.
    
    Args:
        embedding: Multi-vector embeddings [num_tokens, dim]
        output_dtype: Output dtype (default: infer from input, fp16→fp16, bf16→fp32)
    
    Returns:
        Pooled vector [dim]
    """
    out_dtype = _infer_output_dtype(embedding, output_dtype)
    if isinstance(embedding, torch.Tensor):
        if embedding.dtype == torch.bfloat16:
            emb_np = embedding.cpu().float().numpy()
        else:
            emb_np = embedding.cpu().numpy()
    else:
        emb_np = np.array(embedding)
    
    return emb_np.mean(axis=0).astype(out_dtype)


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
