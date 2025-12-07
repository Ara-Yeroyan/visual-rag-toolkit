"""Tests for pooling functions."""

import pytest
import numpy as np


class TestTileLevelPooling:
    """Test tile-level mean pooling."""
    
    def test_basic_pooling(self):
        """Pooling reduces [num_tokens, dim] → [num_tiles, dim]."""
        from visual_rag.embedding.pooling import tile_level_mean_pooling
        
        # 13 tiles × 64 patches = 832 visual tokens
        num_tiles = 13
        patches_per_tile = 64
        num_tokens = num_tiles * patches_per_tile
        dim = 128
        
        embedding = np.random.randn(num_tokens, dim).astype(np.float32)
        pooled = tile_level_mean_pooling(embedding, num_tiles, patches_per_tile)
        
        assert pooled.shape == (num_tiles, dim)
        assert pooled.dtype == np.float32
    
    def test_pooling_preserves_info(self):
        """Pooled vectors should be mean of patches."""
        from visual_rag.embedding.pooling import tile_level_mean_pooling
        
        num_tiles = 5
        patches_per_tile = 64
        dim = 128
        
        embedding = np.random.randn(num_tiles * patches_per_tile, dim).astype(np.float32)
        pooled = tile_level_mean_pooling(embedding, num_tiles, patches_per_tile)
        
        # Check first tile
        expected_tile0 = embedding[:patches_per_tile].mean(axis=0)
        np.testing.assert_array_almost_equal(pooled[0], expected_tile0, decimal=5)
    
    def test_pooling_with_partial_last_tile(self):
        """Handle case where last tile has fewer patches."""
        from visual_rag.embedding.pooling import tile_level_mean_pooling
        
        # 800 tokens, 64 per tile = 12.5 tiles → 13 tiles with partial last
        num_tokens = 800
        num_tiles = 13
        dim = 128
        
        embedding = np.random.randn(num_tokens, dim).astype(np.float32)
        pooled = tile_level_mean_pooling(embedding, num_tiles, patches_per_tile=64)
        
        # Should handle gracefully - at least some tiles
        assert pooled.shape[1] == dim
        assert pooled.shape[0] >= 1


class TestGlobalPooling:
    """Test global mean pooling."""
    
    def test_global_mean(self):
        """Global pooling reduces to single vector."""
        from visual_rag.embedding.pooling import global_mean_pooling
        
        embedding = np.random.randn(832, 128).astype(np.float32)
        pooled = global_mean_pooling(embedding)
        
        assert pooled.shape == (128,)
        np.testing.assert_array_almost_equal(pooled, embedding.mean(axis=0))


class TestMaxSimScore:
    """Test MaxSim scoring."""
    
    def test_maxsim_identical(self):
        """Identical embeddings should have high score."""
        from visual_rag.embedding.pooling import compute_maxsim_score
        
        embedding = np.random.randn(10, 128).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        score = compute_maxsim_score(embedding, embedding)
        
        # Should be close to num_tokens (each token matches itself perfectly)
        assert score >= 9.0  # Allow some floating point tolerance
    
    def test_maxsim_orthogonal(self):
        """Orthogonal embeddings should have low score."""
        from visual_rag.embedding.pooling import compute_maxsim_score
        
        # Create orthogonal vectors
        query = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        doc = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        
        score = compute_maxsim_score(query, doc)
        
        assert score < 0.1  # Near zero for orthogonal
    
    def test_maxsim_shape_independence(self):
        """Score should work with different query/doc lengths."""
        from visual_rag.embedding.pooling import compute_maxsim_score
        
        query = np.random.randn(5, 128).astype(np.float32)
        doc = np.random.randn(100, 128).astype(np.float32)
        
        score = compute_maxsim_score(query, doc)
        
        assert isinstance(score, float)
        assert not np.isnan(score)

