"""
Retrieval module - Search and retrieval strategies.

Components:
- TwoStageRetriever: Pooled prefetch → MaxSim reranking (our novel contribution)
- SingleStageRetriever: Direct multi-vector or pooled search
"""

from visual_rag.retrieval.two_stage import TwoStageRetriever

__all__ = [
    "TwoStageRetriever",
]
