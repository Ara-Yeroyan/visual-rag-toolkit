import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from visual_rag.embedding.visual_embedder import VisualEmbedder
from visual_rag.retrieval.single_stage import SingleStageRetriever
from visual_rag.retrieval.two_stage import TwoStageRetriever
from visual_rag.retrieval.three_stage import ThreeStageRetriever


class MultiVectorRetriever:
    @staticmethod
    def _maybe_load_dotenv() -> None:
        try:
            from dotenv import load_dotenv
        except ImportError:
            return
        if os.path.exists(".env"):
            load_dotenv(".env")

    def __init__(
        self,
        collection_name: str,
        model_name: str = "vidore/colSmol-500M",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        request_timeout: int = 120,
        max_retries: int = 3,
        retry_sleep: float = 0.5,
        qdrant_client=None,
        embedder: Optional[VisualEmbedder] = None,
    ):
        if qdrant_client is None:
            self._maybe_load_dotenv()
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:
                raise ImportError(
                    "Qdrant client not installed. Install with: pip install visual-rag-toolkit[qdrant]"
                ) from e

            qdrant_url = (
                qdrant_url
                or os.getenv("SIGIR_QDRANT_URL")
                or os.getenv("DEST_QDRANT_URL")
                or os.getenv("QDRANT_URL")
            )
            if not qdrant_url:
                raise ValueError(
                    "QDRANT_URL is required (pass qdrant_url or set env var). "
                    "You can also set DEST_QDRANT_URL to override."
                )

            qdrant_api_key = (
                qdrant_api_key
                or os.getenv("SIGIR_QDRANT_KEY")
                or os.getenv("SIGIR_QDRANT_API_KEY")
                or os.getenv("DEST_QDRANT_API_KEY")
                or os.getenv("QDRANT_API_KEY")
            )

            grpc_port = None
            if prefer_grpc:
                try:
                    if urlparse(qdrant_url).port == 6333:
                        grpc_port = 6334
                except Exception:
                    grpc_port = None
            def _make_client(use_grpc: bool):
                return QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    prefer_grpc=bool(use_grpc),
                    grpc_port=grpc_port,
                    timeout=int(request_timeout),
                    check_compatibility=False,
                )

            qdrant_client = _make_client(prefer_grpc)
            if prefer_grpc:
                try:
                    _ = qdrant_client.get_collections()
                except Exception as e:
                    msg = str(e)
                    if "StatusCode.PERMISSION_DENIED" in msg or "http2 header with status: 403" in msg:
                        qdrant_client = _make_client(False)
                    else:
                        raise

        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedder = embedder or VisualEmbedder(model_name=model_name)

        self._two_stage = TwoStageRetriever(
            self.client,
            collection_name=self.collection_name,
            request_timeout=int(request_timeout),
            max_retries=int(max_retries),
            retry_sleep=float(retry_sleep),
        )
        self._three_stage = ThreeStageRetriever(
            self.client,
            collection_name=self.collection_name,
            request_timeout=int(request_timeout),
            max_retries=int(max_retries),
            retry_sleep=float(retry_sleep),
        )
        self._single_stage = SingleStageRetriever(
            self.client,
            collection_name=self.collection_name,
            request_timeout=int(request_timeout),
        )

    def build_filter(
        self,
        year: Optional[Any] = None,
        source: Optional[str] = None,
        district: Optional[str] = None,
        filename: Optional[str] = None,
        has_text: Optional[bool] = None,
    ):
        return self._two_stage.build_filter(
            year=year,
            source=source,
            district=district,
            filename=filename,
            has_text=has_text,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "single_full",
        prefetch_k: Optional[int] = None,
        stage1_mode: str = "pooled_query_vs_tiles",
        filter_obj=None,
        return_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        q = self.embedder.embed_query(query)
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(q, torch.Tensor):
            query_embedding = q.detach().cpu().numpy()
        else:
            query_embedding = q.numpy()

        return self.search_embedded(
            query_embedding=query_embedding,
            top_k=top_k,
            mode=mode,
            prefetch_k=prefetch_k,
            stage1_mode=stage1_mode,
            filter_obj=filter_obj,
            return_embeddings=return_embeddings,
        )

    def search_embedded(
        self,
        *,
        query_embedding,
        top_k: int = 10,
        mode: str = "single_full",
        prefetch_k: Optional[int] = None,
        stage1_mode: str = "pooled_query_vs_tiles",
        stage1_k: Optional[int] = None,
        stage2_k: Optional[int] = None,
        filter_obj=None,
        return_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        if mode == "single_full":
            return self._single_stage.search(
                query_embedding=query_embedding,
                top_k=top_k,
                strategy="multi_vector",
                filter_obj=filter_obj,
            )

        if mode == "single_tiles":
            return self._single_stage.search(
                query_embedding=query_embedding,
                top_k=top_k,
                strategy="tiles_maxsim",
                filter_obj=filter_obj,
            )

        if mode == "single_global":
            return self._single_stage.search(
                query_embedding=query_embedding,
                top_k=top_k,
                strategy="pooled_global",
                filter_obj=filter_obj,
            )

        if mode == "two_stage":
            return self._two_stage.search_server_side(
                query_embedding=query_embedding,
                top_k=top_k,
                prefetch_k=prefetch_k,
                filter_obj=filter_obj,
                stage1_mode=stage1_mode,
            )

        if mode == "three_stage":
            s1 = int(stage1_k) if stage1_k is not None else 1000
            s2 = int(stage2_k) if stage2_k is not None else 300
            return self._three_stage.search_server_side(
                query_embedding=query_embedding,
                top_k=top_k,
                stage1_k=s1,
                stage2_k=s2,
                filter_obj=filter_obj,
            )

        raise ValueError(f"Unknown mode: {mode}")


