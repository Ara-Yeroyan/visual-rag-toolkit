from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class QdrantConnection:
    url: str
    api_key: Optional[str]


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    try:
        from pathlib import Path

        if Path(".env").exists():
            load_dotenv(".env")
    except Exception:
        return


def _resolve_qdrant_connection(
    *,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> QdrantConnection:
    import os

    _maybe_load_dotenv()
    resolved_url = url or os.getenv("SIGIR_QDRANT_URL") or os.getenv("DEST_QDRANT_URL") or os.getenv("QDRANT_URL")
    if not resolved_url:
        raise ValueError("Qdrant URL not set (pass url= or set SIGIR_QDRANT_URL/DEST_QDRANT_URL/QDRANT_URL).")
    resolved_key = (
        api_key
        or os.getenv("SIGIR_QDRANT_KEY")
        or os.getenv("SIGIR_QDRANT_API_KEY")
        or os.getenv("DEST_QDRANT_API_KEY")
        or os.getenv("QDRANT_API_KEY")
    )
    return QdrantConnection(url=str(resolved_url), api_key=resolved_key)


def _infer_grpc_port(url: str) -> Optional[int]:
    try:
        if urlparse(url).port == 6333:
            return 6334
    except Exception:
        return None
    return None


class QdrantAdmin:
    def __init__(
        self,
        *,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout: int = 60,
    ):
        from qdrant_client import QdrantClient

        conn = _resolve_qdrant_connection(url=url, api_key=api_key)
        grpc_port = _infer_grpc_port(conn.url) if prefer_grpc else None
        self.client = QdrantClient(
            url=conn.url,
            api_key=conn.api_key,
            prefer_grpc=bool(prefer_grpc),
            grpc_port=grpc_port,
            timeout=int(timeout),
            check_compatibility=False,
        )

    def get_collection_info(self, *, collection_name: str) -> Dict[str, Any]:
        info = self.client.get_collection(collection_name)
        try:
            return info.model_dump()
        except Exception:
            try:
                return info.dict()
            except Exception:
                return {"collection": str(collection_name), "raw": str(info)}

    def modify_collection_config(
        self,
        *,
        collection_name: str,
        hnsw_config: Optional[Dict[str, Any]] = None,
        collection_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Patch collection-level config via Qdrant update_collection.

        Supported keys:
        - hnsw_config: dict for HnswConfigDiff (e.g. on_disk, m, ef_construct, full_scan_threshold)
        - collection_params: dict for CollectionParamsDiff (e.g. on_disk_payload)
        """
        from qdrant_client.http import models as m

        hnsw_diff = m.HnswConfigDiff(**hnsw_config) if isinstance(hnsw_config, dict) else None
        params_diff = m.CollectionParamsDiff(**collection_params) if isinstance(collection_params, dict) else None
        if hnsw_diff is None and params_diff is None:
            raise ValueError("No changes provided (pass hnsw_config and/or collection_params).")
        return bool(
            self.client.update_collection(
                collection_name=str(collection_name),
                hnsw_config=hnsw_diff,
                collection_params=params_diff,
                timeout=int(timeout) if timeout is not None else None,
            )
        )

    def modify_collection_vector_config(
        self,
        *,
        collection_name: str,
        vectors: Dict[str, Dict[str, Any]],
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Patch vector params under params.vectors[vector_name] using Qdrant update_collection.

        Supported keys per vector:
        - on_disk: bool
        - hnsw_config: dict with optional keys: m, ef_construct, full_scan_threshold, on_disk
        """
        from qdrant_client.http import models as m

        collection_name = str(collection_name)
        info = self.client.get_collection(collection_name)
        existing = set()
        try:
            existing = set((info.config.params.vectors or {}).keys())
        except Exception:
            existing = set()

        missing = [str(k) for k in (vectors or {}).keys() if existing and str(k) not in existing]
        if missing:
            raise ValueError(f"Vectors do not exist in collection '{collection_name}': {missing}. Existing: {sorted(existing)}")

        ok = True
        for name, cfg in (vectors or {}).items():
            if not isinstance(cfg, dict):
                raise ValueError(f"vectors['{name}'] must be a dict, got {type(cfg)}")
            hnsw_cfg = cfg.get("hnsw_config")
            hnsw_diff = m.HnswConfigDiff(**hnsw_cfg) if isinstance(hnsw_cfg, dict) else None
            vectors_diff = {
                str(name): m.VectorParamsDiff(
                    on_disk=cfg.get("on_disk", None),
                    hnsw_config=hnsw_diff,
                )
            }

            ok = bool(
                self.client.update_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_diff,
                    timeout=int(timeout) if timeout is not None else None,
                )
            ) and ok

        return ok

    def ensure_collection_all_on_disk(
        self,
        *,
        collection_name: str,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ensure:
        - All existing named vectors have on_disk=True and hnsw_config.on_disk=True
        - Collection hnsw_config.on_disk=True
        - Collection params.on_disk_payload=True
        Returns the post-update collection info (dict).
        """
        collection_name = str(collection_name)
        info = self.client.get_collection(collection_name)
        vectors = {}
        try:
            existing = list((info.config.params.vectors or {}).keys())
        except Exception:
            existing = []
        for vname in existing:
            vectors[str(vname)] = {"on_disk": True, "hnsw_config": {"on_disk": True}}

        if vectors:
            self.modify_collection_vector_config(collection_name=collection_name, vectors=vectors, timeout=timeout)

        self.modify_collection_config(
            collection_name=collection_name,
            hnsw_config={"on_disk": True},
            collection_params={"on_disk_payload": True},
            timeout=timeout,
        )

        return self.get_collection_info(collection_name=collection_name)

