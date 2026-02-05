"""Indexing runner with UI updates."""

import hashlib
import importlib.util
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st
import torch

from visual_rag import VisualEmbedder


TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# --- Robust imports (Spaces-friendly) ---
# Some environments can have a third-party `benchmarks` package installed, or
# resolve `visual_rag.indexing` oddly. These fallbacks keep the demo working.
try:
    from visual_rag.indexing import QdrantIndexer
except Exception:  # pragma: no cover
    from visual_rag.indexing.qdrant_indexer import QdrantIndexer


def _load_local_benchmark_module(module_filename: str):
    root = Path(__file__).resolve().parents[1]  # demo/.. = repo root
    target = root / "benchmarks" / "vidore_tatdqa_test" / module_filename
    if not target.exists():
        raise ModuleNotFoundError(f"Missing local benchmark module file: {target}")
    name = f"_visual_rag_toolkit_local_{target.stem}"
    spec = importlib.util.spec_from_file_location(name, str(target))
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load module spec for: {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


try:
    from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_beir_dataset
except ModuleNotFoundError:  # pragma: no cover
    _dl = _load_local_benchmark_module("dataset_loader.py")
    load_vidore_beir_dataset = _dl.load_vidore_beir_dataset

from demo.qdrant_utils import get_qdrant_credentials


def _stable_uuid(text: str) -> str:
    """Generate a stable UUID from text (same as benchmark script)."""
    hex_str = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


def _union_point_id(*, dataset_name: str, source_doc_id: str, union_namespace: Optional[str]) -> str:
    """Generate union point ID (same as benchmark script)."""
    ns = f"{union_namespace}::{dataset_name}" if union_namespace else dataset_name
    return _stable_uuid(f"{ns}::{source_doc_id}")


def run_indexing_with_ui(config: Dict[str, Any]):
    st.divider()
    
    print("=" * 60)
    print("[INDEX] Starting indexing via UI")
    print("=" * 60)
    
    url, api_key = get_qdrant_credentials()
    if not url:
        st.error("QDRANT_URL not configured")
        return
    
    datasets = config.get("datasets", [])
    collection = config["collection"]
    model = config.get("model", "vidore/colpali-v1.3")
    recreate = config.get("recreate", False)
    torch_dtype = config.get("torch_dtype", "float16")
    qdrant_vector_dtype = config.get("qdrant_vector_dtype", "float16")
    prefer_grpc = config.get("prefer_grpc", True)
    batch_size = config.get("batch_size", 4)
    max_docs = config.get("max_docs")
    
    print(f"[INDEX] Config: collection={collection}, model={model}")
    print(f"[INDEX] Datasets: {datasets}")
    print(f"[INDEX] max_docs={max_docs}, batch_size={batch_size}, recreate={recreate}")
    print(f"[INDEX] torch_dtype={torch_dtype}, qdrant_dtype={qdrant_vector_dtype}, grpc={prefer_grpc}")
    
    phase1_container = st.container()
    phase2_container = st.container()
    phase3_container = st.container()
    results_container = st.container()
    
    try:
        with phase1_container:
            st.markdown("##### 🤖 Phase 1: Loading Model")
            model_status = st.empty()
            model_status.info(f"Loading `{model.split('/')[-1]}`...")
            
            print(f"[INDEX] Loading embedder: {model}")
            torch_dtype_obj = TORCH_DTYPE_MAP.get(torch_dtype, torch.float16)
            output_dtype_obj = np.float16 if qdrant_vector_dtype == "float16" else np.float32
            embedder = VisualEmbedder(
                model_name=model,
                torch_dtype=torch_dtype_obj,
                output_dtype=output_dtype_obj,
            )
            embedder._load_model()
            print(f"[INDEX] Embedder loaded (torch_dtype={torch_dtype}, output_dtype={qdrant_vector_dtype})")
            model_status.success(f"✅ Model `{model.split('/')[-1]}` loaded")
        
        with phase2_container:
            st.markdown("##### 📦 Phase 2: Setting Up Collection")
            
            indexer_status = st.empty()
            indexer_status.info(f"Connecting to Qdrant...")
            
            print(f"[INDEX] Connecting to Qdrant...")
            indexer = QdrantIndexer(
                url=url,
                api_key=api_key,
                collection_name=collection,
                prefer_grpc=prefer_grpc,
                vector_datatype=qdrant_vector_dtype,
            )
            print(f"[INDEX] Connected to Qdrant")
            indexer_status.success(f"✅ Connected to Qdrant")
            
            coll_status = st.empty()
            action = "Recreating" if recreate else "Creating/verifying"
            coll_status.info(f"{action} collection `{collection}`...")
            
            print(f"[INDEX] {action} collection: {collection}")
            indexer.create_collection(force_recreate=recreate)
            indexer.create_payload_indexes(fields=[
                {"field": "dataset", "type": "keyword"},
                {"field": "doc_id", "type": "keyword"},
                {"field": "source_doc_id", "type": "keyword"},
            ])
            print(f"[INDEX] Collection ready")
            coll_status.success(f"✅ Collection `{collection}` ready")
        
        with phase3_container:
            st.markdown("##### 🚀 Phase 3: Indexing Documents")
            
            total_uploaded = 0
            total_docs = 0
            total_time = 0
            
            for ds_name in datasets:
                ds_short = ds_name.split("/")[-1]
                ds_header = st.empty()
                ds_header.info(f"📚 Loading `{ds_short}`...")
                
                print(f"[INDEX] Loading dataset: {ds_name}")
                corpus, queries, qrels = load_vidore_beir_dataset(ds_name)
                
                if max_docs and max_docs > 0 and len(corpus) > max_docs:
                    corpus = corpus[:max_docs]
                    print(f"[INDEX] Limited to {len(corpus)} docs (max_docs={max_docs})")
                
                total_docs += len(corpus)
                print(f"[INDEX] Dataset {ds_name}: {len(corpus)} documents to index")
                ds_header.success(f"📚 `{ds_short}`: {len(corpus)} documents")
                
                progress_bar = st.progress(0.0)
                batch_status = st.empty()
                log_area = st.empty()
                log_lines = []
                
                num_batches = (len(corpus) + batch_size - 1) // batch_size
                ds_start = time.time()
                
                for i in range(0, len(corpus), batch_size):
                    batch = corpus[i:i + batch_size]
                    images = [doc.image for doc in batch if hasattr(doc, 'image') and doc.image]
                    
                    if not images:
                        continue
                    
                    batch_num = i // batch_size + 1
                    batch_status.info(f"Batch {batch_num}/{num_batches}: embedding & uploading...")
                    
                    batch_start = time.time()
                    embeddings, token_infos = embedder.embed_images(images, return_token_info=True)
                    embed_time = time.time() - batch_start
                    
                    points = []
                    for j, (doc, emb, token_info) in enumerate(zip(batch, embeddings, token_infos)):
                        doc_id = doc.doc_id if hasattr(doc, 'doc_id') else str(i + j)
                        source_doc_id = str(doc.payload.get("source_doc_id", doc_id) if hasattr(doc, 'payload') else doc_id)
                        
                        union_doc_id = _union_point_id(
                            dataset_name=ds_name,
                            source_doc_id=source_doc_id,
                            union_namespace=collection,
                        )
                        
                        emb_np = emb.cpu().numpy() if hasattr(emb, 'cpu') else np.array(emb)
                        visual_indices = token_info.get("visual_token_indices") or list(range(emb_np.shape[0]))
                        visual_emb = emb_np[visual_indices].astype(embedder.output_dtype)
                        
                        tile_pooled = embedder.mean_pool_visual_embedding(visual_emb, token_info, target_vectors=32)
                        experimental = embedder.experimental_pool_visual_embedding(
                            visual_emb, token_info, target_vectors=32, mean_pool=tile_pooled
                        )
                        global_pooled = embedder.global_pool_from_mean_pool(tile_pooled)
                        
                        points.append({
                            "id": union_doc_id,
                            "visual_embedding": visual_emb,
                            "tile_pooled_embedding": tile_pooled,
                            "experimental_pooled_embedding": experimental,
                            "global_pooled_embedding": global_pooled,
                            "metadata": {
                                "dataset": ds_name,
                                "doc_id": doc_id,
                                "source_doc_id": source_doc_id,
                                "union_doc_id": union_doc_id,
                            },
                        })
                    
                    upload_start = time.time()
                    indexer.upload_batch(points)
                    upload_time = time.time() - upload_start
                    total_uploaded += len(points)
                    
                    progress = (i + len(batch)) / len(corpus)
                    progress_bar.progress(progress)
                    batch_status.info(f"Batch {batch_num}/{num_batches} ({int(progress*100)}%) — embed: {embed_time:.1f}s, upload: {upload_time:.1f}s")
                    
                    log_interval = max(2, num_batches // 10)
                    should_log = batch_num % log_interval == 0 or batch_num == num_batches
                    
                    if should_log and batch_num > 1:
                        log_lines.append(f"[Batch {batch_num}/{num_batches}] +{len(points)} pts, total={total_uploaded}")
                        log_area.code("\n".join(log_lines[-8:]), language="text")
                        print(f"[INDEX] Batch {batch_num}/{num_batches}: +{len(points)} pts, total={total_uploaded}, embed={embed_time:.1f}s, upload={upload_time:.1f}s")
                
                ds_time = time.time() - ds_start
                total_time += ds_time
                progress_bar.progress(1.0)
                batch_status.success(f"✅ `{ds_short}` indexed: {len(corpus)} docs in {ds_time:.1f}s")
                print(f"[INDEX] Dataset {ds_name} complete: {len(corpus)} docs in {ds_time:.1f}s")
        
        with results_container:
            st.markdown("##### 📊 Summary")
            
            docs_per_sec = total_uploaded / total_time if total_time > 0 else 0
            
            print("=" * 60)
            print("[INDEX] INDEXING COMPLETE")
            print(f"[INDEX]   Total Uploaded: {total_uploaded:,}")
            print(f"[INDEX]   Datasets: {len(datasets)}")
            print(f"[INDEX]   Collection: {collection}")
            print(f"[INDEX]   Total Time: {total_time:.1f}s")
            print(f"[INDEX]   Throughput: {docs_per_sec:.2f} docs/s")
            print("=" * 60)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Uploaded", f"{total_uploaded:,}")
            c2.metric("Datasets", len(datasets))
            c3.metric("Total Time", f"{total_time:.1f}s")
            c4.metric("Throughput", f"{docs_per_sec:.2f}/s")
            
            st.success(f"🎉 Indexing complete! {total_uploaded:,} documents indexed to `{collection}`")
            
            detailed_report = {
                "generated_at": datetime.now().isoformat(),
                "config": {
                    "collection": collection,
                    "model": model,
                    "datasets": datasets,
                    "batch_size": batch_size,
                    "max_docs_per_dataset": max_docs,
                    "recreate": recreate,
                    "prefer_grpc": prefer_grpc,
                    "torch_dtype": torch_dtype,
                    "qdrant_vector_dtype": qdrant_vector_dtype,
                },
                "results": {
                    "total_docs_uploaded": total_uploaded,
                    "total_time_s": round(total_time, 2),
                    "throughput_docs_per_s": round(docs_per_sec, 2),
                    "num_datasets": len(datasets),
                },
            }
            
            with st.expander("📋 Full Summary"):
                st.json(detailed_report)
            
            report_json = json.dumps(detailed_report, indent=2)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"index_report__{collection}__{timestamp}.json"
            
            st.download_button(
                label="📥 Download Indexing Report",
                data=report_json,
                file_name=filename,
                mime="application/json",
                use_container_width=True,
            )
        
    except Exception as e:
        print(f"[INDEX] ERROR: {e}")
        st.error(f"❌ Error: {e}")
        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc(), language="text")
