"""
Visual Embedder - Generate visual and text embeddings for document retrieval.

This module provides a flexible interface that supports:
- ColPali models (ColSmol, ColPali, ColQwen2)
- Other vision-language models (future)
- Image embedding with tile-aware processing
- Query embedding with special token filtering

The embedder is BACKEND-AGNOSTIC - configure which model to use via the
`backend` parameter or model_name.
"""

import gc
import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """
    Visual document embedder supporting multiple backends.
    
    Currently supports:
    - ColPali family (ColSmol-500M, ColPali, ColQwen2)
    - More backends can be added
    
    Args:
        model_name: HuggingFace model name (e.g., "vidore/colSmol-500M")
        backend: Backend type ("colpali", "auto"). "auto" detects from model_name.
        device: Device to use (auto, cuda, mps, cpu)
        torch_dtype: Data type for model weights
        batch_size: Batch size for image processing
        filter_special_tokens: Filter special tokens from query embeddings
    
    Example:
        >>> # Auto-detect backend from model name
        >>> embedder = VisualEmbedder(model_name="vidore/colSmol-500M")
        >>> 
        >>> # Embed images
        >>> image_embeddings = embedder.embed_images(images)
        >>> 
        >>> # Embed query
        >>> query_embedding = embedder.embed_query("What is the budget?")
        >>> 
        >>> # Get token info for saliency maps
        >>> embeddings, token_infos = embedder.embed_images(
        ...     images, return_token_info=True
        ... )
    """
    
    # Known model families and their backends
    MODEL_BACKENDS = {
        "colsmol": "colpali",
        "colpali": "colpali",
        "colqwen": "colpali",
        "colidefics": "colpali",
    }
    
    def __init__(
        self,
        model_name: str = "vidore/colSmol-500M",
        backend: str = "auto",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        batch_size: int = 4,
        filter_special_tokens: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.filter_special_tokens = filter_special_tokens
        
        # Allow override via environment variable
        if os.getenv("VISUALRAG_INCLUDE_SPECIAL_TOKENS"):
            self.filter_special_tokens = False
            logger.info("Special token filtering disabled via VISUALRAG_INCLUDE_SPECIAL_TOKENS")
        
        # Auto-detect backend from model name
        if backend == "auto":
            backend = self._detect_backend(model_name)
        self.backend = backend
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Auto-detect dtype
        if torch_dtype is None:
            if device == "cuda":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype
        
        # Lazy model loading
        self._model = None
        self._processor = None
        self._image_token_id = None
        
        logger.info(f"🤖 VisualEmbedder initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Backend: {backend}")
        logger.info(f"   Device: {device}, dtype: {torch_dtype}")
    
    def _detect_backend(self, model_name: str) -> str:
        """Auto-detect backend from model name."""
        model_lower = model_name.lower()
        
        for key, backend in self.MODEL_BACKENDS.items():
            if key in model_lower:
                logger.debug(f"Detected backend '{backend}' from model name")
                return backend
        
        # Default to colpali for unknown models
        logger.warning(f"Unknown model '{model_name}', defaulting to 'colpali' backend")
        return "colpali"
    
    def _load_model(self):
        """Lazy load the model when first needed."""
        if self._model is not None:
            return
        
        if self.backend == "colpali":
            self._load_colpali_model()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _load_colpali_model(self):
        """Load ColPali-family model."""
        try:
            from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        except ImportError:
            raise ImportError(
                "colpali_engine not installed. Install with: "
                "pip install visual-rag-toolkit[embedding] or "
                "pip install colpali-engine"
            )
        
        logger.info(f"🤖 Loading ColPali model: {self.model_name}")
        logger.info(f"   Device: {self.device}, dtype: {self.torch_dtype}")
        
        # Determine attention implementation
        attn_implementation = "eager"
        if self.device != "cpu":
            try:
                import flash_attn  # noqa
                attn_implementation = "flash_attention_2"
                logger.info("   Using FlashAttention2")
            except ImportError:
                pass
        
        self._model = ColIdefics3.from_pretrained(
            self.model_name,
            dtype=self.torch_dtype,
            device_map=self.device,
            attn_implementation=attn_implementation,
        ).eval()
        
        self._processor = ColIdefics3Processor.from_pretrained(self.model_name)
        self._image_token_id = self._processor.image_token_id
        
        logger.info("✅ Model loaded successfully")
    
    @property
    def model(self):
        self._load_model()
        return self._model
    
    @property
    def processor(self):
        self._load_model()
        return self._processor
    
    @property
    def image_token_id(self):
        self._load_model()
        return self._image_token_id
    
    def embed_query(
        self,
        query_text: str,
        filter_special_tokens: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Generate embedding for a text query.
        
        By default, filters out special tokens (CLS, SEP, PAD) to keep only
        meaningful text tokens for better MaxSim matching.
        
        Args:
            query_text: Natural language query string
            filter_special_tokens: Override instance-level setting
        
        Returns:
            Query embedding tensor of shape [num_tokens, embedding_dim]
        """
        should_filter = (
            filter_special_tokens 
            if filter_special_tokens is not None 
            else self.filter_special_tokens
        )
        
        with torch.no_grad():
            processed = self.processor.process_queries([query_text]).to(self.model.device)
            embedding = self.model(**processed)
        
        # Remove batch dimension: [1, tokens, dim] -> [tokens, dim]
        if embedding.dim() == 3:
            embedding = embedding.squeeze(0)
        
        if should_filter:
            # Filter special tokens based on attention mask
            attention_mask = processed.get("attention_mask")
            if attention_mask is not None:
                # Keep only tokens with attention_mask = 1
                valid_mask = attention_mask.squeeze(0).bool()
                embedding = embedding[valid_mask]
                
                # Additionally filter padding tokens if present
                input_ids = processed.get("input_ids")
                if input_ids is not None:
                    input_ids = input_ids.squeeze(0)[valid_mask]
                    # Filter common special token IDs
                    # IDs >= 4 are usually real tokens for most tokenizers
                    non_special_mask = input_ids >= 4
                    if non_special_mask.any():
                        embedding = embedding[non_special_mask]
            
            logger.debug(f"Query embedding: {embedding.shape[0]} tokens after filtering")
        else:
            logger.debug(f"Query embedding: {embedding.shape[0]} tokens (unfiltered)")
        
        return embedding
    
    def embed_images(
        self,
        images: List[Image.Image],
        batch_size: Optional[int] = None,
        return_token_info: bool = False,
        show_progress: bool = True,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[Dict[str, Any]]]]:
        """
        Generate embeddings for a list of images.
        
        Args:
            images: List of PIL Images
            batch_size: Override instance batch size
            return_token_info: Also return token metadata (for saliency maps)
            show_progress: Show progress bar
        
        Returns:
            If return_token_info=False:
                List of embedding tensors [num_patches, dim]
            If return_token_info=True:
                Tuple of (embeddings, token_infos)
                
        Token info contains:
            - visual_token_indices: Indices of visual tokens in embedding
            - num_visual_tokens: Count of visual tokens
            - n_rows, n_cols: Tile grid dimensions
            - num_tiles: Total tiles (n_rows × n_cols + 1 global)
        """
        batch_size = batch_size or self.batch_size
        
        embeddings = []
        token_infos = [] if return_token_info else None
        
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="🎨 Embedding", unit="batch")
        
        for i in iterator:
            batch = images[i:i + batch_size]
            
            with torch.no_grad():
                processed = self.processor.process_images(batch).to(self.model.device)
                
                # Extract token info before model forward
                if return_token_info:
                    input_ids = processed["input_ids"]
                    batch_n_rows = processed.get("n_rows")
                    batch_n_cols = processed.get("n_cols")
                    
                    for j in range(input_ids.shape[0]):
                        # Find visual token indices
                        image_token_mask = (input_ids[j] == self.image_token_id)
                        visual_indices = torch.where(image_token_mask)[0].cpu().numpy().tolist()
                        
                        n_rows = batch_n_rows[j].item() if batch_n_rows is not None else None
                        n_cols = batch_n_cols[j].item() if batch_n_cols is not None else None
                        
                        token_infos.append({
                            "visual_token_indices": visual_indices,
                            "num_visual_tokens": len(visual_indices),
                            "n_rows": n_rows,
                            "n_cols": n_cols,
                            "num_tiles": (n_rows * n_cols + 1) if n_rows and n_cols else None,
                        })
                
                # Generate embeddings
                batch_embeddings = self.model(**processed)
            
            # Extract per-image embeddings
            if isinstance(batch_embeddings, torch.Tensor) and batch_embeddings.dim() == 3:
                for j in range(batch_embeddings.shape[0]):
                    embeddings.append(batch_embeddings[j].cpu())
            else:
                embeddings.extend([e.cpu() for e in batch_embeddings])
            
            # Memory cleanup
            del processed, batch_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if return_token_info:
            return embeddings, token_infos
        return embeddings
    
    def extract_visual_embedding(
        self,
        full_embedding: torch.Tensor,
        token_info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Extract only visual token embeddings from full embedding.
        
        Filters out special tokens, keeping only visual patches for MaxSim.
        
        Args:
            full_embedding: Full embedding [all_tokens, dim]
            token_info: Token info dict from embed_images
        
        Returns:
            Visual embedding array [num_visual_tokens, dim]
        """
        visual_indices = token_info["visual_token_indices"]
        
        if isinstance(full_embedding, torch.Tensor):
            if full_embedding.dtype == torch.bfloat16:
                visual_emb = full_embedding[visual_indices].cpu().float().numpy()
            else:
                visual_emb = full_embedding[visual_indices].cpu().numpy()
        else:
            visual_emb = np.array(full_embedding)[visual_indices]
        
        return visual_emb.astype(np.float32)


# Backward compatibility alias
ColPaliEmbedder = VisualEmbedder


