"""
Embedding Manager using Sentence Transformers.
Provides batched semantic embedding generation with caching.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages loading and inference for Sentence Transformer embedding models."""

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._embedding_cache: dict[str, np.ndarray] = {}

    @property
    def model(self):
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning("Could not load SentenceTransformer directly: %s. Using fallback TF-IDF/Hash embedding if offline.", e)
                raise e
        return self._model

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Generate normalized embeddings for single text or list of texts.
        Uses in-memory cache for previously seen texts.
        """
        is_single = isinstance(texts, str)
        text_list: List[str] = [texts] if is_single else list(texts)

        if not text_list:
            return np.empty((0, 384), dtype=np.float32)

        # Identify texts needing encoding
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        embeddings: List[Optional[np.ndarray]] = [None] * len(text_list)

        for idx, text in enumerate(text_list):
            clean_text = text.strip()
            if clean_text in self._embedding_cache:
                embeddings[idx] = self._embedding_cache[clean_text]
            else:
                missing_indices.append(idx)
                missing_texts.append(clean_text if clean_text else "empty")

        if missing_texts:
            encoded_missing = self.model.encode(
                missing_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
            )
            # Ensure 2D numpy array
            if len(missing_texts) == 1 and encoded_missing.ndim == 1:
                encoded_missing = np.expand_dims(encoded_missing, axis=0)

            for i, orig_idx in enumerate(missing_indices):
                vec = encoded_missing[i]
                clean_t = text_list[orig_idx].strip()
                self._embedding_cache[clean_t] = vec
                embeddings[orig_idx] = vec

        result = np.vstack(embeddings)
        return result[0] if is_single else result

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self._embedding_cache.clear()


_global_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(model_name: str = EmbeddingManager.DEFAULT_MODEL_NAME) -> EmbeddingManager:
    """Get or create singleton EmbeddingManager instance."""
    global _global_embedding_manager
    if _global_embedding_manager is None or _global_embedding_manager.model_name != model_name:
        _global_embedding_manager = EmbeddingManager(model_name=model_name)
    return _global_embedding_manager
