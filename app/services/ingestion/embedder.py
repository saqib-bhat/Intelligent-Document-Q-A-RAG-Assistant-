"""Embedding generation using fastembed (ONNX-based, low memory footprint)."""

import logging
from typing import List

import numpy as np
from fastembed import TextEmbedding

from app.config.settings import get_settings
from app.services.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """Generates dense vector embeddings for document chunks."""

    # Map legacy sentence-transformers names to fastembed equivalents
    _MODEL_MAP = {
        "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L3-v2": "sentence-transformers/paraphrase-MiniLM-L3-v2",
    }

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        fastembed_name = self._MODEL_MAP.get(
            self.model_name, "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(f"Loading embedding model (fastembed): {fastembed_name}")
        self._model = TextEmbedding(model_name=fastembed_name)
        # Determine dimension by embedding a test string
        test = list(self._model.embed(["test"]))
        self.dimension = len(test[0])
        logger.info(f"Embedding dimension: {self.dimension}")

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Return a (N, D) float32 numpy array of embeddings for the chunks."""
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)

        texts = [c.text for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks …")
        embeddings = np.array(list(self._model.embed(texts)), dtype=np.float32)

        # Normalise for cosine similarity (FAISS IndexFlatIP)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (embeddings / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Return a (1, D) float32 array for a single query string."""
        embedding = np.array(list(self._model.embed([query])), dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)
