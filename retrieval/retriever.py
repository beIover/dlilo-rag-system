"""Vector store and retriever for the RAG pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

import config

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.EMBEDDING_MODEL
        self.device = device or config.EMBEDDING_DEVICE
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype="float32")


class FaissVectorStore:
    def __init__(self, dim: int, index_path: Path | None = None):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.index_path = index_path
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadatas: Iterable[dict]) -> None:
        self.index.add(vectors)
        self.metadata.extend(list(metadatas))

    def search(self, vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if vector.ndim == 1:
            vector = np.expand_dims(vector, axis=0)
        scores, indices = self.index.search(vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def count(self) -> int:
        return self.index.ntotal

    def save(self, path: Path | None = None) -> None:
        import faiss

        path = path or self.index_path
        if path is None:
            raise ValueError("Index path is required to save the vector store")
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        meta_path = path.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        import faiss

        index = faiss.read_index(str(path))
        store = cls(index.d, index_path=path)
        store.index = index
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            store.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return store


def build_index(
    chunks: list[dict],
    embedding_model: str | None = None,
    backend: str | None = None,
    reset: bool = False,
) -> FaissVectorStore:
    """Embed chunks and build a vector index."""
    backend = backend or config.VECTOR_STORE_BACKEND
    if backend != "faiss":
        raise ValueError(f"Unsupported vector store backend: {backend}")

    embedding_model = embedding_model or config.EMBEDDING_MODEL
    embedder = EmbeddingClient(embedding_model)

    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_texts(texts)

    store = FaissVectorStore(vectors.shape[1], index_path=config.VECTOR_STORE_PATH)
    store.add(vectors, chunks)
    if reset:
        store.save()
    return store


class Retriever:
    def __init__(self, vector_store: FaissVectorStore, embedder: EmbeddingClient | None = None):
        self.vector_store = vector_store
        self.embedder = embedder or EmbeddingClient()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.TOP_K
        vector = self.embedder.embed_query(query)
        return self.vector_store.search(vector, top_k=top_k)
