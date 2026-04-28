"""Configuration defaults for the RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RAW_DIR = Path(os.getenv("RAW_DIR", DATA_DIR / "raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", DATA_DIR / "processed"))

CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "sentence")
EFFECTIVE_CHUNKING_STRATEGY = (
    CHUNKING_STRATEGY if CHUNKING_STRATEGY in {"fixed", "sentence"} else "sentence"
)
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "300"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))
MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", "100"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "faiss")
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", PROCESSED_DIR / "faiss.index"))

TOP_K = int(os.getenv("TOP_K", "5"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.2"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
