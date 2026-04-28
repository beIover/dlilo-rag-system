"""Gradio chat UI for the RAG chatbot."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import gradio as gr

import config
from generation.generator import RAGGenerator
from ingest.pipeline import run_ingestion
from retrieval.retriever import Retriever, build_index

logger = logging.getLogger(__name__)


class ChatApp:
    def __init__(self) -> None:
        self.retriever: Retriever | None = None
        self.generator = RAGGenerator()

    def _load_chunks(self) -> list[dict]:
        strategy = config.EFFECTIVE_CHUNKING_STRATEGY
        chunk_path = config.PROCESSED_DIR / f"chunks_{strategy}.json"
        if not chunk_path.exists():
            run_ingestion(strategy=strategy)
        with chunk_path.open(encoding="utf-8") as f:
            return json.load(f)

    def _ensure_retriever(self) -> None:
        if self.retriever is not None:
            return
        chunks = self._load_chunks()
        store = build_index(chunks, backend=config.VECTOR_STORE_BACKEND, reset=True)
        self.retriever = Retriever(vector_store=store)

    def answer(self, message: str, history: list[tuple[str, str]]) -> str:
        self._ensure_retriever()
        retrieved = self.retriever.retrieve(message)
        result = self.generator.answer(message, retrieved, return_metadata=True)
        sources = ", ".join(result["sources"]) if result["sources"] else "None"
        return f"{result['answer']}\n\nSources: {sources}"


def main() -> None:
    app = ChatApp()

    demo = gr.ChatInterface(
        fn=app.answer,
        title="RAG Chatbot",
        description="Ask a question and get a grounded answer with citations.",
    )
    demo.launch(server_port=config.GRADIO_PORT, share=config.GRADIO_SHARE)


if __name__ == "__main__":
    main()
