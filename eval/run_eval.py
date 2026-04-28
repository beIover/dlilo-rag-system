"""Evaluation suite for retrieval and generation quality."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import config
from generation.generator import RAGGenerator
from ingest.pipeline import run_ingestion
from retrieval.retriever import EmbeddingClient, Retriever, build_index

logger = logging.getLogger(__name__)


DATASET_PATH = Path(__file__).resolve().parent / "qa_dataset.json"


def _load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing evaluation dataset: {DATASET_PATH}")
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _ensure_chunks() -> list[dict]:
    strategy = config.EFFECTIVE_CHUNKING_STRATEGY
    chunk_path = config.PROCESSED_DIR / f"chunks_{strategy}.json"
    if not chunk_path.exists():
        run_ingestion(strategy=strategy)
    return json.loads(chunk_path.read_text(encoding="utf-8"))


def precision_at_k(retrieved: list[dict], relevant_sources: set[str], k: int) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item.get("source") in relevant_sources)
    return hits / k


def _sentence_split(text: str) -> list[str]:
    return [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.split() if t.strip()]


def _faithfulness(answer: str, context: str) -> float:
    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0
    context_tokens = set(_tokenize(context))
    supported = 0
    for sentence in sentences:
        tokens = _tokenize(sentence)
        if not tokens:
            continue
        overlap = len(context_tokens.intersection(tokens)) / len(tokens)
        if overlap >= 0.6:
            supported += 1
    return supported / len(sentences)


def _answer_relevance(embedder: EmbeddingClient, answer: str, reference: str) -> float:
    vectors = embedder.embed_texts([answer, reference])
    if vectors.shape[0] < 2:
        return 0.0
    return float(np.dot(vectors[0], vectors[1]))


def run_retrieval_eval(dataset: list[dict], retriever: Retriever) -> dict:
    scores = []
    for sample in dataset:
        retrieved = retriever.retrieve(sample["question"], top_k=config.TOP_K)
        relevant_sources = set(sample.get("sources", []))
        scores.append(precision_at_k(retrieved, relevant_sources, config.TOP_K))
    return {"precision_at_5": float(np.mean(scores))}


def run_generation_eval(dataset: list[dict], retriever: Retriever) -> dict:
    generator = RAGGenerator()
    embedder = EmbeddingClient()

    faithfulness_scores = []
    relevance_scores = []

    for sample in dataset:
        retrieved = retriever.retrieve(sample["question"], top_k=config.TOP_K)
        result = generator.answer(sample["question"], retrieved, return_metadata=True)
        context = "\n".join(chunk["text"] for chunk in retrieved)
        faithfulness_scores.append(_faithfulness(result["answer"], context))
        relevance_scores.append(_answer_relevance(embedder, result["answer"], sample["answer"]))

    return {
        "faithfulness": float(np.mean(faithfulness_scores)),
        "answer_relevance": float(np.mean(relevance_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-generation", action="store_true")
    args = parser.parse_args()

    dataset = _load_dataset()
    chunks = _ensure_chunks()
    store = build_index(chunks, backend=config.VECTOR_STORE_BACKEND, reset=True)
    retriever = Retriever(vector_store=store)

    retrieval_metrics = run_retrieval_eval(dataset, retriever)
    print("Retrieval metrics:", retrieval_metrics)

    if args.with_generation:
        gen_metrics = run_generation_eval(dataset, retriever)
        print("Generation metrics:", gen_metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
