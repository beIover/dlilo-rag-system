"""Generation and grounding logic for the RAG pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

import config

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a RAG assistant. Use ONLY the provided context to answer the question. "
    "Cite the source document name for every factual claim. "
    "If the answer is not supported by the context, respond exactly with: "
    "'I cannot find this in the provided documents.'"
)


@dataclass
class GenerationResult:
    answer: str
    sources: list[str]


class LLMClient:
    def __init__(self) -> None:
        self.provider = config.LLM_PROVIDER
        self.model = config.OPENAI_MODEL if self.provider == "openai" else config.OLLAMA_MODEL

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "openai":
            from openai import OpenAI

            if not config.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set")
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()

        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            }
            resp = requests.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()

        if self.provider == "mock":
            return "I cannot find this in the provided documents."

        raise ValueError(f"Unsupported LLM provider: {self.provider}")


class RAGGenerator:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def _build_context(self, chunks: list[dict]) -> str:
        lines = []
        for idx, chunk in enumerate(chunks, 1):
            lines.append(
                "\n".join(
                    [
                        f"[{idx}] Source: {chunk['source']}",
                        f"Title: {chunk.get('title', '')}",
                        f"Date: {chunk.get('date', '')}",
                        f"Text: {chunk['text']}",
                    ]
                )
            )
        return "\n\n".join(lines)

    def answer(self, query: str, chunks: list[dict], return_metadata: bool = False) -> dict:
        if not chunks:
            result = GenerationResult(
                answer="I cannot find this in the provided documents.",
                sources=[],
            )
            return result.__dict__ if return_metadata else {"answer": result.answer}

        max_score = max((chunk.get("score", 0.0) for chunk in chunks), default=0.0)
        if max_score < config.MIN_RETRIEVAL_SCORE:
            result = GenerationResult(
                answer="I cannot find this in the provided documents.",
                sources=[],
            )
            return result.__dict__ if return_metadata else {"answer": result.answer}

        sources = sorted({chunk["source"] for chunk in chunks})
        context = self._build_context(chunks)
        user_prompt = (
            f"Context:\n{context}\n\n"
            "Question: "
            f"{query}\n\n"
            "Answer with citations in the form [source]."
        )

        try:
            answer = self.llm.chat(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer = "I cannot find this in the provided documents."

        result = GenerationResult(answer=answer, sources=sources)
        return result.__dict__ if return_metadata else {"answer": result.answer}
