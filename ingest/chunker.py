"""
ingest/chunker.py — Two chunking strategies for the RAG pipeline.

Strategy A: Fixed-size with overlap
  - Splits text into fixed-token windows with a sliding overlap.
  - Simple, predictable, easy to tune.
  - Drawback: may cut mid-sentence, losing semantic coherence.

Strategy B: Sentence-aware recursive splitting
  - First splits on paragraph/sentence boundaries.
  - Then merges small sentences into chunks until the target size is reached.
  - Preserves semantic coherence; preferred strategy.

Each chunk is returned as:
  {
    "text": str,
    "chunk_index": int,         # 0-based within source document
    "source": str,              # filename
    "title": str,
    "date": str | None,
    "doc_type": str,
    "strategy": "fixed" | "sentence",
    "token_count": int,
  }
"""

import re
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ─── Tokenisation (word-level approximation, no heavy dep) ───────────────────

def _approx_tokens(text: str) -> int:
    """Approximate token count: split on whitespace."""
    return len(re.findall(r"\S+", text))


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


# ─── Strategy A: Fixed-size with overlap ─────────────────────────────────────

def chunk_fixed(
    doc: dict,
    chunk_size: int = 300,
    overlap: int = 60,
    min_tokens: int = 100,
) -> list[dict]:
    """
    Split doc['text'] into fixed-size token windows with overlap.

    Args:
        doc: document dict (must have keys: text, source, title, date, doc_type)
        chunk_size: target token count per chunk (100–512 per project spec)
        overlap: token overlap between consecutive chunks (~10–25 % of chunk_size)

    Returns:
        List of chunk dicts.
    """
    words = _words(doc["text"])
    if not words:
        return []

    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        text = " ".join(chunk_words)
        chunks.append({
            **{k: doc[k] for k in ("source", "title", "date", "doc_type")},
            "text": text,
            "chunk_index": idx,
            "strategy": "fixed",
            "token_count": len(chunk_words),
        })
        idx += 1
        if end == len(words):
            break
        start += chunk_size - overlap

    if chunks and chunks[-1]["token_count"] < min_tokens and len(chunks) > 1:
        tail = chunks.pop()
        chunks[-1]["text"] = f"{chunks[-1]['text']} {tail['text']}".strip()
        chunks[-1]["token_count"] += tail["token_count"]

    if chunks and chunks[-1]["token_count"] < min_tokens:
        logger.warning(
            "Small final chunk (%d tokens) kept for %s.",
            chunks[-1]["token_count"],
            doc.get("source", "unknown"),
        )

    logger.debug("Fixed chunking: %s → %d chunks", doc["source"], len(chunks))
    return chunks


# ─── Strategy B: Sentence-aware recursive splitting ───────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    # Split on paragraph breaks first
    paragraphs = re.split(r"\n{2,}", text)
    sentences = []
    for para in paragraphs:
        # Split on sentence-ending punctuation
        sents = re.split(r"(?<=[.!?])\s+", para.strip())
        for s in sents:
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences


def chunk_sentence(
    doc: dict,
    target_size: int = 300,
    overlap: int = 50,
    min_tokens: int = 100,
) -> list[dict]:
    """
    Split doc['text'] sentence-by-sentence, merging into chunks of ~target_size tokens.
    When a new chunk starts, carry over the last `overlap` tokens from the previous chunk
    for context continuity.

    Args:
        doc: document dict
        target_size: target token count per chunk
        overlap: token overlap (carried forward as context)

    Returns:
        List of chunk dicts.
    """
    sentences = _split_sentences(doc["text"])
    if not sentences:
        return []

    chunks = []
    current_sents: list[str] = []
    current_tokens = 0
    overlap_buffer: list[str] = []   # sentences carried over for overlap
    idx = 0

    def _flush(sents: list[str]) -> None:
        nonlocal idx
        text = " ".join(sents)
        tok = _approx_tokens(text)
        if tok < min_tokens:
            if chunks:
                chunks[-1]["text"] = f"{chunks[-1]['text']} {text}".strip()
                chunks[-1]["token_count"] += tok
            else:
                logger.warning(
                    "Small initial chunk (%d tokens) kept for %s.",
                    tok,
                    doc.get("source", "unknown"),
                )
                chunks.append({
                    **{k: doc[k] for k in ("source", "title", "date", "doc_type")},
                    "text": text,
                    "chunk_index": idx,
                    "strategy": "sentence",
                    "token_count": tok,
                })
                idx += 1
            return
        chunks.append({
            **{k: doc[k] for k in ("source", "title", "date", "doc_type")},
            "text": text,
            "chunk_index": idx,
            "strategy": "sentence",
            "token_count": tok,
        })
        idx += 1

    for sent in sentences:
        sent_tokens = _approx_tokens(sent)

        # If a single sentence exceeds target, hard-split it word-by-word
        if sent_tokens > target_size:
            if current_sents:
                _flush(current_sents)
                overlap_buffer = current_sents[-2:] if len(current_sents) >= 2 else current_sents[:]
                current_sents = list(overlap_buffer)
                current_tokens = _approx_tokens(" ".join(current_sents))

            words = _words(sent)
            for wi in range(0, len(words), target_size - overlap):
                slice_words = words[wi:wi + target_size]
                chunk_text = " ".join(slice_words)
                chunks.append({
                    **{k: doc[k] for k in ("source", "title", "date", "doc_type")},
                    "text": chunk_text,
                    "chunk_index": idx,
                    "strategy": "sentence",
                    "token_count": len(slice_words),
                })
                idx += 1
            continue

        if current_tokens + sent_tokens > target_size and current_sents:
            _flush(current_sents)
            # Build overlap buffer from end of previous chunk
            overlap_sents = []
            overlap_tok = 0
            for s in reversed(current_sents):
                t = _approx_tokens(s)
                if overlap_tok + t > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tok += t
            current_sents = overlap_sents + [sent]
            current_tokens = overlap_tok + sent_tokens
        else:
            current_sents.append(sent)
            current_tokens += sent_tokens

    if current_sents:
        _flush(current_sents)

    logger.debug("Sentence chunking: %s → %d chunks", doc["source"], len(chunks))
    return chunks


# ─── Public API ───────────────────────────────────────────────────────────────

def chunk_document(
    doc: dict,
    strategy: Literal["fixed", "sentence"] = "sentence",
    chunk_size: int = 300,
    overlap: int = 60,
    min_tokens: int = 100,
) -> list[dict]:
    """Chunk a single document using the chosen strategy."""
    if strategy == "fixed":
        return chunk_fixed(doc, chunk_size=chunk_size, overlap=overlap, min_tokens=min_tokens)
    elif strategy == "sentence":
        return chunk_sentence(doc, target_size=chunk_size, overlap=overlap, min_tokens=min_tokens)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def chunk_documents(
    docs: list[dict],
    strategy: Literal["fixed", "sentence"] = "sentence",
    chunk_size: int = 300,
    overlap: int = 60,
    min_tokens: int = 100,
) -> list[dict]:
    """Chunk a list of documents."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(
            doc,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            min_tokens=min_tokens,
        )
        all_chunks.extend(chunks)
    logger.info(
        "Chunking complete — strategy=%s, docs=%d, total_chunks=%d",
        strategy, len(docs), len(all_chunks),
    )
    return all_chunks
