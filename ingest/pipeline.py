"""Document ingestion pipeline: load raw docs, chunk, and persist to disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import config
from ingest.chunker import chunk_documents
from ingest.loader import load_directory

logger = logging.getLogger(__name__)


Strategy = Literal["fixed", "sentence", "both"]


def _ensure_dirs() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run_ingestion(
    strategy: Strategy = "sentence",
    chunk_size: int | None = None,
    overlap: int | None = None,
    raw_dir: Path | None = None,
) -> dict[str, Path]:
    """Run ingestion and return output paths per strategy."""
    _ensure_dirs()
    chunk_size = chunk_size or config.CHUNK_SIZE_TOKENS
    overlap = overlap or config.CHUNK_OVERLAP_TOKENS

    docs = load_directory(raw_dir or config.RAW_DIR, recursive=True)
    if not docs:
        logger.warning("No documents found in %s", raw_dir or config.RAW_DIR)
        return {}

    strategies = ["fixed", "sentence"] if strategy == "both" else [strategy]
    outputs: dict[str, Path] = {}

    for strat in strategies:
        chunks = chunk_documents(
            docs,
            strategy=strat,
            chunk_size=chunk_size,
            overlap=overlap,
            min_tokens=config.MIN_CHUNK_TOKENS,
        )
        output_path = config.PROCESSED_DIR / f"chunks_{strat}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d chunks to %s", len(chunks), output_path)
        outputs[strat] = output_path

    return outputs
