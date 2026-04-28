"""
ingest/loader.py — Load documents from multiple file formats.

Supported formats: PDF, DOCX, TXT, Markdown, JSON, HTML
Each loaded document is returned as a dict with:
  - text: str
  - source: str (filename)
  - title: str
  - date: str | None
  - doc_type: str
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Helper: try to extract a date from text ─────────────────────────────────
_DATE_PATTERNS = [
    r"\b(\d{4}-\d{2}-\d{2})\b",
    r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
]


def _extract_date(text: str) -> Optional[str]:
    for pat in _DATE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def _base_meta(path: Path) -> dict:
    return {
        "source": path.name,
        "title": path.stem.replace("_", " ").replace("-", " ").title(),
        "date": None,
        "doc_type": path.suffix.lower().lstrip("."),
    }


# ─── PDF ─────────────────────────────────────────────────────────────────────
def _load_pdf(path: Path) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf not installed — run: pip install pypdf")

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        t = page.extract_text() or ""
        pages.append(t)
    text = "\n".join(pages)

    meta = _base_meta(path)
    # Try to get title from PDF metadata
    if reader.metadata:
        if reader.metadata.title:
            meta["title"] = reader.metadata.title
        if reader.metadata.creation_date:
            try:
                meta["date"] = str(reader.metadata.creation_date)[:10]
            except Exception:
                pass
    if not meta["date"]:
        meta["date"] = _extract_date(text[:2000])
    meta["text"] = text
    return meta


# ─── DOCX ────────────────────────────────────────────────────────────────────
def _load_docx(path: Path) -> dict:
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx not installed — run: pip install python-docx")

    doc = docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)

    meta = _base_meta(path)
    # Core properties
    try:
        cp = doc.core_properties
        if cp.title:
            meta["title"] = cp.title
        if cp.created:
            meta["date"] = str(cp.created)[:10]
    except Exception:
        pass
    if not meta["date"]:
        meta["date"] = _extract_date(text[:2000])
    meta["text"] = text
    return meta


# ─── Plain text / Markdown ────────────────────────────────────────────────────
def _load_text(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta = _base_meta(path)
    meta["date"] = _extract_date(text[:2000])
    meta["text"] = text
    return meta


# ─── JSON ─────────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    """
    Accepts either:
      - A list of {"page": int, "text": str} objects (our nlplecture.json format)
      - A plain dict with a "text" key
      - Any other structure → JSON-dumped as text
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        # Page-by-page format
        pages = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                pages.append(item["text"])
            else:
                pages.append(str(item))
        text = "\n".join(pages)
    elif isinstance(raw, dict) and "text" in raw:
        text = raw["text"]
    else:
        text = json.dumps(raw, ensure_ascii=False, indent=2)

    meta = _base_meta(path)
    meta["date"] = _extract_date(text[:2000])
    meta["text"] = text
    return meta


# ─── HTML ────────────────────────────────────────────────────────────────────
def _load_html(path: Path) -> dict:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 not installed — run: pip install beautifulsoup4")

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    text = soup.get_text(separator="\n")

    meta = _base_meta(path)
    if title_tag and title_tag.string:
        meta["title"] = title_tag.string.strip()
    meta["date"] = _extract_date(text[:2000])
    meta["text"] = text
    return meta


# ─── Dispatcher ──────────────────────────────────────────────────────────────
_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".doc": _load_docx,
    ".txt": _load_text,
    ".md": _load_text,
    ".markdown": _load_text,
    ".json": _load_json,
    ".html": _load_html,
    ".htm": _load_html,
}


def load_document(path: Path) -> Optional[dict]:
    """Load a single document. Returns None on failure."""
    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        logger.warning("Unsupported file type: %s — skipping", path.name)
        return None
    try:
        doc = loader(path)
        if not doc.get("text", "").strip():
            logger.warning("Empty text extracted from %s — skipping", path.name)
            return None
        logger.info("Loaded %s (%d chars)", path.name, len(doc["text"]))
        return doc
    except Exception as exc:
        logger.error("Failed to load %s: %s", path.name, exc)
        return None


def load_directory(directory: Path, recursive: bool = False) -> list[dict]:
    """Load all supported documents from a directory."""
    pattern = "**/*" if recursive else "*"
    docs = []
    for p in sorted(directory.glob(pattern)):
        if p.is_file() and p.suffix.lower() in _LOADERS:
            doc = load_document(p)
            if doc:
                docs.append(doc)
    logger.info("Loaded %d documents from %s", len(docs), directory)
    return docs
