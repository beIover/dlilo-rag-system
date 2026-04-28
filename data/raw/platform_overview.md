# DLILO RAG System Overview

Date: 2025-03-12

The DLILO RAG System is a reference implementation of a retrieval-augmented generation chatbot.
It connects document ingestion, chunking, embedding, vector indexing, retrieval, and grounded
answer generation. The default knowledge base lives under `data/raw` and processed artifacts
are written to `data/processed`.

## Ingestion
- The ingestion pipeline supports PDF, Markdown, TXT, HTML, and DOCX files.
- Each document stores metadata: source filename, title, date, and document type.
- Ingestion writes chunk files named `chunks_<strategy>.json`.

## Chunking
We implement two chunking strategies:
1. Fixed-size chunks with overlap.
2. Sentence-aware recursive chunks.

The default chunk size is **300 tokens** with **60 tokens overlap**.
The minimum chunk size is **100 tokens**.

## Embeddings & Vector Store
All chunks are encoded with the **sentence-transformers/all-MiniLM-L6-v2** model.
Vectors are stored in a **FAISS IndexFlatIP** index using cosine similarity.

## Retrieval
The retriever returns the **top-5** most similar chunks for every query.

## Generation & Grounding
The generator uses either **OpenAI (gpt-4o-mini)** or **Ollama (llama3.1)**.
Every factual claim must cite the source document name.
If the answer is not supported by the retrieved context, the assistant must say:
"I cannot find this in the provided documents."

## Evaluation
We track retrieval precision@5, answer relevance, and faithfulness metrics.

## UI
The chat interface is built with **Gradio** and defaults to port **7860**.
