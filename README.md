# DLILO RAG System

Production-ready Retrieval-Augmented Generation (RAG) chatbot pipeline with ingestion, chunking, embedding, retrieval, grounded generation, evaluation, and a Gradio UI.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

- `OPENAI_API_KEY` (required for OpenAI provider)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `LLM_PROVIDER` (`openai`, `ollama`, or `mock`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3.1`)
- `CHUNKING_STRATEGY` (`fixed` or `sentence`)
- `CHUNK_SIZE_TOKENS` (default: `300`)
- `CHUNK_OVERLAP_TOKENS` (default: `60`)
- `MIN_CHUNK_TOKENS` (default: `100`)
- `TOP_K` (default: `5`)
- `MIN_RETRIEVAL_SCORE` (default: `0.2`)
- `VECTOR_STORE_BACKEND` (default: `faiss`)
- `VECTOR_STORE_PATH` (default: `data/processed/faiss.index`)
- `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR` (optional overrides)

## Commands

```bash
python main.py ingest
python main.py index
python main.py query "What is the default chunk size?"
python main.py eval
python main.py eval --with-generation
python main.py ui
```

## Project Layout

```
config.py
main.py
ingest/        # loaders + chunking + pipeline
retrieval/     # embeddings + vector store + retriever
generation/    # grounded generation + prompts
ui/            # Gradio chat UI
eval/          # eval dataset + experiment log + eval runner
data/raw/      # knowledge base documents
```

## Notes

- The evaluation dataset is in `eval/qa_dataset.json` and includes 30 QA pairs.
- The experiment log is in `eval/experiment_log.csv`.
- The `mock` LLM provider is useful for offline testing (returns a refusal).
