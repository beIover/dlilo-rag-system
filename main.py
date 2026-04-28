"""
main.py — Top-level orchestrator for the RAG Chatbot pipeline.

Commands:
    python main.py ingest          # load + chunk documents
    python main.py index           # embed + store vectors
    python main.py query "..."     # single query (CLI)
    python main.py eval            # run evaluation
    python main.py ui              # launch Gradio chat UI
    python main.py all             # ingest + index + eval + ui
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

import config


def cmd_ingest(args):
    from ingest.pipeline import run_ingestion
    run_ingestion(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


def cmd_index(args):
    import json
    from retrieval.retriever import build_index

    chunk_path = config.PROCESSED_DIR / f"chunks_{args.strategy}.json"
    if not chunk_path.exists():
        logger.error("Chunks file not found: %s. Run 'python main.py ingest' first.", chunk_path)
        sys.exit(1)

    with open(chunk_path, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks from %s", len(chunks), chunk_path)
    store = build_index(chunks, embedding_model=config.EMBEDDING_MODEL, backend=config.VECTOR_STORE_BACKEND, reset=True)
    logger.info("Index ready. Vectors: %d", store.count())


def cmd_query(args):
    import json
    from retrieval.retriever import Retriever, build_index
    from generation.generator import RAGGenerator

    # Rebuild or load
    chunk_path = config.PROCESSED_DIR / f"chunks_{config.EFFECTIVE_CHUNKING_STRATEGY}.json"
    if not chunk_path.exists():
        from ingest.pipeline import run_ingestion
        run_ingestion()
    with open(chunk_path, encoding="utf-8") as f:
        chunks = json.load(f)

    store = build_index(chunks, backend=config.VECTOR_STORE_BACKEND, reset=True)
    retriever = Retriever(vector_store=store)
    generator = RAGGenerator()

    query = args.query
    print(f"\n🔍 Query: {query}\n")
    retrieved = retriever.retrieve(query)

    print(f"📄 Retrieved {len(retrieved)} chunks:")
    for i, c in enumerate(retrieved, 1):
        print(f"  [{i}] {c['source']}  (score={c['score']:.3f})")

    result = generator.answer(query, retrieved, return_metadata=True)
    print(f"\n🤖 Answer:\n{result['answer']}")
    print(f"\n📌 Sources: {', '.join(result['sources'])}")


def cmd_eval(args):
    from eval.run_eval import main as eval_main
    # Delegate to eval runner
    sys.argv = ["eval/run_eval.py"]
    if args.with_generation:
        sys.argv.append("--with-generation")
    eval_main()


def cmd_ui(args):
    from ui.app import main as ui_main
    sys.argv = ["ui/app.py", "--port", str(args.port)]
    if args.share:
        sys.argv.append("--share")
    ui_main()


def cmd_all(args):
    """Run the full pipeline: ingest → index → eval → ui."""
    cmd_ingest(args)
    cmd_index(args)
    cmd_eval(args)
    cmd_ui(args)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG Chatbot — NLP Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strategy", choices=["fixed", "sentence", "both"], default=config.CHUNKING_STRATEGY)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE_TOKENS)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP_TOKENS)

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ingest", help="Load and chunk documents")
    subparsers.add_parser("index", help="Embed chunks and build vector index")

    q_parser = subparsers.add_parser("query", help="Run a single query")
    q_parser.add_argument("query", help="Question to ask")

    e_parser = subparsers.add_parser("eval", help="Run evaluation suite")
    e_parser.add_argument("--with-generation", action="store_true")

    ui_parser = subparsers.add_parser("ui", help="Launch Gradio chat UI")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--share", action="store_true")

    subparsers.add_parser("all", help="Run full pipeline (ingest+index+eval+ui)")

    args = parser.parse_args()

    commands = {
        "ingest": cmd_ingest,
        "index": cmd_index,
        "query": cmd_query,
        "eval": cmd_eval,
        "ui": cmd_ui,
        "all": cmd_all,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands[args.command](args)


if __name__ == "__main__":
    main()
