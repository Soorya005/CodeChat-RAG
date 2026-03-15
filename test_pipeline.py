"""
test_pipeline.py
End-to-end pipeline validation script.

Loads the mock repository, chunks files, generates embeddings, stores in FAISS,
performs similarity search, builds a prompt, and prints the final prompt.

No LLM is called — this script validates everything up to the LLM stage.

Usage:
    cd /media/soorya/Personal/Backend/app/rag
    python test_pipeline.py [--repo PATH] [--query "your question"]
"""

import argparse
import logging
import os
import sys
import time

# ── Bootstrap ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_pipeline")

# Make sure imports resolve whether run from the rag/ dir or the repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ingestion import load_repository
from chunking import chunk_repository
from embeddings import CodeEmbedder
from vector_store import create_vector_store
from prompt_builder import build_prompt


SEPARATOR = "=" * 70
MOCK_REPO = os.path.join(SCRIPT_DIR, "tests", "mock_repo")


def header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def run_pipeline(repo_path: str, query: str, top_k: int = 5) -> str:
    """
    Execute the full RAG pipeline (without LLM) and return the final prompt.

    Steps:
        1. Ingestion
        2. Chunking
        3. Embedding (real model, all-MiniLM-L6-v2)
        4. FAISS indexing
        5. Similarity search
        6. Prompt construction

    Args:
        repo_path: Path to the repository to index.
        query:     Natural-language search query.
        top_k:     Number of chunks to retrieve.

    Returns:
        The final LLM prompt string.
    """

    # ── 1. Ingestion ──────────────────────────────────────────
    header("STEP 1 — Ingestion")
    t0 = time.perf_counter()
    documents = load_repository(repo_path)
    elapsed = time.perf_counter() - t0

    if not documents:
        logger.error("No supported files found in: %s", repo_path)
        sys.exit(1)

    print(f"  ✅ Loaded {len(documents)} file(s) in {elapsed:.2f}s")
    for doc in documents:
        print(f"     • {doc['file_path']}")

    # ── 2. Chunking ───────────────────────────────────────────
    header("STEP 2 — Chunking")
    t0 = time.perf_counter()
    chunks = chunk_repository(documents)
    elapsed = time.perf_counter() - t0

    if not chunks:
        logger.error("Chunking produced zero chunks — check file contents.")
        sys.exit(1)

    print(f"  ✅ Created {len(chunks)} chunk(s) in {elapsed:.2f}s")
    lang_counts: dict = {}
    type_counts: dict = {}
    for c in chunks:
        lang_counts[c.language] = lang_counts.get(c.language, 0) + 1
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
    print(f"  Languages : {lang_counts}")
    print(f"  Chunk types: {type_counts}")

    # ── 3. Embedding ──────────────────────────────────────────
    header("STEP 3 — Embedding")
    t0 = time.perf_counter()
    embedder = CodeEmbedder(model_name="all-MiniLM-L6-v2")
    embedded_chunks = embedder.embed_chunks(chunks, show_progress=True)
    elapsed = time.perf_counter() - t0

    print(f"  ✅ Embedded {len(embedded_chunks)} chunk(s) in {elapsed:.2f}s")
    print(f"  Embedding dim: {embedder.embedding_dim}")

    # ── 4. FAISS Indexing ─────────────────────────────────────
    header("STEP 4 — FAISS Indexing")
    t0 = time.perf_counter()
    vector_store = create_vector_store(embedded_chunks, index_type="flat")
    elapsed = time.perf_counter() - t0

    stats = vector_store.get_stats()
    print(f"  ✅ Indexed in {elapsed:.4f}s")
    print(f"  Total vectors : {stats['total_chunks']}")
    print(f"  Embedding dim : {stats['embedding_dim']}")
    print(f"  Index type    : {stats['index_type']}")

    # Validate embedding dim consistency
    assert stats["embedding_dim"] == embedder.embedding_dim, (
        f"DIMENSION MISMATCH: embedder={embedder.embedding_dim}, "
        f"store={stats['embedding_dim']}"
    )
    print("  ✅ Embedding dimensions consistent")

    # ── 5. Similarity Search ──────────────────────────────────
    header(f"STEP 5 — Similarity Search  (query: '{query}')")
    t0 = time.perf_counter()
    query_embedding = embedder.embed_query(query)
    results = vector_store.search(query_embedding, top_k=top_k)
    elapsed = time.perf_counter() - t0

    if not results:
        logger.warning("No results returned for query: '%s'", query)
    else:
        print(f"  ✅ Retrieved {len(results)} result(s) in {elapsed:.4f}s\n")
        for rank, (meta, score) in enumerate(results, 1):
            print(
                f"  #{rank:>2}  [{meta.language}] {meta.symbol_name} "
                f"({meta.chunk_type})  — score: {score:.4f}"
            )
            print(f"       {meta.file_path}:{meta.start_line}–{meta.end_line}")

    # ── 6. Prompt Building ────────────────────────────────────
    header("STEP 6 — Prompt Construction")
    prompt = build_prompt(query, results)

    print(f"  ✅ Prompt built ({len(prompt)} characters, {len(results)} context chunk(s))\n")

    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end RAG pipeline validation (no LLM required)"
    )
    parser.add_argument(
        "--repo",
        default=MOCK_REPO,
        help=f"Repository path to index (default: tests/mock_repo)",
    )
    parser.add_argument(
        "--query",
        default="How does the login authentication function work?",
        help="Search query to use for retrieval",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.repo):
        print(f"ERROR: Repository path does not exist: {args.repo}")
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print(f"  RAG Pipeline Validation Script")
    print(f"  Repository : {args.repo}")
    print(f"  Query      : {args.query}")
    print(f"  Top-K      : {args.top_k}")
    print(f"{'#' * 70}")

    prompt = run_pipeline(args.repo, args.query, top_k=args.top_k)

    header("FINAL PROMPT (ready for LLM)")
    print(prompt)

    print(f"\n{'#' * 70}")
    print("  ✅ Pipeline validation COMPLETE — all stages passed!")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
