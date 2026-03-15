"""
embeddings.py
Generates embeddings for code chunks in the RAG pipeline.

Install deps:
    pip install sentence-transformers torch numpy
"""

import logging
import os
import pickle
from typing import List, Optional

import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning(
        "sentence-transformers not found. Install with: pip install sentence-transformers"
    )
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class EmbeddedChunk:
    """Code chunk paired with its embedding vector."""
    file_path: str
    source_text: str
    symbol_name: str
    start_line: int
    end_line: int
    chunk_type: str
    language: str
    embedding: np.ndarray
    metadata: Optional[dict] = None


class CodeEmbedder:
    """Handles embedding generation for code chunks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialise the embedder with a SentenceTransformer model.

        Args:
            model_name: HuggingFace model name. Options:
                - "all-MiniLM-L6-v2"   (default, fast, 384 dims)
                - "all-mpnet-base-v2"  (better quality, 768 dims)
            device: "cuda", "cpu", or None (auto-detect).
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers library required. "
                "Install with: pip install sentence-transformers"
            )

        logger.info("[embedder] Loading model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        logger.info("[embedder] Model loaded. Embedding dimension: %d", self.embedding_dim)

    def _format_chunk_for_embedding(self, chunk) -> str:
        """Format a CodeChunk into text optimised for semantic embedding."""
        header = f"[{chunk.language}] [{chunk.chunk_type}] {chunk.symbol_name}\n"
        file_context = f"File: {chunk.file_path}\n"
        return f"{header}{file_context}\n{chunk.source_text}"

    def embed_chunk(self, chunk) -> EmbeddedChunk:
        """Generate an embedding for a single code chunk."""
        text = self._format_chunk_for_embedding(chunk)
        embedding = self.model.encode(text, convert_to_numpy=True)

        return EmbeddedChunk(
            file_path=chunk.file_path,
            source_text=chunk.source_text,
            symbol_name=chunk.symbol_name,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type,
            language=chunk.language,
            embedding=embedding,
            metadata={
                "model": self.model_name,
                "embedding_dim": self.embedding_dim,
            },
        )

    def embed_chunks(
        self,
        chunks: List,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[EmbeddedChunk]:
        """
        Generate embeddings for a list of chunks using batching.

        Args:
            chunks:        List of CodeChunk objects.
            batch_size:    Number of chunks to encode in one batch.
            show_progress: Whether to display a progress bar.

        Returns:
            List of EmbeddedChunk objects in the same order as input.
        """
        if not chunks:
            logger.warning("[embedder] embed_chunks called with empty list.")
            return []

        logger.info("[embedder] Embedding %d chunks (batch_size=%d)…", len(chunks), batch_size)

        texts = [self._format_chunk_for_embedding(chunk) for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        embedded_chunks: List[EmbeddedChunk] = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append(EmbeddedChunk(
                file_path=chunk.file_path,
                source_text=chunk.source_text,
                symbol_name=chunk.symbol_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                chunk_type=chunk.chunk_type,
                language=chunk.language,
                embedding=embedding,
                metadata={
                    "model": self.model_name,
                    "embedding_dim": self.embedding_dim,
                },
            ))

        logger.info("[embedder] Embedded %d chunks successfully.", len(embedded_chunks))
        return embedded_chunks

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a natural-language or code search query.

        Returns:
            1-D numpy array of shape (embedding_dim,).
        """
        return self.model.encode(query, convert_to_numpy=True)


# ─── Utility functions ───────────────────────────────────────────

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return the cosine similarity between two vectors (range [−1, 1])."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def search_chunks(
    query: str,
    embedded_chunks: List[EmbeddedChunk],
    embedder: CodeEmbedder,
    top_k: int = 5,
) -> List[tuple]:
    """
    Brute-force cosine-similarity search over embedded_chunks.

    Returns:
        List of (EmbeddedChunk, similarity_score) tuples, sorted descending.
    """
    if not embedded_chunks:
        return []

    query_embedding = embedder.embed_query(query)

    results = [
        (chunk, cosine_similarity(query_embedding, chunk.embedding))
        for chunk in embedded_chunks
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def save_embeddings(embedded_chunks: List[EmbeddedChunk], output_path: str) -> None:
    """
    Persist embedded chunks to disk (pickle).

    Args:
        embedded_chunks: List of EmbeddedChunk objects.
        output_path:     Destination path *without* extension — `.pkl` appended.
    """
    if not embedded_chunks:
        logger.warning("[embedder] save_embeddings: nothing to save.")
        return

    data = {
        "chunks": embedded_chunks,
        "count": len(embedded_chunks),
        "model": embedded_chunks[0].metadata["model"],
        "embedding_dim": embedded_chunks[0].metadata["embedding_dim"],
    }

    save_file = f"{output_path}.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(data, f)

    logger.info("[embedder] Saved %d embedded chunks to %s", len(embedded_chunks), save_file)


def load_embeddings(input_path: str) -> List[EmbeddedChunk]:
    """
    Load embedded chunks from a pickle file saved by save_embeddings().

    Args:
        input_path: Path used when saving (without `.pkl` extension).

    Returns:
        List of EmbeddedChunk objects.
    """
    load_file = f"{input_path}.pkl"

    if not os.path.exists(load_file):
        raise FileNotFoundError(f"[embedder] Embeddings file not found: {load_file}")

    with open(load_file, "rb") as f:
        data = pickle.load(f)

    logger.info(
        "[embedder] Loaded %d chunks (model: %s, dim: %d)",
        data["count"],
        data["model"],
        data["embedding_dim"],
    )
    return data["chunks"]


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    from chunking import chunk_repository
    from ingestion import load_repository

    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."

    print("\n=== Step 1: Loading Repository ===")
    documents = load_repository(repo_path)
    chunks = chunk_repository(documents)

    if not chunks:
        print("No chunks found!")
        sys.exit(1)

    print("\n=== Step 2: Generating Embeddings ===")
    embedder = CodeEmbedder(model_name="all-MiniLM-L6-v2")
    embedded_chunks = embedder.embed_chunks(chunks)

    print("\n=== Step 3: Testing Search ===")
    for query in ["function that loads files", "class definition", "error handling code"]:
        print(f"\nQuery: '{query}'")
        results = search_chunks(query, embedded_chunks, embedder, top_k=3)
        for i, (chunk, score) in enumerate(results, 1):
            print(f"  {i}. [{chunk.language}] {chunk.symbol_name} (score: {score:.3f})")
            print(f"     {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
