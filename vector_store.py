"""
vector_store.py
Vector storage and retrieval for the RAG pipeline using FAISS.

Install deps:
    pip install faiss-cpu numpy
"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("faiss not found. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False


@dataclass
class ChunkMetadata:
    """Metadata stored alongside each vector in the index."""
    chunk_id: int
    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    chunk_type: str
    language: str
    source_text: str


class VectorStore:
    """
    FAISS-backed vector store for efficient similarity search.

    Supports three index types:
    - "flat"  – exact L2 search, always accurate (good default)
    - "ivf"   – approximate search, faster for >10 k vectors
    - "hnsw"  – graph-based approximate search, fastest for >100 k vectors
    """

    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        Args:
            embedding_dim: Dimension of the embedding vectors.
            index_type:    "flat", "ivf", or "hnsw".
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss library required. Install: pip install faiss-cpu")

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata: List[ChunkMetadata] = []
        self.chunk_count = 0
        self._needs_training = False

        self._build_index()

    def _build_index(self):
        """Build the FAISS index appropriate for index_type."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(
                "[vector_store] Created FLAT index (exact search, dim=%d)", self.embedding_dim
            )

        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            self._needs_training = True
            logger.info(
                "[vector_store] Created IVF index (approximate search, dim=%d)", self.embedding_dim
            )

        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            logger.info(
                "[vector_store] Created HNSW index (fast approximate search, dim=%d)",
                self.embedding_dim,
            )

        else:
            raise ValueError(
                f"Unknown index_type: '{self.index_type}'. Choose 'flat', 'ivf', or 'hnsw'."
            )

    def add_chunks(self, embedded_chunks: List) -> None:
        """
        Add EmbeddedChunk objects to the vector store.

        Args:
            embedded_chunks: List of EmbeddedChunk objects (from embeddings.py).

        Raises:
            ValueError: If embedding dimensions don't match the index.
        """
        if not embedded_chunks:
            logger.warning("[vector_store] add_chunks called with empty list — nothing added.")
            return

        embeddings = np.array(
            [chunk.embedding for chunk in embedded_chunks], dtype=np.float32
        )

        # Validate dimensions before touching the index
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"[vector_store] Embedding dim mismatch: index expects {self.embedding_dim}, "
                f"but received {embeddings.shape[1]}."
            )

        # Train IVF index on first batch
        if self.index_type == "ivf" and self._needs_training:
            logger.info(
                "[vector_store] Training IVF index with %d vectors…", len(embeddings)
            )
            self.index.train(embeddings)
            self._needs_training = False

        self.index.add(embeddings)

        for i, chunk in enumerate(embedded_chunks):
            self.metadata.append(ChunkMetadata(
                chunk_id=self.chunk_count + i,
                file_path=chunk.file_path,
                symbol_name=chunk.symbol_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                chunk_type=chunk.chunk_type,
                language=chunk.language,
                source_text=chunk.source_text,
            ))

        self.chunk_count += len(embedded_chunks)
        logger.info(
            "[vector_store] Added %d chunks. Total: %d", len(embedded_chunks), self.chunk_count
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """
        Search for the top-k most similar chunks.

        Args:
            query_embedding: 1-D numpy array from CodeEmbedder.embed_query().
            top_k:           Maximum number of results to return.
            filters:         Optional dict like {"language": "python"} — values
                             may be a single value or a list of acceptable values.

        Returns:
            List of (ChunkMetadata, score) tuples sorted by similarity
            (higher score = more similar, score is -L2_distance).
        """
        if self.chunk_count == 0:
            logger.warning("[vector_store] Search called on empty store — returning [].")
            return []

        query = np.array([query_embedding], dtype=np.float32)

        # If filters are active we retrieve more candidates and then prune
        search_k = min(top_k * 10 if filters else top_k, self.chunk_count)

        distances, indices = self.index.search(query, search_k)

        results: List[Tuple[ChunkMetadata, float]] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS sentinel for empty slots
                continue

            metadata = self.metadata[idx]

            if filters and not self._matches_filters(metadata, filters):
                continue

            # Convert L2 distance to a similarity score: higher = better
            similarity_score = -float(distance)
            results.append((metadata, similarity_score))

            if len(results) >= top_k:
                break

        logger.info(
            "[vector_store] Search returned %d result(s) (top_k=%d, filters=%s).",
            len(results), top_k, filters,
        )
        return results

    def _matches_filters(self, metadata: ChunkMetadata, filters: Dict) -> bool:
        """Return True if metadata satisfies every filter condition."""
        for key, value in filters.items():
            if not hasattr(metadata, key):
                continue
            attr = getattr(metadata, key)
            if isinstance(value, list):
                if attr not in value:
                    return False
            else:
                if attr != value:
                    return False
        return True

    def get_chunk_by_id(self, chunk_id: int) -> Optional[ChunkMetadata]:
        """Retrieve a chunk by its integer ID."""
        if 0 <= chunk_id < len(self.metadata):
            return self.metadata[chunk_id]
        return None

    def get_stats(self) -> Dict:
        """Return summary statistics about the vector store."""
        languages: Dict[str, int] = {}
        chunk_types: Dict[str, int] = {}

        for meta in self.metadata:
            languages[meta.language] = languages.get(meta.language, 0) + 1
            chunk_types[meta.chunk_type] = chunk_types.get(meta.chunk_type, 0) + 1

        return {
            "total_chunks": self.chunk_count,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "languages": languages,
            "chunk_types": chunk_types,
        }

    def save(self, save_path: str) -> None:
        """
        Persist the index and metadata to a directory.

        Args:
            save_path: Directory path (created if it doesn't exist).
        """
        os.makedirs(save_path, exist_ok=True)

        index_file = os.path.join(save_path, "faiss.index")
        faiss.write_index(self.index, index_file)

        metadata_file = os.path.join(save_path, "metadata.pkl")
        with open(metadata_file, "wb") as f:
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "embedding_dim": self.embedding_dim,
                    "index_type": self.index_type,
                    "chunk_count": self.chunk_count,
                },
                f,
            )

        logger.info(
            "[vector_store] Saved %d chunks to %s (index + metadata).",
            self.chunk_count, save_path,
        )

    @classmethod
    def load(cls, load_path: str) -> "VectorStore":
        """
        Load a previously saved VectorStore from disk.

        Args:
            load_path: Directory containing faiss.index and metadata.pkl.

        Returns:
            Populated VectorStore instance.
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss library required. Install: pip install faiss-cpu")

        metadata_file = os.path.join(load_path, "metadata.pkl")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"[vector_store] metadata.pkl not found in: {load_path}")

        with open(metadata_file, "rb") as f:
            data = pickle.load(f)

        store = cls(
            embedding_dim=data["embedding_dim"],
            index_type=data["index_type"],
        )

        index_file = os.path.join(load_path, "faiss.index")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"[vector_store] faiss.index not found in: {load_path}")

        store.index = faiss.read_index(index_file)
        store.metadata = data["metadata"]
        store.chunk_count = data["chunk_count"]

        logger.info(
            "[vector_store] Loaded %d chunks from %s (dim=%d).",
            store.chunk_count, load_path, store.embedding_dim,
        )
        return store


class InMemoryVectorStore:
    """
    Brute-force cosine-similarity vector store (no FAISS required).
    Suitable for small datasets (<10 k chunks) or when faiss-cpu is unavailable.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[ChunkMetadata] = []
        self.chunk_count = 0

    def add_chunks(self, embedded_chunks: List) -> None:
        """Add embedded chunks to the in-memory store."""
        if not embedded_chunks:
            logger.warning("[in_memory_store] add_chunks called with empty list.")
            return

        for i, chunk in enumerate(embedded_chunks):
            self.embeddings.append(chunk.embedding)
            self.metadata.append(ChunkMetadata(
                chunk_id=self.chunk_count + i,
                file_path=chunk.file_path,
                symbol_name=chunk.symbol_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                chunk_type=chunk.chunk_type,
                language=chunk.language,
                source_text=chunk.source_text,
            ))

        self.chunk_count += len(embedded_chunks)
        logger.info(
            "[in_memory_store] Added %d chunks. Total: %d", len(embedded_chunks), self.chunk_count
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """Search using cosine similarity."""
        if self.chunk_count == 0:
            return []

        norm_query = np.linalg.norm(query_embedding)

        results: List[Tuple[ChunkMetadata, float]] = []
        for emb, meta in zip(self.embeddings, self.metadata):
            if filters:
                matches = True
                for key, value in filters.items():
                    if not hasattr(meta, key):
                        continue
                    attr = getattr(meta, key)
                    if isinstance(value, list):
                        if attr not in value:
                            matches = False
                            break
                    else:
                        if attr != value:
                            matches = False
                            break
                if not matches:
                    continue

            norm_emb = np.linalg.norm(emb)
            if norm_query == 0 or norm_emb == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query_embedding, emb) / (norm_query * norm_emb))

            results.append((meta, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        """Return store statistics."""
        languages: Dict[str, int] = {}
        chunk_types: Dict[str, int] = {}
        for meta in self.metadata:
            languages[meta.language] = languages.get(meta.language, 0) + 1
            chunk_types[meta.chunk_type] = chunk_types.get(meta.chunk_type, 0) + 1
        return {
            "total_chunks": self.chunk_count,
            "embedding_dim": self.embedding_dim,
            "index_type": "in_memory",
            "languages": languages,
            "chunk_types": chunk_types,
        }

    def save(self, save_path: str) -> None:
        """Persist the in-memory store to a pickle file."""
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "in_memory_store.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(
                {
                    "embeddings": self.embeddings,
                    "metadata": self.metadata,
                    "embedding_dim": self.embedding_dim,
                    "chunk_count": self.chunk_count,
                },
                f,
            )
        logger.info("[in_memory_store] Saved to %s", save_file)

    @classmethod
    def load(cls, load_path: str) -> "InMemoryVectorStore":
        """Load an in-memory store from disk."""
        load_file = os.path.join(load_path, "in_memory_store.pkl")
        if not os.path.exists(load_file):
            raise FileNotFoundError(f"[in_memory_store] File not found: {load_file}")

        with open(load_file, "rb") as f:
            data = pickle.load(f)

        store = cls(embedding_dim=data["embedding_dim"])
        store.embeddings = data["embeddings"]
        store.metadata = data["metadata"]
        store.chunk_count = data["chunk_count"]
        logger.info("[in_memory_store] Loaded %d chunks.", store.chunk_count)
        return store


# ─── Convenience factory ─────────────────────────────────────────

def create_vector_store(embedded_chunks: List, index_type: str = "flat"):
    """
    Create and populate a vector store from a list of EmbeddedChunk objects.

    Args:
        embedded_chunks: Non-empty list of EmbeddedChunk objects.
        index_type:      "flat", "ivf", "hnsw" (FAISS), or "in_memory".

    Returns:
        A populated VectorStore or InMemoryVectorStore.

    Raises:
        ValueError: If embedded_chunks is empty.
    """
    if not embedded_chunks:
        raise ValueError("[vector_store] Cannot create store from empty chunk list.")

    embedding_dim = embedded_chunks[0].embedding.shape[0]

    if index_type == "in_memory" or not FAISS_AVAILABLE:
        if not FAISS_AVAILABLE:
            logger.warning("[vector_store] FAISS not available — using in-memory store.")
        store = InMemoryVectorStore(embedding_dim)
    else:
        store = VectorStore(embedding_dim, index_type=index_type)

    store.add_chunks(embedded_chunks)
    return store
