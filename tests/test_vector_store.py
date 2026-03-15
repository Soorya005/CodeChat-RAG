"""
tests/test_vector_store.py
Unit tests for VectorStore, InMemoryVectorStore, and create_vector_store.

FAISS-specific tests are skipped automatically when faiss-cpu is not installed;
all InMemoryVectorStore tests run in all environments.
"""

import os
import numpy as np
import pytest

from vector_store import (
    InMemoryVectorStore,
    ChunkMetadata,
    create_vector_store,
    FAISS_AVAILABLE,
)

EMBED_DIM = 384  # must match conftest fixtures

# Skip FAISS tests if the library is not installed
requires_faiss = pytest.mark.skipif(
    not FAISS_AVAILABLE, reason="faiss-cpu not installed"
)


# ─── Helpers ────────────────────────────────────────────────────

def _query_vec(seed: int = 99) -> np.ndarray:
    """Return a unit-norm query vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


# ─── VectorStore (FAISS) ─────────────────────────────────────────

class TestVectorStoreInit:
    @requires_faiss
    def test_creates_flat_index(self):
        from vector_store import VectorStore
        vs = VectorStore(embedding_dim=EMBED_DIM, index_type="flat")
        assert vs.chunk_count == 0
        assert vs.embedding_dim == EMBED_DIM

    @requires_faiss
    def test_invalid_index_type_raises(self):
        from vector_store import VectorStore
        with pytest.raises(ValueError, match="Unknown index_type"):
            VectorStore(embedding_dim=EMBED_DIM, index_type="unknown")

    @requires_faiss
    def test_invalid_dim_raises(self):
        from vector_store import VectorStore
        with pytest.raises(ValueError):
            VectorStore(embedding_dim=0)


class TestVectorStoreAddSearch:
    @requires_faiss
    def test_add_chunks_increments_count(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        assert vs.chunk_count == len(sample_embedded_chunks)

    @requires_faiss
    def test_search_returns_results(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        results = vs.search(_query_vec(), top_k=3)
        assert len(results) <= 3
        assert len(results) > 0

    @requires_faiss
    def test_search_result_schema(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        results = vs.search(_query_vec(), top_k=1)
        meta, score = results[0]
        assert isinstance(meta, ChunkMetadata)
        assert isinstance(score, float)

    @requires_faiss
    def test_search_empty_store_returns_empty(self):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        results = vs.search(_query_vec(), top_k=5)
        assert results == []

    @requires_faiss
    def test_add_empty_list_is_noop(self):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks([])
        assert vs.chunk_count == 0

    @requires_faiss
    def test_dimension_mismatch_raises(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(embedding_dim=128)  # wrong dim
        with pytest.raises(ValueError, match="dim mismatch"):
            vs.add_chunks(sample_embedded_chunks)

    @requires_faiss
    def test_search_returns_most_similar_first(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        query = sample_embedded_chunks[0].embedding
        results = vs.search(query, top_k=len(sample_embedded_chunks))
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


class TestVectorStoreFilter:
    @requires_faiss
    def test_filter_by_language(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        results = vs.search(_query_vec(), top_k=10, filters={"language": "javascript"})
        for meta, _ in results:
            assert meta.language == "javascript"

    @requires_faiss
    def test_filter_by_chunk_type(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        results = vs.search(_query_vec(), top_k=10, filters={"chunk_type": "class"})
        for meta, _ in results:
            assert meta.chunk_type == "class"

    @requires_faiss
    def test_filter_by_list_of_values(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        results = vs.search(_query_vec(), top_k=10,
                            filters={"language": ["python", "javascript"]})
        for meta, _ in results:
            assert meta.language in ["python", "javascript"]


class TestVectorStoreStats:
    @requires_faiss
    def test_get_stats_keys(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        stats = vs.get_stats()
        for key in ["total_chunks", "embedding_dim", "index_type", "languages", "chunk_types"]:
            assert key in stats

    @requires_faiss
    def test_get_stats_total(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        assert vs.get_stats()["total_chunks"] == len(sample_embedded_chunks)

    @requires_faiss
    def test_get_stats_languages(self, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        langs = vs.get_stats()["languages"]
        assert "python" in langs
        assert "javascript" in langs


class TestVectorStoreSaveLoad:
    @requires_faiss
    def test_save_and_load_round_trip(self, tmp_path, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        save_dir = str(tmp_path / "store")
        vs.save(save_dir)

        loaded = VectorStore.load(save_dir)
        assert loaded.chunk_count == vs.chunk_count
        assert loaded.embedding_dim == vs.embedding_dim

    @requires_faiss
    def test_load_nonexistent_raises(self, tmp_path):
        from vector_store import VectorStore
        with pytest.raises(FileNotFoundError):
            VectorStore.load(str(tmp_path / "ghost"))

    @requires_faiss
    def test_load_and_search(self, tmp_path, sample_embedded_chunks):
        from vector_store import VectorStore
        vs = VectorStore(EMBED_DIM)
        vs.add_chunks(sample_embedded_chunks)
        vs.save(str(tmp_path / "s"))

        loaded = VectorStore.load(str(tmp_path / "s"))
        results = loaded.search(_query_vec(), top_k=3)
        assert len(results) > 0


# ─── InMemoryVectorStore (always runs) ────────────────────────────

class TestInMemoryVectorStore:
    def test_add_and_search(self, sample_embedded_chunks):
        store = InMemoryVectorStore(EMBED_DIM)
        store.add_chunks(sample_embedded_chunks)
        results = store.search(_query_vec(), top_k=3)
        assert len(results) <= 3

    def test_search_empty_returns_empty(self):
        store = InMemoryVectorStore(EMBED_DIM)
        assert store.search(_query_vec()) == []

    def test_cosine_scores_between_minus1_and_1(self, sample_embedded_chunks):
        store = InMemoryVectorStore(EMBED_DIM)
        store.add_chunks(sample_embedded_chunks)
        results = store.search(_query_vec(), top_k=10)
        for _, score in results:
            assert -1.0 <= score <= 1.0

    def test_result_schema(self, sample_embedded_chunks):
        store = InMemoryVectorStore(EMBED_DIM)
        store.add_chunks(sample_embedded_chunks)
        results = store.search(_query_vec(), top_k=1)
        meta, score = results[0]
        assert isinstance(meta, ChunkMetadata)
        assert isinstance(score, float)

    def test_filter_by_language(self, sample_embedded_chunks):
        store = InMemoryVectorStore(EMBED_DIM)
        store.add_chunks(sample_embedded_chunks)
        results = store.search(_query_vec(), top_k=10, filters={"language": "python"})
        for meta, _ in results:
            assert meta.language == "python"

    def test_search_returns_sorted(self, sample_embedded_chunks):
        store = InMemoryVectorStore(EMBED_DIM)
        store.add_chunks(sample_embedded_chunks)
        results = store.search(_query_vec(), top_k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_stats_keys(self, sample_embedded_chunks):
        store = InMemoryVectorStore(EMBED_DIM)
        store.add_chunks(sample_embedded_chunks)
        stats = store.get_stats()
        for k in ["total_chunks", "embedding_dim", "index_type", "languages", "chunk_types"]:
            assert k in stats

    def test_add_empty_list_is_noop(self):
        store = InMemoryVectorStore(EMBED_DIM)
        store.add_chunks([])
        assert store.chunk_count == 0

    def test_save_load_round_trip(self, tmp_path, sample_embedded_chunks):
        store = InMemoryVectorStore(EMBED_DIM)
        store.add_chunks(sample_embedded_chunks)
        store.save(str(tmp_path / "mem"))

        loaded = InMemoryVectorStore.load(str(tmp_path / "mem"))
        assert loaded.chunk_count == store.chunk_count

    def test_search_after_load(self, tmp_path, sample_embedded_chunks):
        store = InMemoryVectorStore(EMBED_DIM)
        store.add_chunks(sample_embedded_chunks)
        store.save(str(tmp_path / "m"))
        loaded = InMemoryVectorStore.load(str(tmp_path / "m"))
        results = loaded.search(_query_vec(), top_k=3)
        assert len(results) > 0


# ─── create_vector_store factory ─────────────────────────────────

class TestCreateVectorStore:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            create_vector_store([])

    def test_in_memory_type(self, sample_embedded_chunks):
        store = create_vector_store(sample_embedded_chunks, index_type="in_memory")
        assert isinstance(store, InMemoryVectorStore)

    def test_in_memory_stores_all_chunks(self, sample_embedded_chunks):
        store = create_vector_store(sample_embedded_chunks, index_type="in_memory")
        assert store.chunk_count == len(sample_embedded_chunks)

    @requires_faiss
    def test_flat_type_returns_vector_store(self, sample_embedded_chunks):
        from vector_store import VectorStore
        store = create_vector_store(sample_embedded_chunks, index_type="flat")
        assert isinstance(store, VectorStore)
