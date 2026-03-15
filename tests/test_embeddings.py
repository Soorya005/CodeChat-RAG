"""
tests/test_embeddings.py
Unit tests for embeddings.py.

SentenceTransformer is mocked at the CLASS level inside CodeEmbedder so tests
work whether or not sentence-transformers is installed.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from embeddings import (
    EmbeddedChunk,
    cosine_similarity,
    save_embeddings,
    load_embeddings,
)

EMBED_DIM = 384  # matches all-MiniLM-L6-v2


# ─── cosine_similarity (pure math, no mocks needed) ──────────────

class TestCosineSimilarity:
    def test_identical_vectors_returns_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_returns_minus_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors_returns_zero(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        v1 = np.zeros(4, dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == 0.0

    def test_both_zero_vectors_returns_zero(self):
        v = np.zeros(4, dtype=np.float32)
        assert cosine_similarity(v, v) == 0.0

    def test_value_in_valid_range(self):
        rng = np.random.default_rng(42)
        v1 = rng.standard_normal(128).astype(np.float32)
        v2 = rng.standard_normal(128).astype(np.float32)
        score = cosine_similarity(v1, v2)
        assert -1.0 <= score <= 1.0


# ─── EmbeddedChunk dataclass ─────────────────────────────────────

class TestEmbeddedChunkSchema:
    def test_has_all_fields(self, sample_embedded_chunk):
        for field in [
            "file_path", "source_text", "symbol_name",
            "start_line", "end_line", "chunk_type", "language",
            "embedding", "metadata",
        ]:
            assert hasattr(sample_embedded_chunk, field)

    def test_embedding_is_ndarray(self, sample_embedded_chunk):
        assert isinstance(sample_embedded_chunk.embedding, np.ndarray)

    def test_embedding_shape(self, sample_embedded_chunk):
        assert sample_embedded_chunk.embedding.ndim == 1
        assert sample_embedded_chunk.embedding.shape[0] == EMBED_DIM


# ─── Helper: build a mocked CodeEmbedder ─────────────────────────

def _make_mock_embedder():
    """
    Construct a CodeEmbedder with a fully mocked SentenceTransformer.
    Works even when sentence-transformers is not installed, by patching
    the model attribute directly on the instance after bypassing __init__.
    """
    from embeddings import CodeEmbedder, SENTENCE_TRANSFORMERS_AVAILABLE

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = EMBED_DIM

    def fake_encode(text_or_texts, **kwargs):
        rng = np.random.default_rng(0)
        if isinstance(text_or_texts, list):
            return rng.standard_normal(
                (len(text_or_texts), EMBED_DIM)
            ).astype(np.float32)
        return rng.standard_normal(EMBED_DIM).astype(np.float32)

    mock_model.encode.side_effect = fake_encode

    # Bypass __init__ to avoid requiring the real library
    embedder = object.__new__(CodeEmbedder)
    embedder.model = mock_model
    embedder.model_name = "mock-model"
    embedder.embedding_dim = EMBED_DIM
    return embedder


@pytest.fixture
def mock_embedder():
    return _make_mock_embedder()


# ─── CodeEmbedder tests ───────────────────────────────────────────

class TestCodeEmbedder:
    def test_embed_chunk_returns_embedded_chunk(self, mock_embedder, sample_code_chunk):
        result = mock_embedder.embed_chunk(sample_code_chunk)
        assert isinstance(result, EmbeddedChunk)

    def test_embed_chunk_preserves_metadata(self, mock_embedder, sample_code_chunk):
        result = mock_embedder.embed_chunk(sample_code_chunk)
        assert result.symbol_name == sample_code_chunk.symbol_name
        assert result.language == sample_code_chunk.language
        assert result.file_path == sample_code_chunk.file_path

    def test_embed_chunk_embedding_shape(self, mock_embedder, sample_code_chunk):
        result = mock_embedder.embed_chunk(sample_code_chunk)
        assert result.embedding.shape == (EMBED_DIM,)

    def test_embed_chunks_batch(self, mock_embedder, sample_code_chunks):
        results = mock_embedder.embed_chunks(sample_code_chunks)
        assert len(results) == len(sample_code_chunks)
        assert all(isinstance(r, EmbeddedChunk) for r in results)

    def test_embed_chunks_empty_list(self, mock_embedder):
        results = mock_embedder.embed_chunks([])
        assert results == []

    def test_embed_query_returns_ndarray(self, mock_embedder):
        result = mock_embedder.embed_query("find authentication function")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_embed_chunks_order_preserved(self, mock_embedder, sample_code_chunks):
        results = mock_embedder.embed_chunks(sample_code_chunks)
        for original, embedded in zip(sample_code_chunks, results):
            assert embedded.symbol_name == original.symbol_name

    def test_metadata_contains_model_info(self, mock_embedder, sample_code_chunk):
        result = mock_embedder.embed_chunk(sample_code_chunk)
        assert "model" in result.metadata
        assert "embedding_dim" in result.metadata
        assert result.metadata["embedding_dim"] == EMBED_DIM


# ─── save / load round-trip ──────────────────────────────────────

class TestSaveLoadEmbeddings:
    def test_save_and_load_round_trip(self, tmp_path, sample_embedded_chunks):
        save_path = str(tmp_path / "embeds")
        save_embeddings(sample_embedded_chunks, save_path)

        loaded = load_embeddings(save_path)
        assert len(loaded) == len(sample_embedded_chunks)
        for orig, result in zip(sample_embedded_chunks, loaded):
            assert orig.symbol_name == result.symbol_name
            np.testing.assert_array_almost_equal(orig.embedding, result.embedding)

    def test_load_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_embeddings(str(tmp_path / "nonexistent"))

    def test_save_empty_list_does_not_crash(self, tmp_path):
        # Should log a warning but not raise
        save_embeddings([], str(tmp_path / "empty"))
