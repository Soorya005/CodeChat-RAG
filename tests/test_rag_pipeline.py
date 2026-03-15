"""
tests/test_rag_pipeline.py
Integration tests for the RAGPipeline.

All tests mock the SentenceTransformer model so no internet/GPU is needed.
LLM calls are also mocked — the tests exercise the pipeline logic only.
Uses InMemoryVectorStore to avoid a hard faiss-cpu dependency.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from rag_pipeline import RAGConfig, RAGPipeline, RetrievalResult, RAGResponse
from vector_store import InMemoryVectorStore
from embeddings import CodeEmbedder

EMBED_DIM = 384


# ─── Helpers ─────────────────────────────────────────────────────

def _fake_encode(text_or_texts, **kwargs):
    rng = np.random.default_rng(42)
    if isinstance(text_or_texts, list):
        return rng.standard_normal((len(text_or_texts), EMBED_DIM)).astype(np.float32)
    return rng.standard_normal(EMBED_DIM).astype(np.float32)


def _make_mock_embedder() -> CodeEmbedder:
    """CodeEmbedder backed by a mock model — no library download required."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = EMBED_DIM
    mock_model.encode.side_effect = _fake_encode

    embedder = object.__new__(CodeEmbedder)
    embedder.model = mock_model
    embedder.model_name = "mock-model"
    embedder.embedding_dim = EMBED_DIM
    return embedder


# ─── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Minimal RAG config using in-memory store so faiss-cpu is not required."""
    return RAGConfig(
        embedding_model="all-MiniLM-L6-v2",
        index_type="in_memory",   # ← avoids faiss dependency in tests
        top_k=3,
    )


@pytest.fixture
def indexed_pipeline(config, mock_repo_path):
    """
    Fully indexed RAGPipeline using the mock repo and a mocked embedder.
    The embedder is patched in via the pipeline's internal attribute after
    object construction so no library download is needed.
    """
    pipeline = RAGPipeline(config)

    # Inject mock embedder *before* index_repository so it is used for encoding
    mock_embedder = _make_mock_embedder()

    # Patch CodeEmbedder construction inside index_repository
    with patch("rag_pipeline.CodeEmbedder", return_value=mock_embedder):
        pipeline.index_repository(mock_repo_path)

    return pipeline


# ─── RAGPipeline initialisation ──────────────────────────────────

class TestRAGPipelineInit:
    def test_creates_with_default_config(self):
        pipeline = RAGPipeline()
        assert pipeline.config is not None
        assert pipeline.vector_store is None
        assert pipeline.embedder is None

    def test_create_with_custom_config(self, config):
        pipeline = RAGPipeline(config)
        assert pipeline.config.top_k == 3
        assert pipeline.config.index_type == "in_memory"


# ─── index_repository ────────────────────────────────────────────

class TestIndexRepository:
    def test_index_populates_vector_store(self, indexed_pipeline):
        assert indexed_pipeline.vector_store is not None
        assert indexed_pipeline.vector_store.chunk_count > 0

    def test_index_initialises_embedder(self, indexed_pipeline):
        assert indexed_pipeline.embedder is not None

    def test_index_nonexistent_repo_raises(self, config):
        pipeline = RAGPipeline(config)
        with patch("rag_pipeline.CodeEmbedder", return_value=_make_mock_embedder()):
            with pytest.raises(ValueError, match="No supported code files"):
                pipeline.index_repository("/nonexistent/path/xyz")

    def test_index_gets_correct_languages(self, indexed_pipeline):
        stats = indexed_pipeline.vector_store.get_stats()
        assert "python" in stats["languages"]

    def test_index_chunk_types_present(self, indexed_pipeline):
        stats = indexed_pipeline.vector_store.get_stats()
        chunk_types = stats["chunk_types"]
        assert "function" in chunk_types or "class" in chunk_types


# ─── retrieve ────────────────────────────────────────────────────

class TestRetrieve:
    def test_retrieve_returns_retrieval_result(self, indexed_pipeline):
        result = indexed_pipeline.retrieve("login function")
        assert isinstance(result, RetrievalResult)

    def test_retrieve_result_has_chunks(self, indexed_pipeline):
        result = indexed_pipeline.retrieve("authentication")
        assert isinstance(result.chunks, list)

    def test_retrieve_respects_top_k(self, indexed_pipeline):
        result = indexed_pipeline.retrieve("math", top_k=2)
        assert result.total_found <= 2

    def test_retrieve_before_index_raises(self, config):
        pipeline = RAGPipeline(config)
        with pytest.raises(RuntimeError, match="Index not loaded"):
            pipeline.retrieve("anything")

    def test_retrieve_with_filter(self, indexed_pipeline):
        result = indexed_pipeline.retrieve("function", filters={"language": "python"})
        for meta, _ in result.chunks:
            assert meta.language == "python"

    def test_retrieve_query_stored_in_result(self, indexed_pipeline):
        result = indexed_pipeline.retrieve("hello world")
        assert result.query == "hello world"


# ─── save / load index ───────────────────────────────────────────

class TestSaveLoadIndex:
    def test_save_and_load_round_trip(self, tmp_path, config, mock_repo_path):
        pipeline = RAGPipeline(config)
        with patch("rag_pipeline.CodeEmbedder", return_value=_make_mock_embedder()):
            pipeline.index_repository(mock_repo_path, save_path=str(tmp_path / "idx"))

        # In-memory store save/load round-trip
        assert (tmp_path / "idx" / "in_memory_store.pkl").exists()

    def test_loaded_pipeline_can_retrieve(self, tmp_path, config, mock_repo_path):
        pipeline = RAGPipeline(config)
        with patch("rag_pipeline.CodeEmbedder", return_value=_make_mock_embedder()):
            pipeline.index_repository(mock_repo_path, save_path=str(tmp_path / "idx"))

        # Manually reload using InMemoryVectorStore
        pipeline2 = RAGPipeline(config)
        pipeline2.vector_store = InMemoryVectorStore.load(str(tmp_path / "idx"))
        pipeline2.embedder = _make_mock_embedder()

        result = pipeline2.retrieve("add function")
        assert isinstance(result, RetrievalResult)


# ─── query (LLM mocked) ─────────────────────────────────────────

class TestQuery:
    def _inject_llm(self, pipeline):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Mocked LLM answer."
        pipeline.llm_client = mock_llm
        return mock_llm

    def test_query_returns_rag_response(self, indexed_pipeline):
        self._inject_llm(indexed_pipeline)
        response = indexed_pipeline.query("How does login work?")
        assert isinstance(response, RAGResponse)

    def test_query_answer_comes_from_llm(self, indexed_pipeline):
        mock_llm = self._inject_llm(indexed_pipeline)
        mock_llm.generate.return_value = "Custom answer."
        response = indexed_pipeline.query("Explain auth")
        assert response.answer == "Custom answer."

    def test_query_no_chunks_returns_fallback(self, config):
        """When vector store is empty the pipeline should return a graceful fallback."""
        pipeline = RAGPipeline(config)
        pipeline.vector_store = InMemoryVectorStore(EMBED_DIM)  # empty
        pipeline.embedder = _make_mock_embedder()

        response = pipeline.query("anything")
        assert "No relevant code" in response.answer

    def test_query_with_return_context(self, indexed_pipeline):
        self._inject_llm(indexed_pipeline)
        response = indexed_pipeline.query("q", return_context=True)
        assert len(response.context_used) > 0

    def test_query_without_return_context(self, indexed_pipeline):
        self._inject_llm(indexed_pipeline)
        response = indexed_pipeline.query("q", return_context=False)
        assert response.context_used == ""

    def test_query_metadata_has_expected_keys(self, indexed_pipeline):
        self._inject_llm(indexed_pipeline)
        response = indexed_pipeline.query("q")
        for key in ["total_chunks_found", "chunks_used", "llm_model", "embedding_model"]:
            assert key in response.metadata
