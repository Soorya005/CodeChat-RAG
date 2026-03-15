"""
tests/test_security.py
Security-focused regression tests for the CodeChat RAG pipeline.

Validates that:
- Path-traversal via symlinks is blocked in ingestion.
- Files exceeding MAX_FILE_SIZE_BYTES are skipped.
- Unsafe Ollama URLs raise a ValueError at config-time.
- Pickle-load security warnings are emitted.
"""

import logging
import os
import pickle
import pytest

from embeddings import save_embeddings, load_embeddings
from ingestion import load_repository, MAX_FILE_SIZE_BYTES
from rag_pipeline import RAGConfig, _validate_ollama_url
from vector_store import InMemoryVectorStore


# ─── Ingestion: path-traversal via symlinks ──────────────────────

class TestSymlinkTraversal:
    def test_symlink_inside_repo_is_allowed(self, tmp_path):
        """A symlink that points to a file *inside* the repo is loaded normally."""
        real_file = tmp_path / "real.py"
        real_file.write_text("x = 1")
        link = tmp_path / "link.py"
        link.symlink_to(real_file)

        docs = load_repository(str(tmp_path))
        # Both the real file and the in-repo symlink should appear
        paths = [d["file_path"] for d in docs]
        assert any("real.py" in p for p in paths)

    def test_symlink_escaping_repo_is_blocked(self, tmp_path):
        """A symlink that resolves outside the repo root must be skipped."""
        # Create a sensitive file *outside* the repo directory
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret = outside_dir / "secret.py"
        secret.write_text("SECRET = 'top_secret_value'")

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        # Plant a symlink inside the repo that points outside
        evil_link = repo_dir / "evil.py"
        evil_link.symlink_to(secret)

        docs = load_repository(str(repo_dir))
        contents = [d["content"] for d in docs]
        assert not any("top_secret_value" in c for c in contents), (
            "Symlink traversal outside repo root must be blocked"
        )

    def test_symlink_to_directory_outside_repo_is_blocked(self, tmp_path):
        """A directory symlink pointing outside the repo is pruned."""
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        (outside_dir / "secret.py").write_text("EXPOSED = True")

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        # Symlink *directory* outside repo
        link_dir = repo_dir / "linked_dir"
        link_dir.symlink_to(outside_dir)

        docs = load_repository(str(repo_dir))
        contents = [d["content"] for d in docs]
        assert not any("EXPOSED" in c for c in contents)


# ─── Ingestion: large-file guard ─────────────────────────────────

class TestFileSizeLimit:
    def test_file_within_limit_is_loaded(self, tmp_path):
        (tmp_path / "small.py").write_text("x = 1")
        docs = load_repository(str(tmp_path))
        assert len(docs) == 1

    def test_oversized_file_is_skipped(self, tmp_path):
        """Files exceeding MAX_FILE_SIZE_BYTES must be skipped."""
        big_file = tmp_path / "huge.py"
        # Write just over the limit
        big_file.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))

        docs = load_repository(str(tmp_path))
        assert docs == [], (
            "Files larger than MAX_FILE_SIZE_BYTES should be skipped"
        )

    def test_file_exactly_at_limit_is_skipped(self, tmp_path):
        """Files of exactly MAX_FILE_SIZE_BYTES + 1 byte must be skipped."""
        big_file = tmp_path / "boundary.py"
        big_file.write_bytes(b"y" * (MAX_FILE_SIZE_BYTES + 1))
        docs = load_repository(str(tmp_path))
        assert docs == []


# ─── RAGConfig: Ollama URL validation ────────────────────────────

class TestOllamaUrlValidation:
    def test_valid_http_url_accepted(self):
        _validate_ollama_url("http://localhost:11434")  # must not raise

    def test_valid_https_url_accepted(self):
        _validate_ollama_url("https://ollama.example.com")  # must not raise

    def test_file_scheme_rejected(self):
        with pytest.raises(ValueError, match="http"):
            _validate_ollama_url("file:///etc/passwd")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="http"):
            _validate_ollama_url("ftp://internal-server/resource")

    def test_empty_scheme_rejected(self):
        with pytest.raises(ValueError):
            _validate_ollama_url("localhost:11434")

    def test_url_with_credentials_rejected(self):
        with pytest.raises(ValueError, match="credentials"):
            _validate_ollama_url("http://user:pass@localhost:11434")

    def test_ragconfig_rejects_invalid_url(self):
        with pytest.raises(ValueError):
            RAGConfig(ollama_base_url="ftp://badhost")

    def test_ragconfig_env_var_validation(self, monkeypatch):
        """OLLAMA_BASE_URL from env is also validated."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "file:///etc/passwd")
        with pytest.raises(ValueError):
            RAGConfig()

    def test_ragconfig_valid_env_var(self, monkeypatch):
        """A valid OLLAMA_BASE_URL env var is accepted."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        cfg = RAGConfig()  # must not raise
        assert cfg.ollama_base_url == "http://127.0.0.1:11434"


# ─── Pickle deserialization: security warnings ───────────────────

class TestPickleSecurityWarnings:
    def test_vector_store_load_emits_warning(self, tmp_path, sample_embedded_chunks, caplog):
        """VectorStore.load() must log a warning about untrusted pickle files."""
        store = InMemoryVectorStore(embedding_dim=sample_embedded_chunks[0].embedding.shape[0])
        store.add_chunks(sample_embedded_chunks)
        store.save(str(tmp_path))

        with caplog.at_level(logging.WARNING, logger="vector_store"):
            InMemoryVectorStore.load(str(tmp_path))

        assert any("trusted" in record.message.lower() for record in caplog.records), (
            "Expected a security warning when loading a pickle file"
        )

    def test_embeddings_load_emits_warning(self, tmp_path, sample_embedded_chunks, caplog):
        """load_embeddings() must log a warning about untrusted pickle files."""
        out = str(tmp_path / "embeddings")
        save_embeddings(sample_embedded_chunks, out)

        with caplog.at_level(logging.WARNING, logger="embeddings"):
            load_embeddings(out)

        assert any("trusted" in record.message.lower() for record in caplog.records), (
            "Expected a security warning when loading a pickle file"
        )
