"""
tests/conftest.py
Shared pytest fixtures for the RAG pipeline test suite.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# Make the rag package importable from any working directory
RAG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)


# ─── Path helpers ────────────────────────────────────────────────

MOCK_REPO_DIR = os.path.join(os.path.dirname(__file__), "mock_repo")


@pytest.fixture(scope="session")
def mock_repo_path() -> str:
    """Return the path to the bundled mock repository."""
    assert os.path.isdir(MOCK_REPO_DIR), f"mock_repo not found at {MOCK_REPO_DIR}"
    return MOCK_REPO_DIR


@pytest.fixture
def tmp_repo(tmp_path) -> str:
    """
    Create a temporary repository with a mix of file types.
    Returns the path to the temp directory.
    """
    files = {
        "src/main.py": (
            "def greet(name: str) -> str:\n"
            "    return f'Hello, {name}!'\n"
            "\n"
            "class Greeter:\n"
            "    def say_hello(self, name):\n"
            "        return greet(name)\n"
        ),
        "src/helpers.py": (
            "def clamp(value, lo, hi):\n"
            "    '''Clamp value between lo and hi.'''\n"
            "    return max(lo, min(value, hi))\n"
        ),
        "src/app.js": (
            "function init() { console.log('ready'); }\n"
        ),
        "README.md": "# Test repo\n",      # should be ignored
        ".git/config": "[core]\n",          # should be ignored
    }

    for rel_path, content in files.items():
        full = tmp_path / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")

    return str(tmp_path)


# ─── CodeChunk fixtures ──────────────────────────────────────────

@pytest.fixture
def sample_code_chunk():
    """A minimal CodeChunk for testing embeddings and vector store."""
    from chunking import CodeChunk

    return CodeChunk(
        file_path="/repo/auth.py",
        source_text="def login(user, pwd):\n    return authenticate(user, pwd)\n",
        symbol_name="login",
        start_line=1,
        end_line=2,
        chunk_type="function",
        language="python",
    )


@pytest.fixture
def sample_code_chunks():
    """A list of diverse CodeChunks for batch-embedding tests."""
    from chunking import CodeChunk

    return [
        CodeChunk(
            file_path="/repo/auth.py",
            source_text="def login(u, p): pass",
            symbol_name="login",
            start_line=1,
            end_line=1,
            chunk_type="function",
            language="python",
        ),
        CodeChunk(
            file_path="/repo/math.py",
            source_text="def add(a, b): return a + b",
            symbol_name="add",
            start_line=1,
            end_line=1,
            chunk_type="function",
            language="python",
        ),
        CodeChunk(
            file_path="/repo/models.py",
            source_text="class User:\n    pass",
            symbol_name="User",
            start_line=1,
            end_line=2,
            chunk_type="class",
            language="python",
        ),
    ]


# ─── EmbeddedChunk fixtures ──────────────────────────────────────

EMBED_DIM = 384  # matches all-MiniLM-L6-v2


def _make_embedded_chunk(symbol_name: str, language: str = "python",
                         chunk_type: str = "function",
                         seed: int = 0) -> "EmbeddedChunk":
    """Helper to build a fake EmbeddedChunk with a deterministic embedding."""
    from embeddings import EmbeddedChunk

    rng = np.random.default_rng(seed)
    embedding = rng.standard_normal(EMBED_DIM).astype(np.float32)
    embedding /= np.linalg.norm(embedding)  # unit norm

    return EmbeddedChunk(
        file_path=f"/repo/{symbol_name}.py",
        source_text=f"def {symbol_name}(): pass",
        symbol_name=symbol_name,
        start_line=1,
        end_line=1,
        chunk_type=chunk_type,
        language=language,
        embedding=embedding,
        metadata={"model": "mock", "embedding_dim": EMBED_DIM},
    )


@pytest.fixture
def sample_embedded_chunk():
    """A single EmbeddedChunk with a unit-norm random embedding."""
    return _make_embedded_chunk("login")


@pytest.fixture
def sample_embedded_chunks():
    """Several EmbeddedChunks with distinct random embeddings."""
    return [
        _make_embedded_chunk("login",    language="python",     chunk_type="function", seed=0),
        _make_embedded_chunk("add",      language="python",     chunk_type="function", seed=1),
        _make_embedded_chunk("User",     language="python",     chunk_type="class",    seed=2),
        _make_embedded_chunk("debounce", language="javascript", chunk_type="function", seed=3),
        _make_embedded_chunk("init",     language="javascript", chunk_type="function", seed=4),
    ]
