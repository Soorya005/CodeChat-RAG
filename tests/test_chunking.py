"""
tests/test_chunking.py
Unit tests for chunking.chunk_code_file() and chunk_repository().
"""

import os
import pytest
from chunking import CodeChunk, chunk_code_file, chunk_python_file, chunk_repository


# ─── Helpers ────────────────────────────────────────────────────

def write_py(tmp_path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


# ─── CodeChunk data structure ────────────────────────────────────

class TestCodeChunkDataclass:
    def test_fields_exist(self, sample_code_chunk):
        for field in ["file_path", "source_text", "symbol_name",
                       "start_line", "end_line", "chunk_type", "language"]:
            assert hasattr(sample_code_chunk, field)

    def test_values_set_correctly(self, sample_code_chunk):
        assert sample_code_chunk.symbol_name == "login"
        assert sample_code_chunk.language == "python"
        assert sample_code_chunk.chunk_type == "function"


# ─── Python chunking ─────────────────────────────────────────────

class TestChunkPythonFile:
    def test_simple_function(self, tmp_path):
        path = write_py(tmp_path, "f.py", "def hello():\n    return 'hi'\n")
        chunks = chunk_python_file(path, open(path).read())
        assert len(chunks) >= 1
        names = [c.symbol_name for c in chunks]
        assert "hello" in names

    def test_class_is_chunked(self, tmp_path):
        path = write_py(tmp_path, "c.py", "class MyClass:\n    def method(self): pass\n")
        chunks = chunk_python_file(path, open(path).read())
        types = {c.chunk_type for c in chunks}
        assert "class" in types

    def test_async_function_chunked(self, tmp_path):
        src = "async def fetch(url):\n    pass\n"
        path = write_py(tmp_path, "a.py", src)
        chunks = chunk_python_file(path, src)
        names = [c.symbol_name for c in chunks]
        assert "fetch" in names

    def test_syntax_error_returns_file_chunk(self, tmp_path):
        src = "def broken(:\n    pass\n"
        path = write_py(tmp_path, "bad.py", src)
        chunks = chunk_python_file(path, src)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "file"
        assert chunks[0].symbol_name == "<file>"

    def test_chunk_line_numbers_are_correct(self, tmp_path):
        src = "x = 1\n\ndef foo():\n    return x\n"
        path = write_py(tmp_path, "ln.py", src)
        chunks = chunk_python_file(path, src)
        foo_chunks = [c for c in chunks if c.symbol_name == "foo"]
        assert foo_chunks, "Expected chunk for 'foo'"
        assert foo_chunks[0].start_line == 3
        assert foo_chunks[0].end_line == 4

    def test_source_text_matches_function_body(self, tmp_path):
        src = "def greet(name):\n    return f'Hello {name}'\n"
        path = write_py(tmp_path, "g.py", src)
        chunks = chunk_python_file(path, src)
        assert chunks[0].source_text.strip().startswith("def greet")

    def test_empty_file_returns_file_chunk(self, tmp_path):
        src = "\n\n\n"
        path = write_py(tmp_path, "empty.py", src)
        chunks = chunk_python_file(path, src)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "file"

    def test_language_is_python(self, tmp_path):
        src = "def foo(): pass\n"
        path = write_py(tmp_path, "x.py", src)
        chunks = chunk_python_file(path, src)
        assert all(c.language == "python" for c in chunks)


# ─── chunk_code_file dispatcher ──────────────────────────────────

class TestChunkCodeFile:
    def test_python_extension_dispatched(self, tmp_path):
        path = write_py(tmp_path, "m.py", "def run(): pass\n")
        chunks = chunk_code_file(path, open(path).read())
        assert len(chunks) >= 1

    def test_unsupported_extension_returns_empty(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("some text")
        chunks = chunk_code_file(str(p), "some text")
        assert chunks == []

    def test_markdown_file_returns_empty(self, tmp_path):
        p = tmp_path / "README.md"
        p.write_text("# Title\nsome content")
        chunks = chunk_code_file(str(p), open(p).read())
        assert chunks == []


# ─── chunk_repository ────────────────────────────────────────────

class TestChunkRepository:
    def test_aggregates_chunks_from_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo(): pass\ndef bar(): pass\n")
        (tmp_path / "b.py").write_text("def baz(): pass\n")

        docs = [
            {"file_path": str(tmp_path / "a.py"), "content": open(tmp_path / "a.py").read()},
            {"file_path": str(tmp_path / "b.py"), "content": open(tmp_path / "b.py").read()},
        ]
        chunks = chunk_repository(docs)
        names = [c.symbol_name for c in chunks]
        assert "foo" in names
        assert "baz" in names

    def test_empty_docs_returns_empty_list(self):
        chunks = chunk_repository([])
        assert chunks == []

    def test_unsupported_files_produce_no_chunks(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text('{"key": "value"}')
        docs = [{"file_path": str(p), "content": open(p).read()}]
        chunks = chunk_repository(docs)
        assert chunks == []

    def test_uses_mock_repo(self, mock_repo_path):
        """Integration: mock_repo should produce chunks including auth functions."""
        from ingestion import load_repository
        docs = load_repository(mock_repo_path)
        chunks = chunk_repository(docs)
        assert len(chunks) > 0
        names = [c.symbol_name for c in chunks]
        assert "login" in names
        assert "add" in names
