"""
tests/test_prompt_builder.py
Unit tests for prompt_builder.build_prompt() and format_code_chunk().
"""

import pytest
from dataclasses import dataclass
from prompt_builder import build_prompt, format_code_chunk, MAX_CONTEXT_CHUNKS


# ─── Minimal ChunkMetadata stub ──────────────────────────────────

@dataclass
class _Meta:
    file_path: str = "/repo/auth.py"
    language: str = "python"
    symbol_name: str = "login"
    chunk_type: str = "function"
    start_line: int = 1
    end_line: int = 5
    source_text: str = "def login(u, p):\n    return authenticate(u, p)\n"


def _chunk(symbol: str = "login", language: str = "python",
           chunk_type: str = "function", score: float = 0.9) -> tuple:
    return (_Meta(symbol_name=symbol, language=language, chunk_type=chunk_type), score)


# ─── format_code_chunk ───────────────────────────────────────────

class TestFormatCodeChunk:
    def test_contains_file_path(self):
        meta, score = _chunk()
        text = format_code_chunk(meta, score)
        assert "/repo/auth.py" in text

    def test_contains_language(self):
        meta, score = _chunk()
        assert "python" in format_code_chunk(meta, score)

    def test_contains_symbol_name(self):
        meta, score = _chunk()
        assert "login" in format_code_chunk(meta, score)

    def test_contains_source_text(self):
        meta, score = _chunk()
        assert "def login" in format_code_chunk(meta, score)

    def test_contains_score(self):
        meta, score = _chunk(score=0.876)
        text = format_code_chunk(meta, 0.876)
        assert "0.876" in text

    def test_contains_line_numbers(self):
        meta, score = _chunk()
        text = format_code_chunk(meta, score)
        assert "1" in text    # start_line
        assert "5" in text    # end_line


# ─── build_prompt ─────────────────────────────────────────────────

class TestBuildPrompt:
    def test_prompt_contains_query(self):
        prompt = build_prompt("How does login work?", [_chunk()])
        assert "How does login work?" in prompt

    def test_prompt_contains_file_path(self):
        prompt = build_prompt("q", [_chunk()])
        assert "/repo/auth.py" in prompt

    def test_prompt_contains_source_text(self):
        prompt = build_prompt("q", [_chunk()])
        assert "def login" in prompt

    def test_prompt_returns_string(self):
        result = build_prompt("q", [_chunk()])
        assert isinstance(result, str)

    def test_prompt_is_not_empty(self):
        assert len(build_prompt("q", [_chunk()])) > 0

    def test_empty_chunks_builds_prompt_anyway(self):
        prompt = build_prompt("What is the answer?", [])
        assert "What is the answer?" in prompt
        assert isinstance(prompt, str)

    def test_multiple_chunks_all_appear_in_prompt(self):
        chunks = [
            _chunk("login",     language="python"),
            _chunk("add",       language="python"),
            _chunk("debounce",  language="javascript"),
        ]
        prompt = build_prompt("usage?", chunks)
        assert "login" in prompt
        assert "add" in prompt
        assert "debounce" in prompt

    def test_max_context_chunks_limit(self):
        """Chunks beyond MAX_CONTEXT_CHUNKS should be silently dropped."""
        many = [_chunk(f"func_{i}") for i in range(MAX_CONTEXT_CHUNKS + 5)]
        prompt = build_prompt("q", many)
        # The (MAX_CONTEXT_CHUNKS+1)-th chunk's name should NOT appear in the prompt
        extra_name = f"func_{MAX_CONTEXT_CHUNKS}"
        assert extra_name not in prompt

    def test_prompt_contains_system_header(self):
        prompt = build_prompt("q", [_chunk()])
        # The prompt template always starts with an LLM instruction
        assert "expert" in prompt.lower() or "engineer" in prompt.lower()

    def test_different_queries_produce_different_prompts(self):
        p1 = build_prompt("question one", [_chunk()])
        p2 = build_prompt("question two", [_chunk()])
        assert p1 != p2
