"""
tests/test_ingestion.py
Unit tests for ingestion.load_repository().
"""

import os
import pytest
from ingestion import load_repository


# ─── Happy-path tests ────────────────────────────────────────────

class TestLoadSupportedFiles:
    def test_loads_python_files(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        docs = load_repository(str(tmp_path))
        assert len(docs) == 1

    def test_loads_all_supported_extensions(self, tmp_path):
        for ext in [".py", ".js", ".jsx", ".ts", ".tsx", ".java"]:
            (tmp_path / f"file{ext}").write_text(f"// {ext}")
        docs = load_repository(str(tmp_path))
        assert len(docs) == 6

    def test_document_schema(self, tmp_path):
        (tmp_path / "utils.py").write_text("def foo(): pass")
        docs = load_repository(str(tmp_path))
        assert "file_path" in docs[0]
        assert "content" in docs[0]

    def test_content_is_exact(self, tmp_path):
        content = "x = 42\ny = 'hello'\n"
        (tmp_path / "sample.py").write_text(content)
        docs = load_repository(str(tmp_path))
        assert docs[0]["content"] == content

    def test_file_path_is_absolute(self, tmp_path):
        (tmp_path / "main.py").write_text("pass")
        docs = load_repository(str(tmp_path))
        assert os.path.isabs(docs[0]["file_path"])


# ─── Filter / ignore tests ───────────────────────────────────────

class TestFiltering:
    def test_ignores_unsupported_extensions(self, tmp_path):
        (tmp_path / "readme.md").write_text("# readme")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "script.sh").write_text("echo hi")
        (tmp_path / "main.py").write_text("x = 1")
        docs = load_repository(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["file_path"].endswith("main.py")

    def test_ignores_git_directory(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config.py").write_text("secret = True")
        (tmp_path / "main.py").write_text("real code")
        docs = load_repository(str(tmp_path))
        assert len(docs) == 1

    def test_ignores_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.js").write_text("module.exports = {}")
        (tmp_path / "app.js").write_text("console.log('hi')")
        docs = load_repository(str(tmp_path))
        assert len(docs) == 1

    def test_ignores_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "module.py").write_text("cached = True")
        (tmp_path / "logic.py").write_text("x = 1")
        docs = load_repository(str(tmp_path))
        assert len(docs) == 1

    def test_ignores_venv_directory(self, tmp_path):
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "activate.py").write_text("# venv")
        (tmp_path / "app.py").write_text("import os")
        docs = load_repository(str(tmp_path))
        assert len(docs) == 1

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.py").write_text("")
        (tmp_path / "whitespace.py").write_text("   \n  \t\n")
        (tmp_path / "real.py").write_text("x = 1")
        docs = load_repository(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["file_path"].endswith("real.py")


# ─── Edge-case / error tests ─────────────────────────────────────

class TestEdgeCases:
    def test_empty_directory_returns_empty_list(self, tmp_path):
        docs = load_repository(str(tmp_path))
        assert docs == []

    def test_nonexistent_path_returns_empty_list(self):
        docs = load_repository("/this/path/does/not/exist/at/all")
        assert docs == []

    def test_nested_directories_are_walked(self, tmp_path):
        (tmp_path / "pkg" / "sub").mkdir(parents=True)
        (tmp_path / "pkg" / "a.py").write_text("a = 1")
        (tmp_path / "pkg" / "sub" / "b.py").write_text("b = 2")
        docs = load_repository(str(tmp_path))
        assert len(docs) == 2

    def test_uses_mock_repo(self, mock_repo_path):
        """Integration check: mock_repo must produce at least 2 documents."""
        docs = load_repository(mock_repo_path)
        assert len(docs) >= 2
        paths = [d["file_path"] for d in docs]
        assert any("auth.py" in p for p in paths)
        assert any("math_utils.py" in p for p in paths)
