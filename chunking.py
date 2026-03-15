"""
chunking.py
Handles text chunking for RAG pipeline.

Install deps:
    pip install tree-sitter tree-sitter-languages
"""

import ast
import logging
import os
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

try:
    from tree_sitter import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    logger.warning("tree_sitter library not found. Code chunking will be limited to Python only.")
    TREE_SITTER_AVAILABLE = False


# ─── Core Data Structures ──────────────────────────────────────
# Always defined at module level regardless of tree-sitter availability.

@dataclass
class CodeChunk:
    """A single chunk of code extracted from a source file."""
    file_path: str
    source_text: str
    symbol_name: str
    start_line: int
    end_line: int
    chunk_type: str   # "function", "class", "method", "file"
    language: str


EXT_TO_LANGUAGE = {
    ".py":  "python",
    ".js":  "javascript",
    ".java": "java",
    ".jsx": "javascript",
    ".ts":  "typescript",
    ".tsx": "typescript",
}


# ─── Tree-sitter node types to chunk ────────────────────────────

TS_CHUNK_NODE_TYPES = {
    "javascript": {
        "function_declaration",
        "arrow_function",
        "function_expression",
        "class_declaration",
        "method_definition",
    },
    "typescript": {
        "function_declaration",
        "arrow_function",
        "function_expression",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
    },
    "tsx": {
        "function_declaration",
        "arrow_function",
        "function_expression",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "jsx_element",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "method_declaration",
        "constructor_declaration",
        "enum_declaration",
    },
}


# ─── Helpers ────────────────────────────────────────────────────

def _make_file_chunk(file_path: str, source_text: str, language: str) -> CodeChunk:
    """Fallback: treat the entire file as a single chunk."""
    line_count = source_text.count("\n") + 1
    return CodeChunk(
        file_path=file_path,
        source_text=source_text,
        symbol_name="<file>",
        start_line=1,
        end_line=line_count,
        chunk_type="file",
        language=language,
    )


def _get_node_name(node, source_bytes: bytes, language: str) -> str:
    """Extract the name of a tree-sitter AST node."""
    for field in ["name", "identifier"]:
        name_node = node.child_by_field_name(field)
        if name_node:
            return source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")

    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")

    return f"<{node.type}>"


# ─── Python chunker (uses stdlib ast) ───────────────────────────

def chunk_python_file(file_path: str, source_text: str) -> List[CodeChunk]:
    """
    Parse a Python file with ast and extract top-level functions/classes.
    Falls back to a single file chunk on syntax errors.
    """
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        logger.warning("[chunker] SyntaxError in %s: %s — falling back to file chunk", file_path, exc)
        return [_make_file_chunk(file_path, source_text, "python")]

    lines = source_text.splitlines()
    chunks: List[CodeChunk] = []
    seen: set = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
        start_line = node.lineno
        end_line = node.end_lineno  # correct end line (requires Python 3.8+)

        key = (node.name, start_line)
        if key in seen:
            continue
        seen.add(key)

        chunk_text = "\n".join(lines[start_line - 1: end_line])
        chunks.append(CodeChunk(
            file_path=file_path,
            source_text=chunk_text,
            symbol_name=node.name,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            language="python",
        ))

    return chunks if chunks else [_make_file_chunk(file_path, source_text, "python")]


# ─── Non-Python chunker (uses tree-sitter) ──────────────────────

def chunk_with_tree_sitter(file_path: str, source_text: str, language: str) -> List[CodeChunk]:
    """
    Parse non-Python files with tree-sitter and extract meaningful symbols.
    Falls back to a single file chunk when tree-sitter is unavailable or fails.
    """
    if not TREE_SITTER_AVAILABLE:
        return [_make_file_chunk(file_path, source_text, language)]

    try:
        parser = get_parser(language)
        source_bytes = source_text.encode("utf-8")
        tree = parser.parse(source_bytes)
    except Exception as exc:
        logger.warning("[chunker] tree-sitter failed for %s: %s — falling back to file chunk", file_path, exc)
        return [_make_file_chunk(file_path, source_text, language)]

    target_node_types = TS_CHUNK_NODE_TYPES.get(language, set())
    lines = source_text.splitlines()
    chunks: List[CodeChunk] = []

    def _traverse(node):
        if node.type in target_node_types:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            symbol_name = _get_node_name(node, source_bytes, language)
            chunk_text = "\n".join(lines[start_line - 1: end_line])

            if "class" in node.type or "interface" in node.type or "enum" in node.type:
                chunk_type = "class"
            elif "method" in node.type or "constructor" in node.type:
                chunk_type = "method"
            else:
                chunk_type = "function"

            chunks.append(CodeChunk(
                file_path=file_path,
                source_text=chunk_text,
                symbol_name=symbol_name,
                start_line=start_line,
                end_line=end_line,
                chunk_type=chunk_type,
                language=language,
            ))

        for child in node.children:
            _traverse(child)

    _traverse(tree.root_node)
    return chunks if chunks else [_make_file_chunk(file_path, source_text, language)]


# ─── Public API ─────────────────────────────────────────────────

def chunk_code_file(file_path: str, source_text: str) -> List[CodeChunk]:
    """
    Dispatch chunking to the appropriate strategy based on file extension.
    Returns an empty list for unsupported extensions.
    """
    ext = os.path.splitext(file_path)[1].lower()
    language = EXT_TO_LANGUAGE.get(ext)

    if not language:
        logger.debug("[chunker] Unsupported extension for %s — skipping", file_path)
        return []

    if language == "python":
        return chunk_python_file(file_path, source_text)
    else:
        return chunk_with_tree_sitter(file_path, source_text, language)


def chunk_repository(ingested_files: List[dict]) -> List[CodeChunk]:
    """
    Chunk all files loaded by ingestion.load_repository().

    Args:
        ingested_files: List of {"file_path": str, "content": str} dicts
                        as returned by ingestion.load_repository().

    Returns:
        Flat list of CodeChunk objects across all files.
    """
    all_chunks: List[CodeChunk] = []

    for doc in ingested_files:
        file_path = doc["file_path"]
        source_text = doc["content"]
        chunks = chunk_code_file(file_path, source_text)
        all_chunks.extend(chunks)
        logger.info("[chunker] %s → %d chunk(s)", file_path, len(chunks))

    logger.info("[chunker] Total chunks: %d", len(all_chunks))
    return all_chunks


# ─── CLI smoke test ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    from ingestion import load_repository

    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    documents = load_repository(repo_path)
    chunks = chunk_repository(documents)

    print(f"\nTotal chunks: {len(chunks)}")
    for c in chunks[:5]:
        print(f"  [{c.language}] {c.chunk_type} '{c.symbol_name}' "
              f"({c.file_path}:{c.start_line}-{c.end_line})")
