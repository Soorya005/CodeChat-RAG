"""
ingestion.py
Handles data ingestion for the CodeChat RAG pipeline.
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".py", ".js", ".jsx", ".ts", ".tsx", ".java"]
IGNORE_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv"}

# Maximum file size to read (10 MB). Files larger than this are skipped to
# prevent accidental out-of-memory errors when ingesting large binary or
# generated files that happen to carry a supported extension.
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def load_repository(repo_path: str) -> List[dict]:
    """
    Walk the repository and load all supported source files.

    Args:
        repo_path: Absolute or relative path to the repository root.

    Returns:
        List of {"file_path": str, "content": str} dicts.
        Returns an empty list if the path does not exist or contains no
        supported files.

    Security notes:
        - Symbolic links that resolve to a path outside *repo_path* are
          skipped to prevent path-traversal attacks.
        - Files larger than MAX_FILE_SIZE_BYTES are skipped to prevent
          memory-exhaustion attacks.
    """
    if not os.path.isdir(repo_path):
        logger.warning("[ingestion] Path does not exist or is not a directory: %s", repo_path)
        return []

    # Resolve the canonical (symlink-free) absolute repo path once so that
    # every file can be validated against it efficiently.
    real_repo_path = os.path.realpath(os.path.abspath(repo_path))

    documents: List[dict] = []

    for root, dirs, files in os.walk(repo_path):
        # Prune directories we should never descend into
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                full_path = os.path.join(root, file)

                # ── Security: block symlinks that escape the repo root ──
                real_full_path = os.path.realpath(full_path)
                try:
                    # commonpath raises ValueError on Windows when paths are
                    # on different drives; treat that as an escape attempt.
                    common = os.path.commonpath([real_repo_path, real_full_path])
                except ValueError:
                    logger.warning(
                        "[ingestion] Skipping file outside repo (different drive): %s",
                        full_path,
                    )
                    continue
                if common != real_repo_path:
                    logger.warning(
                        "[ingestion] Skipping symlink that escapes repo root: %s -> %s",
                        full_path, real_full_path,
                    )
                    continue

                # ── Security: skip excessively large files ──
                try:
                    file_size = os.path.getsize(full_path)
                except OSError:
                    file_size = 0
                if file_size > MAX_FILE_SIZE_BYTES:
                    logger.warning(
                        "[ingestion] Skipping oversized file (%d bytes): %s",
                        file_size, full_path,
                    )
                    continue

                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    if not content.strip():
                        logger.debug("[ingestion] Skipping empty file: %s", full_path)
                        continue

                    documents.append({
                        "file_path": full_path,
                        "content": content,
                    })
                    logger.debug("[ingestion] Loaded: %s", full_path)

                except OSError as exc:
                    logger.error("[ingestion] Could not read %s: %s", full_path, exc)

    logger.info("[ingestion] Total files loaded: %d from %s", len(documents), repo_path)
    return documents


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    documents = load_repository(repo_path)
    if documents:
        print("First file path:", documents[0]["file_path"])
        print("First 500 characters:\n")
        print(documents[0]["content"][:500])