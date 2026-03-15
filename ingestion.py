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


def load_repository(repo_path: str) -> List[dict]:
    """
    Walk the repository and load all supported source files.

    Args:
        repo_path: Absolute or relative path to the repository root.

    Returns:
        List of {"file_path": str, "content": str} dicts.
        Returns an empty list if the path does not exist or contains no
        supported files.
    """
    if not os.path.isdir(repo_path):
        logger.warning("[ingestion] Path does not exist or is not a directory: %s", repo_path)
        return []

    documents: List[dict] = []

    for root, dirs, files in os.walk(repo_path):
        # Prune directories we should never descend into
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                full_path = os.path.join(root, file)
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