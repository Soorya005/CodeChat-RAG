"""
prompt_builder.py
Builds prompts from retrieved code context for the RAG pipeline.

Works directly with ChunkMetadata objects returned by vector_store.search()
and the (metadata, score) tuples produced by RAGPipeline.retrieve().
Does NOT require langchain or any external LLM framework.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Maximum number of retrieved chunks to include in a single prompt.
MAX_CONTEXT_CHUNKS = 5


def format_code_chunk(metadata, score: float) -> str:
    """
    Format a single (ChunkMetadata, score) pair into readable context text.

    Args:
        metadata: ChunkMetadata instance from vector_store.py (or any object
                  with file_path, language, symbol_name, chunk_type,
                  start_line, end_line, source_text attributes).
        score:    Relevance score from the vector store search.

    Returns:
        A formatted string block for inclusion in the prompt.
    """
    return (
        f"[FILE: {metadata.file_path}]\n"
        f"[LANGUAGE: {metadata.language}]\n"
        f"[SYMBOL: {metadata.symbol_name} ({metadata.chunk_type})]\n"
        f"[LINES: {metadata.start_line}–{metadata.end_line}  |  RELEVANCE: {score:.3f}]\n\n"
        f"{metadata.source_text}"
    )


def build_prompt(query: str, retrieved_chunks: List[Tuple]) -> str:
    """
    Assemble the final LLM prompt from the user query and retrieved chunks.

    Args:
        query:            The user's question or search query.
        retrieved_chunks: List of (ChunkMetadata, score) tuples as returned
                          by VectorStore.search() or RAGPipeline.retrieve().
                          Capped to MAX_CONTEXT_CHUNKS internally.

    Returns:
        A complete prompt string ready to be sent to an LLM.
    """
    # Cap to maximum context chunks
    chunks_to_use = retrieved_chunks[:MAX_CONTEXT_CHUNKS]

    if not chunks_to_use:
        logger.warning("[prompt_builder] No retrieved chunks – building prompt without context.")

    context_sections = [
        format_code_chunk(metadata, score)
        for metadata, score in chunks_to_use
    ]
    context_block = "\n\n" + ("-" * 60 + "\n\n").join(context_sections)

    logger.info(
        "[prompt_builder] Built prompt with %d chunk(s) for query: '%s'",
        len(chunks_to_use),
        query[:80],
    )

    prompt = f"""\
You are an expert software engineer assisting with understanding a code repository.

Use ONLY the provided repository context to answer the user's question.
If the answer cannot be found in the context, respond with:
  "I could not find the answer in the provided code."

{'=' * 60}
REPOSITORY CONTEXT ({len(chunks_to_use)} chunk(s))
{'=' * 60}
{context_block}

{'=' * 60}
USER QUESTION
{'=' * 60}

{query}

{'=' * 60}
ANSWER
{'=' * 60}"""

    return prompt.strip()


# ── Smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Minimal stub so we can run without importing the full pipeline
    from dataclasses import dataclass

    @dataclass
    class _FakeMetadata:
        file_path: str
        language: str
        symbol_name: str
        chunk_type: str
        start_line: int
        end_line: int
        source_text: str

    chunks = [
        (
            _FakeMetadata(
                file_path="auth.py",
                language="python",
                symbol_name="login",
                chunk_type="function",
                start_line=1,
                end_line=5,
                source_text=(
                    "def login(username, password):\n"
                    "    if authenticate(username, password):\n"
                    "        return generate_token(username)\n"
                ),
            ),
            0.921,
        )
    ]

    query = "How does authentication work?"
    prompt = build_prompt(query, chunks)
    print("\nGenerated Prompt:\n")
    print(prompt)