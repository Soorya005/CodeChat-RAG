"""
rag_pipeline.py
Orchestrates the complete RAG pipeline for code repositories.

Install deps:
    pip install sentence-transformers faiss-cpu numpy anthropic openai requests
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ingestion import load_repository
from chunking import chunk_repository, CodeChunk
from embeddings import CodeEmbedder, EmbeddedChunk
from vector_store import VectorStore, create_vector_store
from prompt_builder import build_prompt

logger = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("requests not installed. Install with: pip install requests")
    REQUESTS_AVAILABLE = False


# ─── Configuration ───────────────────────────────────────────────

@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: Optional[str] = None

    # Vector store
    index_type: str = "flat"  # "flat", "ivf", "hnsw", "in_memory"

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.0

    # LLM
    llm_provider: str = "ollama"          # "ollama", "anthropic", "openai"
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # API keys (loaded from env vars automatically)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    def __post_init__(self):
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_url:
            self.ollama_base_url = ollama_url


@dataclass
class RetrievalResult:
    """Result returned by RAGPipeline.retrieve()."""
    chunks: List[Tuple]   # List of (ChunkMetadata, score) tuples
    query: str
    total_found: int


@dataclass
class RAGResponse:
    """Complete response from RAGPipeline.query()."""
    query: str
    answer: str
    retrieved_chunks: List[Tuple]
    context_used: str
    metadata: Dict


# ─── LLM Client ─────────────────────────────────────────────────

class LLMClient:
    """Unified interface for Ollama, Anthropic, and OpenAI."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.provider = config.llm_provider

        if self.provider == "ollama":
            if not REQUESTS_AVAILABLE:
                raise RuntimeError(
                    "requests library required for Ollama. Install: pip install requests"
                )
            self.base_url = config.ollama_base_url
            self._check_ollama_connection()

        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise RuntimeError("anthropic library required. Install: pip install anthropic")
            if not config.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set.")
            self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("openai library required. Install: pip install openai")
            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            self.client = openai.OpenAI(api_key=config.openai_api_key)

        else:
            raise ValueError(
                f"Unknown LLM provider: '{self.provider}'. "
                "Choose 'ollama', 'anthropic', or 'openai'."
            )

    def _check_ollama_connection(self):
        """Verify Ollama is reachable and log available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [
                    m["name"] for m in response.json().get("models", [])
                ]
                logger.info(
                    "[LLM] Connected to Ollama. Available models: %s", available_models
                )
                if self.config.llm_model not in available_models:
                    logger.warning(
                        "[LLM] Model '%s' not in available Ollama models: %s",
                        self.config.llm_model, available_models,
                    )
            else:
                logger.warning(
                    "[LLM] Could not list Ollama models (HTTP %d)", response.status_code
                )
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "[LLM] Could not connect to Ollama at %s: %s. "
                "Make sure Ollama is running: ollama serve",
                self.base_url, exc,
            )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the configured LLM."""
        if self.provider == "ollama":
            return self._generate_ollama(prompt, system_prompt)

        if self.provider == "anthropic":
            kwargs = {
                "model": self.config.llm_model,
                "max_tokens": self.config.llm_max_tokens,
                "temperature": self.config.llm_temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            response = self.client.messages.create(**kwargs)
            return response.content[0].text

        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            return response.choices[0].message.content

        return ""

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a generate request to the Ollama REST API."""
        url = f"{self.base_url}/api/generate"
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        payload = {
            "model": self.config.llm_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.llm_temperature,
                "num_predict": self.config.llm_max_tokens,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama request timed out. Model '{self.config.llm_model}' "
                "may be slow or unavailable."
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama API error: {exc}")


# ─── Prompt Builder (internal) ────────────────────────────────────

class PromptBuilder:
    """Builds system and user prompts for the RAG pipeline."""

    @staticmethod
    def build_system_prompt() -> str:
        return (
            "You are an expert code assistant. Answer questions about a codebase "
            "using ONLY the provided code context.\n\n"
            "Guidelines:\n"
            "- Reference specific file names and line numbers.\n"
            "- Provide code snippets from the context when helpful.\n"
            "- If context is insufficient, say so clearly."
        )

    @staticmethod
    def build_user_prompt(query: str, context_chunks: List[Tuple]) -> str:
        """Delegate to prompt_builder module for consistent formatting."""
        return build_prompt(query, context_chunks)


# ─── RAG Pipeline ────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline for code repositories.

    Basic workflow::

        pipeline = RAGPipeline(config)
        pipeline.index_repository("/path/to/repo")
        response = pipeline.query("How does authentication work?")
        print(response.answer)
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.embedder: Optional[CodeEmbedder] = None
        self.vector_store = None
        self.llm_client: Optional[LLMClient] = None
        self.prompt_builder = PromptBuilder()

        logger.info(
            "[RAG Pipeline] Initialised — embedding_model=%s, index=%s, llm=%s/%s",
            self.config.embedding_model,
            self.config.index_type,
            self.config.llm_provider,
            self.config.llm_model,
        )

    # ── Indexing ────────────────────────────────────────────────

    def index_repository(self, repo_path: str, save_path: Optional[str] = None) -> None:
        """
        Index a code repository end-to-end.

        Args:
            repo_path:  Path to the repository root.
            save_path:  Optional directory to persist the FAISS index.

        Raises:
            ValueError: If no supported files or no chunks are found.
        """
        logger.info("[RAG Pipeline] Indexing repository: %s", repo_path)

        # 1. Ingest
        logger.info("[RAG Pipeline] Step 1/4 – Loading files…")
        documents = load_repository(repo_path)
        if not documents:
            raise ValueError(f"No supported code files found in: {repo_path}")

        # 2. Chunk
        logger.info("[RAG Pipeline] Step 2/4 – Chunking %d files…", len(documents))
        chunks = chunk_repository(documents)
        if not chunks:
            raise ValueError("Chunking produced zero chunks. Check file contents.")

        # 3. Embed
        logger.info("[RAG Pipeline] Step 3/4 – Embedding %d chunks…", len(chunks))
        self.embedder = CodeEmbedder(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
        )
        embedded_chunks = self.embedder.embed_chunks(chunks)

        # 4. Index
        logger.info("[RAG Pipeline] Step 4/4 – Building vector store…")
        self.vector_store = create_vector_store(
            embedded_chunks, index_type=self.config.index_type
        )

        if save_path:
            logger.info("[RAG Pipeline] Saving index to %s…", save_path)
            self.vector_store.save(save_path)

        stats = self.vector_store.get_stats()
        logger.info(
            "[RAG Pipeline] Indexing complete — %d chunks, languages=%s, types=%s",
            stats["total_chunks"], stats["languages"], stats["chunk_types"],
        )

    def load_index(self, load_path: str) -> None:
        """
        Load a pre-built FAISS index from disk.

        Args:
            load_path: Directory containing the saved index.
        """
        logger.info("[RAG Pipeline] Loading index from: %s", load_path)
        self.vector_store = VectorStore.load(load_path)
        self.embedder = CodeEmbedder(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
        )
        stats = self.vector_store.get_stats()
        logger.info("[RAG Pipeline] Loaded %d chunks.", stats["total_chunks"])

    # ── Retrieval ───────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> RetrievalResult:
        """
        Retrieve the most relevant code chunks for a query.

        Args:
            query:   Natural-language or code search query.
            top_k:   Override number of results (defaults to config.top_k).
            filters: Optional metadata filters, e.g. {"language": "python"}.

        Returns:
            RetrievalResult with (ChunkMetadata, score) tuples.

        Raises:
            RuntimeError: If index_repository() or load_index() has not been called.
        """
        if not self.vector_store or not self.embedder:
            raise RuntimeError(
                "Index not loaded. Call index_repository() or load_index() first."
            )

        top_k = top_k or self.config.top_k
        logger.info("[Retrieval] Query: '%s' (top_k=%d)", query, top_k)

        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k, filters=filters)

        # Apply similarity threshold
        filtered = [
            (meta, score)
            for meta, score in results
            if score >= self.config.similarity_threshold
        ]

        logger.info(
            "[Retrieval] Returned %d chunks (threshold=%.3f).",
            len(filtered), self.config.similarity_threshold,
        )
        return RetrievalResult(chunks=filtered, query=query, total_found=len(filtered))

    # ── Full Query ───────────────────────────────────────────────

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
        return_context: bool = False,
    ) -> RAGResponse:
        """
        Full RAG query: retrieve context and generate an LLM answer.

        Args:
            query:          User question.
            top_k:          Number of chunks to retrieve.
            filters:        Optional metadata filters.
            return_context: If True, include the raw prompt in the response.

        Returns:
            RAGResponse with answer and provenance metadata.
        """
        if not self.llm_client:
            self.llm_client = LLMClient(self.config)

        retrieval_result = self.retrieve(query, top_k, filters)

        if not retrieval_result.chunks:
            logger.warning("[RAG Pipeline] No relevant chunks found for query: '%s'", query)
            return RAGResponse(
                query=query,
                answer="No relevant code found in the repository for this query.",
                retrieved_chunks=[],
                context_used="",
                metadata={"total_chunks_found": 0},
            )

        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(query, retrieval_result.chunks)

        logger.info("[LLM] Generating answer via %s…", self.config.llm_provider)
        answer = self.llm_client.generate(user_prompt, system_prompt)

        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=retrieval_result.chunks,
            context_used=user_prompt if return_context else "",
            metadata={
                "total_chunks_found": retrieval_result.total_found,
                "chunks_used": len(retrieval_result.chunks),
                "llm_model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
            },
        )

    # ── Interactive Mode ─────────────────────────────────────────

    def interactive_mode(self):
        """Launch an interactive REPL for querying the indexed codebase."""
        border = "=" * 60
        print(f"\n{border}\n🤖 RAG Pipeline – Interactive Mode\n{border}")
        print("Commands: stats | filter <lang> | clear | quit\n" + border + "\n")

        current_filters = None

        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == "quit":
                    print("👋 Goodbye!")
                    break
                if user_input.lower() == "stats":
                    stats = self.vector_store.get_stats()
                    print(
                        f"\n📊 Stats: {stats['total_chunks']} chunks | "
                        f"Languages: {stats['languages']} | "
                        f"Types: {stats['chunk_types']}"
                    )
                    continue
                if user_input.lower().startswith("filter "):
                    lang = user_input[7:].strip()
                    current_filters = {"language": lang}
                    print(f"🔍 Filtering by language: {lang}")
                    continue
                if user_input.lower() == "clear":
                    current_filters = None
                    print("🔍 Filters cleared")
                    continue

                response = self.query(user_input, filters=current_filters)
                print(f"\n🤖 Assistant:\n{response.answer}")
                print(f"\n📚 Retrieved {response.metadata['chunks_used']} chunk(s):")
                for i, (meta, score) in enumerate(response.retrieved_chunks[:3], 1):
                    print(
                        f"   {i}. {meta.symbol_name} "
                        f"– {meta.file_path}:{meta.start_line} "
                        f"(score: {score:.3f})"
                    )

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as exc:
                logger.error("[RAG Pipeline] Unhandled error: %s", exc, exc_info=True)
                print(f"❌ Error: {exc}")


# ─── CLI Entry Point ─────────────────────────────────────────────

def main():
    """Command-line entry point."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    parser = argparse.ArgumentParser(description="RAG Pipeline for Code Repositories")
    parser.add_argument(
        "command",
        choices=["index", "query", "interactive", "load"],
    )
    parser.add_argument("--repo",       help="Repository path (index command)")
    parser.add_argument("--index-path", help="Path to save/load index")
    parser.add_argument("--query",      help="Query string")
    parser.add_argument("--top-k",      type=int, default=5)
    parser.add_argument("--model",      default="all-MiniLM-L6-v2")
    parser.add_argument("--llm",        default="ollama",
                        choices=["ollama", "anthropic", "openai"])
    parser.add_argument("--llm-model",  help="LLM model name (e.g. llama3.2)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")

    args = parser.parse_args()

    config = RAGConfig(
        embedding_model=args.model,
        top_k=args.top_k,
        llm_provider=args.llm,
        ollama_base_url=args.ollama_url,
    )
    if args.llm_model:
        config.llm_model = args.llm_model

    pipeline = RAGPipeline(config)

    if args.command == "index":
        if not args.repo:
            parser.error("--repo required for 'index' command")
        pipeline.index_repository(args.repo, save_path=args.index_path)

    elif args.command == "load":
        if not args.index_path:
            parser.error("--index-path required for 'load' command")
        pipeline.load_index(args.index_path)
        print("✅ Index loaded successfully!")

    elif args.command == "query":
        if not args.index_path:
            parser.error("--index-path required for 'query' command")
        if not args.query:
            parser.error("--query required for 'query' command")
        pipeline.load_index(args.index_path)
        response = pipeline.query(args.query)
        print(f"\n🤖 Answer:\n{response.answer}")
        print("\n📚 Sources:")
        for i, (meta, score) in enumerate(response.retrieved_chunks, 1):
            print(f"   {i}. {meta.symbol_name} – {meta.file_path}:{meta.start_line}")

    elif args.command == "interactive":
        if not args.index_path:
            parser.error("--index-path required for 'interactive' command")
        pipeline.load_index(args.index_path)
        pipeline.interactive_mode()


if __name__ == "__main__":
    main()