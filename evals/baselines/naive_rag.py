from __future__ import annotations

from collections.abc import Callable

from src.config import config
from src.models.llm import get_llm
from src.retrievers.bm25_retriever import BM25Retriever


def llm_is_configured() -> bool:
    if config.llm_provider == "openrouter":
        return bool(config.openrouter_api_key)
    if config.llm_provider == "openai":
        return bool(config.openai_api_key)
    if config.llm_provider == "ollama":
        return True
    return False


def run_naive_rag(query: str, llm: Callable | None = None, k: int = 5) -> dict:
    chunks = BM25Retriever().retrieve(query, k=k)
    observed_source_ids = [doc.metadata.get("source_id") or doc.metadata.get("source", "unknown") for doc in chunks]
    citations = [f"[Source {index + 1}] from {source_id}" for index, source_id in enumerate(observed_source_ids)]

    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "citations": [],
            "observed_source_ids": [],
            "retrieved_count": 0,
        }

    call_llm = llm or get_llm()
    context = "\n\n".join(f"[Source {index + 1}] {doc.content}" for index, doc in enumerate(chunks))
    prompt = f"""Answer based only on the context below. Cite sources like [Source 1].

Query: {query}

Context:
{context}"""

    answer = call_llm(
        system_prompt="You are a precise research assistant. Only use provided context.",
        user_prompt=prompt,
        temperature=0.3,
        max_tokens=1024,
    )

    return {
        "answer": answer,
        "citations": citations,
        "observed_source_ids": observed_source_ids,
        "retrieved_count": len(chunks),
    }
