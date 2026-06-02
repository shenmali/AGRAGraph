import numpy as np

from src.graph.state import AgentState, Document
from src.models.llm import get_llm
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.config import config


def classify_query(state: AgentState) -> dict:
    llm = get_llm()
    prompt = f"""Classify this query into one type: factual, analytical, or creative.
- factual: asks for specific facts, data, definitions
- analytical: asks for comparison, analysis, pros/cons
- creative: asks for ideas, suggestions, imaginative content

Query: {state['query']}
Return only the type name."""

    query_type = llm(system_prompt="You classify queries.", user_prompt=prompt, temperature=0).strip().lower()
    if query_type not in ("factual", "analytical", "creative"):
        query_type = "factual"

    return {"query_type": query_type, "intermediate_results": [{"node": "classify_query", "output": query_type}]}


def retrieve_bm25(state: AgentState) -> dict:
    chunks = BM25Retriever().retrieve(state["query"], k=config.top_k_initial)
    return {"retrieved_chunks": chunks, "intermediate_results": [{"node": "retrieve_bm25", "output": f"{len(chunks)} chunks"}]}


def retrieve_dense(state: AgentState) -> dict:
    chunks = DenseRetriever().retrieve(state["query"], k=config.top_k_initial)
    return {"retrieved_chunks": chunks, "intermediate_results": [{"node": "retrieve_dense", "output": f"{len(chunks)} chunks"}]}


def rerank(state: AgentState) -> dict:
    docs = state.get("retrieved_chunks", [])
    if not docs:
        return {"reranked_chunks": [], "intermediate_results": [{"node": "rerank", "output": "no docs"}]}

    seen = set()
    unique = []
    for d in docs:
        if d.content not in seen:
            seen.add(d.content)
            unique.append(d)

    from src.retrievers.document_store import DocumentStore
    store = DocumentStore.get_instance()
    query_embedding = store.model.encode(state["query"])

    for doc in unique:
        doc_embedding = store.model.encode(doc.content[:512])
        sim = float(query_embedding @ doc_embedding.T) / (
            float(np.linalg.norm(query_embedding)) * float(np.linalg.norm(doc_embedding)) + 1e-8
        )
        doc.score = doc.score * 0.3 + sim * 0.7

    unique.sort(key=lambda d: d.score, reverse=True)
    reranked = unique[:config.top_k_reranked]

    return {
        "reranked_chunks": reranked,
        "intermediate_results": [{"node": "rerank", "output": f"{len(unique)} unique → top {len(reranked)}"}],
    }


def generate(state: AgentState) -> dict:
    llm = get_llm()
    chunks = state.get("reranked_chunks", [])
    if not chunks:
        return {"generated_answer": "No relevant documents found.", "intermediate_results": [{"node": "generate", "output": "no docs"}]}

    context = "\n\n".join(f"[Source {i+1}] {d.content}" for i, d in enumerate(chunks))

    prompt = f"""Answer based only on the context below. Cite sources like [Source 1].

Query: {state['query']}

Context:
{context}"""

    answer = llm(system_prompt="You are a precise research assistant. Only use provided context.", user_prompt=prompt, temperature=0.3, max_tokens=2048)
    return {"generated_answer": answer, "intermediate_results": [{"node": "generate", "output": "done"}]}


def check_hallucination(state: AgentState) -> dict:
    llm = get_llm()
    context = "\n".join(f"[S{i+1}] {d.content[:200]}" for i, d in enumerate(state.get("reranked_chunks", [])))

    prompt = f"""Does this answer contain info NOT in the context? Answer only "yes" or "no".

Answer: {state['generated_answer']}
Context: {context}"""

    result = llm(system_prompt="Detect hallucination. Be strict.", user_prompt=prompt, temperature=0).strip().lower()
    is_hallucination = "yes" in result and "no" not in result

    return {
        "hallucination_check": "hallucinated" if is_hallucination else "grounded",
        "intermediate_results": [{"node": "check_hallucination", "output": result}],
    }


def check_relevance(state: AgentState) -> dict:
    llm = get_llm()
    prompt = f"""Does this answer address the query? Answer only "yes" or "no".

Query: {state['query']}
Answer: {state['generated_answer']}"""

    result = llm(system_prompt="Judge if answer is relevant.", user_prompt=prompt, temperature=0).strip().lower()
    is_relevant = "yes" in result and "no" not in result

    return {
        "relevance_check": "relevant" if is_relevant else "irrelevant",
        "intermediate_results": [{"node": "check_relevance", "output": result}],
    }


def refine_query(state: AgentState) -> dict:
    llm = get_llm()
    prompt = f"""Reformulate this query to be more specific and searchable. Return only the reformulated query.

Original: {state['query']}
Retry #{state.get('retry_count', 0) + 1}"""

    refined = llm(system_prompt="You reformulate queries for better retrieval.", user_prompt=prompt, temperature=0.3)
    return {
        "query": refined,
        "retry_count": state.get("retry_count", 0) + 1,
        "intermediate_results": [{"node": "refine_query", "output": refined[:80]}],
    }


def finalize(state: AgentState) -> dict:
    confidence = 0.4 if state.get("hallucination_check") == "grounded" else 0.1
    confidence += 0.4 if state.get("relevance_check") == "relevant" else 0.1

    chunks = state.get("reranked_chunks", [])
    if chunks:
        confidence += (sum(d.score for d in chunks) / len(chunks)) * 0.2

    confidence = max(0.0, min(1.0, confidence))

    citations = []
    for i, doc in enumerate(chunks):
        src = doc.metadata.get("source", "unknown")
        citations.append(f"[Source {i+1}] from {src}")

    return {
        "confidence": round(confidence, 2),
        "citations": citations,
        "intermediate_results": [{"node": "finalize", "output": f"confidence={confidence:.2f}"}],
    }
