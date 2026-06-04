from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.retrievers.document_store import DocumentStore


SAMPLE_DOCUMENTS = [
    {
        "id": "agra-overview",
        "source": "AGRAGraph overview",
        "content": (
            "AGRAGraph is an agentic retrieval augmented generation pipeline that "
            "uses a LangGraph state machine with nine nodes to classify, retrieve, "
            "generate, evaluate, refine, and finalize grounded responses."
        ),
    },
    {
        "id": "graph-flow",
        "source": "Graph flow",
        "content": (
            "The graph has nine nodes: classify_query, retrieve_bm25, "
            "retrieve_dense, rerank, generate, check_hallucination, "
            "check_relevance, refine_query, and finalize, ending only after the "
            "quality checks route to finalize."
        ),
    },
    {
        "id": "retrieval-routing",
        "source": "Retrieval routing",
        "content": (
            "Retrieval routing is type aware: factual queries use BM25 keyword "
            "search, while analytical and creative queries use dense retrieval so "
            "semantic or open-ended requests can retrieve conceptually related "
            "context."
        ),
    },
    {
        "id": "self-correction",
        "source": "Self correction",
        "content": (
            "AGRAGraph self-corrects through routing rules: hallucinated answers "
            "route to refine_query while retries remain, and irrelevant answers "
            "route to refine_query while retries remain before reranking and "
            "generating again."
        ),
    },
    {
        "id": "document-store",
        "source": "Document store",
        "content": (
            "DocumentStore is a singleton in-memory corpus backed by numpy arrays "
            "for embeddings, document text, metadata, and ids. The demo has no "
            "external vector DB, keeping sample loading local and dependency-light "
            "for quick evaluation."
        ),
    },
    {
        "id": "benchmark-purpose",
        "source": "Benchmark purpose",
        "content": (
            "The benchmark purpose is to give the Streamlit demo predictable "
            "AGRAGraph-specific material so later evaluations can ask about graph "
            "nodes, routing, self-correction, storage, and final answer behavior."
        ),
    },
]

SAMPLE_QUESTIONS = [
    "What problem does AGRAGraph solve?",
    "How does the graph decide between BM25 and dense retrieval?",
    "When does the pipeline refine a query?",
    "Why does the demo use an in-memory DocumentStore?",
    "What is the purpose of the sample benchmark corpus?",
]


def get_sample_documents() -> list[dict[str, str]]:
    return deepcopy(SAMPLE_DOCUMENTS)


def get_sample_questions() -> list[str]:
    return SAMPLE_QUESTIONS.copy()


def _get_document_store() -> "DocumentStore":
    from src.retrievers.document_store import DocumentStore

    return DocumentStore.get_instance()


def reset_document_store(store: Optional["DocumentStore"] = None) -> "DocumentStore":
    target = store if store is not None else _get_document_store()
    target.documents = []
    target.metadatas = []
    target.ids = []
    target.embeddings = None
    return target


def load_sample_corpus(
    reset: bool = False, store: Optional["DocumentStore"] = None
) -> int:
    target = store if store is not None else _get_document_store()

    if reset:
        reset_document_store(target)

    existing_ids = set(target.ids)
    missing_documents = [
        document for document in SAMPLE_DOCUMENTS if document["id"] not in existing_ids
    ]

    if not missing_documents:
        return 0

    target.add_documents(
        texts=[document["content"] for document in missing_documents],
        metadatas=[
            {"source": document["source"], "source_id": document["id"]}
            for document in missing_documents
        ],
        ids=[document["id"] for document in missing_documents],
    )
    return len(missing_documents)
