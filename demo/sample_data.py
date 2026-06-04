from __future__ import annotations

from copy import deepcopy
from typing import Optional

from src.retrievers.document_store import DocumentStore


SAMPLE_DOCUMENTS = [
    {
        "id": "agra-overview",
        "source": "AGRAGraph overview",
        "content": (
            "AGRAGraph is an agentic retrieval augmented generation pipeline that "
            "combines LangGraph state transitions with retrieval, generation, and "
            "answer evaluation for grounded responses."
        ),
    },
    {
        "id": "graph-flow",
        "source": "Graph flow",
        "content": (
            "The graph starts by classifying the query, routes to a retriever, "
            "reranks candidate chunks, generates an answer, checks hallucination "
            "and relevance, then finalizes when quality gates pass."
        ),
    },
    {
        "id": "retrieval-routing",
        "source": "Retrieval routing",
        "content": (
            "Retrieval routing chooses between BM25 keyword search and dense "
            "semantic search so precise phrase matches and conceptual questions "
            "can each use the retrieval strategy most likely to help."
        ),
    },
    {
        "id": "self-correction",
        "source": "Self correction",
        "content": (
            "AGRAGraph self-corrects by refining the query and rerunning rerank "
            "when the generated answer appears unsupported by context or not "
            "relevant enough to the user's original question."
        ),
    },
    {
        "id": "document-store",
        "source": "Document store",
        "content": (
            "DocumentStore is a singleton in-memory corpus backed by numpy arrays "
            "for embeddings, document text, metadata, and ids, avoiding any "
            "external vector database dependency in the demo."
        ),
    },
    {
        "id": "benchmark-purpose",
        "source": "Benchmark purpose",
        "content": (
            "The sample benchmark corpus gives the Streamlit demo predictable "
            "AGRAGraph-specific material so users can test routing, reranking, "
            "self-correction, and final answer behavior quickly."
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


def reset_document_store(store: Optional[DocumentStore] = None) -> DocumentStore:
    target = store if store is not None else DocumentStore.get_instance()
    target.documents = []
    target.metadatas = []
    target.ids = []
    target.embeddings = None
    return target


def load_sample_corpus(
    reset: bool = False, store: Optional[DocumentStore] = None
) -> int:
    target = store if store is not None else DocumentStore.get_instance()

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
