import demo.sample_data as sample_data
from demo.sample_data import (
    SAMPLE_DOCUMENTS,
    SAMPLE_QUESTIONS,
    get_sample_documents,
    get_sample_questions,
    load_sample_corpus,
    reset_document_store,
)


class FakeDocumentStore:
    def __init__(self):
        self.documents = ["stale document"]
        self.metadatas = [{"source": "stale", "source_id": "stale-id"}]
        self.ids = ["stale-id"]
        self.embeddings = ["stale embedding"]
        self.add_calls = []

    def add_documents(self, texts, metadatas, ids):
        self.add_calls.append({"texts": texts, "metadatas": metadatas, "ids": ids})
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        self.embeddings = [f"embedding-{idx}" for idx in range(len(self.documents))]


def test_sample_documents_have_unique_ids_and_content():
    ids = [doc["id"] for doc in SAMPLE_DOCUMENTS]
    assert len(ids) == len(set(ids))
    assert len(SAMPLE_DOCUMENTS) >= 5

    for doc in SAMPLE_DOCUMENTS:
        assert doc["id"].strip()
        assert doc["source"].strip()
        assert doc["content"].strip()
        assert len(doc["content"]) > 80


def test_sample_documents_cover_demo_evaluation_facts():
    corpus = "\n".join(doc["content"] for doc in SAMPLE_DOCUMENTS).lower()

    for node_name in (
        "classify_query",
        "retrieve_bm25",
        "retrieve_dense",
        "rerank",
        "generate",
        "check_hallucination",
        "check_relevance",
        "refine_query",
        "finalize",
    ):
        assert node_name in corpus

    for phrase in (
        "nine nodes",
        "factual queries use bm25",
        "analytical and creative queries use dense retrieval",
        "hallucinated answers route to refine_query while retries remain",
        "irrelevant answers route to refine_query while retries remain",
        "no external vector db",
        "benchmark purpose",
    ):
        assert phrase in corpus


def test_sample_questions_are_non_empty_strings():
    assert len(SAMPLE_QUESTIONS) >= 5
    assert all(isinstance(question, str) for question in SAMPLE_QUESTIONS)
    assert all(question.strip() for question in SAMPLE_QUESTIONS)


def test_getters_return_copies():
    docs = get_sample_documents()
    questions = get_sample_questions()

    docs.append({"id": "mutated", "source": "mutated", "content": "mutated"})
    questions.append("mutated?")

    assert len(get_sample_documents()) == len(SAMPLE_DOCUMENTS)
    assert len(get_sample_questions()) == len(SAMPLE_QUESTIONS)


def test_reset_document_store_clears_store_state():
    store = FakeDocumentStore()

    returned = reset_document_store(store)

    assert returned is store
    assert store.documents == []
    assert store.metadatas == []
    assert store.ids == []
    assert store.embeddings is None


def test_load_sample_corpus_resets_and_loads_expected_documents():
    store = FakeDocumentStore()

    added = load_sample_corpus(reset=True, store=store)

    expected_ids = [doc["id"] for doc in SAMPLE_DOCUMENTS]
    expected_texts = [doc["content"] for doc in SAMPLE_DOCUMENTS]
    expected_metadatas = [
        {"source": doc["source"], "source_id": doc["id"]} for doc in SAMPLE_DOCUMENTS
    ]

    assert added == len(SAMPLE_DOCUMENTS)
    assert store.documents == expected_texts
    assert store.metadatas == expected_metadatas
    assert store.ids == expected_ids
    assert store.add_calls == [
        {"texts": expected_texts, "metadatas": expected_metadatas, "ids": expected_ids}
    ]


def test_load_sample_corpus_is_idempotent_by_id():
    store = FakeDocumentStore()
    reset_document_store(store)

    first_added = load_sample_corpus(store=store)
    second_added = load_sample_corpus(store=store)

    expected_ids = [doc["id"] for doc in SAMPLE_DOCUMENTS]
    assert first_added == len(SAMPLE_DOCUMENTS)
    assert second_added == 0
    assert store.ids == expected_ids
    assert len(store.ids) == len(set(store.ids))
    assert len(store.add_calls) == 1


def test_sample_data_does_not_import_document_store_at_module_import_time():
    assert "DocumentStore" not in vars(sample_data)
