import pytest
from src.retrievers.document_store import DocumentStore
from src.retrievers.bm25_retriever import BM25Retriever


@pytest.fixture(autouse=True)
def fresh_store():
    store = DocumentStore.get_instance()
    store.documents = []
    store.metadatas = []
    store.ids = []
    store.embeddings = None
    yield


def test_document_store_add_and_count():
    store = DocumentStore.get_instance()
    store.add_documents(
        texts=["Machine learning is a subset of artificial intelligence."],
        metadatas=[{"source": "test"}],
        ids=["test-1"],
    )
    assert store.count() == 1


def test_bm25_retriever_no_docs():
    retriever = BM25Retriever()
    results = retriever.retrieve("test query", k=5)
    assert results == []


def test_bm25_retriever_with_docs():
    store = DocumentStore.get_instance()
    docs = [
        "Python is a programming language used for data science.",
        "Machine learning enables computers to learn from data.",
        "Deep learning uses neural networks with many layers.",
    ]
    metas = [{"source": f"doc-{i}"} for i in range(len(docs))]
    ids = [f"id-{i}" for i in range(len(docs))]
    store.add_documents(texts=docs, metadatas=metas, ids=ids)

    retriever = BM25Retriever()
    results = retriever.retrieve("machine learning", k=3)
    assert len(results) > 0
    assert results[0].score > 0


def test_dense_retriever_uses_document_store_query_signature(monkeypatch):
    from src.retrievers import dense_retriever as module

    class FakeModel:
        def encode(self, text):
            assert text == "semantic query"
            return [0.1, 0.2, 0.3]

    class FakeStore:
        def __init__(self):
            self.received_query_embedding = None
            self.received_n_results = None

        def query(self, query_embedding, n_results=10):
            self.received_query_embedding = query_embedding
            self.received_n_results = n_results
            return {
                "documents": [["Dense retrieval uses embeddings."]],
                "metadatas": [[{"source": "dense-source"}]],
                "distances": [[0.25]],
                "ids": [["dense-source"]],
            }

    fake_store = FakeStore()
    monkeypatch.setattr(module.DocumentStore, "get_instance", staticmethod(lambda: fake_store))
    monkeypatch.setattr(module, "SentenceTransformer", lambda model_name: FakeModel())

    retriever = module.DenseRetriever()
    results = retriever.retrieve("semantic query", k=1)

    assert fake_store.received_query_embedding == [0.1, 0.2, 0.3]
    assert fake_store.received_n_results == 1
    assert len(results) == 1
    assert results[0].metadata["source"] == "dense-source"
    assert results[0].score == 0.75
