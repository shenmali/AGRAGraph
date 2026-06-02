from rank_bm25 import BM25Okapi
from typing import List
from src.graph.state import Document
from src.retrievers.document_store import DocumentStore
import nltk
import re

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
from nltk.tokenize import word_tokenize


class BM25Retriever:
    def __init__(self):
        self.store = DocumentStore.get_instance()

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        try:
            return word_tokenize(text)
        except (LookupError, OSError):
            return text.split()

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        all_docs = self.store.all_documents()
        if not all_docs or not all_docs.get("documents"):
            return []

        tokenized_corpus = [self._tokenize(doc) for doc in all_docs["documents"]]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed[:k]:
            if score > 0:
                results.append(
                    Document(
                        content=all_docs["documents"][idx],
                        metadata=all_docs["metadatas"][idx] if all_docs.get("metadatas") else {},
                        score=float(score),
                    )
                )

        return results
