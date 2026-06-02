from sentence_transformers import SentenceTransformer
from typing import List
from src.graph.state import Document
from src.retrievers.document_store import DocumentStore
from src.config import config


class DenseRetriever:
    def __init__(self):
        self.store = DocumentStore.get_instance()
        self.model = SentenceTransformer(config.embedding_model)

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        query_embedding = self.model.encode(query).tolist()
        results = self.store.query(query_embeddings=query_embedding, n_results=k)

        if not results or not results.get("documents"):
            return []

        documents = []
        for i, doc_text in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results.get("distances") else 0
            score = 1.0 - distance if distance <= 1.0 else 0.0
            documents.append(
                Document(
                    content=doc_text,
                    metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                    score=float(score),
                )
            )

        return documents
