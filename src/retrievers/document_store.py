import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.config import config


class DocumentStore:
    _instance: Optional["DocumentStore"] = None

    def __init__(self):
        self.model = SentenceTransformer(config.embedding_model)
        self.documents: List[str] = []
        self.metadatas: List[dict] = []
        self.ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    @classmethod
    def get_instance(cls) -> "DocumentStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_documents(self, texts: List[str], metadatas: List[dict], ids: List[str]):
        new_embeddings = self.model.encode(texts)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def query(self, query_embedding: List[float], n_results: int = 10) -> dict:
        if self.embeddings is None or len(self.documents) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        query_vec = np.array(query_embedding).reshape(1, -1)
        norms = np.linalg.norm(self.embeddings, axis=1) + 1e-8
        query_norm = np.linalg.norm(query_vec) + 1e-8
        similarities = (self.embeddings @ query_vec.T).flatten() / (norms * query_norm)
        similarities = np.clip(similarities, 0, 1)
        top_indices = np.argsort(similarities)[::-1][:n_results]

        return {
            "documents": [[self.documents[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]],
            "distances": [[1.0 - similarities[i] for i in top_indices]],
            "ids": [[self.ids[i] for i in top_indices]],
        }

    def count(self) -> int:
        return len(self.documents)

    def all_documents(self) -> dict:
        return {"documents": self.documents, "metadatas": self.metadatas, "ids": self.ids}
