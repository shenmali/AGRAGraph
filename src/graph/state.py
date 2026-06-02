from typing import List, TypedDict, Annotated, Optional
import operator


class Document:
    def __init__(self, content: str, metadata: dict, score: float = 0.0):
        self.content = content
        self.metadata = metadata
        self.score = score

    def __repr__(self):
        return f"Document(score={self.score:.3f}, content={self.content[:60]}...)"


class AgentState(TypedDict):
    query: str
    query_type: Optional[str]
    retrieved_chunks: Annotated[List[Document], operator.add]
    reranked_chunks: List[Document]
    generated_answer: Optional[str]
    hallucination_check: Optional[str]
    relevance_check: Optional[str]
    retry_count: int
    max_retries: int
    confidence: float
    citations: List[str]
    intermediate_results: List[dict]
    error: Optional[str]
