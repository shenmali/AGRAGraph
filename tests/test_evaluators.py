from src.graph.state import Document


def test_document_model():
    doc = Document(content="test content", metadata={"source": "test"}, score=0.85)
    assert doc.content == "test content"
    assert doc.metadata["source"] == "test"
    assert doc.score == 0.85
    assert "test content" in repr(doc)
