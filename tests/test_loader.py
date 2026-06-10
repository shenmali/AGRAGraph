from pathlib import Path

from src.retrievers import loader
from src.retrievers.loader import extract_text

FIXTURES = Path(__file__).parent / "fixtures"


def test_txt_passthrough():
    assert extract_text("hello world".encode("utf-8"), "note.txt") == "hello world"


def test_text_layer_pdf_uses_text_layer():
    content = (FIXTURES / "text_layer.pdf").read_bytes()
    result = extract_text(content, "text_layer.pdf")
    assert "LangGraph orchestrates stateful agents" in result
