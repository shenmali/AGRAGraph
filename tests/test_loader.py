from pathlib import Path

from src.retrievers import loader  # monkeypatched in the lazy-engine guard test
from src.retrievers.loader import extract_text

FIXTURES = Path(__file__).parent / "fixtures"


def test_txt_passthrough():
    assert extract_text("hello world".encode("utf-8"), "note.txt") == "hello world"


def test_text_layer_pdf_uses_text_layer():
    content = (FIXTURES / "text_layer.pdf").read_bytes()
    result = extract_text(content, "text_layer.pdf")
    assert "LangGraph orchestrates stateful agents" in result


def test_scanned_pdf_falls_back_to_ocr():
    content = (FIXTURES / "scanned.pdf").read_bytes()
    result = extract_text(content, "scanned.pdf")
    assert "RETRIEVAL" in result.upper()


def test_mixed_pdf_extracts_both_pages():
    content = (FIXTURES / "mixed.pdf").read_bytes()
    result = extract_text(content, "mixed.pdf")
    assert "LangGraph" in result
    assert "RETRIEVAL" in result.upper()


def test_text_layer_pdf_never_constructs_ocr_engine(monkeypatch):
    def _fail():
        raise AssertionError("OCR engine must not be constructed for text-layer PDFs")

    monkeypatch.setattr(loader, "_get_ocr_engine", _fail)
    content = (FIXTURES / "text_layer.pdf").read_bytes()
    result = loader.extract_text(content, "text_layer.pdf")
    assert "LangGraph" in result
