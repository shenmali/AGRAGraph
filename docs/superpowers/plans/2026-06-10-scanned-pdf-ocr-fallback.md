# Scanned-PDF OCR Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PDF pages whose text layer is empty (scanned pages) are OCR'd automatically so their content reaches retrieval, instead of being skipped as "empty after extraction".

**Architecture:** Per-page fallback inside `src/retrievers/loader.py` only — the public contract `extract_text(file_bytes, filename) -> str` is unchanged, and the graph, state, config, and `demo/app.py` are untouched. PyPDF2 still reads text layers; pages yielding < 20 chars (stripped) are rendered with pypdfium2 and recognized with RapidOCR (Latin model, covers Turkish + English), constructed lazily once per process.

**Tech Stack:** PyPDF2 (existing), rapidocr ≥ 2.0 + onnxruntime (OCR engine), pypdfium2 (page rendering), pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-scanned-pdf-ocr-fallback-design.md`

**Baseline (verified on main before this plan):**
- `python3 -m pytest tests/ -q` → `24 passed`
- `python3 -c "from src.graph.builder import graph; print(list(graph.nodes.keys()))"` →
  `['__start__', 'classify_query', 'retrieve_bm25', 'retrieve_dense', 'rerank', 'generate', 'check_hallucination', 'check_relevance', 'refine_query', 'finalize']`
- `python3 -c "import demo.app; print('import ok')"` → `import ok`
- Python 3.12.7 (rapidocr requires `>=3.6,<3.13` — do not run this plan on 3.13)
- Pillow 11.3.0 (fixture generator needs ≥ 10.1 for `ImageFont.load_default(size=...)`)

---

### Task 1: OCR dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the three OCR dependencies**

Append to `requirements.txt` (after the existing `numpy>=1.24.0` line, keeping `pytest` last):

```
rapidocr>=2.0
onnxruntime>=1.17
pypdfium2>=4.0
```

Note: since rapidocr 2.0.6 onnxruntime is **not** pulled in automatically — it must be listed explicitly. The legacy `rapidocr-onnxruntime` package is deprecated; do not use it.

- [ ] **Step 2: Install**

Run: `python3 -m pip install -r requirements.txt`
Expected: installs `rapidocr`, `onnxruntime`, `pypdfium2` (and already-satisfied existing deps).

- [ ] **Step 3: Smoke-test the imports**

Run: `python3 -c "from rapidocr import RapidOCR, LangRec; import pypdfium2, onnxruntime; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add ocr dependencies (rapidocr, onnxruntime, pypdfium2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Test fixtures (committed binaries + generator)

**Files:**
- Create: `tests/fixtures/generate_fixtures.py`
- Create (by running the generator): `tests/fixtures/text_layer.pdf`, `tests/fixtures/scanned.pdf`, `tests/fixtures/mixed.pdf`

- [ ] **Step 1: Write the generator script**

Create `tests/fixtures/generate_fixtures.py` exactly as follows. It hand-writes a minimal valid PDF with a real text layer (PyPDF2 cannot author text, so we emit raw PDF objects with computed xref offsets), uses PIL to save an image-only "scanned" PDF, and merges one page of each into the mixed fixture.

```python
"""One-off generator for the loader test fixtures.

Run from the repo root:  python3 tests/fixtures/generate_fixtures.py
The three PDFs it writes are committed so tests never regenerate them.
"""
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PyPDF2 import PdfReader, PdfWriter

OUT = Path(__file__).parent

TEXT_LINE = "LangGraph orchestrates stateful agents"
SCAN_LINES = ["RETRIEVAL AUGMENTED GENERATION", "SELF CORRECTING PIPELINE"]


def build_text_layer_pdf() -> bytes:
    stream = f"BT /F1 24 Tf 72 720 Td ({TEXT_LINE}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
    ]
    buf = BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(buf.tell())
        buf.write(b"%d 0 obj\n%s\nendobj\n" % (number, body))
    xref_at = buf.tell()
    buf.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        buf.write(b"%010d 00000 n \n" % offset)
    buf.write(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, xref_at)
    )
    return buf.getvalue()


def build_scanned_pdf() -> bytes:
    image = Image.new("RGB", (1240, 1754), "white")  # ~A4 at 150 DPI
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=56)
    for line_index, line in enumerate(SCAN_LINES):
        draw.text((80, 300 + line_index * 200), line, fill="black", font=font)
    buf = BytesIO()
    image.save(buf, format="PDF")
    return buf.getvalue()


def build_mixed_pdf(text_pdf: bytes, scanned_pdf: bytes) -> bytes:
    writer = PdfWriter()
    writer.add_page(PdfReader(BytesIO(text_pdf)).pages[0])
    writer.add_page(PdfReader(BytesIO(scanned_pdf)).pages[0])
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def main():
    text_pdf = build_text_layer_pdf()
    scanned_pdf = build_scanned_pdf()
    (OUT / "text_layer.pdf").write_bytes(text_pdf)
    (OUT / "scanned.pdf").write_bytes(scanned_pdf)
    (OUT / "mixed.pdf").write_bytes(build_mixed_pdf(text_pdf, scanned_pdf))

    # Sanity: the fixtures must behave the way the tests assume.
    assert TEXT_LINE in (PdfReader(BytesIO(text_pdf)).pages[0].extract_text() or "")
    assert len((PdfReader(BytesIO(scanned_pdf)).pages[0].extract_text() or "").strip()) < 20
    print("fixtures written:", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run: `python3 tests/fixtures/generate_fixtures.py`
Expected: `fixtures written: ['mixed.pdf', 'scanned.pdf', 'text_layer.pdf']` and exit code 0 (the two asserts inside `main()` are the sanity check — if either fires, stop and investigate before continuing).

- [ ] **Step 3: Confirm the four new files**

Run: `git status --short tests/fixtures/`
Expected: four untracked files — `generate_fixtures.py`, `mixed.pdf`, `scanned.pdf`, `text_layer.pdf`.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/
git commit -m "test: add loader pdf fixtures and generator

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Characterization tests for current loader behavior

**Files:**
- Create: `tests/test_loader.py`

- [ ] **Step 1: Write tests that pin today's behavior**

Create `tests/test_loader.py`:

```python
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
```

(The `from src.retrievers import loader` import is unused until Task 5 adds the monkeypatch test — leave it in place now so Task 5 only appends a function.)

- [ ] **Step 2: Run them — they must already pass (no behavior change yet)**

Run: `python3 -m pytest tests/test_loader.py -v`
Expected: `2 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_loader.py
git commit -m "test: characterize loader txt and text-layer pdf paths

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Per-page OCR fallback (TDD core)

**Files:**
- Modify: `tests/test_loader.py` (append two tests)
- Modify: `src/retrievers/loader.py` (full rewrite shown below)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_loader.py`:

```python
def test_scanned_pdf_falls_back_to_ocr():
    content = (FIXTURES / "scanned.pdf").read_bytes()
    result = extract_text(content, "scanned.pdf")
    assert "RETRIEVAL" in result.upper()


def test_mixed_pdf_extracts_both_pages():
    content = (FIXTURES / "mixed.pdf").read_bytes()
    result = extract_text(content, "mixed.pdf")
    assert "LangGraph" in result
    assert "RETRIEVAL" in result.upper()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_loader.py -v`
Expected: `2 passed, 2 failed` — both new tests fail with `AssertionError` because the scanned pages currently extract to empty strings.

- [ ] **Step 3: Implement the fallback**

Replace the entire contents of `src/retrievers/loader.py` with:

```python
from io import BytesIO

import numpy as np
import pypdfium2 as pdfium
from PyPDF2 import PdfReader

MIN_TEXT_CHARS_PER_PAGE = 20
RENDER_SCALE = 2.0

_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR, LangRec

        _ocr_engine = RapidOCR(params={"Rec.lang_type": LangRec.LATIN})
    return _ocr_engine


def _ocr_page(pdf_doc, page_index: int) -> str:
    bitmap = pdf_doc[page_index].render(scale=RENDER_SCALE)
    # rapidocr expects cv2-style BGR; to_pil() yields RGB
    image = np.ascontiguousarray(np.asarray(bitmap.to_pil().convert("RGB"))[:, :, ::-1])
    result = _get_ocr_engine()(image)
    if result is None or result.txts is None:
        return ""
    return "\n".join(result.txts)


def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        pdf_doc = None
        pages = []
        for index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if len(text) < MIN_TEXT_CHARS_PER_PAGE:
                if pdf_doc is None:
                    pdf_doc = pdfium.PdfDocument(file_bytes)
                text = _ocr_page(pdf_doc, index)
            pages.append(text)
        if pdf_doc is not None:
            pdf_doc.close()
        return "\n\n".join(pages)
    return file_bytes.decode("utf-8")
```

Design notes for the implementer:
- The `from rapidocr import ...` stays **inside** `_get_ocr_engine()` deliberately: importing rapidocr is slow, and the module must import fast for Streamlit and for tests that never touch OCR.
- `_ocr_engine` is a module-level singleton: the engine loads models once per process, and never loads at all if every page has a text layer.
- No try/except anywhere — project convention is that loader/OCR failures propagate and Streamlit displays them.
- `result is None or result.txts is None` is a return-value contract of rapidocr (it reports "nothing recognized" via empty output, not via exceptions), not error swallowing.

- [ ] **Step 4: Run the loader tests**

Run: `python3 -m pytest tests/test_loader.py -v`
Expected: `4 passed`. The first run constructs the engine and downloads the Latin recognition model (~10–15 MB) — allow ~30–60 s extra once; subsequent runs are seconds. (Same acceptance as `test_retrievers.py` downloading sentence-transformers models.)

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/ -v`
Expected: `28 passed` (baseline 24 + the 4 loader tests), no failures.

- [ ] **Step 6: Commit**

```bash
git add src/retrievers/loader.py tests/test_loader.py
git commit -m "feat: add per-page ocr fallback for scanned pdfs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Guard — text-layer PDFs must never construct the OCR engine

**Files:**
- Modify: `tests/test_loader.py` (append one test)

- [ ] **Step 1: Write the guard test**

Append to `tests/test_loader.py`:

```python
def test_text_layer_pdf_never_constructs_ocr_engine(monkeypatch):
    def _fail():
        raise AssertionError("OCR engine must not be constructed for text-layer PDFs")

    monkeypatch.setattr(loader, "_get_ocr_engine", _fail)
    content = (FIXTURES / "text_layer.pdf").read_bytes()
    result = loader.extract_text(content, "text_layer.pdf")
    assert "LangGraph" in result
```

This pins the spec's "zero added cost for text-layer PDFs" promise: if someone later makes the OCR path unconditional, this test fails loudly.

- [ ] **Step 2: Run it**

Run: `python3 -m pytest tests/test_loader.py -v`
Expected: `5 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_loader.py
git commit -m "test: guard that text-layer pdfs never construct the ocr engine

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: README feature mention

**Files:**
- Modify: `README.md:197-202` (Key Design Decisions bullets) and `README.md:239` (Tech Stack table row)

- [ ] **Step 1: Add the feature bullet**

In the `### Key Design Decisions` section, after the bullet `- **Graph-first architecture.** ...`, add:

```markdown
- **Scanned PDFs just work.** Pages with no text layer fall back to local OCR (RapidOCR, per page, Turkish + English). No API calls, no toggles, zero cost for digital PDFs.
```

- [ ] **Step 2: Update the Tech Stack file-parsing row**

Change:

```markdown
| **File parsing** | PyPDF2 (PDF), UTF-8 (TXT/MD) |
```

to:

```markdown
| **File parsing** | PyPDF2 (PDF) + RapidOCR fallback for scanned pages, UTF-8 (TXT/MD) |
```

(The spec scopes README to "one feature bullet"; this row update is included because the change makes the existing row factually wrong otherwise.)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: advertise scanned-pdf ocr fallback in readme

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Final verification (no code changes)

- [ ] **Step 1: Full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: `29 passed` (baseline 24 + 5 loader tests).

- [ ] **Step 2: Graph integrity — must be byte-identical to baseline**

Run: `python3 -c "from src.graph.builder import graph; print(list(graph.nodes.keys()))"`
Expected: `['__start__', 'classify_query', 'retrieve_bm25', 'retrieve_dense', 'rerank', 'generate', 'check_hallucination', 'check_relevance', 'refine_query', 'finalize']`

- [ ] **Step 3: Streamlit import check**

Run: `python3 -c "import demo.app; print('import ok')"`
Expected: `import ok`

- [ ] **Step 4: Loader module import stays fast (lazy rapidocr)**

Run: `python3 -c "import time; t=time.time(); from src.retrievers.loader import extract_text; print(f'{time.time()-t:.2f}s')"`
Expected: well under 2 s (rapidocr must not be imported at module level; numpy/pypdfium2/PyPDF2 are cheap).

Nothing to commit — if any step fails, fix within the task that introduced the problem.
