# Scanned-PDF OCR Fallback — Design

**Date:** 2026-06-10
**Status:** Approved
**Scope:** `src/retrievers/loader.py`, `requirements.txt`, `tests/test_loader.py`, `tests/fixtures/`, `README.md` (one feature bullet)

## Problem

`extract_text()` in `src/retrievers/loader.py` reads only the embedded text layer of PDFs
via PyPDF2. Scanned (image-based) PDFs produce empty text, so the upload flow in
`demo/app.py` skips them with "empty after extraction". The document is unusable even
though its content is visible on the page. Mixed PDFs (some text pages, some scanned
pages) silently lose the scanned pages.

## Decisions

| Question | Decision |
|---|---|
| When does OCR run? | Automatic per-page fallback only. No UI toggle, no config knob. |
| Dependency packaging | Direct entries in `requirements.txt` (not optional/lazy). |
| Input scope | PDF only. No PNG/JPG upload support. |
| Languages | Latin recognition model, PP-OCRv5 (covers Turkish incl. ı/ğ/ş/ö/ç and English; the PP-OCRv3 latin default lacks ğ/ş/İ). |
| Fallback granularity | Per page: a page whose text layer yields < 20 chars (after `.strip()`) is OCR'd individually. |

## Out of Scope

- Image file uploads (PNG/JPG) — possible later iteration.
- "Force OCR" override toggle.
- Layout/table-to-Markdown parsing (PP-StructureV3 / Docling class of tools).
- Any change to the graph (9 nodes), state, edges, config, or `demo/app.py`.

## Architecture

The change is isolated to the ingestion utility layer. The public contract of
`extract_text(file_bytes: bytes, filename: str) -> str` is unchanged. Downstream
behavior (chunking in `demo/app.py`, DocumentStore, BM25/dense retrieval) is untouched.
OCR tuning values are module constants in `loader.py`, not config fields, because no
user-facing setting exists.

### Dependencies (added to `requirements.txt`)

- `rapidocr>=3.0.0` — unified RapidOCR package (~15 MB wheel, default det/cls/rec models
  bundled). The legacy `rapidocr_onnxruntime` package is deprecated. (Floor raised from
  the originally specced `>=2.0` during review: 2.0.x does not expose the `LangRec` API.)
- `onnxruntime>=1.17` — inference engine; since rapidocr 2.0.6 it must be installed
  explicitly.
- `pypdfium2>=4.0` — renders PDF pages to images. Permissive license, binary wheels,
  no system dependencies (rejected alternatives: `pdf2image` needs poppler installed
  separately; PyMuPDF is AGPL).

Constraint check: rapidocr supports Python >=3.6,<3.13; project runs 3.12.7. ✓

The Latin recognition model is not bundled in the wheel; rapidocr downloads it on first
use of the engine (one-time, ~10–15 MB).

## Data Flow

```
PDF bytes → for each page:
              PyPDF2 text layer → ≥ 20 chars? ──yes──→ use that text
                                       │
                                      no (scanned page)
                                       ▼
              pypdfium2 renders page at 2.0 scale (~144 DPI) → numpy array
                                       ▼
              RapidOCR (LangRec.LATIN) → join result.txts lines
            ↓
pages joined with "\n\n" → existing flow (chunking → DocumentStore → retrieval)
```

Mixed PDFs work correctly: each page takes its own path, and OCR cost is paid only for
pages that need it.

## Components (`loader.py` internals)

- `extract_text(file_bytes, filename)` — public API, signature unchanged. The PDF
  branch iterates pages with the per-page fallback.
- `_get_ocr_engine()` — lazy module-level singleton. Constructs
  `RapidOCR(params={"Rec.lang_type": LangRec.LATIN, "Rec.ocr_version": OCRVersion.PPOCRV5})` on first call (~1–2 s init).
  Never constructed if no scanned page is ever seen.
- `_ocr_page(pdf_doc, page_index)` — renders one page via pypdfium2 to a numpy array,
  runs the engine, returns `"\n".join(result.txts)` (empty string when OCR finds
  nothing).
- Constants: `MIN_TEXT_CHARS_PER_PAGE = 20`, `RENDER_SCALE = 2.0`.

Note: PyPDF2 (text layer) and pypdfium2 (rendering) both open the same `file_bytes`;
they are independent readers and do not interact.

## Error Handling

Follows the project convention: no try/except swallowing.

- Model download failure or page render failure → exception propagates; Streamlit
  displays it.
- A genuinely blank page contributes `""`. If the whole document comes back empty, the
  existing "Skipped: empty after extraction" warning in `demo/app.py` fires exactly as
  today.
- Non-PDF files keep the current UTF-8 decode path, unchanged.

## Testing

New `tests/test_loader.py` with committed binary fixtures in `tests/fixtures/`
(generated once during implementation — PIL image saved as PDF for the scanned fixture,
a small text-layer PDF, and a merged mixed PDF):

1. **Text-layer PDF** → returns text; OCR engine is never constructed (assert via
   monkeypatching `_get_ocr_engine` to raise).
2. **Scanned PDF** (image-only) → returned text contains an expected keyword
   (end-to-end through RapidOCR; downloads the Latin model on first run — same
   acceptance as `test_retrievers.py` downloading sentence-transformers models).
3. **Mixed PDF** (1 text page + 1 scanned page) → output contains content from both
   pages.
4. **TXT path** → unchanged behavior (regression guard).

Verification commands:

- `python3 -m pytest tests/ -v` — all green
- `python3 -c "from src.graph.builder import graph; print(list(graph.nodes.keys()))"` — graph unchanged
- Streamlit app imports without error

## Documentation

Add one bullet to the README feature list: scanned-PDF support via automatic OCR
fallback (RapidOCR, local, no API calls). No other documentation changes.

## Implementation Deviations (accepted in review)

Recorded after implementation; each was reviewed and kept deliberately:

- **`rapidocr>=3.0.0` + `Pillow>=10.1,<11` pins** — 2.0.x lacks `LangRec`; Pillow is
  capped because streamlit's transitive constraint forces `<11` and the fixture
  generator needs `>=10.1`.
- **`.gitattributes`: `tests/fixtures/*.pdf binary`** — the hand-written text-layer
  fixture is pure-ASCII PDF that git would otherwise treat as text (CRLF conversion
  would corrupt its xref offsets).
- **`_ocr_lock` (module-level `threading.Lock`)** — PDFium is not thread-safe and
  concurrent Streamlit sessions share the process; the lock serializes document open,
  render+OCR, and close. Text-layer pages never take the lock, preserving the
  zero-added-cost promise.
- **Hardened OCR guard + `try/finally` close** — `getattr(result, "txts", None)`
  tolerates rapidocr output types without a `txts` attribute; the `finally` guarantees
  the pdfium handle closes if a page raises mid-loop. No `except` anywhere — exceptions
  still propagate to the UI per the error-handling section above.
- **PP-OCRv5 latin recognition model** (`"Rec.ocr_version": OCRVersion.PPOCRV5`) — the
  PP-OCRv3 latin model that `LangRec.LATIN` selects by default has a 185-char dictionary
  with no ğ/Ğ/ş/Ş/İ, making Turkish output ASCII-folded; the v5 latin dictionary
  (502 chars) covers all Turkish-specific characters (verified by OCR probe).

## Performance Expectations

- Engine init: ~1–2 s, once per process, lazy.
- Per scanned page: ~0.5–1.5 s on CPU.
- Text-layer PDFs: zero added cost (OCR path never touched).
- No extra UI feedback needed: Streamlit's native running indicator covers the upload
  script execution time.
