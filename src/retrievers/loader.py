import threading
from io import BytesIO

import numpy as np
import pypdfium2 as pdfium
from PyPDF2 import PdfReader

MIN_TEXT_CHARS_PER_PAGE = 20
RENDER_SCALE = 2.0

_ocr_engine = None
_ocr_lock = threading.Lock()


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
    if getattr(result, "txts", None) is None:
        return ""
    return "\n".join(result.txts)


def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        pdf_doc = None
        pages = []
        try:
            for index, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if len(text) < MIN_TEXT_CHARS_PER_PAGE:
                    # pdfium is not thread-safe; serialize all native access
                    with _ocr_lock:
                        if pdf_doc is None:
                            pdf_doc = pdfium.PdfDocument(file_bytes)
                        text = _ocr_page(pdf_doc, index)
                pages.append(text)
        finally:
            if pdf_doc is not None:
                with _ocr_lock:
                    pdf_doc.close()
        return "\n\n".join(pages)
    return file_bytes.decode("utf-8")
