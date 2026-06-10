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
