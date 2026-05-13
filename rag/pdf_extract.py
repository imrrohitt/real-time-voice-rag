"""PDF text extraction with PyMuPDF; OCR fallback via Tesseract when a page has little text."""
from __future__ import annotations

import io
import shutil

import fitz  # PyMuPDF
from PIL import Image

MIN_CHARS_BEFORE_OCR = 60


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _ocr_pixmap(pix: fitz.Pixmap) -> str:
    if not _tesseract_available():
        return ""
    try:
        import pytesseract
    except ImportError:
        return ""
    png = pix.tobytes("png")
    img = Image.open(io.BytesIO(png))
    return (pytesseract.image_to_string(img) or "").strip()


def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 120) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        parts: list[str] = []
        for pno in range(min(doc.page_count, max_pages)):
            page = doc.load_page(pno)
            direct = (page.get_text("text") or "").strip()
            if len(direct) >= MIN_CHARS_BEFORE_OCR:
                parts.append(direct)
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            ocr = _ocr_pixmap(pix)
            merged = (ocr or direct).strip()
            if merged:
                parts.append(merged)
        return "\n\n".join(parts)
    finally:
        doc.close()
