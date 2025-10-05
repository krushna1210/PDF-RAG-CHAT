import fitz  # PyMuPDF
import re

def extract_text_from_pdf(file_path: str):
    """
    Extracts text from a PDF file using PyMuPDF.
    Returns a list of (page_num, text) so we can track pages.
    """
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(pages, max_chars=1200, overlap=200):
    """
    Splits each page into overlapping chunks.
    Input: pages = list of dicts {page: int, text: str}
    Returns: list of dicts {chunk_id, page, text}
    """
    chunks = []
    chunk_id = 0

    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk_text = text[start:end]
            chunks.append({
                "chunk_id": chunk_id,
                "page": p["page"],
                "text": chunk_text
            })
            chunk_id += 1
            start += max_chars - overlap

    return chunks
