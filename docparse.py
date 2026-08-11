"""Extract plain text from uploaded documents: PDF, DOCX, TXT, MD."""
import io
import re

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None
try:
    import docx  # python-docx
except Exception:  # pragma: no cover
    docx = None


def _from_pdf(data: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed; cannot read PDFs")
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def _from_docx(data: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx is not installed; cannot read DOCX files")
    document = docx.Document(io.BytesIO(data))
    blocks = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return "\n\n".join(blocks)


EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
}


def detect_type(filename: str) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return EXTENSIONS.get(ext, "text")


def extract_text(filename: str, data: bytes) -> str:
    """Return plain text for a file, given name + raw bytes."""
    kind = detect_type(filename)
    if kind == "pdf":
        text = _from_pdf(data)
    elif kind == "docx":
        text = _from_docx(data)
    else:
        text = data.decode("utf-8", errors="replace")
    # Normalise whitespace a little but keep paragraph breaks.
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries.

    Falls back to hard character splits for very long paragraphs.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # A single paragraph longer than the chunk size: split it hard.
        while len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:chunk_size])
            para = para[chunk_size:]
        if not current:
            current = para
        elif len(current) + 1 + len(para) <= chunk_size:
            current += "\n\n" + para
        else:
            chunks.append(current)
            # overlap tail of the previous chunk
            current = para[:overlap] + "\n\n" + para if len(para) > overlap else para

    if current:
        chunks.append(current)
    return chunks