
from pathlib import Path
import pymupdf


def load_pdf(file_path: str) -> list[dict]:
    """Extract text from a PDF, preserving its page numbers."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    pages = []

    with pymupdf.open(path) as document:
        if document.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported yet.")

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "source": path.name,
                })

    if not pages:
        raise ValueError(
            "No readable text found. The PDF may be scanned."
        )

    return pages
