
from pathlib import Path
import pymupdf
import pytest

from src.document_loader import load_pdf


def test_load_pdf_extracts_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Machine learning learns patterns from data."
        )
        document.save(pdf_path)

    pages = load_pdf(str(pdf_path))

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Machine learning" in pages[0]["text"]
    assert pages[0]["source"] == "sample.pdf"


def test_missing_pdf_raises_error():
    with pytest.raises(FileNotFoundError):
        load_pdf("missing.pdf")


def test_non_pdf_file_raises_error(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Some text")

    with pytest.raises(ValueError):
        load_pdf(str(txt_path))
