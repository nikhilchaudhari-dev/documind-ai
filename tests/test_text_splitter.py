
import pytest

from src.text_splitter import chunk_pages


def test_long_text_creates_overlapping_chunks():
    pages = [{
        "text": "A " * 1200,
        "source": "sample.pdf",
        "page_number": 1,
    }]

    chunks = chunk_pages(
        pages,
        chunk_size=1000,
        overlap=150,
    )

    assert len(chunks) > 1
    assert all(len(c["text"]) <= 1000 for c in chunks)
    assert all(c["source"] == "sample.pdf" for c in chunks)
    assert all(c["page_number"] == 1 for c in chunks)


def test_short_text_stays_in_one_chunk():
    pages = [{
        "text": "Machine learning finds patterns.",
        "source": "notes.pdf",
        "page_number": 3,
    }]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 3
    assert "Machine learning" in chunks[0]["text"]


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=100, overlap=100)


def test_empty_pages_are_skipped():
    pages = [{
        "text": "   ",
        "source": "empty.pdf",
        "page_number": 1,
    }]

    assert chunk_pages(pages) == []
