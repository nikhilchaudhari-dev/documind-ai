
import pytest

from src.vector_stores import (
    create_index,
    save_knowledge_base,
    load_knowledge_base,
    search_index,
)


def test_save_and_reload_index(tmp_path):
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    chunks = [
        {"text": "Machine learning", "page_number": 1},
        {"text": "Database systems", "page_number": 2},
    ]

    index = create_index(vectors)
    save_knowledge_base(index, chunks, str(tmp_path))

    loaded_index, loaded_chunks = load_knowledge_base(str(tmp_path))

    results = search_index(
        loaded_index,
        loaded_chunks,
        [1.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["text"] == "Machine learning"
    assert results[0]["page_number"] == 1


def test_invalid_vectors_are_rejected():
    with pytest.raises(ValueError):
        create_index([])

    with pytest.raises(ValueError):
        create_index([[2.0, 0.0]])


def test_mismatched_chunk_count_is_rejected(tmp_path):
    index = create_index([[1.0, 0.0]])

    with pytest.raises(ValueError):
        save_knowledge_base(index, [], str(tmp_path))


def test_missing_knowledge_base_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_knowledge_base(str(tmp_path))
