
from types import SimpleNamespace

import pytest

from src.generation import NO_ANSWER, generate_answer


def fake_chat_response(text):
    return SimpleNamespace(
        message=SimpleNamespace(content=text)
    )


def test_answer_returns_cited_source():
    chunks = [
        {
            "text": "The student name is Nikhil Chaudhari.",
            "source": "sample.pdf",
            "page_number": 1,
        }
    ]

    result = generate_answer(
        "What is the student's name?",
        chunks,
        chat_fn=lambda **kwargs: fake_chat_response(
            "The student's name is Nikhil Chaudhari [1]."
        ),
    )

    assert "Nikhil Chaudhari" in result["answer"]
    assert result["sources"][0]["page_number"] == 1
    assert result["sources"][0]["source"] == "sample.pdf"


def test_answer_without_citation_abstains():
    result = generate_answer(
        "What is the answer?",
        [{"text": "Some passage", "page_number": 1}],
        chat_fn=lambda **kwargs: fake_chat_response(
            "The answer is forty-two."
        ),
    )

    assert result["answer"] == NO_ANSWER
    assert result["sources"] == []


def test_invalid_citation_abstains():
    result = generate_answer(
        "What is the answer?",
        [{"text": "Some passage", "page_number": 1}],
        chat_fn=lambda **kwargs: fake_chat_response(
            "The answer is forty-two [99]."
        ),
    )

    assert result["answer"] == NO_ANSWER


def test_empty_retrieval_does_not_call_model():
    def should_not_run(**kwargs):
        raise AssertionError("The model should not be called.")

    result = generate_answer(
        "Question?",
        [],
        chat_fn=should_not_run,
    )

    assert result["answer"] == NO_ANSWER


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        generate_answer(
            "",
            [{"text": "Some passage"}],
        )
