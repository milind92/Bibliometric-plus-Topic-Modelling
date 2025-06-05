import pytest
from standardize_terms import standardize_text
from main_notebook_script import extract_keywords


def test_standardize_text_case_insensitive():
    input_text = "Generative ai and llms lead the field"
    expected = "Generative Artificial Intelligence and LLM lead the field"
    assert standardize_text(input_text) == expected


def test_extract_keywords_standardization():
    keywords = "LLMs; generative AI; robotics"
    assert extract_keywords(keywords) == [
        "LLM",
        "Generative Artificial Intelligence",
        "Robotics",
    ]

