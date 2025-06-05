import pytest

from standardize_terms import standardize_text
from main_notebook_script import extract_keywords

@pytest.mark.parametrize("input_text", [
    "Generative ai is cool",
    "GENERATIVE AI is cool",
    "generative Ai is cool"
])
def test_standardize_text_case_insensitive(input_text):
    expected = "Generative Artificial Intelligence is cool"
    assert standardize_text(input_text) == expected

def test_extract_keywords_semicolon_and_standardization():
    text = "LLMs; generative AI; AI Ethics; ML"
    result = extract_keywords(text)
    assert result == [
        "LLM",
        "Generative Artificial Intelligence",
        "Artificial Intelligence Ethics",
        "Machine Learning",
    ]
