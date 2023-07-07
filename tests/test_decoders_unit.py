"""Runs all unit tests for the decoders."""
import doctest
import math

import pytest
from openai.openai_object import OpenAIObject

from alpaca_eval.decoders.anthropic import anthropic_completions
from alpaca_eval.decoders.cohere import cohere_completions
from alpaca_eval.decoders.huggingface_api import huggingface_api_completions
from alpaca_eval.decoders.openai import _prompt_to_chatml, _string_to_dict, openai_completions

MOCKED_COMPLETION = "Mocked completion text"


@pytest.fixture
def mock_openai_completion():
    # Create a mock Completion object
    completion_mock = OpenAIObject()
    completion_mock["total_tokens"] = 3
    completion_mock["text"] = MOCKED_COMPLETION
    return completion_mock


def test_openai_completions(mocker, mock_openai_completion):
    # Patch the _openai_completion_helper function to return the mock completion object
    mocker.patch(
        "alpaca_eval.decoders.openai._openai_completion_helper",
        return_value=[mock_openai_completion],
    )
    result = openai_completions(["Prompt 1", "Prompt 2"], "text-davinci-003", batch_size=1)
    _run_all_asserts_completions(result)


def test_anthropic_completions(mocker):
    mocker.patch(
        "alpaca_eval.decoders.anthropic._anthropic_completion_helper",
        return_value=MOCKED_COMPLETION,
    )
    result = anthropic_completions(["Prompt 1", "Prompt 2"], num_procs=1)
    _run_all_asserts_completions(result)


def test_cohere_completions(mocker):
    mocker.patch(
        "alpaca_eval.decoders.cohere._cohere_completion_helper",
        return_value="Mocked completion text",
    )
    result = cohere_completions(["Prompt 1", "Prompt 2"], num_procs=1)
    _run_all_asserts_completions(result)


def test_huggingface_api_completions(mocker):
    mocker.patch(
        "alpaca_eval.decoders.huggingface_api.inference_helper",
        return_value=dict(generated_text="Mocked completion text"),
    )
    result = huggingface_api_completions(
        ["Prompt 1", "Prompt 2"],
        model_name="timdettmers/guanaco-33b-merged",
        num_procs=1,
    )
    _run_all_asserts_completions(result)


def _run_all_asserts_completions(result):
    expected_completions = [MOCKED_COMPLETION, MOCKED_COMPLETION]
    assert result["completions"] == expected_completions

    for i in range(len(result["time_per_example"])):
        assert 0 < result["time_per_example"][i] < 1

    assert len(result["price_per_example"]) == 2
    if not math.isnan(result["price_per_example"][0]):
        assert result["price_per_example"][0] == result["price_per_example"][1]
        assert 0 <= result["price_per_example"][0] < 1e-2
    else:
        assert math.isnan(result["price_per_example"][1]) == math.isnan(result["price_per_example"][0])


def test_prompt_to_chatml():
    doctest.run_docstring_examples(_prompt_to_chatml, globals(), name="_prompt_to_chatml", verbose=True)


def test_string_to_dict():
    doctest.run_docstring_examples(_string_to_dict, globals(), name="_string_to_dict", verbose=True)
