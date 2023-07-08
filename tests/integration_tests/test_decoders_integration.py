"""Runs all unit tests for the decoders."""

import pytest

from alpaca_eval import constants, utils
from alpaca_eval.decoders.anthropic import anthropic_completions
from alpaca_eval.decoders.cohere import cohere_completions
from alpaca_eval.decoders.huggingface_api import huggingface_api_completions
from alpaca_eval.decoders.huggingface_local import huggingface_local_completions
from alpaca_eval.decoders.openai import openai_completions


def _get_formatted_prompts(model):
    filename = list((constants.MODELS_CONFIG_DIR / model).glob("*.txt"))[0]
    template = utils.read_or_return(filename)
    prompts = ["Respond with a single digit: 1+1=", "Respond with a single digit: 2+2="]
    prompts = [template.format(instruction=prompt) for prompt in prompts]
    return prompts


@pytest.mark.slow
def test_openai_completions_integration():
    prompts = _get_formatted_prompts("gpt4")
    print(prompts)
    results = openai_completions(prompts, model_name="gpt-3.5-turbo", tokens_to_avoid=["2", " 2", "2 "])
    assert len(results["completions"]) == len(prompts)
    assert "4" in results["completions"][1]
    assert "2" not in results["completions"][0]


@pytest.mark.slow
def test_anthropic_completions_integration():
    prompts = _get_formatted_prompts("claude")
    results = anthropic_completions(prompts)
    assert len(results["completions"]) == len(prompts)
    assert "2" in results["completions"][0]
    assert "4" in results["completions"][1]


@pytest.mark.slow
def test_cohere_completions_integration():
    prompts = _get_formatted_prompts("cohere")
    results = cohere_completions(prompts)
    assert len(results["completions"]) == len(prompts)
    assert "2" in results["completions"][0]
    assert "4" in results["completions"][1]


@pytest.mark.slow
def test_huggingface_api_completions_integration():
    prompts = _get_formatted_prompts("guanaco-7b")
    results = huggingface_api_completions(prompts, model_name="timdettmers/guanaco-33b-merged")
    assert len(results["completions"]) == len(prompts)
    assert "2" in results["completions"][0]
    assert "4" in results["completions"][1]


@pytest.mark.slow
def test_huggingface_local_completions_integration():
    prompts = _get_formatted_prompts("text_davinci_003")  # nor formatting
    results = huggingface_local_completions(prompts, model_name="hf-internal-testing/tiny-random-gpt2")
    assert len(results["completions"]) == len(prompts)
    # nothing to test because random model
