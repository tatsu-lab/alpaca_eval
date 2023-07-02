"""Runs all unit tests for the decoders."""
import doctest

import pytest
from unittest.mock import patch, Mock
from alpaca_eval.decoders.openai import openai_completions, _prompt_to_chatml, _string_to_dict


# def test_openai_completions():
#     prompts = ["Hello, my name is Test.", "How are you today?"]
#     with patch('alpaca_eval.decoders.openai._openai_completion_helper') as mocked_helper:
#         mocked_completion = Mock()
#         mocked_completion.text = 'Hello, Test. I am good.'
#         mocked_completion.total_tokens = 100
#         mocked_helper.return_value = [mocked_completion]
#
#         result = openai_completions(prompts, model_name='gpt-4')
#         assert len(result['completions']) == len(prompts)
#         assert len(result['price_per_example']) == len(prompts)
#         assert len(result['time_per_example']) == len(prompts)
#         mocked_helper.assert_called()
#
#
# def test_prompt_to_chatml():
#     doctest.run_docstring_examples(_prompt_to_chatml, globals(), name="_prompt_to_chatml", verbose=True)
#

def test_string_to_dict():
    doctest.run_docstring_examples(_string_to_dict, globals(), name="_string_to_dict", verbose=True)
