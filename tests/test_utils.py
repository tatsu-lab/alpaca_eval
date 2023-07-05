import doctest

import alpaca_eval.utils as utils


def test_shuffle_pairwise_preferences():
    doctest.run_docstring_examples(
        utils.shuffle_pairwise_preferences,
        globals(),
        name="shuffle_pairwise_preferences",
        verbose=True,
    )


def test_make_prompts():
    doctest.run_docstring_examples(utils.make_prompts, globals(), name="make_prompts", verbose=True)


def test_convert_ordinal_to_binary_preference():
    doctest.run_docstring_examples(
        utils.convert_ordinal_to_binary_preference,
        globals(),
        name="convert_ordinal_to_binary_preference",
        verbose=True,
    )
