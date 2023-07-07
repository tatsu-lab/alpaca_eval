import doctest

from alpaca_eval import completion_parsers


def test_regex_parser():
    doctest.run_docstring_examples(completion_parsers.regex_parser, globals(), name="regex_parser", verbose=True)


def test_lmsys_parser():
    doctest.run_docstring_examples(completion_parsers.lmsys_parser, globals(), name="lmsys_parser", verbose=True)


def test_ranking_parser():
    doctest.run_docstring_examples(
        completion_parsers.ranking_parser,
        globals(),
        name="ranking_parser",
        verbose=True,
    )
