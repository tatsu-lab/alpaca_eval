import ast
import copy
import logging
import re
from typing import Any

import numpy as np

from . import utils


def regex_parser(completion: str, outputs_to_match: dict[str, Any]) -> list[Any]:
    r"""Parse a single batch of completions, by returning a sequence of keys in the order in which outputs_to_match
    was matched.

    Parameters
    ----------
    completion : str
        Completion to parse.

    outputs_to_match : dict[str, Any]
        Dictionary of compiled regex to match. Keys are the keys to return in the order in which they are matched.
        The values can be either a compiled regex or a string. If a string, it will be compiled to a regex and that will
        be modified inplace.

    Examples
    --------
    >>> completion = ('\n(b)\n\n### Best output for example 8:\n(a)\n\n### Best output for example 9:\n(b)\n\n### Best'\
    ...               ' output for example 10:\n(a)\n\n### Best output for example 11:\n(a)')
    >>> regex_parser(completion, {1: r"\n\(a\)", 2: r"\n\(b\)"})
    [2, 1, 2, 1, 1]
    >>> regex_parser(' (a)', {1: r" \(a\)", 2: r" \(b\)"})
    [1]
    >>> completion = ('### Preferred output in JSON format for example 4:\r\n{{\r\n"Concise explanation": "Both'\
    ... ' outputs are incorrect, but Output (a) is less confusing and more concise.",\r\n"Output (a) is better than'\
    ... ' Output (b)": true\r\n}}\r\n\r\n### Preferred output in JSON format for example 5:\r\n{{\r\n"Concise'\
    ... ' explanation": "Both outputs are incomplete, but Output (b) seems to start with a more relevant source."'\
    ... ',\r\n"Output (a) is better than Output (b)": false\r\n}}\r\n\r\n### Preferred output in JSON format for'\
    ... ' example 6:\r\n{{\r\n"Concise explanation": "Both outputs are incorrect, but Output (a) is less confusing and'\
    ... ' more concise.",\r\n"Output (a) is better than Output (b)": true\r\n}}\r\n\r\n### Preferred output in JSON' \
    ... ' format for example 7:\r\n{{\r\n"Concise explanation": "Both outputs are incomplete, but Output (b) seems to'\
    ... ' start with a more relevant source.", \r\n"Output (a) is better than Output (b)": false\r\n}}')
    >>> regex_parser(completion, {1: ' true', 2: ' false'})
    [1, 2, 1, 2]
    """
    for k, v in outputs_to_match.items():
        if not isinstance(v, re.Pattern):
            # inplace modification, which is bad practice but useful to speedup
            outputs_to_match[k] = re.compile(v)

    completion = copy.deepcopy(completion)
    responses = []
    while True:
        match, key = utils._find_first_match(completion, outputs_to_match)
        if not match:
            break
        responses.append(key)
        # avoid matching the same output twice
        completion = completion[match.end() :]
    return responses


# modified from: https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/eval_gpt_review.py#L47
# does not work with batched completions
def lmsys_parser(completion: str):
    r"""Parse a pair of scores from a single completion and returns which is better.

    Examples
    --------
    >>> lmsys_parser("1 7\n ...")
    [2]
    >>> lmsys_parser("7 1\n more text")
    [1]
    >>> lmsys_parser("1 1\n ...")
    [0]
    """
    try:
        score_pair = completion.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            lmsys_score_1 = float(sp[0])
            lmsys_score_2 = float(sp[1])
            if lmsys_score_1 > lmsys_score_2:
                return [1]
            elif lmsys_score_1 < lmsys_score_2:
                return [2]
            else:
                return [0]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logging.error(f"{e}\nContent: {completion}\n" "You must manually fix the score pair.")
        return [np.nan]


def ranking_parser(completion):
    r"""Parse a completion that contains a list of dictionary and returns the name of the preferred model.

    Examples
    --------
    >>> ranking_parser("[{'model': 'model_1', 'rank': 1}, {'model': 'model_2', 'rank': 2}]")
    [1]
    >>> ranking_parser("[{'model': 'model_1', 'rank': 2}, {'model': 'model_2', 'rank': 1}]")
    [2]
    >>> ranking_parser("[{'model': 'model_1', 'rank': 3}, {'model': 'model_2', 'rank': 1}]")
    [nan]
    """
    try:
        if isinstance(completion, str):
            ordered_completions = ast.literal_eval(completion)
        else:
            ordered_completions = completion

        rank = [c for c in ordered_completions if c["model"] == "model_1"][0]["rank"]
        assert rank in [1, 2]

        return [rank]
    except Exception as e:
        logging.error(f"{e}\nContent: {completion}\n" "You must manually fix the score pair.")
        return [np.nan]
