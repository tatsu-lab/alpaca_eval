import copy
import itertools
import logging
import random
import os
import pathlib
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import datasets
import numpy as np
import pandas as pd

# don't load from utils to avoid unnecessary dependencies
AnyPath = Union[str, os.PathLike, pathlib.Path]
AnyData = Union[Sequence[dict[str, Any]], pd.DataFrame, datasets.Dataset]
DUMMY_EXAMPLE = dict(instruction="1+1=", output_1="2", input="", output_2="3")


def read_or_return(to_read: Union[AnyPath, str], **kwargs):
    """Read a file or return the input if it is already a string."""
    try:
        with open(Path(to_read), **kwargs) as f:
            out = f.read()
    except FileNotFoundError as e:
        logging.warning(f"Returning input because file not found. Error: {e}")
        out = to_read

    return out


def random_seeded_choice(seed: Union[int, str, float], choices):
    """Random choice with a (potentially string) seed."""
    return random.Random(seed).choice(choices)


def shuffle_pairwise_preferences(
        df: pd.DataFrame, arr_is_shuffle: Sequence[int]
) -> pd.DataFrame:
    """Shuffle the outputs of a pairwise preference dataframe.

    Examples
    --------
    >>> df = pd.DataFrame([dict(instruction='2+2', output_1='3', output_2='4', preference=2),
                           dict(instruction='2+3', output_1='5', output_2='4', preference=1)])
    >>> print(shuffle_pairwise_preferences(df, [True, False]))
        instruction output_1 output_2  preference
    0         2+2        4        3           1
    1         2+3        5        4           1
    """
    col_1 = df["output_1"].copy()
    col_2 = df["output_2"].copy()
    df["output_1"] = np.where(arr_is_shuffle, col_2, col_1)
    df["output_2"] = np.where(arr_is_shuffle, col_1, col_2)

    if "preference" in df.columns:
        df["preference"] = np.where(
            arr_is_shuffle, 3 - df["preference"], df["preference"]
        )

    return df


def is_derangement(arr1, arr2):
    """Whether 2 arrays are derangements of one another"""
    return all([a != b for a, b in zip(arr1, arr2)])


def random_derangement(arr, max_loop=10, seed=None):
    """
    Make random derangement of an array. I.e. shuffle without keeping any element in place. To be efficient,
    we first try `max_loop` rejection sampling. If didn't work then computes all possible derangement.
    """
    if len(arr) < 2:
        return arr

    rng = random.Random(seed)

    idcs = list(range(len(arr)))
    shuffled = list(range(len(arr)))

    for _ in range(max_loop):
        rng.shuffle(shuffled)
        if is_derangement(idcs, shuffled):
            return arr[shuffled]

    # if no luck then computes all possibilities
    deranged_order = list(
        set([s for s in itertools.permutations(idcs) if is_derangement(s, idcs)])
    )
    return arr[list(rng.choice(deranged_order))]


def _find_first_match(text: str, outputs_to_match: dict[str, Any]) -> tuple[Any, Any]:
    """Given text to parse and a dictionary of compiled regex to match, return the first match and corresponding key."""
    first_match = None
    first_key = None

    for key, compiled_regex in outputs_to_match.items():
        match = compiled_regex.search(text)
        if match and (not first_match or match.start() < first_match.start()):
            first_match = match
            first_key = key

    return first_match, first_key


def parse_batched_completion(
        completion: str, outputs_to_match: dict[str, Any]
) -> list[Any]:
    """Parse a single batch of completions, by returning a sequence of keys in the order in which outputs_to_match
    was matched.

    Parameters
    ----------
    completion : str
        Completion to parse.

    outputs_to_match : dict[str, Any]
        Dictionary of compiled regex to match. Keys are the keys to return in the order in which they are matched.

    Examples
    --------
    >>> completion = '\n(b)\n\n### Best output for example 8:\n(a)\n\n### Best output for example 9:\n(b)\n\n### Best
    output for example 10:\n(a)\n\n### Best output for example 11:\n(a)'
    >>> parse_batched_completion(completion, {1: re.compile('\n\(a\)'), 2: re.compile('\n\(b\)')})
    [2, 1, 2, 1, 1]
    >>> parse_batched_completion(' (a)', {1: re.compile(' \(a\)'), 2: re.compile(' \(b\)')})
    [1]
    >>> completion = '### Preferred output in JSON format for example 4:\r\n{{\r\n"Concise explanation": "Both
    outputs are incorrect, but Output (a) is less confusing and more concise.",\r\n"Output (a) is better than Output
    (b)": true\r\n}}\r\n\r\n### Preferred output in JSON format for example 5:\r\n{{\r\n"Concise explanation": "Both
    outputs are incomplete, but Output (b) seems to start with a more relevant source.",\r\n"Output (a) is better
    than Output (b)": false\r\n}}\r\n\r\n### Preferred output in JSON format for example 6:\r\n{{\r\n"Concise
    explanation": "Both outputs are incorrect, but Output (a) is less confusing and more concise.",\r\n"Output (a) is
    better than Output (b)": true\r\n}}\r\n\r\n### Preferred output in JSON format for example 7:\r\n{{\r\n"Concise
    explanation": "Both outputs are incomplete, but Output (b) seems to start with a more relevant source.",
    \r\n"Output (a) is better than Output (b)": false\r\n}}'
    >>> parse_batched_completion(completion, {1: re.compile(' true'), 2: re.compile(' false')})
    [1, 2, 1, 2]
    """
    completion = copy.deepcopy(completion)
    responses = []
    while True:
        match, key = _find_first_match(completion, outputs_to_match)
        if not match:
            break
        responses.append(key)
        # avoid matching the same output twice
        completion = completion[match.end():]
    return responses


def make_prompts(
        df: pd.DataFrame, template: str, batch_size: int = 1, padding_example=DUMMY_EXAMPLE
) -> tuple[list[str], pd.DataFrame]:
    """Helper function to make batch prompts for a single template.

    Parameters
    ----------
    df : pd.DataFrame
        Examples to annotate

    template : str
        Template for the prompt. Should have batch_size number of placeholder {key} where key is a column in df.

    batch_size : int
        Number of examples to batch in a single prompt.

    padding_example : dict
        Padding example to use if len(df) not divisible by batch_size.

    Returns
    -------
    prompts : list[str]
        List of formatted prompts.

    df_out : pd.DataFrame
        All examples. Will be df with potential padding examples.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"instruction": ["solve", "write backwards", "other 1"],
                           "input": ["1+1", "'abc'", ""]})
    >>> make_prompts(df, template="first: {instruction} {input}, second: {instruction} {input}",
                     batch_size=2, padding_example=dict(instruction="pad", input="pad_in"))[0]
    ["first: solve 1+1, second: write backwards 'abc'",
     'first: other 1 , second: pad pad_in']
    """

    if df.empty:
        return [], df

    text_to_format = re.findall("{(.+?)}", template)
    n_occurrences = Counter(text_to_format)

    if not all([n == batch_size for n in n_occurrences.values()]):
        raise ValueError(
            f"All placeholders should be repeated batch_size={batch_size} times but {n_occurrences}."
        )

    # padding if you don't have enough examples
    n_to_pad = (batch_size - len(df)) % batch_size
    padding = pd.DataFrame([padding_example] * n_to_pad)
    padding["is_padding"] = True
    df_out = pd.concat([df, padding], axis=0, ignore_index=True)
    df_out["is_padding"] = df_out["is_padding"].fillna(False)

    prompts = []
    # ugly for loops, not trivial to vectorize because of the batching
    for i in range(0, len(df_out), batch_size):
        current_prompt = copy.deepcopy(template)
        for j in range(batch_size):
            for to_format in n_occurrences.keys():
                # replace only first occurrence (that's why we don't use .format)
                current_prompt = current_prompt.replace(
                    "{" + to_format + "}", str(df_out.iloc[i + j][to_format]), 1
                )
        prompts.append(current_prompt)

    return prompts, df_out


def convert_ordinal_to_binary_preference(
        preferences: Union[pd.DataFrame, list[dict[str, Any]]],
        ordinal_preference_key: str = "preference",
        binary_preference_key: str = "preference",
):
    """Convert ordinal preference annotations to preference annotations. By merging multiple subcategories together,
    eg A/a/b/B into A/B, or AA/A/a/b/B/BB into A/B.

    Parameters
    ----------
    preferences : pd.DataFrame or list of dicts
        List of dictionaries or a dataframe that contains ordinal preference A/a/b/B in ordinal_preference_key.

    ordinal_preference_key : str
        Key in the dictionaries or column name of the ordinal preference annotations.

    binary_preference_key : str
        Key in the dictionaries or column name of the binary preference annotations. This can be the same
        as ordinal_preference_key if you want to overwrite the ordinal preference annotations.

    Returns
    -------
    binary_preferences
        List of dictionary or a dataframe (same type as the input) that contains binary preferences A/B in
        binary_preference_key.

    Examples
    --------
    >>> preferences = [dict(output="test A", preference=1),
                        dict(output="test a", preference=2),
                        dict(output="test b", preference=3),
                        dict(output="test B", preference=4),
                        dict(output="test None", preference=0)]
    >>> convert_ordinal_to_binary_preference(preferences, ordinal_preference_key="preference",
    binary_preference_key="preference")
    [{'output': 'test A', 'preference': 1},
     {'output': 'test a', 'preference': 1},
     {'output': 'test b', 'preference': 2},
     {'output': 'test B', 'preference': 2},
     {'output': 'test None', 'preference': 0}]
    """
    if isinstance(preferences, pd.DataFrame):
        is_df = True
    else:
        is_df = False
        preferences = pd.DataFrame.from_records(preferences)

    preferences[binary_preference_key] = (
                                                 preferences[ordinal_preference_key].round().astype(int) - 1
                                         ) // 2 + 1

    if not is_df:
        preferences = preferences.to_dict(orient="records")

    return preferences


def convert_to_dataframe(data: AnyData) -> pd.DataFrame:
    """Convert input that AlpacaFarm accepts into a dataframe."""
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, datasets.Dataset):
        return data.data.to_pandas()
    elif isinstance(data, list):
        return pd.DataFrame.from_records(data)
    else:
        # try
        return pd.DataFrame(data)


def check_import(module: str, to_use: Optional[str] = None):
    """Check whether the given module is imported."""
    if module not in sys.modules:
        if to_use is None:
            error = '{} module not imported. Try "pip install {}".'.format(
                module, module
            )
            raise ImportError(error)
        else:
            error = 'You need {} to use {}. Try "pip install {}".'.format(
                module, to_use, module
            )
            raise ImportError(error)


def load_or_convert_to_dataframe(df=Union[AnyPath, AnyData], **kwargs):
    """Load a dataframe from a path or convert the input to a dataframe if it's not a path."""
    if isinstance(df, AnyPath):
        df = Path(df)
        suffix = df.suffix
        if suffix == ".json":
            df = pd.read_json(df, **kwargs)
        elif suffix == ".csv":
            df = pd.read_csv(df, **kwargs)
            if df.columns[0] == "Unnamed: 0":
                df.set_index(df.columns[0], inplace=True)
                df.index.name = None
        elif suffix == ".tsv":
            df = pd.read_table(df, sep="\t", **kwargs)
        else:
            raise ValueError(f"File format {suffix} not supported.")
    else:
        df = convert_to_dataframe(df, **kwargs)
    return df


class Timer:
    """Timer context manager"""

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start

    def __str__(self):
        return f"{self.duration:.3f} seconds"
