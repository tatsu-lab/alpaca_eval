import contextlib
import copy
import glob
import itertools
import logging
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import datasets
import numpy as np
import pandas as pd
import pkg_resources
import yaml

from . import constants

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
        if Path(to_read).is_absolute():
            # The path is not absolute, so it's not just a string
            raise e

        logging.warning(f"Returning input because file not found. Error: {e}")
        out = to_read

    return out


def random_seeded_choice(seed: Union[int, str, float], choices, **kwargs):
    """Random choice with a (potentially string) seed."""
    return random.Random(seed).choices(choices, k=1, **kwargs)[0]


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
    deranged_order = list(set([s for s in itertools.permutations(idcs) if is_derangement(s, idcs)]))
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


def make_prompts(
    df: pd.DataFrame,
    template: str,
    batch_size: int = 1,
) -> tuple[list[str], pd.DataFrame]:
    r"""Helper function to make batch prompts for a single template.

    Parameters
    ----------
    df : pd.DataFrame
        Examples to annotate

    template : str
        Template for the prompt. Should have batch_size number of placeholder {key} where key is a column in df.

    batch_size : int
        Number of examples to batch in a single prompt.

    Returns
    -------
    prompts : list[str]
        List of formatted prompts.

    df_out : pd.DataFrame
        All examples. Will be df with potential padding examples.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"instruction": ["solve", "write backwards", "other 1", "pad"],
    ...                    "input": ["1+1", "'abc'", "", "pad_in"]})
    >>> make_prompts(df, template="first: {instruction} {input}, second: {instruction} {input}", batch_size=2)[0]
    ["first: solve 1+1, second: write backwards 'abc'", 'first: other 1 , second: pad pad_in']
    """

    if df.empty:
        return [], df

    text_to_format = re.findall("{([^ \s]+?)}", template)
    n_occurrences = Counter(text_to_format)

    if not all([n == batch_size for n in n_occurrences.values()]):
        raise ValueError(f"All placeholders should be repeated batch_size={batch_size} times but {n_occurrences}.")

    if len(df) % batch_size > 0:
        raise ValueError(
            f"The number of rows should be dividable by the batch_size={batch_size} but got {len(df)}."
            "You should use PaddingForBatchesProcessor"
        )

    df_out = df.copy()
    prompts = []
    # ugly for loops, not trivial to vectorize because of the batching
    for i in range(0, len(df_out), batch_size):
        current_prompt = copy.deepcopy(template)
        for j in range(batch_size):
            for to_format in n_occurrences.keys():
                # replace only first occurrence (that's why we don't use .format)
                current_prompt = current_prompt.replace("{" + to_format + "}", str(df_out.iloc[i + j][to_format]), 1)
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
    >>> from alpaca_eval.utils import convert_ordinal_to_binary_preference
    >>> preferences = [dict(output="test A", preference=1),
    ...                dict(output="test a", preference=2),
    ...                dict(output="test b", preference=3),
    ...                dict(output="test B", preference=4),
    ...                dict(output="test None", preference=0)]
    >>> convert_ordinal_to_binary_preference(preferences, ordinal_preference_key="preference",
    ...     binary_preference_key="preference")
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

    preferences[binary_preference_key] = (preferences[ordinal_preference_key].round().astype(int) - 1) // 2 + 1

    if not is_df:
        preferences = preferences.to_dict(orient="records")

    return preferences


def convert_to_dataframe(data: AnyData) -> pd.DataFrame:
    """Convert input that AlpacaEval accepts into a dataframe."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    elif isinstance(data, datasets.Dataset):
        return data.data.to_pandas()
    elif isinstance(data, list):
        return pd.DataFrame.from_records(data)
    else:
        # try
        return pd.DataFrame(data)


def check_imports(modules: Sequence[str], to_use: str = "this fnction"):
    """Check whether the given module is imported."""
    modules = list(modules)
    for module in modules:
        if module not in sys.modules:
            error = f"You need {modules} to use {to_use}. Try `pip install {' '.join(modules)}`."
            raise ImportError(error)


def check_pkg_atleast_version(package, atleast_version):
    curr_version = pkg_resources.get_distribution(package).version
    return pkg_resources.parse_version(curr_version) > pkg_resources.parse_version(atleast_version)


def load_or_convert_to_dataframe(df=Union[AnyPath, AnyData, Callable], **kwargs):
    """Load a dataframe from a path or convert the input to a dataframe if it's not a path."""
    if isinstance(df, Callable):
        df = df(**kwargs)

    if isinstance(df, (str, os.PathLike, pathlib.Path)):
        df = Path(df)

        # check if it's a globbing pattern
        if "*" in str(df):
            df = pd.concat(
                [load_or_convert_to_dataframe(f, **kwargs) for f in glob.glob(str(df))],
            )
        else:
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
        return f"{self.duration:.1f} seconds"


@contextlib.contextmanager
def silent():
    """Context manager to remove all outputs and warnings."""
    import IPython

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), DisableLogger(), IPython.utils.io.capture_output():
        yield


class DisableLogger:
    def __enter__(self):
        logging.disable(50)

    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)


def contains_list(text):
    """Check if the text contains a list / bullet points...."""

    # Bullet points or '*' list items
    bullet_point_pattern = r"(\s*•\s*|\s*\*\s*)(\w+)"

    # Numbered lists with '.' or ')'
    number_list_pattern = r"(\s*\d+\.|\s*\d+\))\s*(\w+)"

    # Alphabetic lists with '.' or ')'
    alpha_list_pattern = r"(\s*[a-zA-Z]\.|\s*[a-zA-Z]\))\s*(\w+)"

    # List items starting with a dash '-'
    dash_list_pattern = r"(\s*-\s*)(\w+)"

    patterns = [
        bullet_point_pattern,
        number_list_pattern,
        alpha_list_pattern,
        dash_list_pattern,
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


def prioritize_elements(lst: list, elements: Sequence) -> list:
    """Prioritize elements in a list. If elements are not in the list, they will be appended to the end of the list."""
    elements = list(elements)
    for el in elements:
        if el in lst:
            lst.remove(el)
    return elements + lst


def load_configs(configs: Union[AnyPath, dict], relative_to: Optional[AnyPath] = None):
    """Load the config yaml files, or return if it's already a dict."""
    if not isinstance(configs, dict):
        if relative_to is not None:
            configs = Path(relative_to) / configs
        configs = Path(configs)
        if configs.is_dir():
            configs = configs / "configs.yaml"
        with open(configs, "r") as stream:
            try:
                configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logging.exception(exc)

    return configs


def get_precomputed_leaderboard(precomputed_leaderboard, reference_outputs, annotators_config):
    if precomputed_leaderboard == "auto":
        try:
            precomputed_leaderboard = constants.PRECOMPUTED_LEADERBOARDS[
                (str(reference_outputs), str(annotators_config))
            ]
        except KeyError:
            try:
                if Path(reference_outputs).is_absolute():
                    logging.warning(
                        f"precomputed_leaderboard = 'auto'. But we have found no corresponding leaderboard for"
                        f" {reference_outputs} and {annotators_config}"
                    )
            except:
                logging.warning(f"precomputed_leaderboard = 'auto'. But we have found no corresponding leaderboard")
            precomputed_leaderboard = None

    if precomputed_leaderboard is not None:
        try:
            leaderboard = load_or_convert_to_dataframe(precomputed_leaderboard).to_dict(orient="index")
        except FileNotFoundError:
            logging.warning(f"precomputed_leaderboard = {precomputed_leaderboard} not found => computing from scratch.")
            leaderboard = dict()
    else:
        leaderboard = dict()
    return leaderboard, precomputed_leaderboard


def get_output_path(output_path, model_outputs, name):
    if output_path == "auto":
        if model_outputs is None:
            output_path = None
        else:
            try:
                output_path = Path(model_outputs).parent
            except:
                if name is not None:
                    output_path = Path("results") / name
                else:
                    output_path = "."
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)


def print_leaderboard(df_leaderboard, leaderboard_mode, cols_to_print, current_name=None):
    cols_to_print = list(cols_to_print)

    if leaderboard_mode is not None:
        if "mode" in df_leaderboard.columns:
            # select all modes that come before
            current_idx = constants.ORDERED_LEADERBOARD_MODES.index(leaderboard_mode)
            df_leaderboard["mode_idx"] = df_leaderboard["mode"].apply(constants.ORDERED_LEADERBOARD_MODES.index)

            is_smaller_mode = df_leaderboard["mode_idx"] <= current_idx
            is_selected = is_smaller_mode | (df_leaderboard["mode"].isnull())

            if current_name is not None:
                is_selected |= df_leaderboard.index == current_name

            df_leaderboard = df_leaderboard[is_selected]
    elif "mode" in df_leaderboard.columns:
        cols_to_print = cols_to_print + ["mode"]
    print(df_leaderboard[cols_to_print].to_string(float_format="%.2f"))


def get_generator_name(name, model_outputs):
    if name is None:
        try:
            assert len(model_outputs["generator"].unique()) == 1
            name = model_outputs["generator"].iloc[0]
        except:
            name = "Current model"
    return name
