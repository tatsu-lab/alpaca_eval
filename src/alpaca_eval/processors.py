"""
Helper classes for processing the data. Each of those should have a function preprocess and postprocess, which will
respectively be called in SingleAnnotator._preprocess and SingleAnnotator._postprocess in reverse order.

Note: not worth to make the changes but all the parsers could have been processors.
"""

import abc
import json
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import utils

__all__ = ["RandomSwitchTwoColumnsProcessor", "PaddingForBatchesProcessor", "RawCompletionProcessor"]


class BaseProcessor(abc.ABC):
    """Base class for a processor."""

    def __init__(
        self,
        seed: int = 123,
        annotation_column: str = "annotation",
        completion_column: str = "raw_completion",
    ):
        self.seed = seed
        self.annotation_column = annotation_column
        self.completion_column = completion_column

    @abc.abstractmethod
    def preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """Process the annotation dataframe before annotations."""
        pass

    @abc.abstractmethod
    def postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """Process the annotation dataframe after annotations."""
        pass


class RandomSwitchTwoColumnsProcessor(BaseProcessor):
    r"""Randomly switch the order of two columns.

    Parameters
    ----------
    two_columns_to_switch : Sequence[str]
        The two columns to switch.

    fn_replace_if_switch : Optional[Callable[[pd.DataFrame], pd.DataFrame]], optional
        Function to apply to the dataframe formed of the rows with a switch. By default, does nothing.

    fn_replace_if_unswitch : Optional[Callable[[pd.DataFrame], pd.DataFrame]], optional
        Function to apply to the dataframe formed of the rows without a switch. By default, applies the same as
        `fn_replace_if_switch`.

    random_seed_columns : Optional[Sequence[str]], optional
        The columns to use to seed the random choice of switching or not. If None, will use `columns_to_switch`.

    kwargs :
        Additional arguments to pass to `BaseProcessor`. E.g. seed.

    Examples
    --------
    >>> df = pd.DataFrame([dict(instruction='2+2', output_1='10', output_2='4', preference=2),
    ...                    dict(instruction='2+3', output_1='5', output_2='7', preference=1)])
    >>> processor = RandomSwitchTwoColumnsProcessor(two_columns_to_switch=['output_1', 'output_2'],
    ...                                             fn_replace_if_switch = lambda x: x.replace({"preference":{1: 2, 2: 1}}))
    >>> processor.preprocess(df)
        instruction output_1 output_2  preference is_switch_output_1_output_2
    0         2+2         4       10           1                         True
    1         2+3         5        7           1                        False
    >>> (processor.postprocess(processor.preprocess(df)) == df).all(axis=None)
    True
    """

    def __init__(
        self,
        two_columns_to_switch: Sequence[str],
        fn_replace_if_switch=None,
        fn_replace_if_unswitch=None,
        random_seed_columns: Optional[Sequence[str]] = None,
        _switch_column: Optional[str] = None,
        **kwargs,
    ):
        self.two_columns_to_switch = list(set(two_columns_to_switch))
        if len(self.two_columns_to_switch) != 2:
            raise ValueError(
                f"two_columns_to_switch should have exactly two different columns but {two_columns_to_switch}"
            )
        self.fn_replace_if_switch = fn_replace_if_switch or (lambda x: x)
        # by default we assume that it's an involutive function
        self.fn_replace_if_unswitch = fn_replace_if_unswitch or self.fn_replace_if_switch

        # `switch_column` used for backward compatibility
        if _switch_column is None:
            _switch_column = "_".join(["is_switch"] + list(two_columns_to_switch))
        self._switch_column = _switch_column

        if random_seed_columns is None:
            random_seed_columns = two_columns_to_switch
        self.random_seed_columns = sorted(list(random_seed_columns))

        super().__init__(**kwargs)

    def preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """When preprocessing, we select the rows to switch and perform the switch."""
        df_to_annotate = df_to_annotate.copy()

        # randomize order of output_1, output_2 base on inputs
        df_to_annotate[self._switch_column] = df_to_annotate.apply(
            # we add "_switch_column" at the beginning to not use the same seed for all tasks
            lambda x: utils.random_seeded_choice(
                seed=self._switch_column + "".join(x[self.random_seed_columns]) + str(self.seed),
                choices=[False, True],
            ),
            axis=1,
        )
        return self._switch_or_unswitch(df_to_annotate, is_switch=True)

    def postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """When postprocessing, we undo the switch and remove the switch column."""
        df_annotated = df_annotated.copy()
        df_annotated = self._switch_or_unswitch(df_annotated, is_switch=False)
        df_annotated = df_annotated.drop(columns=[self._switch_column])
        return df_annotated

    @property
    def col1(self):
        return self.two_columns_to_switch[0]

    @property
    def col2(self):
        return self.two_columns_to_switch[1]

    def _switch_or_unswitch(self, df: pd.DataFrame, is_switch: bool) -> pd.DataFrame:
        """Applies the switch to the dataframe. If `is_switch=False` will undo the switch."""

        # switching two columns is an involution => no need to use is_switch here
        col1_values = df[self.col1].copy()
        col2_values = df[self.col2].copy()
        is_switch_arr = df[self._switch_column]
        df[self.col2] = np.where(is_switch_arr, col1_values, col2_values)
        df[self.col1] = np.where(is_switch_arr, col2_values, col1_values)

        if is_switch:
            df.loc[is_switch_arr, :] = self.fn_replace_if_switch(df.loc[is_switch_arr, :])
        else:
            df.loc[is_switch_arr, :] = self.fn_replace_if_unswitch(df.loc[is_switch_arr, :])

        return df


class PaddingForBatchesProcessor(BaseProcessor):
    r"""Pad the dataframe to have a number of examples divisible by `batch_size`.

    Parameters
    ----------
    batch_size : int
        Number of examples to batch in a single prompt.

    padding_example : dict
        Padding example to use if len(df) not divisible by batch_size.

    kwargs :
        Additional arguments to pass to `BaseProcessor`. E.g. seed.

    Examples
    --------
    >>> df = pd.DataFrame({"instruction": ["solve", "write", "other 1"],
    ...                    "input": ["1+1", "'abc'", ""]})
    >>> processor = PaddingForBatchesProcessor(batch_size=2, padding_example=dict(instruction="pad", input="pad_in"))
    >>> processor.preprocess(df)
        instruction   input  is_padding
    0         solve     1+1       False
    1         write   'abc'       False
    2       other 1               False
    3           pad  pad_in        True
    >>> (processor.postprocess(processor.preprocess(df)) == df).all(axis=None)
    True
    """

    def __init__(self, batch_size, padding_example: dict, **kwargs):
        self.batch_size = batch_size
        self.padding_example = padding_example
        super().__init__(**kwargs)

    def preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        # padding if you don't have enough examples
        n_to_pad = (self.batch_size - len(df_to_annotate)) % self.batch_size
        padding = pd.DataFrame([self.padding_example] * n_to_pad)
        padding["is_padding"] = True
        df_out = pd.concat([df_to_annotate, padding], axis=0, ignore_index=True)
        df_out["is_padding"] = df_out["is_padding"].fillna(False)
        return df_out

    def postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        return df_annotated[~df_annotated["is_padding"].astype(bool)].drop(columns=["is_padding"]).copy()


class RawCompletionProcessor(BaseProcessor):
    r"""Processes the raw completins by loading them as a JSON and, if chain of thought is used, adding a dictionary
    "referenced_models" to better understand which model names correspond to which outputs in the chain of thought.

    Examples
    --------
    >>> raw_completion = '{"concise_explanation": "M is better", "ordered_models": [{"rank": 1, "model": "M"}, {"rank": 2, "model": "m"}]}'
    >>> df = pd.DataFrame([dict(preference=2, raw_completion=raw_completion),
    ...                    dict(preference=1, raw_completion=raw_completion)])
    >>> processor = RawCompletionProcessor(is_chain_of_thought=True)
    >>> processor.postprocess(df).drop(columns=["ordered_models"])
        preference	                                   raw_completion	                referenced_models
    0	        2	{'concise_explanation': 'M is better', 'ordere...	{'M': 'output_2', 'm': 'output_1'}
    1	        1	{'concise_explanation': 'M is better', 'ordere...	{'M': 'output_1', 'm': 'output_2'}
    """

    def __init__(self, is_chain_of_thought: bool = False, **kwargs):
        self.is_chain_of_thought = is_chain_of_thought
        super().__init__(**kwargs)

    def preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        return df_to_annotate

    def postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """Load the raw completion as a JSON and add the referenced models to better understand chain of thought."""
        df_annotated = df_annotated.copy()

        if self.completion_column in df_annotated:
            df_annotated[self.completion_column] = df_annotated[self.completion_column].apply(_try_json_load)
            if self.is_chain_of_thought:
                self.add_referenced_model_(df_annotated)

        return df_annotated

    def add_referenced_model_(self, df):
        """Add a dictionary to better understand chain of thought in case it's useful"""
        for i, r in df.iterrows():
            if (
                isinstance(r[self.completion_column], dict)
                and "concise_explanation" in r[self.completion_column]
                and "ordered_models" in r[self.completion_column]
            ):
                preference = int(df.loc[i, "preference"])
                ordered_models = df.loc[i, self.completion_column]["ordered_models"]
                for m in ordered_models:
                    if m["rank"] == 1:
                        first_model = m["model"]
                    elif m["rank"] == 2:
                        second_model = m["model"]
                    else:
                        assert False

                if "referenced_models" not in df.columns:
                    df["referenced_models"] = None

                df.at[i, "referenced_models"] = {
                    first_model: f"output_{preference}",
                    second_model: f"output_{3 - preference}",
                }


def _try_json_load(el):
    """Try to load as json"""
    try:
        return json.loads(el)
    except:
        return el
