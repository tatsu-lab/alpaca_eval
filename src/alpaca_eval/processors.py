"""
Helper classes for processing the data. Each of those should have a function preprocess and postprocess, which will
respectively be called in SingleAnnotator._preprocess and SingleAnnotator._postprocess in reverse order.

Note: not worth to make the changes but all the parsers could have been processors.
"""

import abc
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import utils

DUMMY_EXAMPLE = dict(instruction="1+1=", output_1="2", input="", output_2="3")


class BaseProcessor(abc.ABC):
    """Base class for a processor."""

    def __init__(self, seed: int = 123):
        self.seed = seed

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

    replace_if_swtich_kwargs : dict, optional
        Arguments to pass to `df.replace` to replace some values in the dataframe when there was a switch.

    replace_if_unswitch_kwargs : dict, optional
        Arguments to pass to `df.replace` to replace some values in the dataframe when you are undoing a switch.
        By default, applies the same as `replace_if_switch_kwargs`.

    random_seed_columns : Optional[Sequence[str]], optional
        The columns to use to seed the random choice of switching or not. If None, will use `columns_to_switch`.

    kwargs :
        Additional arguments to pass to `BaseProcessor`. E.g. seed.

    Examples
    --------
    >>> df = pd.DataFrame([dict(instruction='2+2', output_1='10', output_2='4', preference=2),
    ...                    dict(instruction='2+3', output_1='5', output_2='7', preference=1)])
    >>> processor = RandomSwitchTwoColumnsProcessor(two_columns_to_switch=['output_1', 'output_2'],
    ...                                                replace_if_switch_kwargs={'preference': {1: 2, 2: 1}})
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
        replace_if_switch_kwargs: dict = None,
        replace_if_unswitch_kwargs: dict = None,
        random_seed_columns: Optional[Sequence[str]] = None,
        _switch_column: Optional[str] = None,
        **kwargs,
    ):
        self.two_columns_to_switch = list(set(two_columns_to_switch))
        if len(self.two_columns_to_switch) != 2:
            raise ValueError(
                f"two_columns_to_switch should have exactly two different columns but {two_columns_to_switch}"
            )
        self.replace_if_switch_kwargs = replace_if_switch_kwargs or {}
        self.replace_if_unswitch_kwargs = replace_if_unswitch_kwargs or self.replace_if_switch_kwargs

        # `switch_column` used for backward compatibility
        if _switch_column is None:
            _switch_column = "_".join(["is_switch"] + list(two_columns_to_switch))
        self._switch_column = _switch_column

        if random_seed_columns is None:
            random_seed_columns = two_columns_to_switch
        self.random_seed_columns = list(random_seed_columns)

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

        # replace might not be an involution e.g. if using a regex
        if is_switch:
            df.loc[is_switch_arr, :] = df.loc[is_switch_arr, :].replace(self.replace_if_switch_kwargs)
        else:
            df.loc[is_switch_arr, :] = df.loc[is_switch_arr, :].replace(self.replace_if_unswitch_kwargs)

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
        return df_annotated[~df_annotated["is_padding"]].drop(columns=["is_padding"]).copy()
