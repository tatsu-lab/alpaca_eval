import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union
import os

import numpy as np
import pandas as pd

from .. import completion_parsers, utils, constants
from ..decoders import get_fn_completions

CURRENT_DIR = Path(__file__).parent
logging.getLogger().setLevel(logging.INFO)

__all__ = ["PairwiseAnnotator", "SinglePairwiseAnnotator"]


class PairwiseAnnotator:
    """Class for a pool of annotators.

    Notes
    -----
    There are three main functions for annotations depending on how the outputs to compare are given:
        - annotate_pairs: annotate a sequence of examples that contain the pair of outputs `"output_1"` and `"output_2"`
        - annotate_samples: annotate a sequence of examples that contain `"output"` from which we will sample a pair of
            outputs. Useful for collecting pairwise preferences for RLHF.
        - annotate_head2head: annotate a pair of sequence of outputs, each containing `"output"` which will be merged
            into a single sequence of paired outputs. Useful for evaluation against a reference.

    Other functions that are useful for annotating:
        - set_noise: set the noise level for the annotators.
        - load_: load annotations from a file.
        - save: save annotations to a file.

    Parameters
    ----------
    annotators_config : Path or list of dict, optional
        A dictionary or path to a yaml file containing the configuration for the pool of annotators. If a directory,
        we search for 'configs.yaml' in it. The keys in the first  dictionary should be the annotator's name, and
        the value should be a dictionary of the annotator's configuration which should have the following keys:
        The path is relative to `evaluators_configs/` directory.
        - prompt_template (str): a prompt template or path to it. The template should contain placeholders for keys in
            the example dictionary, typically {instruction} and {output_1} {output_2}.
        - fn_completions (str): function in `alpaca_farm.decoders` for completions. Needs to accept as first argument
            `prompts` which is a list of string.
        - completions_kwargs (dict): kwargs for fn_completions. E.g. model_name, max_tokens, temperature,
        tokens_to_avoid
        - fn_completion_parser (str) : Function in `completion_parsers.py` to use for parsing the completions into
        preferences.
        - completion_parser_kwargs (dict) : Kwargs for fn_completion_parser.
        - other kwargs to `SinglePairwiseAnnotator` such as batch_size

    seed : int, optional
        Seed for the random number generator.

    is_avoid_reannotations : bool, optional
        Whether to avoid re-annotating examples that have already been annotated by the annotator. This will decrease
        cost but can be slightly slower if there are no annotations that can be reused.

    caching_path : Path, optional
        Path to cache the annotations to. If None, will not save the annotations. If the path already exists it will
        load annotations from there.

    input_keys : tuple of str, optional
        Keys use to distinguish inputs.

    output_keys : tuple of str, optional
        Keys use to distinguish outputs.

    p_label_flip : float, optional
        Probability of flipping the label (ie adds noise by taking a mixture between predicted label and
        2*p_label_flip of independent coin flip). If None, will not flip the label. In AlpacaFarm we use 0.25
        for training. You can set this later on using `set_noise`.

    other_keys_to_keep : tuple of str, optional
        Other columns to store besides the preferences.

    is_store_missing_preferences : bool, optional
        Whether to store missing preferences. If True it will avoid trying to always reannotate examples that have
        errors.

    base_dir : Path, optional
        Path to the directory containing the annotators configs. I.e. annotators_config will be relative
        to this directory.
    """

    def __init__(
        self,
        annotators_config: Union[utils.AnyPath, list[dict[str, Any]]] = "claude",
        seed: Optional[int] = 0,
        is_avoid_reannotations: bool = True,
        caching_path: Optional[utils.AnyPath] = "auto",
        input_keys: Sequence[str] = ("instruction",),
        output_keys: Sequence[str] = ("output_1", "output_2"),
        p_label_flip: Optional[float] = None,
        other_keys_to_keep: Sequence[str] = ("price_per_example", "time_per_example"),
        is_store_missing_preferences: bool = True,
        base_dir: utils.AnyPath = constants.EVALUATORS_CONFIG_DIR,
    ):
        logging.info(f"Creating the annotator from `{annotators_config}`.")
        self.base_dir = Path(base_dir)

        # setting it relative to the config directory
        annotators_config = self.base_dir / annotators_config

        if annotators_config.is_dir():
            annotators_config = annotators_config / "configs.yaml"

        if caching_path == "auto":
            if isinstance(annotators_config, (str, Path, os.PathLike)):
                stem = Path(annotators_config).stem
                caching_path = (
                    Path(annotators_config).parent
                    / f"annotations_seed{seed}_{stem}.json"
                )
                logging.info(f"Saving annotations to `{caching_path}`.")
            else:
                logging.warning(
                    "caching_path cannot be 'auto' if annotators_config is not a path. Setting to None."
                )
                caching_path = None
        elif caching_path is not None:
            logging.warning(
                "Saving_path is given but not 'auto', make sure that it's different for different seeds."
            )

        self.seed = seed
        self.is_avoid_reannotations = is_avoid_reannotations
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        self.input_output_keys = self.input_keys + self.output_keys
        self.all_keys = self.input_keys + self.output_keys + ["annotator"]
        self.other_keys_to_keep = list(other_keys_to_keep)
        self.p_label_flip = p_label_flip
        self.is_store_missing_preferences = is_store_missing_preferences
        self.annotators_config = annotators_config

        self.annotators = self._initialize_annotators(annotators_config)
        self.caching_path = caching_path
        self.df_annotations = None
        self.load_()

    ### Helper properties to make it easier to inherit from this class ###
    @property
    def SingleAnnotator(self):
        return SinglePairwiseAnnotator

    #########################################

    def annotate_samples(
        self,
        all_outputs: utils.AnyData,
        keys_to_sample_output_2: Optional[Sequence] = None,
        is_unique_instructions: bool = True,
        p_label_flip: Optional[float] = None,
        is_multisample_list: bool = True,
        **decoding_kwargs,
    ) -> list[dict[str, Any]]:
        """Sample pairs of outputs from a sequence of examples and annotate them.

        Parameters
        ----------
        all_outputs : list of dict or pd.DataFrame or datasets.Dataset
            All examples from which we will sample a pair of outputs to annotate. Each dictionary (or row) should
            contain all of `self.input_keys` and `keys_to_sample_output_2` and `"output"`.

        keys_to_sample_output_2 : tuple of str, optional
            Keys to use to sample paired `"output_2"` to compare to the current `"output"` which will become
            `"output_1"`. If `None` it uses `self.input_keys`.

        is_unique_instructions : bool, optional
            Whether to deduplicate the instructions such that there is only one pair per instruction. If False
            there will be as many pairs as there are outputs for each instruction.

        p_label_flip : float, optional
            Probability of flipping the label (ie adds noise by taking a mixture between predicted label and
            2*p_label_flip of independent coin flip). If None, will use `self.p_label_flip`.

        is_multisample_list : bool, optional
            If True `all_outputs` is a list of examples (dictionary) and each example has an `"output"` column
            containing
            a list of all multi samples. If False `"output"` contains a single output but each element in the list is a
            different (instruction, output) pair with potentially the same instruction.

        decoding_kwargs :
            Additional arguments to pass to the decoder.
        """

        all_outputs = utils.convert_to_dataframe(all_outputs)

        if is_multisample_list:
            all_outputs = (
                all_outputs.explode("output")
                .reset_index()
                .rename(columns={"index": "sample_id"})
            )
            all_outputs["sample_id"] = all_outputs.groupby("sample_id").cumcount()

        if keys_to_sample_output_2 is None:
            keys_to_sample_output_2 = self.input_keys
        keys_to_sample_output_2 = list(keys_to_sample_output_2)

        n_pre_drop = len(all_outputs)

        # set output to be unique for each keys_to_sample_output_2
        df_to_annotate = (
            all_outputs.groupby(keys_to_sample_output_2)
            .apply(lambda x: x.drop_duplicates(["output"]))
            .reset_index(drop=True)
            .rename(columns={"output": "output_1"})
        )

        if len(df_to_annotate) != n_pre_drop:
            logging.warning(
                f"""Filtered rows because of duplicate outputs for the same keys_to_sample_output_2=
                {keys_to_sample_output_2}. {n_pre_drop} -> {len(df_to_annotate)}"""
            )

        # sample an output 2 for each output 1 that are different
        df_to_annotate["output_2"] = df_to_annotate.groupby(
            list(keys_to_sample_output_2)
        )["output_1"].transform(
            lambda x: utils.random_derangement(x.values, seed=self.seed)
        )

        if is_unique_instructions:
            n_pre_dedup = len(df_to_annotate)
            df_to_annotate = df_to_annotate.drop_duplicates(subset=self.input_keys)
            if len(df_to_annotate) != n_pre_dedup:
                logging.info(
                    f"Filtered unique instruction/input pairs: {n_pre_dedup} -> {len(df_to_annotate)}"
                )

        if p_label_flip is not None:
            old_p_label_flip = self.p_label_flip
            self.set_noise(p_label_flip)

        try:
            annotated = self.annotate_pairs(df_to_annotate, **decoding_kwargs)
        finally:
            # reset even if there is an error
            if p_label_flip is not None:
                self.set_noise(old_p_label_flip)

        return annotated

    def annotate_head2head(
        self,
        outputs_1: Union[Sequence[dict[str, Any]], pd.DataFrame],
        outputs_2: Union[Sequence[dict[str, Any]], pd.DataFrame],
        keys_to_merge: Optional[Sequence[str]] = None,
        is_ordered: bool = False,
        **decoding_kwargs,
    ) -> list[dict[str, Any]]:
        """Head-to-head comparison between two sequence of outputs.

        Parameters
        ----------
        outputs_1 : list of dict or dataframe
            Examples to annotate. Each dictionary (or row) should contain all of `keys_to_merge` and `"output"`.
            `"output"` will become `"output_1"`.

        outputs_2 : list of dict or dataframe
            Second  to annotate. Each dictionary (or row) should contain all of `keys_to_merge` and `"output"`.
            `"output"` will become `"output_2"`.

        keys_to_merge : tuple of str, optional
            Keys to use to merge the two sequences of outputs. If None uses `self.input_keys`

        is_ordered : bool, optional
            Whether the two sequences of outputs are in matching order. If not we will be merging based on
            `keys_to_merge`, which means that the outputs can actually be shorter than the inputs (if some outputs
            are not found in the other sequence) or longer (if some outputs are duplicated in both sequences =>
            set cartesian products).

        decoding_kwargs :
            Additional arguments to pass to `fn_completions`.

        Returns
        -------
        annotated : list of dict
            The annotated examples. Each dictionary will contain all of `keys_to_merge`, `"output_1"`, `"output_2"`, and
            `"preference"`. Preference will be 0 if output_1 == output_2, 1 if output_1 is preferred, and 2 if output_2
            is preferred.
        """
        if keys_to_merge is None:
            keys_to_merge = self.input_keys

        keys_to_merge = list(keys_to_merge)

        outputs_1 = utils.convert_to_dataframe(outputs_1)
        outputs_2 = utils.convert_to_dataframe(outputs_2)

        if is_ordered:
            outputs_1 = outputs_1.copy()
            outputs_2 = outputs_2.copy()
            outputs_1["tmp_idx"] = range(len(outputs_1))
            outputs_2["tmp_idx"] = range(len(outputs_1))
            keys_to_merge += ["tmp_idx"]  # add a temporary index to merge on

        # find all the columns that are in both
        other_same_cols = [
            k
            for k in outputs_1.columns
            if k in outputs_2 and k not in (keys_to_merge + ["output"])
        ]

        df_to_annotate = pd.merge(
            outputs_1,
            outputs_2,
            on=keys_to_merge,
            suffixes=("_1", "_2"),
        )

        for c in other_same_cols:
            # if the columns are the same, we can drop the _2
            if df_to_annotate[c + "_1"].equals(df_to_annotate[c + "_2"]):
                df_to_annotate = df_to_annotate.drop(columns=c + "_2").rename(
                    columns={c + "_1": c}
                )

        if is_ordered:
            df_to_annotate = df_to_annotate.drop(columns="tmp_idx")
        else:
            # if you are taking the cartesian product, you can have undesired duplicates
            df_to_annotate = df_to_annotate.drop_duplicates()

            if not (len(outputs_1) == len(outputs_2) == len(df_to_annotate)):
                logging.warning(
                    f"""The length of outputs before and after merge are not the same. We have len(outputs_1)==
                    {len(outputs_1)}, len(outputs_2)=={len(outputs_2)}, and len(df_annotated)=={len(df_to_annotate)}. 
                    This means that there are missing examples or duplicates. We are taking a SQL inner join.
                    """
                )

        out = self.annotate_pairs(df_to_annotate, **decoding_kwargs)

        return out

    def annotate_pairs(
        self,
        to_annotate: Union[Sequence[dict[str, Any]], pd.DataFrame],
        **decoding_kwargs,
    ) -> list[dict[str, Any]]:
        """Annotates the given examples, which contain both `"output_1"` and `"output_2"` keys.

        Parameters
        ----------
        to_annotate : list of dict or dataframe
            Examples to annotate. Each dictionary (or row) should contain all of `self.input_output_keys`.

        **decoding_kwargs :
            Additional arguments to pass to `fn_completions`.

        Returns
        -------
        annotated : list of dict
            The annotated examples. Each dictionary will contain all of `self.input_output_keys` and `"preference"`.
            Preference will be 0 if output_1 == output_2, 1 if output_1 is preferred, and 2 if output_2 is preferred.
        """
        if len(to_annotate) == 0:
            return []

        df_to_annotate = self._preprocess(to_annotate)

        df_annotated = self._annotate(df_to_annotate, **decoding_kwargs)
        annotated = self._postprocess_and_store_(df_annotated, to_annotate)
        return annotated

    def set_noise(self, p_label_flip: float):
        """Set the noise level for the annotators.

        Parameters
        ----------
        p_label_flip : float, optional
            Probability of flipping the label (ie adds noise by taking a mixture between predicted label and
            2*p_label_flip of independent coin flip). If None, will not flip the label. In AlpacaFarm we use 0.25
            for training.
        """
        self.p_label_flip = p_label_flip

    def _preprocess(self, to_annotate: utils.AnyData) -> pd.DataFrame:
        """Preprocess the examples to annotate. In particular takes care of filtering unnecessary examples."""

        df_to_annotate = utils.convert_to_dataframe(to_annotate).copy()

        for c in self.other_keys_to_keep + ["preference"]:
            if c in df_to_annotate.columns:
                logging.warning(
                    f"""{c} column is already in the dataframe. We will overwrite it."""
                )
                df_to_annotate[c] = np.nan

        # remove duplicates because you only need to annotate one of them
        df_to_annotate = df_to_annotate.drop_duplicates(subset=self.input_output_keys)

        # set the annotater for each example
        df_to_annotate["annotator"] = df_to_annotate.apply(
            lambda x: utils.random_seeded_choice(
                # we add "annotator" at the beginning to not use the same seed for all tasks
                seed="annotator" + x["instruction"] + str(self.seed),
                choices=list(self.annotators.keys()),
            ),
            axis=1,
        )

        if self.is_avoid_reannotations:
            # merge the old annotations
            df_to_annotate = self._merge_annotations(
                df_to_annotate, self.df_annotations
            )

        # adds random noise => avoids annotating examples that will be noised out.
        if self.p_label_flip:
            logging.info(
                f"Adding random noise to the labels p_label_flip={self.p_label_flip}."
            )
            # if you have 25% change of flipping the label, you have 50% chance of selecting random label
            p_noise = self.p_label_flip * 2
            noisy_preference = df_to_annotate.apply(
                # we add "noisy_label" at the beginning to use ~independent seeds between tasks
                lambda x: utils.random_seeded_choice(  # seed on inputs for reproducibility
                    seed="noisy_preference" + x["instruction"] + str(self.seed),
                    choices=[np.nan, 1, 2],
                    weights=[1 - p_noise, self.p_label_flip, self.p_label_flip],
                ),
                axis=1,
            )
            df_to_annotate["is_noisy_label"] = ~noisy_preference.isna()
            # keeps previously annotated examples when you did not add noise
            df_to_annotate["preference"] = np.where(
                df_to_annotate["is_noisy_label"],
                noisy_preference,
                df_to_annotate["preference"],
            )

        idcs_is_same_outputs = df_to_annotate["output_1"] == df_to_annotate["output_2"]
        df_to_annotate.loc[idcs_is_same_outputs, "preference"] = 0

        return df_to_annotate

    def _initialize_annotators(
        self, annotators_config: Union[utils.AnyPath, dict[str, dict[str, Any]]]
    ) -> dict[str, Callable]:
        """Load all the configs and prompts if necessary."""
        annotators_config = utils.load_configs(annotators_config)
        return {
            name: self.SingleAnnotator(
                seed=self.seed, base_dir=self.base_dir, **annotator_config
            )
            for name, annotator_config in annotators_config.items()
        }

    def _annotate(
        self, df_to_annotate: pd.DataFrame, **decoding_kwargs
    ) -> pd.DataFrame:
        """Annotate the examples."""

        df_annotated = df_to_annotate
        for annotator in self.annotators.keys():
            # only annotate examples that have not been annotated yet
            curr_idcs = (df_annotated["annotator"] == annotator) & df_annotated[
                "preference"
            ].isna()

            logging.info(f"Annotating {curr_idcs.sum()} examples with {annotator}")

            # actual annotation
            curr_annotated = self.annotators[annotator](
                df_annotated.loc[curr_idcs, self.all_keys], **decoding_kwargs
            )

            df_annotated = self._merge_annotations(df_annotated, curr_annotated)

        return df_annotated

    def _postprocess_and_store_(
        self,
        df_annotated: pd.DataFrame,
        to_annotate: Union[Sequence[dict[str, Any]], pd.DataFrame],
    ) -> list[dict[str, Any]]:
        """Convert the dataframe into a list of dictionaries to be returned, and store current anntations."""

        df_to_annotate = utils.convert_to_dataframe(to_annotate)

        # select available annotations
        if self.is_store_missing_preferences:
            df_annotated["preference"] = df_annotated["preference"].fillna(-1)
        else:
            df_annotated["preference"] = df_annotated["preference"].replace(-1, np.nan)

        df_annotated = df_annotated[~df_annotated["preference"].isna()].copy()

        # try converting to int now that no nan
        df_annotated["preference"] = pd.to_numeric(
            df_annotated["preference"], downcast="integer", errors="ignore"
        )

        if "is_noisy_label" in df_annotated.columns:
            # dont' store noisy labels
            df_annotated_to_store = df_annotated.query("is_noisy_label == False").drop(
                columns=["is_noisy_label"]
            )
            df_annotated = df_annotated.drop(columns=["is_noisy_label"])
        else:
            df_annotated_to_store = df_annotated

        other_keys_to_keep = [
            c for c in self.other_keys_to_keep if c in df_annotated_to_store.columns
        ]
        all_keys_to_keep = self.all_keys + ["preference"] + other_keys_to_keep
        df_annotated_to_store = df_annotated_to_store[all_keys_to_keep]

        if self.df_annotations is None:
            df_annotations = df_annotated_to_store
        else:
            df_annotations = pd.concat(
                [self.df_annotations, df_annotated_to_store], axis=0, ignore_index=True
            )

        self.df_annotations = df_annotations.drop_duplicates(
            subset=self.all_keys, keep="last"
        )

        self.save()

        if self.is_store_missing_preferences:
            # put back np.nan
            df_annotated["preference"] = df_annotated["preference"].replace(-1, np.nan)

        # need to merge with df_to_annotate in case you dropped duplicates
        on = list(self.input_keys + self.output_keys)
        df_annotated = df_annotated[all_keys_to_keep]
        df_to_annotate = df_to_annotate[
            [
                c
                for c in df_to_annotate.columns
                if c not in df_annotated.columns or c in on
            ]
        ]
        # need to remove all other columns before merging if not you will
        df_annotated = df_to_annotate.merge(df_annotated, on=on, how="outer")

        annotated = df_annotated.to_dict(orient="records")

        return annotated

    def save(self, path: Optional[utils.AnyPath] = None):
        """Save the annotations to json."""
        path = path or self.caching_path
        if path is not None:
            logging.info(f"Saving all annotations to {path}.")
            # to make sure that we don't overwrite the annotations we load again from file (ideally would use a DB)
            self._refresh_annotations_()
            if not self.is_store_missing_preferences:
                self.df_annotations = self.df_annotations[
                    ~self.df_annotations["preference"].isna()
                ]
            self.df_annotations.to_json(path, orient="records", indent=2)

    def _refresh_annotations_(self):
        """Refresh the annotations in memory."""
        curr_df_annotations = self.df_annotations.copy()
        self.load_()
        self.df_annotations = pd.concat(
            [self.df_annotations, curr_df_annotations], axis=0, ignore_index=True
        ).drop_duplicates(subset=self.all_keys, keep="last")

    def load_(self, path: Optional[utils.AnyPath] = None):
        """Load all the annotations from json."""
        path = path or self.caching_path
        if path is not None:
            path = Path(path)
            if path.exists():
                logging.info(f"Loading all annotations from {path}.")
                self.df_annotations = pd.read_json(path)

    def _merge_annotations(
        self, df_to_annotate: pd.DataFrame, df_partially_annotated: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge (partial) annotations with the original df to keep the same order and avoid duplicates annotations."""
        if df_partially_annotated is None or df_partially_annotated.empty:
            return df_to_annotate

        other_keys_to_keep = [
            c for c in self.other_keys_to_keep if c in df_partially_annotated.columns
        ]

        df_to_annotate = df_to_annotate.merge(
            df_partially_annotated[self.all_keys + ["preference"] + other_keys_to_keep],
            on=self.all_keys,
            how="left",
            suffixes=("_old", "_new"),
        )

        # if columns were in both dataframes, try to merge them
        for c in other_keys_to_keep + ["preference"]:
            if (
                f"{c}_old" in df_to_annotate.columns
                and f"{c}_new" in df_to_annotate.columns
            ):
                df_to_annotate[c] = df_to_annotate[c + "_old"].fillna(
                    df_to_annotate[c + "_new"]
                )
                df_to_annotate = df_to_annotate.drop(columns=[c + "_old", c + "_new"])

        return df_to_annotate


class SinglePairwiseAnnotator:
    """A helper class for a single auto annotators.

    Parameters
    ----------
    prompt_template : str or path
        A prompt template that will be given to `fn_prompter` or path to those prompts. Path is relative to
        `evaluators_configs/`

    fn_completion_parser : callable or str
        Function in `completion_parsers.py` to use for parsing the completions into preferences. For each completion,
        the number of preferences should be equal to the batch_size if not we set all the preferences in that batch to
        NaN.

    completion_parser_kwargs : dict
        Kwargs for fn_completion_parser.

    fn_completions : callable or str
        Function in `decoders.py` to use for decoding the output.

    completions_kwargs : dict
        kwargs for fn_completions. E.g. model_name, max_tokens, temperature, top_p, top_k, stop_seq.

    is_randomize_output_order : bool
        Whether to randomize output_1, output_2 when formatting.

    is_shuffle : bool
        Whether to shuffle the order of the examples before making the prompt. Useful if batch_size > 1.

    seed : int
        Seed for randomization.

    batch_size : int
        Number of examples that will be added in a single prompt.

    base_dir : Path, optional
        Path to the directory containing the annotators configs. I.e. annotators_config will be relative
        to this directory.
    """

    def __init__(
        self,
        prompt_template: utils.AnyPath,
        fn_completion_parser: Union[Callable, str] = "regex_parser",
        completion_parser_kwargs: Optional[dict[str, Any]] = None,
        fn_completions: Union[Callable, str] = "openai_completions",
        completions_kwargs: Optional[dict[str, Any]] = None,
        is_randomize_output_order: bool = True,
        is_shuffle: bool = True,
        seed: Optional[int] = 123,
        batch_size: int = 1,
        base_dir: utils.AnyPath = constants.EVALUATORS_CONFIG_DIR,
    ):
        self.base_dir = Path(base_dir)
        self.prompt_template = self._get_prompt_template(prompt_template)

        if isinstance(fn_completion_parser, str):
            fn_completion_parser = getattr(completion_parsers, fn_completion_parser)
        completion_parser_kwargs = completion_parser_kwargs or {}
        self.fn_completion_parser = partial(
            fn_completion_parser, **completion_parser_kwargs
        )

        self.is_randomize_output_order = is_randomize_output_order
        self.fn_completions = get_fn_completions(fn_completions)
        self.completions_kwargs = completions_kwargs or {}
        self.seed = seed
        self.is_shuffle = is_shuffle
        self.batch_size = batch_size

    def _get_prompt_template(self, prompt_template: utils.AnyPath):
        return utils.read_or_return(self.base_dir / prompt_template)

    def make_prompts(
        self, df_to_annotate: pd.DataFrame, prompt_template: Optional[str] = None
    ) -> tuple[list[str], pd.DataFrame]:
        """Make all the prompts for the given examples.

        Parameters
        ----------
        df_to_annotate : pd.DataFrame
            Examples to annotate

        prompt_template : str
            Template to use for the prompt. If None, use the one from the constructor.

        Returns
        -------
        prompts : list[str]
            Formatted prompts for the given examples.

        df_to_annotate : pd.DataFrame
            Examples to annotate in the same order as prompts.
        """
        if prompt_template is None:
            prompt_template = self.prompt_template
        return utils.make_prompts(
            df=df_to_annotate, template=prompt_template, batch_size=self.batch_size
        )

    def __call__(self, df_to_annotate: pd.DataFrame, **decoding_kwargs) -> pd.DataFrame:
        """Annotates the given examples.

        Parameters
        ----------
        df_to_annotate : pd.DataFrame
            Examples to annotate

        decoding_kwargs :
            Additional arguments to pass to `fn_completions`.
        """
        df_to_annotate = df_to_annotate.copy()  # avoid in place modifications

        if df_to_annotate.empty:
            df_to_annotate["preference"] = []
            return df_to_annotate

        df_to_annotate = self.preprocess(df_to_annotate)

        # prompts and completions here will not be the same length as the dataframe due to batching
        prompts, df_to_annotate = self.make_prompts(df_to_annotate)

        completions = self.fn_completions(
            prompts=prompts, **self.completions_kwargs, **decoding_kwargs
        )

        df_to_annotate["preference"] = self.parse_completions(
            completions=completions["completions"]
        )
        for k, v in completions.items():
            if k != "completions":
                if len(df_to_annotate["preference"]) == len(v) * self.batch_size:
                    v = [el for el in v for _ in range(self.batch_size)]
                df_to_annotate[k] = v
                if "per_example" in k:
                    df_to_annotate[k] = df_to_annotate[k] / self.batch_size

        df_annotated = self.postprocess(df_to_annotate)

        return df_annotated

    def preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the examples before annotating. In particular, takes care of all the randomization."""

        if self.is_randomize_output_order:
            # randomize order of output_1, output_2 base on inputs
            df_to_annotate["is_switched_outputs"] = df_to_annotate.apply(
                # we add "is_switched_outputs" at the beginning to not use the same seed for all tasks
                lambda x: utils.random_seeded_choice(
                    seed="is_switched_outputs" + x["instruction"] + str(self.seed),
                    choices=[False, True],
                ),
                axis=1,
            )
            df_to_annotate = utils.shuffle_pairwise_preferences(
                df_to_annotate, df_to_annotate["is_switched_outputs"]
            )

        if self.is_shuffle:
            df_to_annotate = df_to_annotate.sample(frac=1, random_state=self.seed)

        return df_to_annotate

    def parse_completions(self, completions: list[str]) -> list[int]:
        """Converts the completions into annotations."""
        all_preferences = []
        for completion in completions:
            # use a regex to match all outputs on a line. Assumes that there is at most one output to match per line
            batch_preferences = self.fn_completion_parser(completion)
            if len(batch_preferences) != self.batch_size:
                logging.warning(
                    f"Found {len(batch_preferences)} preferences in:'''\n{completion}\n''' but expected"
                    f" {self.batch_size}. We are setting all preferences to np.nan."
                )
                batch_preferences = [np.nan] * self.batch_size
            all_preferences += batch_preferences
        return all_preferences

    def postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """Postprocess the annotated examples."""

        # remove padding examples when using batch_size > 1
        df_annotated = df_annotated.query("is_padding == False").drop(
            columns=["is_padding"]
        )

        arr_is_na = df_annotated["preference"].isna()
        if arr_is_na.any():
            logging.warning(
                f"{arr_is_na.sum().item()} samples had no auto annotation. We are filtering them for now. "
                f"If you are using chain of thought it might be that max_tokens limit is too low. "
            )
            df_annotated = df_annotated[~arr_is_na]

        assert set(df_annotated["preference"].unique().tolist()) <= {0, 1, 2}

        if self.is_randomize_output_order:
            # unshuffles output 1 and output 2. For binary preference, unshuffling is equivalent to reshuffling
            df_annotated = utils.shuffle_pairwise_preferences(
                df_annotated, df_annotated["is_switched_outputs"]
            )
            df_annotated = df_annotated.drop(columns=["is_switched_outputs"])

        return df_annotated
