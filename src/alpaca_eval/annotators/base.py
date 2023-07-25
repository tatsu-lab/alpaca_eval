import abc
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd

from .. import completion_parsers, constants, processors, utils
from ..decoders import get_fn_completions

CURRENT_DIR = Path(__file__).parent
logging.getLogger().setLevel(logging.INFO)

__all__ = ["BaseAnnotator", "BaseAnnotatorJSON", "SingleAnnotator"]


class BaseAnnotator(abc.ABC):
    """Base class for a pool of annotators.

    Parameters
    ----------
    annotators_config : Path or list of dict, optional
        A dictionary or path to a yaml file containing the configuration for the pool of annotators. If a directory,
        we search for 'configs.yaml' in it. The keys in the first  dictionary should be the annotator's name, and
        the value should be a dictionary of the annotator's configuration which should have the following keys:
        The path is relative to `base_dir` directory.
        - prompt_template (str): a prompt template or path to it. The template should contain placeholders for keys in
            the example dictionary, typically {instruction} and {output_1} {output_2}.
        - fn_completions (str): function in `alpaca_farm.decoders` for completions. Needs to accept as first argument
            `prompts` which is a list of string.
        - completions_kwargs (dict): kwargs for fn_completions. E.g. model_name, max_tokens, temperature,
        tokens_to_avoid
        - fn_completion_parser (str) : Function in `completion_parsers.py` to use for parsing the completions into
        annotations.
        - completion_parser_kwargs (dict) : Kwargs for fn_completion_parser.
        - other kwargs to `SingleAnnotator` such as batch_size

    seed : int, optional
        Seed for the random number generator.

    is_avoid_reannotations : bool, optional
        Whether to avoid re-annotating examples that have already been annotated by the annotator. This will decrease
        cost but can be slightly slower if there are no annotations that can be reused.

    primary_keys : sequence of str, optional
        Keys use to distinguish the example.

    other_keys_to_keep : sequence of str, optional
        Other columns to store besides the annotations.

    is_store_missing_annotations : bool, optional
        Whether to store missing annotations. If True it avoids trying to reannotate examples that have errors.

    base_dir : Path, optional
        Path to the directory containing the annotators configs. I.e. annotators_config will be relative
        to this directory. If None uses self.DEFAULT_BASE_DIR

    is_raise_if_missing_primary_keys : bool, optional
        Whether to ensure that the primary keys are in the example dictionary. If True, raises an error.

    tmp_missing_annototation : Any, optional
        Temporary value to use for missing annotations when `is_store_missing_annotations` is True.

    annotation_type : type, optional
        Type to use for storing the annotations. If None, uses `self.DEFAULT_ANNOTATION_TYPE`.
    """

    DEFAULT_BASE_DIR = constants.EVALUATORS_CONFIG_DIR
    annotator_column = "annotator"
    TMP_MISSING_ANNOTATION = -1
    DEFAULT_ANNOTATION_TYPE = int

    def __init__(
        self,
        primary_keys: Sequence[str],
        annotators_config: Union[utils.AnyPath, list[dict[str, Any]]] = "claude",
        seed: Optional[int] = 0,
        is_avoid_reannotations: bool = True,
        other_keys_to_keep: Sequence[str] = (
            "price_per_example",
            "time_per_example",
            "raw_completion",
        ),
        is_store_missing_annotations: bool = True,
        base_dir: Optional[utils.AnyPath] = None,
        is_raise_if_missing_primary_keys: bool = True,
        annotation_type: Optional[Type] = None,
    ):
        logging.info(f"Creating the annotator from `{annotators_config}`.")
        self.base_dir = Path(base_dir or self.DEFAULT_BASE_DIR)
        self.seed = seed
        self.is_avoid_reannotations = is_avoid_reannotations
        self.primary_keys = list(primary_keys)
        self.all_keys = self.primary_keys + [self.annotator_column]
        self.other_keys_to_keep = list(other_keys_to_keep)
        self.is_store_missing_annotations = is_store_missing_annotations
        self.is_raise_if_missing_primary_keys = is_raise_if_missing_primary_keys
        self.annotation_type = annotation_type or self.DEFAULT_ANNOTATION_TYPE

        self.annotators_config = self._initialize_annotators_config(annotators_config)
        self.annotators = self._initialize_annotators()
        self.df_annotations = None

    ### Abstract methods ###

    @property
    @abc.abstractmethod
    def SingleAnnotator(self) -> Type["SingleAnnotator"]:
        """Class to use for each single annotator."""
        pass

    #######################
    @property
    def available_fields_to_format(self):
        """Fields that can be formatted in the prompt template."""
        return self.all_keys

    @property
    def annotation_key(self) -> str:
        """How to refer to the annotations, this will be the key for annotations in the output."""
        return "annotation"

    @property
    def random_seed_key(self) -> list[str]:
        """What key / column to seed on for the random generator."""
        return list(self.primary_keys)

    ### Public methods ###
    @property
    def annotator_name(self) -> str:
        return Path(self.annotators_config).parent.name

    def __call__(
        self,
        to_annotate: utils.AnyData,
        **decoding_kwargs,
    ) -> list[dict[str, Any]]:
        """Main function for annotating.

        Parameters
        ----------
        to_annotate : list of dict or dataframe
            Examples to annotate. Each dictionary (or row) should contain all of `self.primary_keys`.

        **decoding_kwargs :
            Additional arguments to pass to `fn_completions`.

        Returns
        -------
        annotated : list of dict
            The annotated examples. Each dict will contain all of `self.primary_keys` and `self.annotation_key`.
        """
        if len(to_annotate) == 0:
            return []

        df_to_annotate = self._preprocess(to_annotate)
        df_annotated = self._annotate(df_to_annotate, **decoding_kwargs)
        annotated = self._postprocess_and_store_(df_annotated, to_annotate)
        return annotated

    #######################

    ### Private methods ###
    def _initialize_annotators_config(self, annotators_config):
        # setting it relative to the config directory
        annotators_config = self.base_dir / annotators_config

        if annotators_config.is_dir():
            annotators_config = annotators_config / "configs.yaml"

        return annotators_config

    def _initialize_annotators(self) -> dict[str, "SingleAnnotator"]:
        """Load all the configs and prompts if necessary."""
        annotators_config = utils.load_configs(self.annotators_config)
        try:
            # in case a path is given we make it relative to that path
            base_dir = self.annotators_config.parents[1]
        except:
            base_dir = self.base_dir

        return {
            name: self.SingleAnnotator(
                seed=self.seed,
                base_dir=base_dir,
                annotation_column=self.annotation_key,
                **annotator_config,
            )
            for name, annotator_config in annotators_config.items()
        }

    def _add_missing_primary_keys_(self, df: pd.DataFrame):
        missing_primary_keys = [c for c in self.primary_keys if c not in df.columns]
        if self.is_raise_if_missing_primary_keys:
            if len(missing_primary_keys) > 0:
                raise ValueError(f"Missing primary keys: {missing_primary_keys}")
        else:
            for c in missing_primary_keys:
                df[c] = None

    def _preprocess(self, to_annotate: utils.AnyData) -> pd.DataFrame:
        """Preprocess the examples to annotate. In particular takes care of filtering unnecessary examples."""

        df_to_annotate = utils.convert_to_dataframe(to_annotate)
        self._add_missing_primary_keys_(df_to_annotate)

        for c in self.other_keys_to_keep + [self.annotation_key]:
            if c in df_to_annotate.columns:
                logging.warning(f"{c} column is already in the dataframe. We will overwrite it.")
                df_to_annotate[c] = None

        # remove duplicates because you only need to annotate one of them
        df_to_annotate = df_to_annotate.drop_duplicates(subset=self.primary_keys)

        # set the annotater for each example
        df_to_annotate[self.annotator_column] = df_to_annotate.apply(
            lambda x: utils.random_seeded_choice(
                # we add "annotator" at the beginning to not use the same seed for all tasks
                seed="annotator" + "".join(x[self.random_seed_key]) + str(self.seed),
                choices=list(self.annotators.keys()),
            ),
            axis=1,
        )

        if self.is_avoid_reannotations:
            df_to_annotate = self._apply_cached_annotations(df_to_annotate)

        return df_to_annotate

    def _annotate(self, df_to_annotate: pd.DataFrame, **decoding_kwargs) -> pd.DataFrame:
        """Annotate the examples."""

        df_annotated = df_to_annotate
        for annotator in self.annotators.keys():
            # only annotate examples that have not been annotated yet
            curr_idcs = df_annotated[self.annotator_column] == annotator
            if self.annotation_key in df_annotated.columns:
                curr_idcs &= df_annotated[self.annotation_key].isna()

            logging.info(f"Annotating {curr_idcs.sum()} examples with {annotator}")

            # actual annotation
            curr_annotated = self.annotators[annotator](
                df_annotated.loc[curr_idcs, self.available_fields_to_format],
                **decoding_kwargs,
            )

            df_annotated = self._merge_annotations(df_annotated, curr_annotated)

        return df_annotated

    def _postprocess_and_store_(
        self,
        df_annotated: pd.DataFrame,
        to_annotate: utils.AnyData,
    ) -> list[dict[str, Any]]:
        """Convert the dataframe into a list of dictionaries to be returned, and store current anntations."""

        df_to_annotate = utils.convert_to_dataframe(to_annotate)
        self._add_missing_primary_keys_(df_to_annotate)

        # select available annotations
        if self.is_store_missing_annotations:
            df_annotated[self.annotation_key] = df_annotated[self.annotation_key].fillna(self.TMP_MISSING_ANNOTATION)
        else:
            df_annotated[self.annotation_key] = df_annotated[self.annotation_key].replace(
                self.TMP_MISSING_ANNOTATION, None
            )

        df_annotated = df_annotated[~df_annotated[self.annotation_key].isna()].copy()

        # try converting to int now that no nan. Note this will only do so if possible
        df_annotated[self.annotation_key] = df_annotated[self.annotation_key].astype(self.annotation_type)

        df_annotated = self._filter_annotations_before_storing(df_annotated)
        self._store_annotations_(df_annotated)

        if self.is_store_missing_annotations:
            # put back None
            df_annotated[self.annotation_key] = df_annotated[self.annotation_key].replace(
                self.TMP_MISSING_ANNOTATION, None
            )

        # need to merge with df_to_annotate in case you dropped duplicates
        on = list(self.primary_keys)
        # keeps columns from both df_to_annotate and df_annotated that are useful
        df_annotated = df_annotated[
            self._get_all_keys_to_keep(list(df_to_annotate.columns) + list(df_annotated.columns))
        ]
        df_to_annotate = df_to_annotate[[c for c in df_to_annotate.columns if c not in df_annotated.columns or c in on]]
        # need to remove all other columns before merging if not you will
        df_annotated = df_to_annotate.merge(df_annotated, on=on, how="outer")

        annotated = df_annotated.to_dict(orient="records")

        return annotated

    def _filter_annotations_before_storing(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """Filter annotations before storing them."""
        df_annotated = df_annotated[self._get_all_keys_to_keep(df_annotated.columns)]
        return df_annotated

    def _get_all_keys_to_keep(self, current_columns: Sequence) -> list[str]:
        other_keys_to_keep = [c for c in self.other_keys_to_keep if c in current_columns]
        all_keys_to_keep = self.all_keys + [self.annotation_key] + other_keys_to_keep
        return all_keys_to_keep

    def _apply_cached_annotations(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """annotate examples with cached annotations"""
        df_to_annotate = self._merge_annotations(df_to_annotate, self.df_annotations)
        return df_to_annotate

    def _store_annotations_(self, df_annotated: pd.DataFrame):
        """Store annotation in memory and on disk"""

        if self.df_annotations is None:
            df_annotations = df_annotated
        else:
            df_annotations = pd.concat([self.df_annotations, df_annotated], axis=0, ignore_index=True)

        self.df_annotations = df_annotations.drop_duplicates(subset=self.all_keys, keep="last")

    def _merge_annotations(self, df_to_annotate: pd.DataFrame, df_partially_annotated: pd.DataFrame) -> pd.DataFrame:
        """Merge (partial) annotations with the original df to keep the same order and avoid duplicates annotations."""

        if df_partially_annotated is None or df_partially_annotated.empty:
            return df_to_annotate

        other_keys_to_keep = [c for c in self.other_keys_to_keep if c in df_partially_annotated.columns]

        kwargs = dict(
            on=self.all_keys,
            how="left",
            suffixes=("_old", "_new"),
        )
        try:
            df_to_annotate = df_to_annotate.merge(
                df_partially_annotated[self.all_keys + [self.annotation_key] + other_keys_to_keep],
                **kwargs,
            )
        except ValueError:
            # can have merging issues if columns have different dtypes
            df_partially_annotated = df_partially_annotated.astype({k: str for k in self.all_keys})
            df_to_annotate = df_to_annotate.astype({k: str for k in self.all_keys}).merge(
                df_partially_annotated[self.all_keys + [self.annotation_key] + other_keys_to_keep],
                **kwargs,
            )

        # if columns were in both dataframes, try to merge them
        for c in other_keys_to_keep + [self.annotation_key]:
            if f"{c}_old" in df_to_annotate.columns and f"{c}_new" in df_to_annotate.columns:
                df_to_annotate[c] = df_to_annotate[c + "_old"].fillna(df_to_annotate[c + "_new"])
                df_to_annotate = df_to_annotate.drop(columns=[c + "_old", c + "_new"])

        return df_to_annotate

    #######################


class BaseAnnotatorJSON(BaseAnnotator):
    __doc__ = (
        BaseAnnotator.__doc__.replace(
            "Base class for a pool of annotators.",
            "Base class for a pool of annotators with caching to JSON file.",
        )
        + """
    caching_path : Path, optional
        Path to cache the annotations to. If None, will not save the annotations. If the path already exists it will
        load annotations from there.
    """
    )

    def __init__(self, *args, caching_path: Optional[utils.AnyPath] = "auto", **kwargs):
        super().__init__(*args, **kwargs)
        self.caching_path = self._initialize_cache(caching_path)

    def save(self, path: Optional[utils.AnyPath] = None):
        """Save all annotations to json."""

        path = path or self.caching_path
        if path is not None:
            logging.info(f"Saving all annotations to {path}.")
            # to make sure that we don't overwrite the annotations we load again from file (ideally would use a DB)
            self._refresh_annotations_()
            if not self.is_store_missing_annotations:
                self.df_annotations = self.df_annotations[~self.df_annotations[self.annotation_key].isna()]
            self.df_annotations.to_json(path, orient="records", indent=2)

    def load_(self, path: Optional[utils.AnyPath] = None):
        """Load all the annotations from json."""
        path = path or self.caching_path
        if path is not None:
            path = Path(path)
            if path.exists():
                logging.info(f"Loading all annotations from {path}.")
                self.df_annotations = pd.read_json(path, dtype={k: str for k in self.all_keys})

    def _initialize_cache(self, caching_path):
        if caching_path == "auto":
            if isinstance(self.annotators_config, (str, Path, os.PathLike)):
                stem = Path(self.annotators_config).stem
                caching_path = Path(self.annotators_config).parent / f"annotations_seed{self.seed}_{stem}.json"
                logging.info(f"Saving annotations to `{caching_path}`.")
            else:
                logging.warning("caching_path cannot be 'auto' if annotators_config is not a path. Setting to None.")
                caching_path = None
        elif caching_path is not None:
            logging.warning("Saving_path is given but not 'auto', make sure that it's different for different seeds.")
        self.load_(caching_path)
        return caching_path

    def _store_annotations_(self, df_annotated_to_store: pd.DataFrame):
        super()._store_annotations_(df_annotated_to_store)
        self.save()

    def _refresh_annotations_(self):
        """Refresh the annotations in memory."""
        curr_df_annotations = self.df_annotations.copy()
        self.load_()
        self.df_annotations = pd.concat(
            [self.df_annotations, curr_df_annotations], axis=0, ignore_index=True
        ).drop_duplicates(subset=self.all_keys, keep="last")


class SingleAnnotator:
    """A helper class for a single auto annotator.

    Parameters
    ----------
    prompt_template : str or path
        A prompt template that will be given to `fn_prompter` or path to those prompts. Path is relative to
        `evaluators_configs/`

    fn_completion_parser : callable or str
        Function that maps (parses) the completion to a list of annotations. If a string, it should be a function in
        `completion_parsers.py` to use for parsing the completions into annotations. For each completion, the number of
        annotations (lenght of list) should be equal to the batch_size if not we set all the annotations in that batch
        to NaN.

    completion_parser_kwargs : dict
        Kwargs for fn_completion_parser.

    fn_completions : callable or str
        Function in `decoders.py` to use for decoding the output.

    completions_kwargs : dict
        kwargs for fn_completions. E.g. model_name, max_tokens, temperature, top_p, top_k, stop_seq.

    is_shuffle : bool
        Whether to shuffle the order of the examples before making the prompt. Useful if batch_size > 1.

    seed : int
        Seed for randomization.

    batch_size : int
        Number of examples that will be added in a single prompt.

    base_dir : Path, optional
        Path to the directory containing the annotators configs. I.e. annotators_config will be relative
        to this directory.

    annotation_column : str, optional
        Name of the annotation column in the output dataframe.

    is_store_raw_completions : bool, optional
        Whether to store raw completions at `"raw_completion"` column in the output dataframe.

    processors_to_kwargs : Sequence[dict(str, dict)], optional
        A dictionary of BaseProcessor objects to apply for preprocessing the  dataframe before making the prompts and
        prostprocessing after anntoations. The key should be the names of the BaseProcessor objectsto use in
        `processors.py` the values are the kwargs for the constructor of the Processor. Order matters.

    is_add_default_processors : bool, optional
        Whether to add the default processors to the list of processors.
    """

    def __init__(
        self,
        prompt_template: utils.AnyPath,
        fn_completion_parser: Optional[Union[Callable, str]] = "regex_parser",
        completion_parser_kwargs: Optional[dict[str, Any]] = None,
        fn_completions: Union[Callable, str] = "openai_completions",
        completions_kwargs: Optional[dict[str, Any]] = None,
        is_shuffle: bool = True,
        seed: Optional[int] = 123,
        batch_size: int = 1,
        base_dir: utils.AnyPath = constants.EVALUATORS_CONFIG_DIR,
        annotation_column: str = "annotation",
        is_store_raw_completions: bool = False,
        processors_to_kwargs: Optional[dict[str, dict]] = None,
        is_add_default_processors: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.prompt_template = self._get_prompt_template(prompt_template)

        if fn_completion_parser is None:
            fn_completion_parser = lambda x: [x]
        elif isinstance(fn_completion_parser, str):
            fn_completion_parser = self._search_fn_completion_parser(fn_completion_parser)
        completion_parser_kwargs = completion_parser_kwargs or {}
        self.fn_completion_parser = partial(fn_completion_parser, **completion_parser_kwargs)

        self.fn_completions = get_fn_completions(fn_completions)
        self.completions_kwargs = completions_kwargs or {}
        self.seed = seed
        self.is_shuffle = is_shuffle
        self.batch_size = batch_size
        self.annotation_column = annotation_column
        self.completion_column = "raw_completion" if is_store_raw_completions else None

        self.is_add_default_processors = is_add_default_processors
        self.processors = []
        processors_to_kwargs = processors_to_kwargs or {}
        if batch_size > 1 and self.is_add_default_processors:
            processors_to_kwargs["PaddingForBatchesProcessor"] = {
                "batch_size": batch_size,
                "padding_example": DUMMY_EXAMPLE,
            }
        for processor, processor_kwargs in processors_to_kwargs.items():
            processor_kwargs["seed"] = self.seed
            Processor = self._search_processor(processor)
            self.processors += [Processor(**processor_kwargs)]

    ### Public methods ###
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
            df_to_annotate[self.annotation_column] = []
            return df_to_annotate

        df_to_annotate = self._preprocess(df_to_annotate)

        # prompts and completions here will not be the same length as the dataframe due to batching
        prompts, df_to_annotate = self._make_prompts(df_to_annotate)

        completions = self.fn_completions(prompts=prompts, **self.completions_kwargs, **decoding_kwargs)

        annotations_to_save, completions_to_save = self._parse_completions(completions=completions["completions"])
        df_to_annotate[self.annotation_column] = annotations_to_save
        if self.completion_column is not None:
            df_to_annotate[self.completion_column] = completions_to_save

        for k, v in completions.items():
            if k != "completions":
                if len(df_to_annotate[self.annotation_column]) == len(v) * self.batch_size:
                    v = [el for el in v for _ in range(self.batch_size)]
                df_to_annotate[k] = v
                if "per_example" in k:
                    df_to_annotate[k] = df_to_annotate[k] / self.batch_size

        df_annotated = self._postprocess(df_to_annotate)

        return df_annotated

    ######################

    ### Private methods ###
    def _search_fn_completion_parser(self, name: str) -> Callable:
        """Search for a completion parser by name."""
        return utils.get_module_attribute(completion_parsers, name)

    def _search_processor(self, name: str) -> Type["processors.BaseProcessor"]:
        """Search for a Processor class by name."""
        return utils.get_module_attribute(processors, name)

    def _get_prompt_template(self, prompt_template: utils.AnyPath):
        return utils.read_or_return(self.base_dir / prompt_template)

    def _make_prompts(
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
        return utils.make_prompts(df=df_to_annotate, template=prompt_template, batch_size=self.batch_size)

    def _preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the examples before annotating. In particular, takes care of all the randomization."""

        for processor in self.processors:
            df_to_annotate = processor.preprocess(df_to_annotate)

        if self.is_shuffle:
            df_to_annotate = df_to_annotate.sample(frac=1, random_state=self.seed)

        return df_to_annotate

    def _parse_completions(self, completions: list[str]) -> tuple[list[Any], list[Any]]:
        """Converts the completions into annotations."""
        all_annotations = []
        all_completions = []
        for completion in completions:
            try:
                batch_annotations = self.fn_completion_parser(completion)
                batch_annotations = list(batch_annotations)

                if len(batch_annotations) != self.batch_size:
                    logging.warning(
                        f"Found {len(batch_annotations)} annotations in:'''\n{completion}\n''' but expected"
                        f" {self.batch_size}. We are setting all annotations to None."
                    )
                    batch_annotations = [None] * self.batch_size

            except Exception as e:
                logging.exception(f"Error while parsing completion: '''\n{completion}\n'''")
                batch_annotations = [None] * self.batch_size

            all_annotations += batch_annotations
            all_completions += [completion] * self.batch_size
        return all_annotations, all_completions

    def _postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """Postprocess the annotated examples."""

        arr_is_na = df_annotated[self.annotation_column].isna()
        if arr_is_na.any():
            logging.warning(
                f"{arr_is_na.sum().item()} samples had no auto annotation. We are filtering them for now. "
                f"If you are using chain of thought it might be that max_tokens limit is too low. "
            )
            df_annotated = df_annotated[~arr_is_na]

        for processor in self.processors[::-1]:  # postprocess in reverted order => no interactions between processors
            df_annotated = processor.postprocess(df_annotated)

        return df_annotated

    #######################
