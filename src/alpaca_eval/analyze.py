"""
Main module for analyzing an evaluation benchmark (annotator and data).
"""
import logging
from itertools import combinations
from typing import Callable, Optional, Union

import datasets
import pandas as pd

from . import constants, utils
from .types import AnyPath, AnyData


def DEFAULT_GOLD_CROSSANNOTATIONS():
    df = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_human_crossannotations",
        cache_dir=constants.DEFAULT_CACHE_DIR,
        use_auth_token=constants.DATASETS_TOKEN,
    )["validation"].to_pandas()

    # turkers took around 9 min for 15 examples in AlpacaFarm
    df["time_per_example"] = 9.2 * 60 / 15
    df["price_per_example"] = 0.3  # price we paid for each example
    return df


def DEFAULT_GOLD_ANNOTATIONS():
    df = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_human_annotations",
        cache_dir=constants.DEFAULT_CACHE_DIR,
        use_auth_token=constants.DATASETS_TOKEN,
    )["validation"].to_pandas()

    # turkers took around 9 min for 15 examples in AlpacaFarm
    df["time_per_example"] = 9.2 * 60 / 15
    df["price_per_example"] = 0.3  # price we paid for each example
    return df


class Analyzer:
    """Helper class to compare and understand annotations from different annotators.

    Parameters
    ----------
    gold_crossannotations : path or data or callable
        The cross annotations from the gold annotators. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
        (or row of dataframe) should contain all of `keys` and `preference` keys.

    gold_annotations : path or data or callable, optional
        The annotations from the gold annotators. Same format as `gold_crossannotations`. If None we use the first
        annotation from `gold_crossannotations`.

    keys : tuple
        Keys to use to compare the annotations.

    n_annotators : int
        Minimum number of annotators for treating as gold annotation.

    annotator_kwargs : dict
        Arguments that will be passed to all annotators being analyzed.
    """

    def __init__(
            self,
            gold_crossannotations: Union[AnyPath, AnyData, Callable] = DEFAULT_GOLD_CROSSANNOTATIONS,
            gold_annotations: Optional[Union[AnyPath, AnyData, Callable]] = None,
            keys=("instruction", "input", "output_1", "output_2"),
            n_annotators: Optional[int] = 4,
            seed: Optional[int] = 0,
            **annotator_kwargs
    ):
        self.keys = list(keys)
        self.n_annotators = n_annotators
        self.annotator_kwargs = annotator_kwargs

        df_gold_crossannotations = utils.load_or_convert_to_dataframe(gold_crossannotations)
        # adding a random index to differentiate between the n_annotators
        self.df_gold_crossannotations = self._select_n_annotations(df_gold_crossannotations,
                                                                   n_annotators=self.n_annotators)

        if gold_annotations is None:
            self.df_gold_annotations = self.df_gold_crossannotations.query("annotator_index == 0")
        else:
            self.df_gold_annotations = utils.load_or_convert_to_dataframe(gold_annotations)

        self.all_df_annotations = dict()
        self.seed = seed

    def _select_n_annotations(self, df, n_annotators=None, is_rm_less_than: bool = True):
        """Gets examples with at least n annotations. Adds `index` and `n_annotated` columns."""
        if "n_annotated" in df.columns:
            df = df.drop(columns="n_annotated")

        df["index"] = df.groupby(self.keys)["preference"].cumcount()

        if is_rm_less_than:
            # remove samples that have more than n_annotators
            df = df[df["index"] < n_annotators]

        # select examples that have at least n_annotators
        counts = df.groupby(self.keys)["preference"].count()
        counts.name = "n_annotated"
        n_annotators = n_annotators or df["counts"].min()
        counts = counts[counts >= n_annotators].reset_index()
        df_selected = df.merge(counts, on=self.keys)

        return df_selected.copy()

    def _get_annotations(self, annotations: Union[pd.DataFrame, str]):
        if isinstance(annotations, str):
            if annotations == "gold_crossannotations":
                annotations = self.df_gold_crossannotations
            elif annotations == "gold_annotations":
                annotations = self.df_gold_annotations
            else:
                raise ValueError(f"Unknown annotations: {annotations}")
        return annotations

    def _get_mode(self, annotations, idcs):
        annotations = annotations[annotations["index"].isin(idcs)]
        return annotations.groupby(self.keys)["preference"].aggregate(_random_mode)

    def _agreement_of_single_annotations(
            self,
            df_annotations_1: pd.DataFrame,
            df_annotations_2: pd.DataFrame,
    ):
        out = pd.merge(df_annotations_1, df_annotations_2, on=self.keys, suffixes=("_1", "_2"))
        out["match"] = (out["preference_1"] == out["preference_2"]).astype(int)
        return pd.Series(dict(accuracy=out["match"].mean(), sem_samples=out["match"].sem(), counts=len(out["match"])))

    def agreement_of_annotations(
            self,
            annotations_1: Union[pd.DataFrame, str],
            annotations_2: Optional[Union[pd.DataFrame, str]],
            n_majority_vote_1: Optional[int] = 1,
            n_majority_vote_2: Optional[int] = 1,
            is_same_annotator: Optional[bool] = None,
    ) -> pd.Series:
        """Compare (cross)annotations from two annotators.

        Notes
        -----
        - if you want to compute the agreement of 1 annotation vs the rest (eg to estimate the variance) then use
        n_majority_vote_1=1 and n_majority_vote_2=None and annotations_2=None.
        - if you want to measure the agreement of N annotators between two different annotators (eg to estimate the bias
        use n_majority_vote_1=N and n_majority_vote_2=N.

        Parameters
        ----------
        annotations_1 : pd.DataFrame or "gold_crossannotations" or "gold_annotations"
            First annotations. If "gold_crossannotations" or "gold_annotations" we use the corresponding gold
            annotations. If there are more than one annotation per example (ie index > 0) then we either use majority
            vote (if n_majority_vote_1 == n_annotators) or take an expectation over possible annotators.

        annotations_2 : pd.DataFrame or "gold_crossannotations" or "gold_annotations"
            First annotations. If "gold_crossannotations" or "gold_annotations" we use the corresponding gold
            annotations. If None we use the same as `annotations_1`. If there are more than one annotation per example
            (ie index > 0) then we either use majority vote (if n_majority_vote_1 == n_annotators) or take an
            expectation over possible annotators.

        n_majority_vote_1 : int, optional
            If more than 1 we will use the majority vote of annotations_1. If None we use the maximum possible.
            It can only be None if both annotations_1 and annotations_2 are different.

        n_majority_vote_2 : int, optional
            If more than 1 we will use the majority vote of annotations_2. If None we use the maximum possible, which
            is all annotations if both annotations are the same, or the complement of annotations_1 if they are
            different.

        is_same_annotator : bool, optional
            Whether both annotations_1 and annotations_2 are the same or a subset of each other => you should not
            compare the same indices as this will bias the agreement. If None we will check if they are the same.

        Examples
        --------
        >>> analyzer = Analyzer(n_annotators=4)
        >>> df_crossannotations = analyzer.df_gold_crossannotations.head(8).copy()
        >>> df_crossannotations["preference"] = [1] * 4 + [2,2,2,1]
        >>> analyzer.agreement_of_annotations(df_crossannotations, annotations_2=None,
        >>>                                   n_majority_vote_1=1,  n_majority_vote_2=1)
        accuracy          0.750000
        sem_samples       0.250000
        counts            2.000000
        sem_annotators    0.075378
        dtype: float64
        >>> # accuracy above is 3/4 because for the first 3 comparison you get 2 * 100% and 1 * 50%. I.e. you get 50%
        >>> # when the second index is 3.  And for the last comparison the first index is always 3 so you get 3*50%
        >>> analyzer.agreement_of_annotations(df_crossannotations, annotations_2=None,
        >>>                                   n_majority_vote_1=1,  n_majority_vote_2=3)
        accuracy          0.875
        sem_samples       0.125
        counts            2.000
        sem_annotators    0.125
        dtype: float64
        >>> # above you are doing 4 comparison of 1 vs 3. As you are doing majority vote of 3 you get 100% for 3 out
        >>> # of 4 comparisons and 50% for the last one. So you get 3*100% + 1*50% = 87.5%
        """
        annotations_1 = self._get_annotations(annotations_1)

        if annotations_2 is None:
            annotations_2 = annotations_1

        annotations_2 = self._get_annotations(annotations_2)
        if is_same_annotator is None:
            is_same_annotator = annotations_2.equals(annotations_1)

        if is_same_annotator and n_majority_vote_1 is None:
            raise ValueError("n_majority_vote_1 cannot be None if annotations_1 and annotations_2 are the same")

        annotations_1 = self._select_n_annotations(annotations_1, n_annotators=n_majority_vote_1, is_rm_less_than=False)
        max_majority_vote_1 = annotations_1["n_annotated"].max()
        n_majority_vote_1 = n_majority_vote_1 or max_majority_vote_1

        if is_same_annotator:
            logging.info("You are comparing twice the same annotators.")
            # the maximum number of votes you should compare is the complement given that it's the same data
            n_majority_vote_2 = n_majority_vote_2 or (max_majority_vote_1 - n_majority_vote_1)
            assert (n_majority_vote_2 <= max_majority_vote_1) and (n_majority_vote_1 <= max_majority_vote_1)

        annotations_2 = self._select_n_annotations(annotations_2, n_annotators=n_majority_vote_2, is_rm_less_than=False)
        max_majority_vote_2 = annotations_2["n_annotated"].max()
        n_majority_vote_2 = n_majority_vote_2 or max_majority_vote_2

        results = dict()
        for idcs_1 in combinations(range(max_majority_vote_1), n_majority_vote_1):
            for idcs_2 in combinations(range(max_majority_vote_2), n_majority_vote_2):

                is_overlapping_idcs = len(set(idcs_1).intersection(idcs_2)) > 0
                if is_same_annotator:
                    if is_overlapping_idcs:
                        continue  # skipping overlapping indices because biased
                    elif (idcs_2, idcs_1) in results.keys():
                        # not skipping for unbiased but no need to compute twice
                        results[(idcs_1, idcs_2)] = results[(idcs_2, idcs_1)]
                        continue

                results[(idcs_1, idcs_2)] = self._agreement_of_single_annotations(
                    df_annotations_1=self._get_mode(annotations_1, idcs_1),
                    df_annotations_2=self._get_mode(annotations_2, idcs_2),
                )

        logging.info(f"n_majority_vote_1={n_majority_vote_1}, n_majority_vote_2={n_majority_vote_2}. "
                     f"Compared results of indices: {list(results.keys())}")

        sem_annotators = pd.DataFrame(results).T["accuracy"].sem()
        results = sum(results.values()) / len(results.values())
        results["sem_annotators"] = sem_annotators

        return results

    def auto_biases(
            self,
            annotator="gpt-4-0314_pairwise_vH_b5_chatml-prompt",
            annotator_kwargs=None,  # kwargs for AutoAnnotatorPairwiseDB
            is_annotate=True,  # False will be faster but doesn't ensure that annotated
    ):
        """Computes the auto biases"""
        annotator_kwargs = annotator_kwargs or {}
        df_auto = self.get_df_auto(annotator=annotator, is_annotate=is_annotate, **annotator_kwargs)

        expected = df_auto.mean() - 1

        if isinstance(df_auto, pd.Series):
            df_auto = df_auto.to_frame().reset_index(drop=False)

        df_auto = db_io.add_instruction_input_input_id(df=df_auto, database=self.auto_db)
        df_auto = db_io.add_output_output_id(df=df_auto, database=self.auto_db)
        df_spurious = ann_helpers_pairwise.analyze_spurious_pairwise_correlations_best_worst(
            df_auto,
            col_preference="preference",
        )
        spurious = df_spurious["delta_best_minus_worse"].to_dict()

        return pd.Series(dict(expected=expected, **self.get_probability_biases(df_auto), **spurious))

    def get_probability_biases(self, df_auto):
        df_auto = df_auto.query("preference.isin([1,2])").copy()
        df_auto["best_output"] = np.where(df_auto["preference"] == 1, df_auto.output_1, df_auto.output_2)
        df_auto["worse_output"] = np.where(df_auto["preference"] == 2, df_auto.output_1, df_auto.output_2)
        df_auto["is_best_list"] = df_auto["best_output"].apply(lambda x: bool(ann_helpers.is_listy_text(x)))
        df_auto["is_worse_list"] = df_auto["worse_output"].apply(lambda x: bool(ann_helpers.is_listy_text(x)))
        # Step 1: Create a new column indicating whether either `best_output` or `worse_output` has a list but not both
        df_auto["either_list"] = df_auto["is_best_list"] ^ df_auto["is_worse_list"]
        # Step 2: Count the number of times you prefer `best_output` when either `best_output` or `worse_output` has
        # a list but not both
        prefer_best_either_list = df_auto[(df_auto["either_list"]) & df_auto["is_best_list"]].shape[0]
        # Step 3: Count the total number of instances when either `best_output` or `worse_output` has a list but not
        # both
        total_either_list = df_auto[df_auto["either_list"]].shape[0]
        # Step 4: Calculate the probability
        probability_list = prefer_best_either_list / total_either_list

        # Step 1: Create new columns indicating the length of `best_output` and `worse_output`
        df_auto["best_output_length"] = df_auto["best_output"].apply(len)
        df_auto["worse_output_length"] = df_auto["worse_output"].apply(len)
        # Step 2: Create a new column indicating whether one output is longer than the other
        df_auto["one_is_longer"] = (df_auto["best_output_length"] - df_auto["worse_output_length"]).abs() > 30
        df_auto["is_prefer_longer"] = df_auto["best_output_length"] > df_auto["worse_output_length"]
        # Step 3: Count the number of times you prefer the longer output
        prefer_longer = df_auto[df_auto["one_is_longer"] & df_auto["is_prefer_longer"]].shape[0]
        # Step 4: Count the total number of instances when one output is longer than the other
        total_one_is_longer = df_auto[df_auto["one_is_longer"]].shape[0]
        # Step 5: Calculate the probability
        probability_length = prefer_longer / total_one_is_longer
        return dict(probability_list=probability_list, probability_length=probability_length)

    def turk_biases(self):
        """Computes the auto biases"""

        results = dict()
        for i in range(self.n_annotators):
            curr_turk = self.df_gold.query(f"index == {i}").set_index(self.interpretable_keys)["preference"]
            curr_gold = (
                self.df_gold.query(f"index != {i}")
                .groupby(self.interpretable_keys)["preference"]
                .aggregate(ann_helpers.random_mode)
            )
            for name, curr in dict(turk=curr_turk, turk_mode=curr_gold).items():
                expected = curr.mean() - 1

                curr = curr.to_frame().reset_index(drop=False)
                df_spurious = ann_helpers_pairwise.analyze_spurious_pairwise_correlations_best_worst(
                    curr,
                    col_preference="preference",
                )
                spurious = df_spurious["delta_best_minus_worse"]

                tmp_results = pd.Series(dict(expected=expected, **spurious, **self.get_probability_biases(curr)))

                if i == 0:
                    results[name] = tmp_results
                else:
                    results[name] += tmp_results

        return pd.DataFrame(results).T / self.n_annotators

    def intra_multi_annotator(
            self,
            annotator="multi",
            annotator_kwargs={},  # kwargs for AutoAnnotatorPairwiseDB
            is_annotate=True,  # False will be faster but doesn't ensure that annotated
    ):
        """Computes the interannotator agreement between the constituents of a multi annotator"""
        df_auto_bis = self.get_df_auto(
            annotator=annotator, is_annotate=is_annotate, delta_seed=1, is_return_all_columns=True, **annotator_kwargs
        )
        df_auto = self.get_df_auto(
            annotator=annotator, is_annotate=is_annotate, is_return_all_columns=True, **annotator_kwargs
        )
        df_gold = self.df_gold.groupby(self.keys)["preference"].aggregate(ann_helpers.random_mode)
        df_auto = db_io.add_instruction_input_input_id(df=df_auto, database=self.auto_db)
        df_auto = db_io.add_output_output_id(df=df_auto, database=self.auto_db)

        results = dict()
        for a in df_auto.annotator.unique():
            df_curr = df_auto.query(f"annotator == '{a}'").set_index(self.keys)
            df_curr_bis = df_auto_bis.set_index(self.keys).loc[df_curr.index]
            df_gold_bis = df_gold.loc[df_curr.index]
            inter_results = self.auto_vs_auto(df_curr_bis.reset_index(drop=False), df_curr.reset_index(drop=False))
            gold_results = self.auto_vs_auto(df_gold_bis.reset_index(drop=False), df_curr.reset_index(drop=False))
            results[a] = dict(inter=format_acc_sem(inter_results), gold=format_acc_sem(gold_results))

            spurious = ann_helpers_pairwise.analyze_spurious_pairwise_correlations_best_worst(
                df_auto.query(f"annotator == '{a}'").copy(),
                col_preference="preference",
            )["delta_best_minus_worse"].to_dict()
            results[a].update(spurious)

        return pd.DataFrame(results).T


def get_crossannotations(analyzer, Annotator, max_instances: Optional[int] = None, **kwargs):
    """Get cross annotations by `Annotator` corresponding to `analyzer.df_gold_crossannotations`."""
    n_crossannotations = analyzer.n_annotators
    all_annotations = []
    for seed in range(n_crossannotations):
        annotator = Annotator(seed=seed, **kwargs)
        df_gold_crossannotations = analyzer.df_gold_crossannotations.query(f"index == {seed}")
        if max_instances is not None:
            df_gold_crossannotations = df_gold_crossannotations.head(max_instances)
        annotations = annotator.annotate_pairs(df_gold_crossannotations)
        df_annotations = utils.load_or_convert_to_dataframe(annotations)
        df_annotations["index"] = seed
        all_annotations.append(df_annotations)
    df = pd.concat(all_annotations, axis=0)
    df["n_annotated"] = n_crossannotations
    return df


def get_annotations(analyzer, Annotator, max_instances: Optional[int] = None, **kwargs):
    """Get annotations by `Annotator` corresponding to `analyzer.df_gold_annotations`."""
    annotator = Annotator(**kwargs)
    df_gold_crossannotations = analyzer.df_gold_annotations.query(f"index == 0")
    if max_instances is not None:
        df_gold_crossannotations = df_gold_crossannotations.head(max_instances)
    annotations = annotator.annotate_pairs(df_gold_crossannotations)
    df_annotations = utils.load_or_convert_to_dataframe(annotations)
    return df_annotations


###############################


def format_acc_sem(serie, accuracy="accuracy", sem_col="sem_samples"):
    return f"{serie[accuracy]:.2f}±{serie[sem_col]:.2f}"


def _random_mode(s, available_modes=None, favorite_mode=None, seed=123):
    """Take the mode of a series, but if there are multiple modes, randomly sample one
    (with potential restriction to `available_modes` or favoring a specific mode `favorite_mode`).

    Example
    -------
    >>> import pandas as pd
    >>> random_mode(pd.Series([1.0,2.0,1.0]))
    1.0
    >>> random_mode(pd.Series([1.0,2.0])) in [1.0, 2.0]
    True
    >>> random_mode(pd.Series([1.0,2.0,-1.0]), favorite_mode=-1.0)
    -1.0
    >>> random_mode(pd.Series([1.0,2.0,2.0,-1.0]), favorite_mode=-1.0)
    2.0
    """
    out = pd.Series.mode(s)
    if len(out) > 1:
        if favorite_mode is not None and favorite_mode in out.values:
            return favorite_mode
        if available_modes is not None:
            out = out[out.isin(available_modes)]
        out = out.sample(1, random_state=seed)
    return out.item()
