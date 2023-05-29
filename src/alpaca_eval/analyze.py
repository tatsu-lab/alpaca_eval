"""
Main module for analyzing an evaluation benchmark (annotator and data).
"""
import logging
from typing import Callable, Optional, Union

import datasets
import pandas as pd

from . import constants, utils
from .types import AnyPath, AnyData


def DEFAULT_GOLD_CROSSANNOTATIONS():
    return datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_human_crossannotations",
        cache_dir=constants.DEFAULT_CACHE_DIR,
        use_auth_token=constants.DATASETS_TOKEN,
    )["validation"]


def DEFAULT_GOLD_ANNOTATIONS():
    return datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_human_annotations",
        cache_dir=constants.DEFAULT_CACHE_DIR,
        use_auth_token=constants.DATASETS_TOKEN,
    )["validation"]


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

    min_n_annotators : int
        Minimum number of annotators for treating as gold annotation.

    annotator_kwargs : dict
        Arguments that will be passed to all annotators being analyzed.
    """

    def __init__(
            self,
            gold_crossannotations: Union[AnyPath, AnyData, Callable] = DEFAULT_GOLD_CROSSANNOTATIONS,
            gold_annotations: Optional[Union[AnyPath, AnyData, Callable]] = None,
            keys=("instruction", "input", "output_1", "output_2"),
            min_n_annotators=4,
            seed=0,
            **annotator_kwargs
    ):
        self.keys = list(keys)
        self.min_n_annotators = min_n_annotators
        self.annotator_kwargs = annotator_kwargs

        df_gold_crossannotations = utils.load_or_convert_to_dataframe(gold_crossannotations)
        # adding a random index to differentiate between the min_n_annotators
        df_gold_crossannotations["index"] = df_gold_crossannotations.groupby(self.keys)["preference"].cumcount()
        self.df_gold_crossannotations = self._select_at_least_n_annotations(df_gold_crossannotations)

        if gold_annotations is None:
            self.df_gold_annotations = self.df_gold_crossannotations.query("annotator_index == 0")
        else:
            self.df_gold_annotations = utils.load_or_convert_to_dataframe(gold_annotations)

        self.all_df_annotations = dict()
        self.seed = seed

    def _select_at_least_n_annotations(self, df):
        """Gets examples with at least n annotations"""
        counts = df.groupby(self.keys)["annotator_index"].count()
        counts = counts[counts >= self.min_n_annotators].reset_index().rename(columns={
            "annotator_index": "n_annotated"})
        df_selected = df.merge(counts, on=self.keys)
        return df_selected

    def get_price(
            self,
            annotator="gpt-4-0314_pairwise_vH_b5_chatml-prompt",
            annotator_kwargs=None,  # kwargs for AutoAnnotatorPairwiseDB
            is_annotate=False,  # False will be faster but doesn't ensure that annotated
    ):
        df_auto = self.get_df_auto(
            annotator=annotator, is_annotate=is_annotate, is_return_all_columns=True, **annotator_kwargs
        )

        price = df_auto.apply(
            lambda x: x["auto_total_tokens"] * get_price_per_token(x["annotator"].split("_")[0]), axis=1
        ).mean()
        return price

    def auto_and_turk_vs_turk_mode(
            self,
            annotator="gpt-4-0314_pairwise_vH_b5_chatml-prompt",
            annotator_kwargs=None,  # kwargs for AutoAnnotatorPairwiseDB
            is_annotate=True,  # False will be faster but doesn't ensure that annotated
            df_auto=None,
    ):
        """Computes the accuracy of the auto annotations and turk annotations vs mode of turk annotations"""
        annotator_kwargs = annotator_kwargs or {}

        accuracies_auto = []
        sems_auto = []
        counts_auto = []
        accuracies_turk = []
        sems_turk = []
        counts_turk = []
        if df_auto is None:
            df_auto = self.get_df_auto(annotator=annotator, is_annotate=is_annotate, **annotator_kwargs)

        for i in range(self.min_n_annotators):
            curr_gold = (
                self.df_gold.query(f"index != {i}").groupby(self.keys)["preference"].aggregate(ann_helpers.random_mode)
            )

            out_auto = pd.merge(curr_gold, df_auto, left_index=True, right_index=True, suffixes=("_gold", "_auto"))
            out_auto["match"] = (out_auto["preference_gold"] == out_auto["preference_auto"]).astype(int)
            accuracies_auto.append(out_auto["match"].mean())
            sems_auto.append(out_auto["match"].sem())
            counts_auto.append(len(out_auto["match"]))

            out_turk = pd.merge(
                curr_gold,
                self.df_gold.query(f"index == {i}").set_index(self.keys),
                left_index=True,
                right_index=True,
                suffixes=["_gold", "_turk"],
            )
            out_turk["match"] = (out_turk["preference_gold"] == out_turk["preference_turk"]).astype(int)
            accuracies_turk.append(out_turk["match"].mean())
            sems_turk.append(out_turk["match"].sem())
            counts_turk.append(len(out_turk["match"]))

        auto_results = pd.DataFrame(dict(accuracy=accuracies_auto, sem_samples=sems_auto, count=counts_auto))
        turk_results = pd.DataFrame(dict(accuracy=accuracies_turk, sem_samples=sems_turk, count=counts_turk))

        return pd.DataFrame(
            dict(
                auto=dict(**auto_results.mean(), sem_annotators=auto_results["accuracy"].sem()),
                turk=dict(**turk_results.mean(), sem_annotators=turk_results["accuracy"].sem()),
            )
        )

    def turk_vs_turk(self):
        """Computes the turk vs turk accuracy"""

        accuracies = []
        sems_samples = []

        for i in range(self.min_n_annotators):
            for j in range(i + 1, self.min_n_annotators):
                curr_i = self.df_gold.query(f"index == {i}")
                curr_j = self.df_gold.query(f"index == {j}")

                out = pd.merge(curr_i, curr_j, on=self.keys, suffixes=("_i", "_j"))
                out["match"] = (out["preference_i"] == out["preference_j"]).astype(int)
                accuracies.append(out["match"].mean())
                sems_samples.append(out["match"].sem())
        all_results = pd.DataFrame(dict(accuracies=accuracies, sems=sems_samples))

        return pd.Series(
            dict(
                accuracy=all_results.accuracies.mean(),
                sem_annotators=all_results.accuracies.sem(),
                sem_samples=all_results.sems.mean(),
            )
        )

    def auto_vs_turk(
            self,
            annotator="gpt-4-0314_pairwise_vH_b5_chatml-prompt",
            annotator_kwargs=None,  # kwargs for AutoAnnotatorPairwiseDB
            is_annotate=True,  # False will be faster but doesn't ensure that annotated
    ):
        """Computes the auto vs turk accuracy"""
        annotator_kwargs = annotator_kwargs or {}

        accuracies = []
        sems_samples = []
        counts = []

        df_auto = self.get_df_auto(annotator=annotator, is_annotate=is_annotate, **annotator_kwargs)

        for i in range(self.min_n_annotators):
            curr_turk = self.df_gold.query(f"index == {i}").set_index(self.keys)

            out = pd.merge(curr_turk, df_auto, left_index=True, right_index=True, suffixes=("_turk", "_auto"))
            out["match"] = (out["preference_turk"] == out["preference_auto"]).astype(int)
            accuracies.append(out["match"].mean())
            sems_samples.append(out["match"].sem())
            counts.append(len(out["match"]))

        all_results = pd.DataFrame(dict(accuracies=accuracies, sems=sems_samples))

        return pd.Series(
            dict(
                accuracy=all_results.accuracies.mean(),
                sem_annotators=all_results.accuracies.sem(),
                sem_samples=all_results.sems.mean(),
            )
        )

    def auto_vs_auto(
            self,
            annotator_1="gpt-4-0314_pairwise_vH_b5_chatml-prompt_temp=0.7",
            annotator_2="gpt-4-0314_pairwise_vH_b5_chatml-prompt",
            annotator_kwargs_1=None,  # kwargs for AutoAnnotatorPairwiseDB
            annotator_kwargs_2=None,  # kwargs for AutoAnnotatorPairwiseDB
            is_annotate=True,  # False will be faster but doesn't ensure that annotated
    ):
        """Computes the auto vs auto accuracy"""
        annotator_kwargs_1 = annotator_kwargs_1 or {}
        annotator_kwargs_2 = annotator_kwargs_2 or {}

        df_auto_1 = self.get_df_auto(annotator=annotator_1, is_annotate=is_annotate, **annotator_kwargs_1)
        df_auto_2 = self.get_df_auto(annotator=annotator_2, is_annotate=is_annotate, **annotator_kwargs_2)

        out = pd.merge(df_auto_1, df_auto_2, on=self.keys, suffixes=("_1", "_2"))
        out["match"] = (out["preference_1"] == out["preference_2"]).astype(int)

        return pd.Series(dict(accuracy=out["match"].mean(), sem_samples=out["match"].sem(), counts=len(out["match"])))

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
        for i in range(self.min_n_annotators):
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

        return pd.DataFrame(results).T / self.min_n_annotators

    def auto_cross_annotation(
            self,
            annotator="gpt-4-0314_pairwise_vH_b5_chatml-prompt",
            annotator_kwargs=None,  # kwargs for AutoAnnotatorPairwiseDB
            is_annotate=True,  # False will be faster but doesn't ensure that annotated
    ):
        """Evaluate cross annotation accuracy of auto annotators. This requires rerunning annotations => expensive"""
        df_auto_bis = self.get_df_auto(annotator=annotator, is_annotate=is_annotate, delta_seed=1, **annotator_kwargs)
        df_auto = self.get_df_auto(annotator=annotator, is_annotate=is_annotate, **annotator_kwargs)

        return self.auto_vs_auto(annotator_1=df_auto, annotator_2=df_auto_bis)

    def mode_vs_mode(
            self,
            annotator="gpt-4-0314_pairwise_vH_b5_chatml-prompt",
            is_annotate=True,  # False will be faster but doesn't ensure that annotated
            comparisons=(1, 3),  # 1 vs 1
            **annotator_kwargs,
    ):
        """Evaluate cross annotation accuracy of auto annotators. This requires rerunning annotations => expensive"""
        assert len(comparisons) == 2
        total_comparisons = comparisons[0] + comparisons[1]

        df_autos = self.get_all_df_auto_seed(
            annotator=annotator,
            is_annotate=is_annotate,
            n_delta_seeds=total_comparisons if annotator != "gold" else 0,
            is_return_all_columns=True,
            **annotator_kwargs,
        )

        results = []
        for idcs_1, idcs_2 in all_paired_partition(range(total_comparisons), comparisons[0]):
            results.append(
                self.auto_vs_auto(
                    annotator_1=self._get_mode(df_autos, annotator, idcs_1),
                    annotator_2=self._get_mode(df_autos, annotator, idcs_2),
                )
            )

        return sum(results) / len(results)

    def _get_mode(self, df_autos, annotator, idcs):
        if annotator == "gold":
            out = self.df_gold[self.df_gold["index"].isin(idcs)]
        else:
            out = pd.concat([df_autos[s] for s in idcs])
        return out.groupby(self.keys)["preference"].aggregate(ann_helpers.random_mode)

    def get_all_df_auto_seed(self, annotator, n_delta_seeds=4, **kwargs):
        out = {
            i: self.get_df_auto(
                annotator=annotator,
                delta_seed=i,
                **kwargs,
            )
            for i in range(n_delta_seeds)
        }

        return out

    def mode_vs_mode_cross(
            self,
            annotator="gpt-4-0314_pairwise_vH_b5_chatml-prompt",
            is_annotate=True,  # False will be faster but doesn't ensure that annotated
            n_auto_mode=2,
            comparisons_gold=(1, 3),
            **annotator_kwargs,
    ):
        """Evaluate cross annotation accuracy of auto annotators. This requires rerunning annotations => expensive"""
        assert len(comparisons_gold) == 2
        total_comparisons = comparisons_gold[0] + comparisons_gold[1]

        df_autos = self.get_all_df_auto_seed(
            annotator=annotator,
            is_annotate=is_annotate,
            n_delta_seeds=n_auto_mode,
            is_return_all_columns=True,
            **annotator_kwargs,
        )

        results = []
        for _, idcs_2 in all_paired_partition(range(total_comparisons), comparisons_gold[0]):
            results.append(
                self.auto_vs_auto(
                    annotator_1=self._get_mode(df_autos, annotator, list(range(n_auto_mode))),
                    annotator_2=self._get_mode(df_autos, "gold", idcs_2),
                )
            )

        return sum(results) / len(results)

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
    n_crossannotations = analyzer.min_n_annotators
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
    return pd.concat(all_annotations, axis=0)


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


def get_price_per_token(model):
    """Returns the price per token for a given model"""
    if model == "claude-v1":
        return 0
    elif "gpt-4" in model:
        return (
                0.03 / 1000
        )  # that's not completely true because decoding is 0.06 but close enough given that most is context
    elif "gpt-3.5-turbo" in model:
        return 0.002 / 1000
    elif "text-davinci-003" in model:
        return 0.02 / 1000
    else:
        raise ValueError(f"Unknown model {model}")


def all_paired_partition(lst, k):
    partitions = []
    for comb in combinations(lst, k):
        rest = tuple([x for x in lst if x not in comb])
        if (rest, tuple(comb)) not in partitions:
            partitions.append((tuple(comb), rest))
    return partitions
