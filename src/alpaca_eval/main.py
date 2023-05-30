import datasets
import pandas as pd
from typing import Any, Callable, Optional, Union
from pathlib import Path
import fire

from .types import AnyPath, AnyData
from . import utils, metrics, annotators, constants, analyze

CUR_DIR = Path(__file__).parent


def DEFAULT_REFERENCE_OUTPUTS():
    return datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_farm_evaluation",
        cache_dir=constants.DEFAULT_CACHE_DIR,
        use_auth_token=constants.DATASETS_TOKEN,
    )["eval"]


DEFAULT_CONFIGS = "alpaca_farm/configs.yaml"
DEFAULT_LEADERBOARD = CUR_DIR / "leaderboards/pairwise/alpaca_farm_leaderboard.csv"
DEFAULT_EVALUATOR_LEADERBOARD = CUR_DIR / "leaderboards/pairwise/evaluators_leaderboard.csv"

ALL_LEADERBOARDS = {
    (str(DEFAULT_REFERENCE_OUTPUTS), str(DEFAULT_CONFIGS)): DEFAULT_LEADERBOARD,
}


def pairwise_winrates(
        model_outputs: Union[AnyPath, AnyData, Callable],
        reference_outputs: Union[AnyPath, AnyData, Callable] = DEFAULT_REFERENCE_OUTPUTS,
        annotators_config: AnyPath = DEFAULT_CONFIGS,
        name: str = "Current method",
        is_return_metrics: bool = False,
        rest_of_leaderboard: Optional[Union[str, AnyPath, AnyData]] = "auto",
        fn_metric: Union[str, callable] = "pairwise_to_winrate",
        sort_by: str = "win_rate",
        max_instances: Optional[int] = None,
        annotation_kwargs: Optional[dict[str, Any]] = None,
        **annotator_kwargs
):
    """

    Parameters
    ----------
    model_outputs : path or data or dict
        The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
        (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction`
        and `output` with optional `input`.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `model_outputs`. If None, the reference outputs are the
        003 outputs on AlpacaFarm evaluation set.

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file. For details see the docstring of
        `PairwiseAnnotator`.

    name : str, optional
        The name of the model to add to the leaderboard.

    is_return_metrics : bool, optional
        Whether to return the metrics instead of printing the results.

    rest_of_leaderboard : path or data, optional
        The precomputed leaderboard or a path to it (json, csv, or tsv). The leaderboard should contain at least the
        column `win_rate`. If `auto` we will try to use the corresponding leaderboard for the reference outputs (only if
        in CORRESPONDING_OUTPUTS_LEADERBOARDS). If `None` we won't add other models from the leaderboard.

    fn_metric : str or callable, optional
        The function or function name in `metrics.py` that will be used to convert preference to metrics. The function
        should take a sequence of preferences (0 for draw, 1 for base win, 2 when the model to compare wins) and return
        a dictionary of metrics and the key by which to sort the leaderboard.

    sort_by : str, optional
        The key by which to sort the leaderboard.

    max_instances : int, optional
        The maximum number of instances to annotate. Useful for testing.

    annotation_kwargs : dict, optional
        Additional arguments to pass to `PairwiseAnnotator.annotate_head2head`.

    annotator_kwargs :
        Additional arguments to pass to `PairwiseAnnotator`.
    """
    annotation_kwargs = annotation_kwargs or dict()

    if rest_of_leaderboard == "auto":
        try:
            rest_of_leaderboard = ALL_LEADERBOARDS[(str(reference_outputs), str(annotators_config))]
        except KeyError:
            rest_of_leaderboard = None

    if rest_of_leaderboard is not None:
        leaderboard = utils.load_or_convert_to_dataframe(rest_of_leaderboard).to_dict(orient="index")
    else:
        leaderboard = dict()

    model_outputs = utils.load_or_convert_to_dataframe(model_outputs)
    reference_outputs = utils.load_or_convert_to_dataframe(reference_outputs)

    if max_instances is not None:
        model_outputs = model_outputs[:max_instances]
        reference_outputs = reference_outputs[:max_instances]

    if isinstance(fn_metric, str):
        fn_metric = getattr(metrics, fn_metric)

    annotator = annotators.PairwiseAnnotator(annotators_config=annotators_config, **annotator_kwargs)
    annotated = annotator.annotate_head2head(outputs_1=reference_outputs, outputs_2=model_outputs, **annotation_kwargs)

    leaderboard[name] = fn_metric(preferences=[a["preference"] for a in annotated])
    df_leaderboard = pd.DataFrame(leaderboard).T.sort_values(by=sort_by, ascending=False)

    if is_return_metrics:
        return df_leaderboard
    else:
        print(df_leaderboard.to_string(float_format="%.2f"))


def analyze_evaluators(annotators_config: Optional[AnyPath] = DEFAULT_CONFIGS,
                       Annotator=annotators.PairwiseAnnotator,
                       analyzer_kwargs=None,
                       rest_of_leaderboard: Optional[Union[AnyPath, AnyData]] = DEFAULT_EVALUATOR_LEADERBOARD,
                       is_save_leaderboard: bool = False,
                       is_return_metrics: bool = False,
                       is_overwrite_leaderboard: bool = False,
                       ):

    if rest_of_leaderboard is not None:
        leaderboard = utils.load_or_convert_to_dataframe(rest_of_leaderboard).to_dict(orient="index")
    else:
        leaderboard = dict()

    analyzer_kwargs = analyzer_kwargs or {}

    if annotators_config is not None:
        key = annotators_config.replace("/", "_").replace("_configs.yaml", "")
        if key not in leaderboard or is_overwrite_leaderboard:
            analyzer = analyze.Analyzer(**analyzer_kwargs)

            if key == "humans":
                df_crossannotations = analyzer.df_gold_crossannotations
            elif key == "longest":
                df_crossannotations = analyze._get_longest_predictor(analyzer.df_gold_crossannotations)
            else:
                df_crossannotations = analyze.get_crossannotations(analyzer=analyzer,
                                                                   Annotator=Annotator,
                                                                   annotators_config=annotators_config)
            leaderboard[key] = analyze.get_metrics_evaluator(analyzer, df_crossannotations, evaluator_name=key)

    df_leaderboard = pd.DataFrame(leaderboard).T.sort_values(by="Human Agreement", ascending=False)

    if is_save_leaderboard:
        df_leaderboard.to_csv(rest_of_leaderboard)

    if is_return_metrics:
        return df_leaderboard
    else:
        print(df_leaderboard.to_string(float_format="%.2f"))


def main_helper(task="pairwise_winrates", **kwargs):
    globals()[task](**kwargs)


def main():
    fire.Fire(main_helper)


if __name__ == "__main__":
    fire.Fire(main_helper)
