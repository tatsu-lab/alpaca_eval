import logging

import pandas as pd
from typing import Any, Callable, Optional, Union
from pathlib import Path
import fire

from .types import AnyPath, AnyData
from . import utils, metrics, annotators, constants, analyze

CUR_DIR = Path(__file__).parent
DEFAULT_CONFIGS = "claude"


def pairwise_winrates(
        model_outputs: Union[AnyPath, AnyData, Callable],
        reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAFARM_REFERENCE_OUTPUTS,
        annotators_config: AnyPath = "claude",
        name: str = "Current method",
        output_path: Optional[Union[AnyPath, str]] = "auto",
        precomputed_leaderboard: Optional[Union[str, AnyPath, AnyData]] = "auto",
        is_return_instead_of_print: bool = False,
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

    output_path : bool, optional
        Path to the directory where the new leaderboard and the annotations should be stored. If None we don't save.
        If `auto` we use `model_outputs` if it is a path, and otherwise use the directory from which we call the script.

    precomputed_leaderboard : path or data, optional
        The precomputed leaderboard or a path to it (json, csv, or tsv). The leaderboard should contain at least the
        column `win_rate`. If `auto` we will try to use the corresponding leaderboard for the reference outputs (only if
        in CORRESPONDING_OUTPUTS_LEADERBOARDS). If `None` we won't add other models from the leaderboard.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

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
    if output_path == "auto":
        try:
            output_path = Path(model_outputs).parent
        except:
            output_path = "."

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

    if precomputed_leaderboard == "auto":
        try:
            precomputed_leaderboard = constants.PRECOMPUTED_LEADERBOARDS[
                (str(reference_outputs), str(annotators_config))]
        except KeyError:
            logging.warning(f"precomputed_leaderboard = 'auto'. But we have found no corresponding leaderboard for"
                            f" {reference_outputs} and {annotators_config}")
            precomputed_leaderboard = None

    if precomputed_leaderboard is not None:
        try:
            leaderboard = utils.load_or_convert_to_dataframe(precomputed_leaderboard).to_dict(orient="index")
        except FileNotFoundError:
            logging.warning(f"precomputed_leaderboard = {precomputed_leaderboard} not found => computing from scratch.")
            leaderboard = dict()
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
    annotations = annotator.annotate_head2head(outputs_1=reference_outputs,
                                               outputs_2=model_outputs,
                                               **annotation_kwargs)

    leaderboard[name] = fn_metric(preferences=[a["preference"] for a in annotations])
    df_leaderboard = pd.DataFrame(leaderboard).T.sort_values(by=sort_by, ascending=False)

    if output_path is not None:
        df_leaderboard.to_csv(output_path / "leaderboard.csv")
        utils.convert_to_dataframe(annotations).to_json(output_path / "annotations.json", orient="records", indent=2)

    if is_return_instead_of_print:
        return df_leaderboard, annotations
    else:
        print(df_leaderboard.to_string(float_format="%.2f"))


def make_model_leaderboard(leaderboard_path: AnyPath,
                           annotators_config: AnyPath = DEFAULT_CONFIGS,
                           fn_add_to_leaderboard: Callable = "pairwise_winrates",
                           all_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAFARM_ALL_OUTPUTS,
                           is_return_instead_of_print: bool = False,
                           **kwargs):
    """Precompute and save an entire leaderboard.

    Parameters
    ----------
    leaderboard_path : path
        The path to save the leaderboard to. The leaderboard will be saved as a csv file, if it already exists it will
        append

    fn_add_to_leaderboard : callable or str, optional
        The function to use to add a model to the leaderboard. If a string, it should be the name of a function in
        `main.py`. The function should take the arguments: `model_outputs`, `annotators_config`, `name`,
        `precomputed_leaderboard`, `is_return_instead_of_print`.

    all_outputs : path or data or callable, optional
        The outputs of all models to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
        (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction`
        and `output` with optional `input`. It should also contain a column `generator` with the name of the current
        model.

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

    kwargs :
        Additional arguments to pass to `fn_add_to_leaderboard`.
    """
    if isinstance(fn_add_to_leaderboard, str):
        fn_add_to_leaderboard = globals()[fn_add_to_leaderboard]

    all_outputs = utils.load_or_convert_to_dataframe(all_outputs)
    if "generator" not in all_outputs.columns:
        raise ValueError(f"all_outputs should have a column 'generator' with the name of the model.")

    all_annotations = []
    for model in all_outputs["generator"].unique():
        model_outputs = all_outputs[all_outputs["generator"] == model]
        df_leaderboard, annotations = fn_add_to_leaderboard(model_outputs=model_outputs,
                                                            annotators_config=annotators_config,
                                                            name=model,
                                                            precomputed_leaderboard=leaderboard_path,
                                                            is_return_instead_of_print=True,
                                                            **kwargs)
        all_annotations += annotations
        df_leaderboard.to_csv(leaderboard_path)

    leaderboard = utils.load_or_convert_to_dataframe(leaderboard_path)
    df_leaderboard = pd.DataFrame(leaderboard)

    if is_return_instead_of_print:
        return df_leaderboard, all_annotations
    else:
        print(df_leaderboard.to_string(float_format="%.2f"))


def analyze_evaluators(annotators_config: Optional[AnyPath] = DEFAULT_CONFIGS,
                       Annotator=annotators.PairwiseAnnotator,
                       analyzer_kwargs=None,
                       precomputed_leaderboard: Optional[
                           Union[AnyPath, AnyData]] = CUR_DIR / "leaderboards/evaluators/evaluators_leaderboard.csv",
                       is_save_leaderboard: bool = False,
                       is_return_instead_of_print: bool = False,
                       is_overwrite_leaderboard: bool = False,
                       max_instances: Optional[int] = None,
                       ):
    """Analyze the annotators.

    Parameters
    ----------
    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file.

    Annotator : class, optional
        The annotator class to use.

    analyzer_kwargs : dict, optional
        Additional arguments to pass to the analyzer.

    precomputed_leaderboard : path or data, optional
        The precomputed (meta)leaderboard of annotators or a path to it (json, csv, or tsv).
        
    is_save_leaderboard : bool, optional
        Whether to save the leaderboard (ie analyzed results).
        
    is_return_instead_of_print : bool, optional 
        Whether to return the leaderboard (ie analyzed results). If True, it will not print the results.
        
    is_overwrite_leaderboard : bool, optional
        Whether to overwrite the leaderboard if it already exists.
        
    max_instances : int, optional
        The maximum number of instances to analyze.
    """

    leaderboard = dict()
    if precomputed_leaderboard is not None:
        try:
            leaderboard = utils.load_or_convert_to_dataframe(precomputed_leaderboard).to_dict(orient="index")
        except FileNotFoundError:
            logging.warning(f"Could not find precomputed leaderboard at {precomputed_leaderboard}. Starting from "
                            f"scratch.")

    analyzer_kwargs = analyzer_kwargs or {}

    all_crossannotations = dict()
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
                                                                   max_instances=max_instances,
                                                                   annotators_config=annotators_config)
            leaderboard[key] = analyze.get_metrics_evaluator(analyzer, df_crossannotations, evaluator_name=key)
            all_crossannotations[key] = df_crossannotations

    df_leaderboard = pd.DataFrame(leaderboard).T.sort_values(by="Human agreement [%]", ascending=False)

    if is_save_leaderboard:
        df_leaderboard.to_csv(precomputed_leaderboard)

    if is_return_instead_of_print:
        return df_leaderboard, all_crossannotations
    else:
        print(df_leaderboard.to_string(float_format="%.2f"))


def main_helper(task="pairwise_winrates", **kwargs):
    globals()[task](**kwargs)


def main():
    fire.Fire(main_helper)


if __name__ == "__main__":
    fire.Fire(main_helper)
