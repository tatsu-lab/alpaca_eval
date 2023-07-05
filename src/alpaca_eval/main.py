import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

import fire
import pandas as pd

from . import analyze, annotators, constants, decoders, metrics, utils
from .types import AnyData, AnyPath

CUR_DIR = Path(__file__).parent
DEFAULT_CONFIGS = "alpaca_eval_gpt4"


def evaluate(
    model_outputs: Optional[Union[AnyPath, AnyData, Callable]] = None,
    reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
    annotators_config: AnyPath = DEFAULT_CONFIGS,
    name: Optional[str] = None,
    output_path: Optional[Union[AnyPath, str]] = "auto",
    precomputed_leaderboard: Optional[Union[str, AnyPath, AnyData]] = "auto",
    is_overwrite_leaderboard: bool = False,
    leaderboard_mode_to_print: Optional[str] = "minimal",
    current_leaderboard_mode: str = "community",
    is_return_instead_of_print: bool = False,
    fn_metric: Union[str, callable] = "pairwise_to_winrate",
    sort_by: str = "win_rate",
    is_cache_leaderboard: Optional[bool] = None,
    max_instances: Optional[int] = None,
    annotation_kwargs: Optional[dict[str, Any]] = None,
    **annotator_kwargs,
):
    """Evaluate a model based on its outputs. This is the default entrypoint if no command is specified.

    Parameters
    ----------
    model_outputs : path or data or dict
        The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
        (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction`
        and `output` with optional `input`. If None, we just print the leaderboard.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `model_outputs`. If None, the reference outputs are the
        003 outputs on the AlpacaEval set.

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file. For details see the docstring of
        `PairwiseAnnotator`.

    name : str, optional
        The name of the model to add to the leaderboard. If None we check if `generator is in model_outputs` if not
        we use "Current model".

    output_path : bool, optional
        Path to the directory where the new leaderboard and the annotations should be stored. If None we don't save.
        If `auto` we use `model_outputs` if it is a path, and otherwise use the directory from which we call the script.

    precomputed_leaderboard : path or data, optional
        The precomputed leaderboard or a path to it (json, csv, or tsv). The leaderboard should contain at least the
        column `win_rate`. If `auto` we will try to use the corresponding leaderboard for the reference outputs (only if
        in CORRESPONDING_OUTPUTS_LEADERBOARDS). If `None` we won't add other models from the leaderboard.

    is_overwrite_leaderboard : bool, optional
        Whether to overwrite the leaderboard if the model is already in it.

    leaderboard_mode_to_print : {"minimal", "verified", "community", None}, optional
        The mode of the leaderboard to use. Only used if the precomputed leaderboard has a column `mode`, in which case
        it will filter the leaderboard by this mode. If None keeps all.

    current_leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard for the current method.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

    fn_metric : str or callable, optional
        The function or function name in `metrics.py` that will be used to convert preference to metrics. The function
        should take a sequence of preferences (0 for draw, 1 for base win, 2 when the model to compare wins) and return
        a dictionary of metrics and the key by which to sort the leaderboard.

    sort_by : str, optional
        The key by which to sort the leaderboard.

    is_cache_leaderboard : bool, optional
        Whether to save the result leaderboard to `precomputed_leaderboard`. If None we save only if max_instances.
        A preferred way of adding models to the leaderboard is to set `precomputed_leaderboard` to the previously
        saved leaderboard at `<output_path>/leaderboard.csv`.

    max_instances : int, optional
        The maximum number of instances to annotate. Useful for testing.

    annotation_kwargs : dict, optional
        Additional arguments to pass to `PairwiseAnnotator.annotate_head2head`.

    annotator_kwargs :
        Additional arguments to pass to `PairwiseAnnotator`.
    """
    if (
        isinstance(current_leaderboard_mode, str)
        and current_leaderboard_mode not in constants.ORDERED_LEADERBOARD_MODES
    ):
        raise ValueError(f"current_leaderboard_mode should be one of {constants.ORDERED_LEADERBOARD_MODES}")

    annotation_kwargs = annotation_kwargs or dict()

    leaderboard, precomputed_leaderboard = utils.get_precomputed_leaderboard(
        precomputed_leaderboard, reference_outputs, annotators_config
    )
    annotations = None

    if model_outputs is not None:
        model_outputs = utils.load_or_convert_to_dataframe(model_outputs)
        reference_outputs = utils.load_or_convert_to_dataframe(reference_outputs)
        name = utils.get_generator_name(name, model_outputs)

        if (name not in leaderboard) or is_overwrite_leaderboard:
            logging.info(f"Evaluating the {name} outputs.")

            if max_instances is not None:
                model_outputs = model_outputs[:max_instances]
                reference_outputs = reference_outputs[:max_instances]

            annotator = annotators.PairwiseAnnotator(annotators_config=annotators_config, **annotator_kwargs)
            annotations = annotator.annotate_head2head(
                outputs_1=reference_outputs, outputs_2=model_outputs, **annotation_kwargs
            )

            if isinstance(fn_metric, str):
                fn_metric = getattr(metrics, fn_metric)

            leaderboard[name] = fn_metric(preferences=[a["preference"] for a in annotations])
            leaderboard[name]["mode"] = current_leaderboard_mode
        else:
            logging.info(f"Skipping evaluation of {name} as it is already in the precomputed leaderboard.")

    output_path = utils.get_output_path(output_path, model_outputs, name)

    df_leaderboard = pd.DataFrame.from_dict(leaderboard, orient="index").sort_values(by=sort_by, ascending=False)
    df_leaderboard = df_leaderboard[
        utils.prioritize_elements(list(df_leaderboard.columns), ["win_rate", "standard_error"])
    ]

    if output_path is not None:
        logging.info(f"Saving all results to {output_path}")
        df_leaderboard.to_csv(output_path / "leaderboard.csv")
        if annotations is not None:
            utils.convert_to_dataframe(annotations).to_json(
                output_path / "annotations.json", orient="records", indent=2
            )

    if is_cache_leaderboard is None:
        is_cache_leaderboard = max_instances is None

    if is_cache_leaderboard:
        if isinstance(precomputed_leaderboard, AnyPath):
            logging.info(f"Saving result to the precomputed leaderboard at {precomputed_leaderboard}")
            df_leaderboard.to_csv(precomputed_leaderboard)
        else:
            logging.info(
                f"Not saving the result to the cached leaderboard because precomputed_leaderboard is not a "
                f"path but {type(precomputed_leaderboard)}."
            )

    if is_return_instead_of_print:
        return df_leaderboard, annotations
    else:
        utils.print_leaderboard(
            df_leaderboard,
            leaderboard_mode_to_print,
            current_name=name,
            cols_to_print=["win_rate", "standard_error", "n_total"],
        )


def evaluate_from_model(
    model_configs: Union[AnyPath, dict],
    reference_model_configs: Optional[Union[AnyPath, dict]] = None,
    evaluation_dataset: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
    annotators_config: AnyPath = DEFAULT_CONFIGS,
    output_path: AnyPath = "auto",
    max_instances: int = None,
    is_strip_output: bool = True,
    **kwargs,
):
    """Evaluate a model from HuggingFace or an API provider. This is a wrapper around `evaluate` which includes
    generating from
    a desired model.

    Parameters
    ----------
    model_configs : path or dict
        A dictionary or path (relative to `models_configs`) to a yaml file containing the configuration of the model to
        decode from. If a directory,we search for 'configs.yaml' in it. The keys in the first dictionary should be the
        generator's name, and the value should be a dictionary of the generator's configuration which should have the
        following keys:
        - prompt_template (str): a prompt template or path to one. Each template should contain placeholders for
        keys in the data dictionary, typically {instruction} and {output}.
        - fn_completions (str): function in `alpaca_farm.decoders` for completions. Needs to accept as first argument
            `prompts` which is a list of string.
        - completions_kwargs (dict): kwargs for fn_completions. E.g. model_name, max_tokens, temperature...

    reference_model_configs : path or dict, optional
        Same as in `model_configs` but for the reference model. If None, we use the default Davinci003 outputs.

    evaluation_dataset : path or callable, optional
        Path to the evaluation dataset or a function that returns a dataframe. If None, we use the default evaluation

    annotators_config : path or dict, optional
        Path to the annotators configuration or a dictionary. If None, we use the default annotators configuration.

    output_path : path, optional
        Path to save the generations, annotations and leaderboard. If auto saves at `results/<model_name>`

    max_instances : int, optional
        Maximum number of instances to generate and evaluate. If None, we evaluate all instances.

    is_strip_output : bool, optional
        Whether to strip trailing and leading whitespaces from the outputs.

    kwargs:
        Other kwargs to `evaluate`
    """
    evaluation_dataset = utils.load_or_convert_to_dataframe(evaluation_dataset)

    model_configs = utils.load_configs(model_configs, relative_to=constants.MODELS_CONFIG_DIR)
    if reference_model_configs is not None:
        reference_model_configs = utils.load_configs(reference_model_configs, relative_to=constants.MODELS_CONFIG_DIR)

    def get_completions(configs):
        columns_to_keep = ["dataset", "instruction", "output"]
        columns_to_keep = [c for c in columns_to_keep if c in evaluation_dataset.columns]
        curr_outputs = evaluation_dataset.copy()[columns_to_keep]
        if max_instances is not None:
            curr_outputs = curr_outputs.iloc[:max_instances]
        assert len(configs) == 1
        generator = list(configs.keys())[0]
        configs = list(configs.values())[0]
        prompts, _ = utils.make_prompts(
            curr_outputs,
            template=utils.read_or_return(constants.MODELS_CONFIG_DIR / configs["prompt_template"]),
        )
        fn_completions = decoders.get_fn_completions(configs["fn_completions"])
        completions = fn_completions(prompts=prompts, **configs["completions_kwargs"])["completions"]
        if is_strip_output:
            completions = [c.strip() for c in completions]
        curr_outputs["output"] = completions
        curr_outputs["generator"] = generator
        return curr_outputs

    model_outputs = get_completions(model_configs)
    if reference_model_configs is None:
        if "output" not in evaluation_dataset.columns:
            raise ValueError("evaluation_dataset should have a column 'output' containing references outputs")
        reference_outputs = evaluation_dataset.copy()
    else:
        reference_outputs = get_completions(reference_model_configs)

    if output_path == "auto":
        output_path = Path("results") / (model_outputs["generator"].iloc[0])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        model_outputs.to_json(output_path / "model_outputs.json", orient="records", indent=2)
        reference_outputs.to_json(output_path / "reference_outputs.json", orient="records", indent=2)

    return evaluate(
        model_outputs=model_outputs,
        reference_outputs=reference_outputs,
        annotators_config=annotators_config,
        output_path=output_path,
        max_instances=max_instances,
        **kwargs,
    )


def make_leaderboard(
    leaderboard_path: AnyPath,
    annotators_config: AnyPath = DEFAULT_CONFIGS,
    all_model_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAFARM_ALL_OUTPUTS,
    reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
    fn_add_to_leaderboard: Callable = "evaluate",
    leaderboard_mode: str = "verified",
    is_return_instead_of_print: bool = False,
    **kwargs,
):
    """Precompute and save an entire leaderboard for a given dataset / evaluator / set of models generations.

    Parameters
    ----------
    leaderboard_path : path
        The path to save the leaderboard to. The leaderboard will be saved as a csv file, if it already exists it will
        append

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file.

    all_model_outputs : path or data or callable, optional
        The outputs of all models to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv potentially with globbing) or a function to generate
        those. If the path contains a globbing pattern, we will read all files matching the pattern and concatenate
        them. Each dictionary (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by
        default `instruction` and `output` with optional `input`. It should also contain a column `generator` with the
        name of the current model.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `all_model_outputs` but without needing `generator`. By
        default,
        the reference outputs are the 003 outputs on AlpacaEval set.

    fn_add_to_leaderboard : callable or str, optional
        The function to use to add a model to the leaderboard. If a string, it should be the name of a function in
        `main.py`. The function should take the arguments: `model_outputs`, `annotators_config`, `name`,
        `precomputed_leaderboard`, `is_return_instead_of_print`, `reference_outputs`.

    leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard to save all new entries with.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

    kwargs :
        Additional arguments to pass to `fn_add_to_leaderboard`.
    """
    if isinstance(fn_add_to_leaderboard, str):
        fn_add_to_leaderboard = globals()[fn_add_to_leaderboard]

    all_model_outputs = utils.load_or_convert_to_dataframe(all_model_outputs)
    if "generator" not in all_model_outputs.columns:
        raise ValueError(f"all_model_outputs should have a column 'generator' with the name of the model.")

    all_annotations = []
    for model in all_model_outputs["generator"].unique():
        model_outputs = all_model_outputs[all_model_outputs["generator"] == model]
        df_leaderboard, annotations = fn_add_to_leaderboard(
            model_outputs=model_outputs,
            reference_outputs=reference_outputs,
            annotators_config=annotators_config,
            name=model,
            precomputed_leaderboard=leaderboard_path,
            is_return_instead_of_print=True,
            current_leaderboard_mode=leaderboard_mode,
            **kwargs,
        )
        if annotations is not None:
            all_annotations += annotations
        df_leaderboard.to_csv(leaderboard_path)

    leaderboard = utils.load_or_convert_to_dataframe(leaderboard_path)
    df_leaderboard = pd.DataFrame(leaderboard)

    if is_return_instead_of_print:
        return df_leaderboard, all_annotations
    else:
        utils.print_leaderboard(
            df_leaderboard, leaderboard_mode=None, cols_to_print=["win_rate", "standard_error", "n_total"]
        )


def analyze_evaluators(
    annotators_config: Optional[AnyPath] = DEFAULT_CONFIGS,
    Annotator=annotators.PairwiseAnnotator,
    analyzer_kwargs=None,
    precomputed_leaderboard: Optional[Union[AnyPath, AnyData]] = CUR_DIR
    / "leaderboards/evaluators/evaluators_leaderboard.csv",
    is_save_leaderboard: bool = False,
    is_return_instead_of_print: bool = False,
    is_overwrite_leaderboard: bool = False,
    max_instances: Optional[int] = None,
    is_single_annotator: bool = False,
    leaderboard_mode_to_print: str = "minimal",
    current_leaderboard_mode: str = "minimal",
):
    """Analyze an evaluator and populates the evaluators leaderboard (agreement with human, speed, price,...).

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

    is_single_annotator : bool, optional
        Whether to analyze a single annotator. If True, will not be able to estimate the annotator's bias.

    leaderboard_mode_to_print : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard to print.

    current_leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard to save all new entries with.
    """

    leaderboard = dict()
    if precomputed_leaderboard is not None:
        try:
            leaderboard = utils.load_or_convert_to_dataframe(precomputed_leaderboard).to_dict(orient="index")
        except FileNotFoundError:
            logging.warning(
                f"Could not find precomputed leaderboard at {precomputed_leaderboard}. Starting from " f"scratch."
            )

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
                df_crossannotations = analyze.get_crossannotations(
                    analyzer=analyzer,
                    Annotator=Annotator,
                    max_instances=max_instances,
                    annotators_config=annotators_config,
                    is_single_annotator=is_single_annotator,
                )

            leaderboard[key] = analyze.get_metrics_evaluator(analyzer, df_crossannotations, evaluator_name=key)
            leaderboard[key]["mode"] = current_leaderboard_mode
            all_crossannotations[key] = df_crossannotations

    df_leaderboard = pd.DataFrame.from_dict(leaderboard, orient="index").sort_values(
        by="Human agreement [%]", ascending=False
    )

    df_leaderboard = df_leaderboard[
        utils.prioritize_elements(list(df_leaderboard.columns), constants.EVALUATORS_LEADERBOARD_COLS_TO_PRIORITIZE)
    ]

    if is_save_leaderboard:
        df_leaderboard.to_csv(precomputed_leaderboard)

    if is_return_instead_of_print:
        return df_leaderboard, all_crossannotations
    else:
        utils.print_leaderboard(
            df_leaderboard, leaderboard_mode_to_print, cols_to_print=constants.EVALUATORS_LEADERBOARD_COLS_TO_PRINT
        )


ALL_FUNCTIONS = {
    "evaluate": evaluate,
    "evaluate_from_model": evaluate_from_model,
    "make_leaderboard": make_leaderboard,
    "analyze_evaluators": analyze_evaluators,
}


def main():
    is_fn_name = len(sys.argv) > 1 and "--" not in sys.argv[1]
    is_help = any(a == "--help" for a in sys.argv)

    if is_fn_name or is_help:
        fire.Fire(ALL_FUNCTIONS)
    else:
        # default behavior if no function is specified
        fire.Fire(evaluate)


if __name__ == "__main__":
    fire.Fire(ALL_FUNCTIONS)
