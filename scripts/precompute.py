import fire

from alpaca_eval import analyze, annotators, constants
from alpaca_eval import main as alpaca_main
from alpaca_eval import metrics, utils


def precompute_on_all_human_leaderboard(
    annotators_config="gpt4",
    Annotator=annotators.PairwiseAnnotator,
    all_data=constants.ALPACAFARM_GOLD_ANNOTATIONS,
    analyzer_kwargs=None,
    **annotator_kwargs
):
    """Precompute all instructions on the eval leaderboard that has been annotated by humans."""
    analyzer_kwargs = analyzer_kwargs or {}
    analyzer = analyze.Analyzer(gold_annotations=all_data, **analyzer_kwargs)
    df_annotations = analyze.get_annotations(
        analyzer, Annotator=Annotator, annotators_config=annotators_config, **annotator_kwargs
    )


def precompute_evaluator_leaderboard(
    annotators_configs_to_analyze="MINIMAL_EVALUATORS",
    annotators_configs_to_benchmark="VERIFIED_EVALUATORS",
    max_instances=None,
    **kwargs
):
    """Precompute evaluator's leaderboard for important API models."""
    if isinstance(annotators_configs_to_analyze, str):
        annotators_configs_to_analyze = getattr(constants, annotators_configs_to_analyze)

    if isinstance(annotators_configs_to_benchmark, str):
        annotators_configs_to_benchmark = getattr(constants, annotators_configs_to_benchmark)

    for annotators_config in annotators_configs_to_analyze:
        # saving is done automatically
        _ = alpaca_main.analyze_evaluators(
            annotators_config=annotators_config,
            max_instances=max_instances,
            is_save_leaderboard=max_instances is None,
            is_return_instead_of_print=True,  # don't print
            current_leaderboard_mode="minimal",
            **kwargs
        )

    for annotators_config in annotators_configs_to_benchmark:
        # saving is done automatically
        _ = alpaca_main.analyze_evaluators(
            annotators_config=annotators_config,
            max_instances=max_instances,
            is_save_leaderboard=max_instances is None,
            is_return_instead_of_print=True,  # don't print
            is_single_annotator=True,
            current_leaderboard_mode="verified",
            **kwargs
        )


def update_leaderboard(leaderboard_path, model_outputs="results/{model_name}/model_outputs.json", **kwargs):
    """Rerun evaluate on each model in the leaderboard. Useful to add a column suc as avg_length."""
    df_leaderboard = utils.load_or_convert_to_dataframe(leaderboard_path)
    for model_name in df_leaderboard.index:
        alpaca_main.evaluate(model_outputs=model_outputs.format(model_name=model_name), **kwargs)


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
