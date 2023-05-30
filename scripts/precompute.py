from alpaca_eval import utils, metrics, annotators, constants, analyze, main as alpaca_main
import fire


def precompute_human_leaderboard(annotators_config=f"claude/basic_configs.yaml",
                                 Annotator=annotators.PairwiseAnnotator,
                                 all_data=analyze.DEFAULT_GOLD_ANNOTATIONS,
                                 analyzer_kwargs=None,
                                 **annotator_kwargs):
    """Precompute all instructions on the eval leaderbaord that has been annotated by humans."""
    analyzer_kwargs = analyzer_kwargs or {}
    analyzer = analyze.Analyzer(gold_annotations=all_data, **analyzer_kwargs)
    df_annotations = analyze.get_annotations(analyzer,
                                             Annotator=Annotator,
                                             annotators_config=annotators_config,
                                             **annotator_kwargs)


def precompute_evaluator_leaderboard(annotators_configs="API_EVALUATORS_TO_ANALYZE",
                                     max_instances=None, **kwargs):
    """Precompute evaluator's leaderboard for important API models."""
    if isinstance(annotators_configs, str):
        annotators_configs = getattr(constants, annotators_configs)

    for annotators_config in annotators_configs:
        # saving is done automatically
        _ = alpaca_main.analyze_evaluators(annotators_config=annotators_config,
                                           max_instances=max_instances,
                                           is_save_leaderboard=max_instances is not None,
                                           is_return_metrics=True,  # don't print
                                           **kwargs)


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
