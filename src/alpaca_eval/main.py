import pandas as pd
from typing import Any, Optional, Union
from pathlib import Path
import fire

from .types import AnyPath, AnyData
from . import utils, metrics, annotators

CUR_DIR = Path(__file__).parent
DEFAULT_REFERENCE_OUTPUTS = CUR_DIR/"data/alpaca_farm/gpt3_outputs.json"
DEFAULT_CONFIGS = "alpaca_farm/configs.yaml"
DEFAULT_LEADERBOARD = CUR_DIR/"data/alpaca_farm/pairwise_leaderboard.csv"

ALL_LEADERBOARDS ={
    (str(DEFAULT_REFERENCE_OUTPUTS), str(DEFAULT_CONFIGS)) : DEFAULT_LEADERBOARD,
}
def pairwise_winrates(
    model_outputs: Union[AnyPath, AnyData],
    reference_outputs: Union[AnyPath, AnyData] = DEFAULT_REFERENCE_OUTPUTS,
    annotators_config: AnyPath = DEFAULT_CONFIGS,
    name: str = "Current method",
    is_return_metrics: bool = False,
    rest_of_leaderboard : Optional[Union[str, AnyPath, AnyData]] = "auto" ,
    fn_metric: Union[str, callable] = "pairwise_to_winrate",
    sort_by: str = "win_rate",
    annotation_kwargs : Optional[dict[str, Any]] = None,
    **annotator_kwargs
):
    """

    Parameters
    ----------
    model_outputs : path or data
        The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv). Each dictionary (or row of dataframe) should contain
        the keys that are formatted in the prompts. E.g. by default `instruction` and `output` with optional `input`.

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
        leaderboard = utils.load_or_convert_to_dataframe(rest_of_leaderboard)
    else:
        leaderboard = dict()

    model_outputs = utils.load_or_convert_to_dataframe(model_outputs)
    reference_outputs = utils.load_or_convert_to_dataframe(reference_outputs)

    fn_metric = getattr(metrics, fn_metric, fn_metric)

    annotator = annotators.PairwiseAnnotator(annotators_config=annotators_config, **annotator_kwargs)
    annotated = annotator.annotate_head2head(outputs_1=reference_outputs, outputs_2=model_outputs, **annotation_kwargs)

    leaderboard[name] = fn_metric(preferences=[a["preference"] for a in annotated])
    df_leaderboard = pd.DataFrame(leaderboard).T.sort_values(by=sort_by, ascending=False)

    if is_return_metrics:
        return df_leaderboard
    else:
        print(df_leaderboard.to_string(float_format="%.2f"))

def main(task="pairwise_winrates"):
    fire.Fire(globals()[task])


if __name__ == "__main__":
    main()