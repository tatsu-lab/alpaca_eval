import logging
from pathlib import Path

from alpaca_eval.constants import MODELS_CONFIG_DIR, PRECOMPUTED_LEADERBOARDS
from alpaca_eval.utils import load_configs, load_or_convert_to_dataframe

CURRENT_DIR = Path(__file__).parents[1]
RESULTS_DIR = CURRENT_DIR / "results"

for leaderboard_file in PRECOMPUTED_LEADERBOARDS.values():
    df = load_or_convert_to_dataframe(leaderboard_file)
    df["link"] = ""
    df["samples"] = ""
    df = df[["win_rate", "avg_length", "link", "samples", "mode"]]
    df = df.rename(columns={"mode": "filter"})
    df = df.reset_index(names="name")
    for idx in range(len(df)):
        informal_name = df.loc[idx, "name"]
        try:
            model_config = load_configs(df.loc[idx, "name"], relative_to=MODELS_CONFIG_DIR)[informal_name]
        except KeyError as e:
            logging.exception(
                f"Could not find model config for {informal_name}. This is likely because the name of "
                f"the annotator does not match the name of the model's directory."
            )
            raise e

        if "pretty_name" in model_config:
            df.loc[idx, "name"] = model_config["pretty_name"]

        if "link" in model_config:
            df.loc[idx, "link"] = model_config["link"]

        file_outputs = RESULTS_DIR / informal_name / "model_outputs.json"
        if file_outputs.is_file():
            df.loc[
                idx, "samples"
            ] = f"https://github.com/tatsu-lab/alpaca_eval/blob/main/results/{informal_name}/model_outputs.json"
    df = df.sort_values(by=["win_rate"], ascending=False)
    save_dir = Path("docs") / leaderboard_file.parent.name
    save_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(save_dir / leaderboard_file.name, index=False)
