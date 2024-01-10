import json
from pathlib import Path

import pandas as pd

F_OUTPUTS = "model_outputs.json"
F_ANNOTATIONS = "annotations.json"
CURRENT_DIR = Path(__file__).parents[1]
RESULTS_DIR = CURRENT_DIR / "results"


def json_load(el):
    """Try to load as json"""
    try:
        return json.loads(el)
    except:
        return el


def add_referenced_model_(df):
    """Add a dictionary to better understand chain of thought in case it's useful"""

    for i, r in df.iterrows():
        if (
            isinstance(r["raw_completion"], dict)
            and "concise_explanation" in r["raw_completion"]
            and "ordered_models" in r["raw_completion"]
        ):
            preference = int(df.loc[i, "preference"])
            ordered_models = df.loc[i, "raw_completion"]["ordered_models"]
            for m in ordered_models:
                if m["rank"] == 1:
                    first_model = m["model"]
                elif m["rank"] == 2:
                    second_model = m["model"]
                else:
                    assert False

            if "referenced_models" not in df.columns:
                df["referenced_models"] = None

            df.at[i, "referenced_models"] = {
                first_model: f"output_{preference}",
                second_model: f"output_{3-preference}",
            }


df_reference = pd.read_json(RESULTS_DIR / "text_davinci_003" / F_OUTPUTS, orient="records")


# Create a dict mapping each instruction in df_reference to its index => will keep that order for the other files
order = {tuple(pair): i for i, pair in enumerate(zip(df_reference["dataset"], df_reference["instruction"]))}

for f in RESULTS_DIR.glob(f"*/*/{F_OUTPUTS}"):
    df = pd.read_json(f, orient="records")
    if len(df_reference) != len(df):
        raise ValueError(f"Length of {f} is not equal to the reference file {len(df_reference)}!={len(df)}.")

    # Sort the df using the reference df
    df["order"] = df.apply(lambda row: order[(row["dataset"], row["instruction"])], axis=1)
    df = df.sort_values("order").drop("order", axis=1)

    df.to_json(f, orient="records", indent=2)

for f in RESULTS_DIR.glob(f"*/*/{F_ANNOTATIONS}"):
    df = pd.read_json(f, orient="records")
    if len(df_reference) != len(df):
        raise ValueError(f"Length of {f} is not equal to the reference file {len(df_reference)}!={len(df)}.")

    # can't sort because you don't have the dataset
    # df["order"] = df.apply(lambda row: order[(row["dataset"], row["instruction"])], axis=1)
    # df = df.sort_values("order").drop("order", axis=1)

    # jsonify & add the referenced models
    if "raw_completion" in df:
        df["raw_completion"] = df["raw_completion"].apply(json_load)
        add_referenced_model_(df)

    df.to_json(f, orient="records", indent=2)
