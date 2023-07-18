from pathlib import Path

import pandas as pd

F_OUTPUTS = "model_outputs.json"
CURRENT_DIR = Path(__file__).parents[1]
RESULTS_DIR = CURRENT_DIR / "results"


df_reference = pd.read_json(RESULTS_DIR / "text_davinci_003" / F_OUTPUTS, orient="records")


# Create a dict mapping each instruction in df_reference to its index => will keep that order for the other files
order = {tuple(pair): i for i, pair in enumerate(zip(df_reference["dataset"], df_reference["instruction"]))}

for f in RESULTS_DIR.glob(f"*/{F_OUTPUTS}"):
    df = pd.read_json(f, orient="records")
    if len(df_reference) != len(df):
        raise ValueError(f"Length of {f} is not equal to the reference file {len(df_reference)}!={len(df)}.")

    # Sort the df using the reference df
    df["order"] = df.apply(lambda row: order[(row["dataset"], row["instruction"])], axis=1)
    df = df.sort_values("order").drop("order", axis=1)

    df.to_json(f, orient="records", indent=2)
