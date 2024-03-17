import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import display
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

from alpaca_eval import analyze, annotators, constants, main, metrics, plotting, utils

CURR_DIR = Path(__file__).parent


def get_chatbot_arena_lb_mapping():
    return (
        pd.read_csv(CURR_DIR / "benchmarks.csv", index_col=0)
        .dropna(subset="LC AlpacaEval 2.0")["Arena Elo\n[Feb 2, 2024]"]
        .squeeze()
    ).to_dict()


def logit(p):
    return np.log(p / (1 - p))


def make_data(all_df_annotations, instruction_difficulty=None, baseline="gpt4_1106_preview"):
    df = all_df_annotations.copy()
    df["delta_len"] = df["len_1"] - df["len_2"]

    df["rand_delta_len"] = df["delta_len"].astype(float) + ((np.random.rand(len(df)) - 0.5) * 1e-4)

    group_stats = (
        df.groupby("generator_2")["rand_delta_len"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "group_mean", "std": "group_std"})
    )
    df = df.merge(group_stats, left_on="generator_2", right_index=True)
    df["rand_delta_len_std_only"] = df["rand_delta_len"] / df["group_std"]

    if instruction_difficulty is not None:
        instruction_difficulty = instruction_difficulty.squeeze().sort_index().rename(index="instruction_difficulty")
        instruction_difficulty.index.name = "index"
    else:
        instruction_difficulty = (
            df.groupby("index")["preference"].mean().apply(logit).sort_index().rename(index="instruction_difficulty")
        )
    df["instruction_difficulty"] = df["index"].transform(lambda g: instruction_difficulty[g])

    rows_per_model = {}
    sub_df = instruction_difficulty.to_frame().reset_index(drop=False).copy()
    sub_df["len_1"] = df.drop_duplicates(["index"])["len_1"].loc[sub_df.index]
    all_models = df["generator_2"].unique()
    ordered_models = [baseline] + [m for m in all_models if m != baseline]
    for m in ordered_models:
        rows_per_model[m] = sub_df.copy()
        rows_per_model[m]["generator_2"] = m
        # apply the transformation on the baseline length
        rows_per_model[m]["rand_delta_len_std"] = (
            rows_per_model[m]["len_1"] - group_stats.loc[m, "group_mean"]
        ) / group_stats.loc[m, "group_std"]
    df_lb = pd.concat(rows_per_model.values(), axis=0)
    df_lb["delta_len"] = 0
    df_lb["len_2"] = df_lb["len_1"]
    df_lb["rand_delta_len_std_only"] = 0

    fn_is_gamed_baseline = lambda g: baseline in g and ("verbose" in g or "concise" in g)
    df["not_gamed_baseline"] = ~df["generator_2"].apply(fn_is_gamed_baseline)
    df_lb["not_gamed_baseline"] = True

    return df, df_lb


def load_annotations(lb):
    """Load annotations from models in lb and add some statistics that may be useful."""
    annotations = {}

    for i in lb.index:
        # load actual annotations to see if it was longer or not
        df_annotations = pd.read_json(f"results/{i}/weighted_alpaca_eval_gpt4_turbo/annotations.json")
        df_annotations["len_1"] = df_annotations["output_1"].str.len()
        df_annotations["len_2"] = df_annotations["output_2"].str.len()
        df_annotations["is_longer2"] = df_annotations["len_1"] < df_annotations["len_2"]
        df_annotations["is_longer1"] = df_annotations["len_2"] < df_annotations["len_1"]
        df_annotations["is_same_length"] = df_annotations["len_2"] == df_annotations["len_1"]
        df_annotations["model"] = i
        annotations[i] = df_annotations.reset_index().drop(
            columns=["raw_completion", "output_2", "output_1", "instruction"]
        )  # drop all the long stuff that is not needed

    df_annotations = pd.concat(annotations, ignore_index=True).query("preference >= 0")
    df_annotations["preference"] = (
        df_annotations["preference"].astype(float).replace({0.0: 1.5}) - 1
    )  # easier to work with
    return df_annotations


def print_correlations(arr1, arr2, txt="", is_return_metrics=False):
    if isinstance(arr1, pd.DataFrame):
        arr1 = list(arr1.index)
    if isinstance(arr2, pd.DataFrame):
        arr2 = list(arr2.index)
    s = scipy.stats.spearmanr(arr1, arr2).statistic
    t = scipy.stats.kendalltau(arr1, arr2).statistic

    if is_return_metrics:
        return dict(spearman=s, kendall=t)
    else:
        if txt != "":
            txt = txt + "\n"
        print(f"{txt}Spearman Corr: {s:.3f}\nKendall Corr: {t:.3f}")
