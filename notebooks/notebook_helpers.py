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
from alpaca_eval.metrics.glm_winrate import fit_LogisticRegressionCV, logloss, make_dmatrix_for_model

CURR_DIR = Path(__file__).parent
BASELINE = "gpt4_1106_preview"


def get_chatbot_arena_lb_mapping():
    return (
        pd.read_csv(CURR_DIR / "benchmarks.csv", index_col=0)
        .dropna(subset="LC AlpacaEval 2.0")
        .dropna(subset="Arena Elo\n[Feb 2, 2024]")["Arena Elo\n[Feb 2, 2024]"]
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
        df_annotations["generator_2"] = i  # in case the annotations are not in the right format
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


def process_gamed_models_(lb):
    game_process_v = lambda s: s.replace("_verbose", "")
    game_process_c = lambda s: s.replace("_concise", "")
    gamed_models = [i for i in lb.index if (i + "_verbose") in lb.index and (i + "_concise") in lb.index]
    lb["gamed_verbose_only"] = [game_process_v(i) if game_process_v(i) in gamed_models else None for i in lb.index]
    lb["gamed_concise_only"] = [game_process_c(i) if game_process_c(i) in gamed_models else None for i in lb.index]
    return lb


def make_lb_arena(lb):
    dict_arena = get_chatbot_arena_lb_mapping()
    dict_arena = {k: v for k, v in dict_arena.items() if k in lb.index}
    lb_arena = lb.loc[list(dict_arena.keys()), :]
    lb_arena["ELO"] = dict_arena.values()
    return lb_arena


def report(lb, metric, is_detailed=False, n_toshow=10, is_return_metrics=False):
    lb_arena = make_lb_arena(lb)

    if not is_return_metrics:
        print(f"# Report for **{metric}**")

        print()
        print("## Gameability (lower is better)")

    df_gamed_v = lb.groupby("gamed_verbose_only")[["avg_length", metric]].agg(["mean", "std"])
    df_gamed_c = lb.groupby("gamed_concise_only")[["avg_length", metric]].agg(["mean", "std"])
    # relative in the sense that models with larger metric shouldn't be considered as having larger vairance
    df_gamed_v[(metric, "rel_std")] = df_gamed_v[metric]["std"] / df_gamed_v[metric]["mean"]
    df_gamed_c[(metric, "rel_std")] = df_gamed_c[metric]["std"] / df_gamed_c[metric]["mean"]
    # renormalize to avoid removing gameability by shrinking the scale of the metric
    diff_models = [i for i in lb.index if "_verbose" not in i and i + "_concise" not in i]
    winrate_std_across_models = lb[lb.index.isin(diff_models)]["win_rate"].std()
    metric_std_across_models = lb[lb.index.isin(diff_models)][metric].std()
    metric_weight = winrate_std_across_models / metric_std_across_models

    if is_detailed:
        print(f"metric_weight: {metric_weight:.3f}")
        display(df_gamed_v)
        display(df_gamed_c)

    verbosity_gameability = df_gamed_v[metric]["rel_std"].mean() * metric_weight * 100
    conciseness_gameability = df_gamed_c[metric]["rel_std"].mean() * metric_weight * 100

    adversarial_winrate_gain = lb.loc["gpt4_gamed", metric] - lb.loc["gpt4_gamed", "win_rate"]

    rank_by_metric = lb[metric].rank(method="min", ascending=False)  # Adjust ascending as needed
    rank_by_win_rate = lb["win_rate"].rank(method="min", ascending=False)  # Adjust ascending as needed
    adversarial_rank_gain = rank_by_win_rate.loc["gpt4_gamed"] - rank_by_metric.loc["gpt4_gamed"]

    if not is_return_metrics:
        print(f"Verbosity gameability (relative std metric): {verbosity_gameability:.1f}%")
        print(f"Conciseness gameability (relative std metric): {conciseness_gameability:.1f}%")
        print(f"Adversarial winrate gain: {adversarial_winrate_gain:.1f}")
        print(f"Adversarial rank gain: {adversarial_rank_gain}")

        print()
        print("## Correlation with Arena (higher is better)")

    corr_arena = print_correlations(lb_arena[metric], lb_arena["ELO"], is_return_metrics=is_return_metrics)

    if not is_return_metrics:
        print()
        arena_corr = print_correlations(
            lb_arena["ELO"], lb_arena["avg_length"], "Arena vs Length", is_return_metrics=True
        )

        print(
            f"## Correlation with length (closer to spearman={arena_corr['spearman']:.2f}, kendall={arena_corr['kendall']:.2f} is better)"
        )

    corr_len = print_correlations(lb_arena[metric], lb_arena["avg_length"], is_return_metrics=is_return_metrics)

    if not is_return_metrics:
        print()
        print(f"## Top {n_toshow} models")

        display(lb[metric].sort_values(ascending=False)[:n_toshow])

        print()
        print(f"## Bottom {n_toshow} models")

        display(lb[metric].sort_values(ascending=False)[-n_toshow:])

    if is_return_metrics:
        return dict(
            verbosity_gameability=verbosity_gameability,
            conciseness_gameability=conciseness_gameability,
            adversarial_rank_gain=adversarial_rank_gain,
            adversarial_winrate_gain=adversarial_winrate_gain,
            corr_arena=corr_arena["spearman"],
            corr_len=corr_len["spearman"],
        )


def regression_report(y_true, y_pred):
    return dict(
        logloss=logloss(y_true, y_pred),
        mse=mean_squared_error(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
        corr=pearsonr(y_true, y_pred).statistic,
        acc=accuracy_score(np.round(y_true).astype(int), np.round(y_pred).astype(int)),
    )


def disjoint_optimization_(lb, df, df_lb, formula, regularize_to_baseline_lambda=None, **kwargs):
    all_reports = dict()
    all_models = dict()
    curr_df_lb = df_lb.copy()

    for m in df["generator_2"].unique():
        df_gamed = df.query(f"~`not_gamed_baseline`")
        df_m = df.query(f"`generator_2` == '{m}'").copy()
        df_m["not_gamed_baseline"] = True  # need to reset in case the current is gamed
        df_gamed_and_m = pd.concat([df_gamed, df_m], axis=0)
        curr_df_lb_m = curr_df_lb.query(f"`generator_2` == '{m}'")
        df_input, df_input_lb = make_dmatrix_for_model(df_gamed_and_m, curr_df_lb_m, formula=formula)

        df_input_only_m = df_input[df_gamed_and_m["not_gamed_baseline"]]

        if regularize_to_baseline_lambda:
            # divided by 2 becasue there are two gamed baselines.
            sample_weight = (df_gamed_and_m["not_gamed_baseline"]).astype(float) + (
                regularize_to_baseline_lambda * (~df_gamed_and_m["not_gamed_baseline"])
            ).astype(float) / 2
        else:
            sample_weight = None
            df_input = df_input_only_m

        model = fit_LogisticRegressionCV(
            df_input, "preference", is_ytrue_proba=True, n_splits=5, sample_weight=sample_weight, **kwargs
        )

        if m == BASELINE:
            # by definition (we shoudln't actually fit it, there's one dof too much)
            model.coef_ *= 0

        all_models[m] = model

        all_reports[m] = regression_report(
            df_input_only_m["preference"], model.predict_proba(df_input_only_m.drop(columns=["preference"]))[:, 1]
        )
        curr_df_lb.loc[curr_df_lb["generator_2"] == m, "preference"] = model.predict_proba(df_input_lb)[:, 1]

    lb[formula] = curr_df_lb.groupby("generator_2")["preference"].mean()[lb.index] * 100
    metrics = report(lb, formula, is_return_metrics=True)
    metrics.update(pd.DataFrame(all_reports).T.mean().to_dict())
    return metrics, all_models
