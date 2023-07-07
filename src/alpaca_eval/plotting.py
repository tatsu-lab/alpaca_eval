"""Helpers for plotting."""
import logging
import warnings
from contextlib import contextmanager
from typing import Callable, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import MatplotlibDeprecationWarning
from matplotlib import pyplot as plt
from matplotlib import rc_params_from_file
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from scipy import stats

from . import constants, metrics

RC_IF_NO_FILE = {
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
    "grid.color": "cbcbcb",
    "savefig.dpi": 360,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.0,
    "savefig.transparent": True,
}


@contextmanager
def plot_config(
    style="ticks",
    context="talk",
    palette="colorblind",
    font_scale=1,
    is_ax_off=False,
    is_rm_xticks=False,
    is_rm_yticks=False,
    rc={"lines.linewidth": 4},
    is_use_tex=False,
    set_kwargs=dict(),
    despine_kwargs=dict(),
    file_to_default_rc=None,
    # pretty_renamer=dict(), #TODO
):
    """Temporary seaborn and matplotlib figure style / context / limits / ....

    Parameters
    ----------
    style : dict, None, or one of {darkgrid, whitegrid, dark, white, ticks}
        A dictionary of parameters or the name of a preconfigured set.

    context : dict, None, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.

    palette : string or sequence
        Color palette, see :func:`color_palette`

    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.

    is_ax_off : bool, optional
        Whether to turn off all axes.

    is_rm_xticks, is_rm_yticks : bool, optional
        Whether to remove the ticks and labels from y or x axis.

    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        style dictionaries.

    is_use_tex : bool, optional
        Whether to use tex for the labels.

    set_kwargs : dict, optional
        kwargs for matplotlib axes. Such as xlim, ylim, ...

    despine_kwargs : dict, optional
        Arguments to `sns.despine`.

    file_to_default_rc : str, optional
        Path to a matplotlib rc file to use as default. If not provided, the default rc file is used.
    """
    defaults_rc = plt.rcParams.copy()

    if file_to_default_rc is not None:
        try:
            desired_rc = rc_params_from_file(file_to_default_rc, use_default_template=False).copy()
        except Exception as e:
            if file_to_default_rc is not None:
                logging.warning(f"Could not find {file_to_default_rc}. Error: {e}")
            desired_rc = rc
    else:
        desired_rc = RC_IF_NO_FILE
    desired_rc.update(rc)

    try:
        if is_use_tex:
            desired_rc["text.usetex"] = True
        else:
            desired_rc["text.usetex"] = False

        plt.rcParams.update(desired_rc)

        with sns.axes_style(style=style, rc=desired_rc), sns.plotting_context(
            context=context, font_scale=font_scale, rc=desired_rc
        ), sns.color_palette(palette):
            yield
            last_fig = plt.gcf()
            for i, ax in enumerate(last_fig.axes):
                ax.set(**set_kwargs)

                if is_ax_off:
                    ax.axis("off")

                if is_rm_yticks:
                    ax.axes.yaxis.set_ticks([])

                if is_rm_xticks:
                    ax.axes.xaxis.set_ticks([])

        sns.despine(**despine_kwargs)

    finally:
        with warnings.catch_warnings():
            # filter out depreciation warnings when resetting defaults
            warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
            # reset defaults
            plt.rcParams.update(defaults_rc)


def evaluator_renamer(name):
    if name == "gpt4":
        name = "gpt_b5"
    return name.replace("_basic", "").replace("_", " ").replace("-", " ")


def plot_quality_vs_price_and_time(
    evaluator_leaderboard: pd.DataFrame, min_agreement: float = 0.55, config_kwargs=None, **preprocess_kwargs
):
    df_all = _preprocess_evaluator_leaderboard(evaluator_leaderboard, min_agreement=min_agreement, **preprocess_kwargs)

    df_melted = df_all.melt(
        var_name="Variable",
        value_name="value",
        id_vars=["Annotator", "Human agreement [%]"],
        value_vars=["Price [$/1000 examples]", "Time [seconds/1000 examples]"],
    )

    config_kwargs = config_kwargs or dict()
    with plot_config(**config_kwargs):
        g = sns.relplot(
            data=df_melted,
            x="value",
            col="Variable",
            y="Human agreement [%]",
            kind="scatter",
            hue="Annotator",
            facet_kws={"sharex": False, "sharey": True},
            s=300,
            alpha=0.9,
            legend="full",
        )

        axes = g.axes.flatten()
        g.set_titles("{col_name}")

        axes[0].yaxis.set_major_locator(plt.MaxNLocator(4))
        for ax in axes:
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.set_xlabel(ax.title._text)

        g.set_titles("")
        axes[0].set_xscale("symlog", linthresh=1)
        axes[0].xaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
        axes[0].set_xlim([-0.02, 400])
        axes[1].set_xscale("log")

        sns.move_legend(g, "center right", bbox_to_anchor=(1.05, 0.55))

    plt.show()
    return g


def plot_quality_vs_price(
    evaluator_leaderboard: pd.DataFrame, min_agreement: float = 0.55, config_kwargs=None, **preprocess_kwargs
):
    config_kwargs = config_kwargs or dict()
    df_all = _preprocess_evaluator_leaderboard(evaluator_leaderboard, min_agreement=min_agreement, **preprocess_kwargs)

    with plot_config(**config_kwargs):
        g = sns.relplot(
            data=df_all,
            x="Price [$/1000 examples]",
            y="Human agreement [%]",
            kind="scatter",
            hue="Annotator",
            s=300,
            alpha=0.9,
            legend="full",
            aspect=1.3,
        )

        axes = g.axes.flatten()

        axes[0].yaxis.set_major_locator(plt.MaxNLocator(4))

        g.set_titles("")
        axes[0].set_xscale("symlog", linthresh=1)
        axes[0].set_xlim([-0.02, 400])
        axes[0].xaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))

        sns.move_legend(g, "center right", bbox_to_anchor=(1.05, 0.6))

    plt.show()
    return g


def plot_quality_vs_price(
    evaluator_leaderboard: pd.DataFrame, min_agreement: float = 0.55, config_kwargs=None, **preprocess_kwargs
):
    config_kwargs = config_kwargs or dict()
    df_all = _preprocess_evaluator_leaderboard(evaluator_leaderboard, min_agreement=min_agreement, **preprocess_kwargs)

    with plot_config(**config_kwargs):
        g = sns.relplot(
            data=df_all,
            x="Price [$/1000 examples]",
            y="Human agreement [%]",
            kind="scatter",
            hue="Annotator",
            s=300,
            alpha=0.9,
            legend="full",
            aspect=1.3,
        )

        axes = g.axes.flatten()

        axes[0].yaxis.set_major_locator(plt.MaxNLocator(4))

        g.set_titles("")
        axes[0].set_xscale("symlog", linthresh=1)
        axes[0].set_xlim([-0.02, 400])
        axes[0].xaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))

        sns.move_legend(g, "center right", bbox_to_anchor=(1.05, 0.6))

    plt.show()
    return g


def plot_quality_vs_time(
    evaluator_leaderboard: pd.DataFrame, min_agreement: float = 0.55, config_kwargs=None, **preprocess_kwargs
):
    config_kwargs = config_kwargs or dict()
    df_all = _preprocess_evaluator_leaderboard(evaluator_leaderboard, min_agreement=min_agreement, **preprocess_kwargs)

    with plot_config(**config_kwargs):
        g = sns.relplot(
            data=df_all,
            x="Time [seconds/1000 examples]",
            y="Human agreement [%]",
            kind="scatter",
            hue="Annotator",
            s=300,
            alpha=0.9,
            legend="full",
            aspect=1.3,
        )

        axes = g.axes.flatten()

        axes[0].yaxis.set_major_locator(plt.MaxNLocator(4))

        g.set_titles("")
        axes[0].set_xscale("log")

        sns.move_legend(g, "center right", bbox_to_anchor=(1.05, 0.6))

    plt.show()

    return g


def plot_bias_vs_variance(
    evaluator_leaderboard: pd.DataFrame,
    min_agreement: float = 0.55,
    config_kwargs=dict(is_use_tex=False, palette=sns.color_palette(np.array(sns.color_palette("colorblind"))[1:])),
    **preprocess_kwargs,
):
    config_kwargs = config_kwargs or dict()
    df_all = _preprocess_evaluator_leaderboard(evaluator_leaderboard, min_agreement=min_agreement, **preprocess_kwargs)

    with plot_config(**config_kwargs):
        g = sns.relplot(
            data=df_all.query("Annotator!='humans'"),
            x="Variance",
            y="Bias",
            kind="scatter",
            hue="Annotator",
            s=300,
            alpha=0.9,
            legend="full",
            aspect=1.3,
        )

        axes = g.axes.flatten()
        g.set_titles("")
        plt.axvline(x=df_all.query("Annotator=='humans'")["Variance"].iloc[0], linestyle="--")

        axes[0].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[0].yaxis.set_major_locator(plt.MaxNLocator(5))

        sns.move_legend(g, "center right", bbox_to_anchor=(1.05, 0.6))

    plt.show()
    return g


def plot_all_properties(
    evaluator_leaderboard: pd.DataFrame,
    properties_to_rm: Sequence[str] = ("# parsed",),
    min_agreement: float = 0.55,
    config_kwargs=dict(is_use_tex=False, palette=sns.color_palette(np.array(sns.color_palette("colorblind"))[1:])),
    annotators_to_rm: Sequence[str] = ("longest",),
    **preprocess_kwargs,
):
    properties_to_rm = list(properties_to_rm)
    config_kwargs = config_kwargs or dict()
    annotators_to_keep = [c for c in evaluator_leaderboard.index if c not in annotators_to_rm]
    df_all = _preprocess_evaluator_leaderboard(
        evaluator_leaderboard.drop(columns=properties_to_rm),
        min_agreement=min_agreement,
        annotators_to_keep=annotators_to_keep,
        **preprocess_kwargs,
    )

    df_all["jitter"] = np.random.uniform(-0.5, 0.5, len(df_all))
    df_melted = df_all.melt(var_name="Variable", value_name="value", id_vars=["Annotator", "jitter"])

    with plot_config(**config_kwargs):
        g = sns.relplot(
            data=df_melted.query("Annotator!='humans'"),
            x="value",
            y="jitter",
            kind="scatter",
            row="Variable",
            hue="Annotator",
            facet_kws={"sharex": False, "sharey": True},
            s=300,
            color="grey",
            alpha=0.9,
            legend="full",
            aspect=2.5,
            height=2.5,
        )

        g.set(ylim=[-0.75, 0.75], xlabel="")
        plt.tight_layout()

        axes = g.axes.flatten()
        g.set_titles("{row_name}")

        for ax in axes:
            ax.get_yaxis().set_visible(False)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            # ax.yaxis.set_ticks([])
            # ax.set_ylabel(ax.title._text)

        # g.set_titles("")

        sns.move_legend(g, "center right", bbox_to_anchor=(1.4, 0.6))

    plt.show()
    return g


def plot_winrate_correlations(
    human_leaderboard,
    auto_leaderboard,
    models_to_keep=constants.HUMAN_ANNOTATED_MODELS_TO_KEEP,
    config_kwargs=dict(rc={"lines.linewidth": 2}),
):
    models_to_keep = list(models_to_keep)
    df = pd.merge(
        human_leaderboard["win_rate"],
        auto_leaderboard["win_rate"],
        suffixes=["_human", "_auto"],
        left_index=True,
        right_index=True,
    )
    df = df.loc[models_to_keep]

    df = df.rename(columns=dict(win_rate_human="Human Win Rate", win_rate_auto="Auto Win Rate"))
    with plot_config(**config_kwargs):
        g = sns.lmplot(data=df, y="Human Win Rate", x="Auto Win Rate")

        axes = g.axes.flatten()
        axes[0].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[0].yaxis.set_major_locator(plt.MaxNLocator(6))

        def annotate(data, **kwargs):
            s = scipy.stats.spearmanr(data["Human Win Rate"], data["Auto Win Rate"]).statistic
            r, _ = scipy.stats.pearsonr(data["Human Win Rate"], data["Auto Win Rate"])
            ax = plt.gca()
            ax.text(0.05, 0.92, r"Spearman corr: {:.2f}".format(s), transform=ax.transAxes, fontsize=14)
            ax.text(0.05, 0.84, "Pearson corr: {:.2f}".format(r), transform=ax.transAxes, fontsize=14)

        g.map_dataframe(annotate)

    plt.show()
    return g


def save_fig(fig, filename, dpi=300, is_tight=True):
    """General function for saving many different types of figures."""

    # order matters ! and don't use elif!
    if isinstance(fig, sns.FacetGrid):
        fig = fig.fig

    if isinstance(fig, matplotlib.artist.Artist):  # any type of axes
        fig = fig.get_figure()

    if isinstance(fig, matplotlib.figure.Figure):
        plt_kwargs = {}
        if is_tight:
            plt_kwargs["bbox_inches"] = "tight"

        fig.savefig(filename, dpi=dpi, **plt_kwargs)
        plt.close(fig)
    else:
        raise ValueError(f"Unknown figure type {type(fig)}")


def plot_paired_ttests(df):
    df_ttest = _get_ttest_df(df)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
    with plot_config(font_scale=0.4):
        g = sns.heatmap(
            df_ttest.astype(float),
            annot=True,
            fmt=".2f",
            cbar=False,
            square=True,
            xticklabels=False,
            ax=ax,
            mask=np.triu(np.ones_like(df_ttest, dtype=bool)),
            cmap=sns.color_palette("rocket", as_cmap=True),
        )
        g.set(xlabel="", ylabel="")
    plt.show()
    return g


def plot_paired_ttests_per_dataset(df, is_print_values=False, is_add_alpaca_eval=False):
    min_dataset_size = df.drop_duplicates("instruction").groupby("dataset")["instruction"].count().min()

    all_pvalues = dict()
    for d in df["dataset"].unique():
        df_sub = df.query(f"dataset=='{d}'")
        all_pvalues[d] = _get_ttest_df(df_sub, n_samples=min_dataset_size)

    if is_add_alpaca_eval:
        all_pvalues["AlpacaEval"] = _get_ttest_df(df, n_samples=min_dataset_size)

    if is_print_values:
        for i, (key, curr_df) in enumerate(all_pvalues.items()):
            print(key, f"mean p-val: {curr_df.mean(axis=None):.3f}", f"max p-val: {curr_df.max(axis=None):.3f}")

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(23, 15))

    with plot_config(font_scale=0.5):
        for i, (key, curr_df) in enumerate(all_pvalues.items()):
            ax = axes[i // 3][i % 3]
            g = sns.heatmap(
                curr_df,
                annot=True,
                fmt=".2f",
                cbar=False,
                square=True,
                xticklabels=False,
                ax=ax,
                mask=np.triu(np.ones_like(curr_df, dtype=bool)),
            )
            ax.set_title(key + f" n={min_dataset_size}", fontsize=20)
            g.set(xlabel="", ylabel="")

        for i in range(len(all_pvalues), axes.size):
            ax = axes.flatten()[i]
            ax.set_visible(False)

        # adjust spacing between plots
        plt.tight_layout()

    plt.show()
    return g


def plot_paired_ttests_pvalues(df):
    df_ttest = _get_ttest_df(df)
    all_sub_ttest_df = {
        n: _get_ttest_df(df, n_samples=n, random_state=123, sorted_idx=list(df_ttest.index))
        for n in range(50, len(df["instruction"].unique()), 50)
    }

    df_describe = pd.DataFrame(
        {
            "mean": {k: v.mean(axis=None) for k, v in all_sub_ttest_df.items()},
            "90% quantile": {k: v.stack().quantile(q=0.9) for k, v in all_sub_ttest_df.items()},
            "max": {k: v.max(axis=None) for k, v in all_sub_ttest_df.items()},
        }
    )

    melted = df_describe.melt(ignore_index=False, value_name="p-value", var_name="aggregator").reset_index(
        names="# samples"
    )

    with plot_config(rc={"lines.linewidth": 4, "axes.grid": False}):
        ax = sns.lineplot(melted, x="# samples", y="p-value", hue="aggregator")

        ax.axhline(y=0.05, color="black", linestyle="--", linewidth=2, alpha=0.5)

        # Get the handles and labels from the existing line plot legend
        handles, labels = ax.get_legend_handles_labels()

        # Create a new legend element for the horizontal line
        legend_elements = [Line2D([0], [0], color="black", linestyle="--", label="0.05")]

        # Combine the handles, labels, and new legend element
        all_handles = handles + legend_elements
        all_labels = labels + ["0.05"]

        # Plot the combined legend
        ax.legend(handles=all_handles, labels=all_labels)
    plt.show()
    return ax


def plot_paired_ttest_nsamples(df):
    df_ttest = _get_ttest_df(df)
    all_sub_ttest_df = {
        n: _get_ttest_df(df, n_samples=n, random_state=123, sorted_idx=list(df_ttest.index))
        for n in range(50, len(df["instruction"].unique()), 50)
    }

    arr_min_samples = np.minimum.reduce([np.where(v < 0.05, k, float("inf")) for k, v in all_sub_ttest_df.items()])
    arr_min_samples[np.isinf(arr_min_samples)] = np.nan
    df_min_samples = pd.DataFrame(arr_min_samples, index=df_ttest.index, columns=df_ttest.index)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
    with plot_config(font_scale=0.55):
        sns.heatmap(
            df_min_samples.isnull(),
            cbar=False,
            color="black",
            alpha=0.5,
            mask=~df_min_samples.isnull() | np.triu(np.ones_like(df_ttest, dtype=bool), k=0),
        )
        g = sns.heatmap(
            df_min_samples,
            annot=True,
            fmt=".0f",
            cbar=False,
            square=True,
            xticklabels=False,
            ax=ax,
            vmin=0,
            vmax=1000,
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            mask=np.triu(np.ones_like(df_ttest, dtype=bool)),
        )

        g.set(xlabel="", ylabel="")

    plt.show()

    return g


##########
def _preprocess_evaluator_leaderboard(
    evaluator_leaderboard: pd.DataFrame,
    min_agreement: float = 0.55,
    annotators_to_keep: Sequence[str] = constants.VERIFIED_EVALUATORS,
    evaluator_renamer: Optional[Callable] = evaluator_renamer,
    is_human_at_top: bool = True,
) -> pd.DataFrame:
    df_all = evaluator_leaderboard.copy()
    annotators_to_keep = [evaluator_renamer(a) for a in annotators_to_keep]

    if evaluator_renamer is not None:
        df_all.index = [evaluator_renamer(i) for i in df_all.index]

    df_all["Annotator"] = df_all.index
    df_all = df_all.query("Annotator.isin(@annotators_to_keep)")

    # select only useful
    df_all = df_all[df_all["Human agreement [%]"] > min_agreement]

    if is_human_at_top and "humans" in df_all.index:
        # puts humans at the top (easier for colors)
        idcs = list(df_all.index)
        idx_humans = idcs.index("humans")
        idcs_reordered = [idx_humans] + list(range(0, idx_humans)) + list(range(idx_humans + 1, len(idcs)))
        df_all = df_all.iloc[idcs_reordered]

    return df_all


def _pairwise_ttest(df):
    p_values = pd.DataFrame(index=df.columns, columns=df.columns)

    for i in df.columns:
        for j in df.columns:
            if i == j:
                p_values.loc[i, j] = np.nan
            else:
                t_stat, p_val = stats.ttest_rel(df[i], df[j], nan_policy="omit")
                p_values.loc[i, j] = p_val

    return p_values


def _get_ttest_df(df, n_samples=None, random_state=123, sorted_idx=None):
    """return a dataframe of pairwise relative ttest with potential subsampling"""
    df_pivoted = df.pivot(index="instruction", values="preference", columns=["generator_2"])
    if n_samples is not None:
        df_pivoted = df_pivoted.sample(n=n_samples, random_state=random_state)
    # win_rate = metrics.pairwise_to_winrate(df["preference"])['win_rate']
    if sorted_idx is None:
        sorted_idx = list(
            df.groupby("generator_2")["preference"]
            .apply(lambda x: metrics.pairwise_to_winrate(x)["win_rate"])
            .sort_values(ascending=False)
            .index
        )
    return _pairwise_ttest(df_pivoted[sorted_idx].replace({0: 1, 1: 0})).astype(float)  # draw is 0 but to test order it
    # should be in the middle
