import os
from typing import Any, Union

import matplotlib.pyplot as plt
import seaborn as sns


def save_fig(fig: Any, filename: Union[str, bytes, os.PathLike], dpi: int = 300, is_tight: bool = True) -> None:
    """General function for many different types of figures."""

    # order matters ! and don't use elif!
    if isinstance(fig, sns.FacetGrid):
        fig = fig.fig

    if isinstance(fig, plt.Artist):  # any type of axes
        fig = fig.get_figure()

    if isinstance(fig, plt.Figure):
        plt_kwargs = {}
        if is_tight:
            plt_kwargs["bbox_inches"] = "tight"

        fig.savefig(filename, dpi=dpi, **plt_kwargs)
        plt.close(fig)
    else:
        raise ValueError(f"Unknown figure type {type(fig)}")
