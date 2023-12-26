import logging
from typing import Sequence, Union

import pandas as pd

from alpaca_eval.utils import validate_alpacaeval_preference


def pairwise_to_winrate(preferences: Union[pd.Series, Sequence]) -> dict[str, int]:
    """Extract head2head metrics (n_wins, n_counts, win_rate) from a sequence preference.
    This assumes that the preference is encoded as 0 or 1.5 for draw, 1 for base win, 2 when the model to compare wins.
    """
    if not isinstance(preferences, pd.Series):
        series_preferences = pd.Series(preferences)
    else:
        series_preferences = preferences.copy()

    # for backward compatibility
    series_preferences[series_preferences == 0] = 1.5

    is_preference = series_preferences.apply(validate_alpacaeval_preference, is_allow_nan=False)
    n_not_pair = sum(~is_preference)
    if n_not_pair > 0:
        logging.info(f"drop {n_not_pair} outputs that are not preferences")
    series_preferences = series_preferences[is_preference].astype(float).copy() - 1

    win_rate = series_preferences.mean()  # takes into account the score
    n_draws = (series_preferences == 1.5).sum()
    n_wins_base = (series_preferences < 1.5).sum()
    n_wins = (series_preferences > 1.5).sum()
    n_total = len(series_preferences)

    discrete_preferences = series_preferences.copy()
    arr_not_draw = discrete_preferences != 0.5
    discrete_preferences[arr_not_draw] = discrete_preferences[arr_not_draw].round()
    discrete_win_rate = discrete_preferences.mean()

    out = dict(
        win_rate=win_rate * 100,
        standard_error=series_preferences.sem() * 100,
        n_wins=n_wins,
        n_wins_base=n_wins_base,
        n_draws=n_draws,
        n_total=n_total,
    )

    if discrete_win_rate != win_rate:
        out["discrete_win_rate"] = discrete_win_rate * 100
        out["discrete_standard_error"] = discrete_preferences.sem() * 100

    return out
