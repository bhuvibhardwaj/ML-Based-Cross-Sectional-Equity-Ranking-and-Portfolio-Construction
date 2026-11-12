"""
Factor scoring model: the transparent pre-ML baseline.

The composite factor score is a linear combination of normalized characteristics
with pre-specified direction weights. These weights are NOT estimated from data.
They encode the a priori direction of each factor premium, grounded in academic
literature.

Mathematical definition (see docs/methodology.md, Section 3):

    S_{i,t} = sum_k alpha_k * z_{i,t,k}

where z_{i,t,k} is the cross-sectionally normalized feature k for stock i at date t
and alpha_k in {-1, 0, +1} is the pre-specified direction weight.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factor direction weights
# ---------------------------------------------------------------------------

# Pre-specified direction weights for each feature.
# These are chosen based on academic literature and must NOT be changed
# to improve historical performance.
#
# Sign conventions:
#   +1: Higher feature value -> higher expected return (bullish)
#   -1: Higher feature value -> lower expected return (bearish)
#
# References:
#   MOM_12_1 (+1): Jegadeesh & Titman (1993)
#   REV_1 (-1):    Short-term reversal (Lehmann 1990); recent losers -> winners
#   VOL_12M (-1):  Low-volatility anomaly; Ang et al. (2006), Frazzini & Pedersen (2014)
#   TURN_1M (+1):  Prefer liquid stocks for practical portfolio construction
#   BETA_12M (-1): Betting Against Beta anomaly; Frazzini & Pedersen (2014)

FACTOR_WEIGHTS: dict[str, float] = {
    'MOM_12_1': +1.0,
    'REV_1':    -1.0,
    'VOL_12M':  -1.0,
    'TURN_1M':  +1.0,
    'BETA_12M': -1.0,
}


def compute_factor_scores(
    feature_dict: dict[str, pd.DataFrame],
    weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute the composite factor score for every (date, ticker) pair.

    The composite score is:
        S_{i,t} = sum_k [w_k * z_{i,t,k}]

    where z_{i,t,k} is the normalized (winsorized + z-scored) feature k
    and w_k is the direction weight.

    Features with w_k = 0 are excluded from the score.

    Parameters
    ----------
    feature_dict : dict[str, pd.DataFrame]
        Mapping from feature name to normalized feature DataFrame.
        Each DataFrame has shape (n_dates, n_tickers).
    weights : dict[str, float], optional
        Direction weights. Defaults to FACTOR_WEIGHTS defined above.
        Must not be modified to exploit historical data.

    Returns
    -------
    pd.DataFrame
        Composite score; shape (n_dates, n_tickers).
        Higher score -> stock ranked more favorably by the factor model.
    """
    if weights is None:
        weights = FACTOR_WEIGHTS

    # Validate that we have features for at least some weight keys
    available = set(feature_dict.keys()) & set(weights.keys())
    if not available:
        raise ValueError(
            f"No overlap between available features {list(feature_dict.keys())} "
            f"and weight keys {list(weights.keys())}."
        )

    missing = set(weights.keys()) - set(feature_dict.keys())
    if missing:
        logger.warning("Features not available for scoring: %s. Scores will be partial.", missing)

    # Initialize score with zeros, using the shape of the first available feature
    first_feat = next(iter(feature_dict.values()))
    score = pd.DataFrame(0.0, index=first_feat.index, columns=first_feat.columns)

    # Track how many features contribute to each (date, ticker) cell
    n_contributors = pd.DataFrame(0, index=first_feat.index, columns=first_feat.columns)

    for feature_name, weight in weights.items():
        if feature_name not in feature_dict:
            continue
        if weight == 0:
            continue

        feat = feature_dict[feature_name].reindex(index=first_feat.index, columns=first_feat.columns)

        # Add weight * feature value; NaN contributes 0 (effectively ignored)
        score += weight * feat.fillna(0)
        # Count valid (non-NaN) contributions
        n_contributors += feat.notna().astype(int)

    # Where fewer than 2 features contributed, set score to NaN
    # (too few signals to form a meaningful composite)
    score = score.where(n_contributors >= 2)

    logger.info(
        "Composite factor scores computed. Non-null: %.1f%% of cells.",
        score.notna().mean().mean() * 100,
    )

    return score


def top_n_selection(
    score_matrix: pd.DataFrame,
    n_top: int = 20,
    min_stocks: int = 10,
) -> dict[pd.Timestamp, frozenset]:
    """
    For each date, select the top-n stocks by composite score.

    Parameters
    ----------
    score_matrix : pd.DataFrame
        Rows = dates, columns = tickers; values = composite scores.
    n_top : int
        Number of stocks to select per date.
    min_stocks : int
        Minimum eligible stocks (with non-NaN score) required to form a portfolio.
        If fewer available, that date is skipped (returns empty set).

    Returns
    -------
    dict mapping date -> frozenset of selected ticker symbols.
    """
    selections: dict[pd.Timestamp, frozenset] = {}

    for date, row in score_matrix.iterrows():
        eligible = row.dropna()
        if len(eligible) < min_stocks:
            logger.debug("Date %s: only %d eligible stocks (< %d min), skipping.", date, len(eligible), min_stocks)
            selections[date] = frozenset()
            continue

        top = eligible.nlargest(n_top)
        selections[date] = frozenset(top.index)

    n_valid = sum(1 for v in selections.values() if len(v) > 0)
    logger.info("Stock selection complete: %d/%d dates have valid portfolios.", n_valid, len(selections))
    return selections


def factor_portfolio_returns(
    selections: dict[pd.Timestamp, frozenset],
    monthly_returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute the equal-weighted portfolio return for each date.

    R_{P,t+1} = (1/n) * sum_{i in top-n_t} R_{i,t+1}

    Parameters
    ----------
    selections : dict
        Date -> frozenset of selected tickers (from top_n_selection).
    monthly_returns : pd.DataFrame
        Next-month returns; rows=dates, columns=tickers.

    Returns
    -------
    pd.Series
        Portfolio returns indexed by formation date.
        Note: the return at formation date t is the return in the NEXT month.
    """
    portfolio_rets = {}
    dates = sorted(selections.keys())

    for i, date in enumerate(dates):
        selected = selections[date]
        if not selected:
            portfolio_rets[date] = np.nan
            continue

        # Get the NEXT date's return (this is the holding period return)
        # The return at formation date t is realized in month t+1
        # In our monthly panel, monthly_returns.loc[t] is the return FOR month t
        # (i.e., from t-1 to t). We need the return for the NEXT month.
        # Find the next date in the returns index.
        returns_dates = monthly_returns.index
        idx = returns_dates.searchsorted(date)
        if idx + 1 >= len(returns_dates):
            portfolio_rets[date] = np.nan
            continue

        next_date = returns_dates[idx + 1]
        next_returns = monthly_returns.loc[next_date]

        # Only include tickers that have return data
        valid_selected = [t for t in selected if t in next_returns.index and not np.isnan(next_returns[t])]
        if not valid_selected:
            portfolio_rets[date] = np.nan
            continue

        portfolio_rets[date] = next_returns[valid_selected].mean()

    return pd.Series(portfolio_rets, name='factor_portfolio_return').dropna()
