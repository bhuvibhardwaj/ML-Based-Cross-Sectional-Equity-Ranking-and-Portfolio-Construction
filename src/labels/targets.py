"""
Target construction module for ML-Based Cross-Sectional Equity Ranking (`src/labels/targets.py`).

Kept strictly separate from `src/features/engineering.py`.
Constructs:
  1. Continuous next-month excess return targets (`NEXT_MONTH_EXCESS_RETURN`).
  2. Cross-sectional percentile ranks in `[0, 1]` per formation month (`CS_PERCENTILE_RANK`).
  3. Learning-to-Rank (LTR) relevance grades (`0..4` and `0..18`) and query group structures
     (`group` and `qid`) for XGBoost `rank:pairwise`, where higher relevance labels strictly
     represent better future stock performance, calibrated using only training-period information.
  4. Pairwise preference labels P(stock_i > stock_j | month t) and return margins.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

TARGET_COL = "NEXT_MONTH_EXCESS_RETURN"


def construct_Targets_from_prices(
    monthly_prices: pd.DataFrame,
    monthly_rf_rate: float = (1.0 + 0.062) ** (1.0 / 12.0) - 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct next-month simple return and excess return panels from month-end prices.

    At formation date t, the target is the return realized over (t, t+1]:
        R_{i, t+1} = P_{i, t+1} / P_{i, t} - 1
        y_{i, t+1} = R_{i, t+1} - r_{f, t}
    """
    monthly_rets = monthly_prices.pct_change(1)
    next_month_ret = monthly_rets.shift(-1)
    next_month_excess = next_month_ret - monthly_rf_rate
    return next_month_ret, next_month_excess


def compute_cross_sectional_percentile_ranks(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    date_col: str = "date",
) -> pd.Series:
    """
    Compute cross-sectional percentile rank in [0.0, 1.0] within each formation month t.
    Higher percentile (1.0) = highest future return in that month; 0.0 = lowest future return.
    """
    def _pct_rank(s: pd.Series) -> pd.Series:
        n = len(s)
        if n <= 1:
            return pd.Series(0.5, index=s.index)
        ranks = s.rank(method="average", ascending=True)
        return (ranks - 1.0) / (n - 1.0)

    return df.groupby(date_col, group_keys=False)[target_col].apply(_pct_rank)


def fit_training_relevance_bins(
    y_train_or_df: Union[np.ndarray, pd.Series, pd.DataFrame],
    target_col: str = TARGET_COL,
    n_bins: int = 5,
) -> np.ndarray:
    """
    Estimate interior quantile bin edges strictly on the Training Split (Feb 2016 – Dec 2020).

    Returns (n_bins - 1) interior quantile cutoffs so `np.digitize(y, cutoffs)` produces
    integer relevance grades in {0, 1, ..., n_bins - 1} where higher grades strictly
    represent higher future stock returns.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if isinstance(y_train_or_df, pd.DataFrame):
        values = np.asarray(y_train_or_df[target_col].values, dtype=np.float64)
    else:
        values = np.asarray(y_train_or_df, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return edges


def assign_relevance_grades(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    date_col: str = "date",
    scheme: str = "train_quantile_5",
    train_edges: Optional[np.ndarray] = None,
    train_bin_edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Assign non-negative integer relevance labels for XGBoost Learning-to-Rank (`rank:pairwise`).
    Higher relevance integer strictly indicates better subsequent stock performance.

    Supported schemes:
      - 'train_quantile_5': 5 relevance grades {0, 1, 2, 3, 4} using quantile thresholds
                            fit strictly on training-period excess returns (`train_edges`).
      - 'cs_quintile_5':    5 relevance grades {0, 1, 2, 3, 4} assigned via within-month
                            cross-sectional quintiles.
      - 'cs_ordinal_19':    Exact within-month ordinal rank {0, 1, ..., N_t - 1} (0=worst, 18=best).
    """
    edges = train_edges if train_edges is not None else train_bin_edges
    if scheme == "train_quantile_5":
        if edges is None:
            raise ValueError("train_edges / train_bin_edges must be provided for scheme='train_quantile_5'")
        grades = np.digitize(df[target_col].values, bins=edges, right=False).astype(np.int32)
        return grades

    if scheme == "cs_quintile_5":
        pct = compute_cross_sectional_percentile_ranks(df, target_col=target_col, date_col=date_col)
        grades = np.minimum(np.floor(pct.values * 5.0), 4.0).astype(np.int32)
        return grades

    if scheme == "cs_ordinal_19":
        ranks = (
            df.groupby(date_col)[target_col]
            .rank(method="first", ascending=True)
            .values
            .astype(np.int32)
            - 1
        )
        return ranks

    raise ValueError(f"Unknown relevance label scheme: {scheme}")


def verify_higher_label_means_better_return(
    df: pd.DataFrame,
    label_col: str,
    target_col: str = TARGET_COL,
    date_col: str = "date",
) -> bool:
    """
    Verify that higher relevance grades correspond to strictly higher realized excess returns:
      1. Mean within-month cross-sectional Spearman rank correlation > 0.85.
      2. Mean realized return for the highest relevance grade strictly exceeds the lowest grade.
      3. Mean realized returns across grades are monotonically increasing.
    """
    monthly_corrs = []
    for _, grp in df.groupby(date_col):
        if grp[label_col].nunique() > 1 and grp[target_col].nunique() > 1:
            r, _ = spearmanr(grp[label_col].values, grp[target_col].values)
            if not np.isnan(r):
                monthly_corrs.append(float(r))
    mean_cs_rho = float(np.mean(monthly_corrs)) if monthly_corrs else 0.0
    grade_means = df.groupby(label_col)[target_col].mean().sort_index()
    is_monotonic = bool(np.all(np.diff(grade_means.values) > 0))
    return bool(mean_cs_rho > 0.85 and is_monotonic)


def extract_query_groups(
    df: pd.DataFrame,
    date_col: str = "date",
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Construct query group sizes (`group`), integer query IDs (`qid`), and sorted formation dates
    for XGBoost Ranker (`objective="rank:pairwise"`).
    Requires `df` to be sorted chronologically by `date_col`.
    """
    dates = pd.to_datetime(df[date_col])
    if not dates.is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by date ascending before extracting query groups.")
    unique_dates = list(pd.to_datetime(df[date_col].unique()))
    group_sizes = df.groupby(date_col, sort=False).size().values.astype(np.int32)
    qid = np.repeat(np.arange(len(group_sizes), dtype=np.int32), group_sizes)
    return group_sizes, qid, unique_dates


def build_pairwise_preferences_for_month(
    month_df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Construct all ordered pairwise preference comparisons (i, j) for a single formation month
    where stock i outperformed stock j (171 pairs for 19 stocks).
    """
    records = []
    rows = month_df.sort_values([target_col, "ticker"], ascending=[False, True]).reset_index(drop=True)
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            diff = float(rows.loc[i, target_col] - rows.loc[j, target_col])
            records.append(
                {
                    "date": rows.loc[i, "date"],
                    "preferred_ticker": rows.loc[i, "ticker"],
                    "dominated_ticker": rows.loc[j, "ticker"],
                    "preferred_excess_ret": float(rows.loc[i, target_col]),
                    "dominated_excess_ret": float(rows.loc[j, target_col]),
                    "return_diff": diff,
                    "return_margin": diff,
                    "preference_label": 1,
                }
            )
    return pd.DataFrame(records)


def construct_all_targets(
    df: pd.DataFrame,
    train_end: str = "2020-12-31",
    target_col: str = TARGET_COL,
    n_relevance_bins: int = 5,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Construct all target columns on a copy of the panel using ONLY training-period
    quantile cutoffs (`date <= train_end`) for `REL_GRADE_TRAIN_Q5`.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    train_sub = out[out["date"] <= pd.Timestamp(train_end)]
    train_edges = fit_training_relevance_bins(train_sub, target_col=target_col, n_bins=n_relevance_bins)

    out["CS_PERCENTILE_RANK"] = compute_cross_sectional_percentile_ranks(out, target_col=target_col)
    out["REL_GRADE_TRAIN_Q5"] = assign_relevance_grades(
        out, target_col=target_col, scheme="train_quantile_5", train_bin_edges=train_edges
    )
    out["REL_GRADE_CS_Q5"] = assign_relevance_grades(
        out, target_col=target_col, scheme="cs_quintile_5"
    )
    out["REL_GRADE_ORDINAL"] = assign_relevance_grades(
        out, target_col=target_col, scheme="cs_ordinal_19"
    )
    return out, train_edges


def summarize_target_constructions(
    panel_df: pd.DataFrame,
    train_end: str = "2020-12-31",
    val_end: str = "2022-06-30",
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Summarize target representations across the Training, Validation, and Test splits.
    """
    df = panel_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    t_mask = df["date"] <= pd.Timestamp(train_end)
    v_mask = (df["date"] > pd.Timestamp(train_end)) & (df["date"] <= pd.Timestamp(val_end))
    te_mask = df["date"] > pd.Timestamp(val_end)

    summary_rows = []
    for split_name, s_df in [
        ("Train (Feb 2016 – Dec 2020)", df.loc[t_mask]),
        ("Validation (Jan 2021 – Jun 2022)", df.loc[v_mask]),
        ("Test (Jul 2022 – Nov 2024)", df.loc[te_mask]),
    ]:
        n_months = int(s_df["date"].nunique())
        n_obs = len(s_df)
        n_pairs = int(n_months * (19 * 18 // 2))
        summary_rows.append(
            {
                "Split": split_name,
                "Months (Query Groups)": n_months,
                "Stock-Months": n_obs,
                "Pairwise Comparisons C(19,2)/mo": n_pairs,
                "Excess Return Mean (%)": round(float(s_df[target_col].mean() * 100.0), 3),
                "Excess Return Std (%)": round(float(s_df[target_col].std() * 100.0), 3),
                "CS Percentile Range": "[0.00, 1.00]",
                "LTR Relevance Grades": "0..4 (Train Quantile & CS Quintile) / 0..18 (Ordinal)",
            }
        )
    return pd.DataFrame(summary_rows)
