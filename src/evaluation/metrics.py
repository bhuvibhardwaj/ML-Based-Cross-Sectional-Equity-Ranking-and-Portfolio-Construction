"""
Evaluation module (`src/evaluation/metrics.py`):
Cross-sectional ranking metrics, regression error metrics, ranking stability diagnostics,
and portfolio performance metrics.

Clearly separates:
  1. Cross-Sectional Ranking Metrics (Monthly Spearman Rank IC, Mean Rank IC, ICIR,
     Newey-West t-stat, NDCG@5, NDCG@10, Top-5 Return, Bottom-5 Return, Top-5 minus Bottom-5 Spread).
  2. Pointwise Regression Metrics (RMSE, MAE, Out-of-Sample R²).
  3. Portfolio Performance Metrics (Annualized Return, CAGR, Volatility, Sharpe Ratio,
     Maximum Drawdown, Calmar Ratio, Hit Rate).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Cross-Sectional Ranking Metrics
# ---------------------------------------------------------------------------

def information_coefficient(
    predicted_scores: pd.Series,
    realized_returns: pd.Series,
) -> float:
    """
    Compute the Spearman rank correlation (IC) between predicted scores and
    realized returns for a single cross-section (one formation month).
    """
    common = predicted_scores.index.intersection(realized_returns.index)
    pred = predicted_scores.loc[common].dropna()
    real = realized_returns.loc[common].dropna()

    common = pred.index.intersection(real.index)
    pred, real = pred.loc[common], real.loc[common]

    if len(pred) < 5:
        return np.nan

    if pred.std() < 1e-10 or real.std() < 1e-10:
        return 0.0

    ic, _ = stats.spearmanr(pred.values, real.values)
    return float(ic) if not np.isnan(ic) else 0.0


def rank_ic(
    predicted_panel: pd.DataFrame,
    realized_panel: pd.DataFrame,
) -> pd.Series:
    """
    Compute Monthly Spearman Rank IC across a wide panel (index = formation dates,
    columns = tickers).
    """
    common_dates = predicted_panel.index.intersection(realized_panel.index)
    ics = {}
    for dt in common_dates:
        ics[dt] = information_coefficient(predicted_panel.loc[dt], realized_panel.loc[dt])
    s = pd.Series(ics, name="Rank_IC")
    s.index.name = "date"
    return s


def ic_time_series(
    panel: pd.DataFrame,
    score_col: str,
    target_col: str = "target",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.Series:
    """Compute Spearman Rank IC for each formation date in a long-format panel DataFrame."""
    def _ic_for_group(group: pd.DataFrame) -> float:
        return information_coefficient(group[score_col], group[target_col])

    ic_series = panel.groupby(date_col).apply(_ic_for_group)
    ic_series.name = "IC"
    return ic_series


def newey_west_tstat(series: Union[pd.Series, np.ndarray], max_lags: int = 3) -> Tuple[float, float, float]:
    """
    Compute Newey-West (1987) HAC standard error, t-statistic, and two-sided p-value
    for the sample mean of a monthly time series.
    """
    x = np.asarray(pd.Series(series).dropna(), dtype=np.float64)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    mean_x = float(np.mean(x))
    u = x - mean_x
    gamma_0 = float(np.dot(u, u) / n)
    if gamma_0 < 1e-16:
        return 0.0, 0.0, 1.0

    l_max = min(max_lags, n - 1)
    hac_var = gamma_0
    for lag in range(1, l_max + 1):
        weight = 1.0 - lag / (l_max + 1.0)
        gamma_l = float(np.dot(u[lag:], u[:-lag]) / n)
        hac_var += 2.0 * weight * gamma_l

    hac_var_mean = max(hac_var / n, 1e-16)
    se_nw = float(np.sqrt(hac_var_mean))
    t_stat = float(mean_x / se_nw)
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(n - 1, 1))))
    return se_nw, t_stat, p_val


def ic_summary(ic_series: pd.Series) -> dict:
    """Compute IC summary statistics used to evaluate a ranking model."""
    valid_ic = ic_series.dropna()
    n = len(valid_ic)

    if n < 2:
        return {
            k: np.nan
            for k in [
                "mean_ic",
                "std_ic",
                "icir",
                "t_stat",
                "nw_t_stat",
                "nw_p_val",
                "positive_ic_pct",
                "pct_positive",
                "n_dates",
            ]
        }

    mean_ic = float(valid_ic.mean())
    std_ic = float(valid_ic.std(ddof=1))
    icir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 1e-10 else 0.0
    _, nw_t, nw_p = newey_west_tstat(valid_ic, max_lags=3)
    pos_pct = float((valid_ic > 0).mean())

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "t_stat": t_stat,
        "nw_t_stat": nw_t,
        "nw_p_val": nw_p,
        "positive_ic_pct": pos_pct,
        "pct_positive": pos_pct,
        "n_dates": n,
    }


def ndcg_at_k(
    predicted_scores: Union[pd.Series, np.ndarray],
    realized_returns: Union[pd.Series, np.ndarray],
    k: int = 5,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at cutoff k (NDCG@k) for a single month.

    Relevance grades g_i in {0, 1, 2, 3, 4} are defined by the stock's realized cross-sectional
    return quintile within that month (4 = Top quintile return, 0 = Bottom quintile return),
    with exponential gain (2^{g_i} - 1) and logarithmic discount 1 / log2(rank + 1).
    """
    pred = np.asarray(predicted_scores, dtype=np.float64)
    real = np.asarray(realized_returns, dtype=np.float64)
    mask = ~(np.isnan(pred) | np.isnan(real))
    pred, real = pred[mask], real[mask]
    n = len(real)
    if n == 0:
        return np.nan

    k_eff = min(k, n)
    ranks_01 = (stats.rankdata(real, method="average") - 1.0) / max(n - 1, 1)
    rel_grades = np.minimum(np.floor(ranks_01 * 5.0), 4.0)
    gains = (2.0 ** rel_grades) - 1.0

    pred_order = np.argsort(-pred, kind="mergesort")[:k_eff]
    ideal_order = np.argsort(-gains, kind="mergesort")[:k_eff]

    discounts = np.log2(np.arange(2, k_eff + 2, dtype=np.float64))
    dcg = float(np.sum(gains[pred_order] / discounts))
    idcg = float(np.sum(gains[ideal_order] / discounts))
    if idcg <= 1e-12:
        return 0.0
    return dcg / idcg


def ndcg_time_series(
    predicted_panel: pd.DataFrame,
    realized_panel: pd.DataFrame,
    k: int = 5,
) -> pd.Series:
    """Compute monthly NDCG@k across all formation dates in wide panels."""
    common_dates = predicted_panel.index.intersection(realized_panel.index)
    vals = {}
    for dt in common_dates:
        vals[dt] = ndcg_at_k(predicted_panel.loc[dt].values, realized_panel.loc[dt].values, k=k)
    s = pd.Series(vals, name=f"NDCG@{k}")
    s.index.name = "date"
    return s


def top_bottom_spread_series(
    predicted_panel: pd.DataFrame,
    realized_panel: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """
    For each formation month t in wide panels (index = date, columns = tickers),
    rank stocks by predicted score descending and compute:
      - `top_k_return`: average realized return of Top-k predicted stocks
      - `bottom_k_return`: average realized return of Bottom-k predicted stocks
      - `spread`: Top-k minus Bottom-k return spread
    """
    common_dates = predicted_panel.index.intersection(realized_panel.index)
    rows = []
    for dt in common_dates:
        scores = predicted_panel.loc[dt].dropna()
        rets = realized_panel.loc[dt].reindex(scores.index).dropna()
        scores = scores.reindex(rets.index)
        ordered_tickers = scores.sort_values(ascending=False).index.tolist()
        top_tickers = ordered_tickers[:k]
        bot_tickers = ordered_tickers[-k:]
        top_ret = float(rets.loc[top_tickers].mean())
        bot_ret = float(rets.loc[bot_tickers].mean())
        rows.append(
            {
                "date": dt,
                "top_k_return": top_ret,
                "bottom_k_return": bot_ret,
                "spread": top_ret - bot_ret,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def compute_ranking_stability(
    predicted_panel: pd.DataFrame,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate month-to-month ranking stability on a wide score panel (index = date, columns = tickers):
      1. `Mean_Rank_Autocorr`: Average Spearman rank autocorrelation between month t-1 and month t.
      2. `Mean_Top_K_Overlap`: Average fraction of Top-k stocks retained from month t-1 to month t.
      3. `Implied_Monthly_Set_Turnover`: 1 - Mean_Top_K_Overlap.
    """
    rank_piv = predicted_panel.sort_index().rank(axis=1, ascending=False, method="first")
    autocorrs = []
    overlaps = []
    dates = list(rank_piv.index)
    for i in range(1, len(dates)):
        r_prev = rank_piv.iloc[i - 1]
        r_curr = rank_piv.iloc[i]
        corr, _ = stats.spearmanr(r_prev.values, r_curr.values)
        if not np.isnan(corr):
            autocorrs.append(float(corr))

        top_prev = set(r_prev[r_prev <= top_k].index)
        top_curr = set(r_curr[r_curr <= top_k].index)
        overlaps.append(len(top_prev & top_curr) / float(top_k))

    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    return {
        "Mean_Rank_Autocorr": float(np.mean(autocorrs)) if autocorrs else 0.0,
        "Mean_Top_K_Overlap": mean_overlap,
        "Implied_Monthly_Set_Turnover": 1.0 - mean_overlap,
    }


# ---------------------------------------------------------------------------
# 2. Pointwise Regression Metrics (Separated from Ranking Metrics)
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_mean: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute cardinal regression metrics (RMSE, MAE, Out-of-Sample R²) for models
    that predict next-month excess returns directly.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    # Gu, Kelly & Xiu (2020) Out-of-Sample R² relative to zero excess return benchmark
    denom = float(np.sum(yt ** 2))
    r2_oos = float(1.0 - np.sum((yt - yp) ** 2) / denom) if denom > 0 else np.nan
    r2_std = float(r2_score(yt, yp))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2_OOS": r2_oos,
        "R2_Standard": r2_std,
    }


# ---------------------------------------------------------------------------
# 3. Portfolio Performance Metrics
# ---------------------------------------------------------------------------

def annualized_return(monthly_returns: pd.Series) -> float:
    """Arithmetic mean of monthly returns, annualized by multiplying by 12."""
    return float(monthly_returns.mean() * 12.0)


def cagr_return(monthly_returns: pd.Series) -> float:
    """Compound Annual Growth Rate (geometric annualized return) from monthly returns."""
    valid = monthly_returns.dropna()
    n = len(valid)
    if n == 0:
        return np.nan
    total_wealth = float((1.0 + valid).prod())
    if total_wealth <= 0:
        return -1.0
    return float(total_wealth ** (12.0 / n) - 1.0)


def annualized_volatility(monthly_returns: pd.Series) -> float:
    """Annualized volatility: standard deviation of monthly returns * sqrt(12)."""
    return float(monthly_returns.std(ddof=1) * np.sqrt(12.0))


def sharpe_ratio(
    monthly_returns: pd.Series,
    risk_free_annual: Union[float, pd.Series] = 0.0666,
) -> float:
    """
    Annualized Sharpe ratio: (R_{p,annual} - R_{f,annual}) / sigma_{p,annual}.
    Supports either a scalar annualized risk-free rate (e.g. 0.0666) or a monthly
    risk-free rate Series.
    """
    ann_vol = annualized_volatility(monthly_returns)
    if ann_vol < 1e-10:
        return 0.0
    if isinstance(risk_free_annual, pd.Series):
        excess_monthly = monthly_returns - risk_free_annual
        return float((excess_monthly.mean() * 12.0) / ann_vol)
    ann_ret = annualized_return(monthly_returns)
    return float((ann_ret - float(risk_free_annual)) / ann_vol)


def max_drawdown(monthly_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (returned as a negative fraction, e.g. -0.105 for -10.5%)."""
    wealth = (1.0 + monthly_returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return float(dd.min())


def maximum_drawdown(monthly_returns: pd.Series) -> float:
    """Positive magnitude of maximum drawdown."""
    return abs(max_drawdown(monthly_returns))


def calmar_ratio(monthly_returns: pd.Series) -> float:
    """Ratio of annualized return to |max_drawdown|."""
    mdd = abs(max_drawdown(monthly_returns))
    if mdd < 1e-10:
        return 0.0
    return float(annualized_return(monthly_returns) / mdd)


def hit_rate(monthly_returns: pd.Series) -> float:
    """Fraction of months with strictly positive return."""
    valid = monthly_returns.dropna()
    if len(valid) == 0:
        return 0.0
    return float((valid > 0.0).mean())


def cumulative_returns(monthly_returns: pd.Series) -> pd.Series:
    """Cumulative wealth index starting at 1.0."""
    return (1.0 + monthly_returns.fillna(0.0)).cumprod()
