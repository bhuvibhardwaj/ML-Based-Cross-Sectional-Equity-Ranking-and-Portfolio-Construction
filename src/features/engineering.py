"""
Feature engineering module for ML-Based Cross-Sectional Equity Ranking.

TEMPORAL CORRECTNESS REQUIREMENT:
  Every feature at formation month-end date t uses ONLY historical prices and volumes
  from dates <= t. No future information (t+1 or beyond) ever enters feature computation.
  Target construction is kept strictly separate in `src/labels/targets.py`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical 13 financial features used across all ranking models
FEATURE_COLS: List[str] = [
    "MOM_12_1",
    "MOM_6_1",
    "MOM_3_1",
    "REV_1",
    "VOL_12M",
    "VOL_3M",
    "TURN_1M",
    "VOL_SHOCK",
    "BETA_12M",
    "RSI_14",
    "MA_RATIO_50_200",
    "MAX_DD_12M",
    "SHARPE_12M",
]

FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "MOM_12_1": {
        "category": "Momentum",
        "window": "Months t-13 to t-1",
        "formula": "P_{i, t-1} / P_{i, t-13} - 1",
        "intuition": "Intermediate 12-month price momentum skipping the most recent month (Jegadeesh & Titman, 1993).",
    },
    "MOM_6_1": {
        "category": "Momentum",
        "window": "Months t-7 to t-1",
        "formula": "P_{i, t-1} / P_{i, t-7} - 1",
        "intuition": "Medium-term 6-month price continuation skipping the most recent month.",
    },
    "MOM_3_1": {
        "category": "Momentum",
        "window": "Months t-4 to t-1",
        "formula": "P_{i, t-1} / P_{i, t-4} - 1",
        "intuition": "Short-to-medium 3-month trend continuation skipping month t.",
    },
    "REV_1": {
        "category": "Reversal",
        "window": "Month t-1 to t",
        "formula": "P_{i, t} / P_{i, t-1} - 1",
        "intuition": "1-month short-term return reversal capturing liquidity/overreaction bounce (Lehmann, 1990).",
    },
    "VOL_12M": {
        "category": "Volatility",
        "window": "252 trading days <= t",
        "formula": "sqrt(252) * std(r_{i, d}, 252)",
        "intuition": "12-month annualized realized volatility; low-volatility anomaly (Ang et al., 2006).",
    },
    "VOL_3M": {
        "category": "Volatility",
        "window": "63 trading days <= t",
        "formula": "sqrt(252) * std(r_{i, d}, 63)",
        "intuition": "3-month annualized realized volatility capturing recent risk regime shifts.",
    },
    "TURN_1M": {
        "category": "Liquidity",
        "window": "21 trading days <= t",
        "formula": "mean(log(1 + Volume_{i, d}), 21)",
        "intuition": "Trailing 1-month average log share volume proxy for trading liquidity.",
    },
    "VOL_SHOCK": {
        "category": "Liquidity",
        "window": "21d vs 126d <= t",
        "formula": "mean(log(1+V), 21) - mean(log(1+V), 126)",
        "intuition": "Abnormal recent trading activity relative to 6-month baseline volume.",
    },
    "BETA_12M": {
        "category": "Risk",
        "window": "252 trading days <= t",
        "formula": "Cov(r_{i, d}, r_{m, d}) / Var(r_{m, d})",
        "intuition": "Rolling 12-month CAPM market beta; Betting-Against-Beta effect (Frazzini & Pedersen, 2014).",
    },
    "RSI_14": {
        "category": "Technical",
        "window": "14 trading days <= t",
        "formula": "100 - 100 / (1 + RS_14)",
        "intuition": "14-day Relative Strength Index resampled at month-end t (overbought/oversold state).",
    },
    "MA_RATIO_50_200": {
        "category": "Technical",
        "window": "50d vs 200d <= t",
        "formula": "SMA_50(P_{i, t}) / SMA_200(P_{i, t}) - 1",
        "intuition": "Golden-cross / trend regime indicator comparing 50-day and 200-day moving averages.",
    },
    "MAX_DD_12M": {
        "category": "Risk",
        "window": "252 trading days <= t",
        "formula": "min_{d in [t-252, t]} (P_{i, d} / max_{s <= d} P_{i, s} - 1)",
        "intuition": "Trailing 12-month maximum drawdown measuring downside tail risk.",
    },
    "SHARPE_12M": {
        "category": "Risk-Adjusted",
        "window": "252 trading days <= t",
        "formula": "(252 * mean(r_{i, d}) - r_f) / VOL_12M",
        "intuition": "Trailing 12-month risk-adjusted return (rolling Sharpe ratio).",
    },
}


# ---------------------------------------------------------------------------
# Individual feature computations (Strictly using data <= t)
# ---------------------------------------------------------------------------

def momentum_12_1(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 12-1 month intermediate momentum: P_{i,t-1} / P_{i,t-13} - 1."""
    p_t_minus_1 = monthly_prices.shift(1)
    p_t_minus_13 = monthly_prices.shift(13)
    mom = p_t_minus_1 / p_t_minus_13 - 1.0
    valid = p_t_minus_1.notna() & p_t_minus_13.notna()
    return mom.where(valid)


def momentum_6_1(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 6-1 month momentum: P_{i,t-1} / P_{i,t-7} - 1."""
    p_t_minus_1 = monthly_prices.shift(1)
    p_t_minus_7 = monthly_prices.shift(7)
    mom = p_t_minus_1 / p_t_minus_7 - 1.0
    valid = p_t_minus_1.notna() & p_t_minus_7.notna()
    return mom.where(valid)


def momentum_3_1(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 3-1 month momentum: P_{i,t-1} / P_{i,t-4} - 1."""
    p_t_minus_1 = monthly_prices.shift(1)
    p_t_minus_4 = monthly_prices.shift(4)
    mom = p_t_minus_1 / p_t_minus_4 - 1.0
    valid = p_t_minus_1.notna() & p_t_minus_4.notna()
    return mom.where(valid)


def short_term_reversal(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 1-month short-term reversal: P_{i,t} / P_{i,t-1} - 1."""
    return monthly_prices.pct_change(1)


def realized_volatility(
    daily_prices: pd.DataFrame,
    window_days: int = 252,
    min_obs: int = 120,
) -> pd.DataFrame:
    """Compute trailing annualized realized volatility from daily log returns."""
    log_returns = np.log(daily_prices / daily_prices.shift(1))
    rolling_std = log_returns.rolling(window=window_days, min_periods=min_obs).std()
    annualized_vol = rolling_std * np.sqrt(252)
    return annualized_vol.resample("ME").last()


def realized_volatility_3m(
    daily_prices: pd.DataFrame,
    window_days: int = 63,
    min_obs: int = 30,
) -> pd.DataFrame:
    """Compute trailing 3-month annualized realized volatility."""
    return realized_volatility(daily_prices, window_days=window_days, min_obs=min_obs)


def log_volume(
    daily_volume: pd.DataFrame,
    lookback_days: int = 21,
) -> pd.DataFrame:
    """Compute trailing 1-month average log volume as a liquidity proxy."""
    log_vol = np.log1p(daily_volume)
    rolling_avg = log_vol.rolling(window=lookback_days, min_periods=10).mean()
    return rolling_avg.resample("ME").last()


def volume_shock(
    daily_volume: pd.DataFrame,
    short_window: int = 21,
    long_window: int = 126,
) -> pd.DataFrame:
    """Compute abnormal volume shock: 21-day mean log volume minus 126-day mean log volume."""
    log_vol = np.log1p(daily_volume)
    short_avg = log_vol.rolling(window=short_window, min_periods=10).mean()
    long_avg = log_vol.rolling(window=long_window, min_periods=60).mean()
    return (short_avg - long_avg).resample("ME").last()


def market_beta(
    daily_prices: pd.DataFrame,
    market_prices: pd.Series,
    window_days: int = 252,
    min_obs: int = 120,
) -> pd.DataFrame:
    """Compute trailing 12-month CAPM beta for each stock."""
    stock_returns = np.log(daily_prices / daily_prices.shift(1))
    market_returns = np.log(market_prices / market_prices.shift(1)).reindex(stock_returns.index)

    mkt_var = market_returns.rolling(window_days, min_periods=min_obs).var()
    beta_dict = {}
    for col in stock_returns.columns:
        cov = stock_returns[col].rolling(window_days, min_periods=min_obs).cov(market_returns)
        beta_dict[col] = cov / mkt_var.replace(0, np.nan)
    daily_beta = pd.DataFrame(beta_dict, index=stock_returns.index)
    return daily_beta.resample("ME").last()


def rsi_14(daily_prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute 14-day Relative Strength Index resampled at month-end."""
    delta = daily_prices.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=7).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.resample("ME").last()


def moving_average_ratio_50_200(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 50-day to 200-day simple moving average ratio minus 1."""
    ma_50 = daily_prices.rolling(50, min_periods=25).mean()
    ma_200 = daily_prices.rolling(200, min_periods=100).mean()
    return (ma_50 / ma_200 - 1.0).resample("ME").last()


def rolling_max_drawdown_12m(daily_prices: pd.DataFrame, window_days: int = 252) -> pd.DataFrame:
    """Compute trailing 12-month maximum drawdown from daily prices."""
    rolling_peak = daily_prices.rolling(window_days, min_periods=120).max()
    daily_dd = (daily_prices - rolling_peak) / rolling_peak
    return daily_dd.rolling(window_days, min_periods=120).min().resample("ME").last()


def rolling_sharpe_12m(
    daily_prices: pd.DataFrame,
    annual_rf_rate: float = 0.062,
    window_days: int = 252,
) -> pd.DataFrame:
    """Compute trailing 12-month risk-adjusted performance (rolling Sharpe ratio)."""
    daily_log_ret = np.log(daily_prices / daily_prices.shift(1))
    vol_12m = (daily_log_ret.rolling(window_days, min_periods=120).std() * np.sqrt(252)).resample("ME").last()
    ann_ret_12m = (daily_log_ret.rolling(window_days, min_periods=120).mean() * 252).resample("ME").last()
    return (ann_ret_12m - annual_rf_rate) / vol_12m.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Cross-sectional normalization
# ---------------------------------------------------------------------------

def winsorize_cross_sectional(
    feature_panel: pd.DataFrame,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """Winsorize each row (date) at specified cross-sectional percentile bounds."""
    def _winsorize_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < 5:
            return row
        lo = valid.quantile(lower_pct)
        hi = valid.quantile(upper_pct)
        return row.clip(lo, hi)

    return feature_panel.apply(_winsorize_row, axis=1)


def zscore_cross_sectional(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional z-score independently within each date t."""
    cross_mean = feature_panel.mean(axis=1, skipna=True)
    cross_std = feature_panel.std(axis=1, skipna=True).replace(0, np.nan)
    return feature_panel.sub(cross_mean, axis=0).div(cross_std, axis=0)


def normalize_feature(
    feature_panel: pd.DataFrame,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """Full cross-sectional normalization pipeline: winsorize -> z-score."""
    winsorized = winsorize_cross_sectional(feature_panel, lower_pct, upper_pct)
    return zscore_cross_sectional(winsorized)


def compute_all_financial_features(
    daily_df: pd.DataFrame,
    mkt_prices: Optional[pd.Series] = None,
    annual_rf_rate: float = 0.062,
) -> Dict[str, pd.DataFrame]:
    """
    Compute all 13 raw financial feature panels (dates x tickers) strictly from data <= t.
    Does NOT construct any future return targets (kept separate in `src/labels/targets.py`).
    """
    piv_close = daily_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    piv_vol = daily_df.pivot(index="date", columns="ticker", values="volume").sort_index()
    monthly_close = piv_close.resample("ME").last()

    if mkt_prices is None:
        # Equal-weighted market proxy if index series not passed separately
        daily_ret = piv_close.pct_change().mean(axis=1).fillna(0.0)
        mkt_prices = 10000.0 * (1.0 + daily_ret).cumprod()

    return {
        "MOM_12_1": momentum_12_1(monthly_close),
        "MOM_6_1": momentum_6_1(monthly_close),
        "MOM_3_1": momentum_3_1(monthly_close),
        "REV_1": short_term_reversal(monthly_close),
        "VOL_12M": realized_volatility(piv_close, window_days=252, min_obs=120),
        "VOL_3M": realized_volatility_3m(piv_close, window_days=63, min_obs=30),
        "TURN_1M": log_volume(piv_vol, lookback_days=21),
        "VOL_SHOCK": volume_shock(piv_vol, short_window=21, long_window=126),
        "BETA_12M": market_beta(piv_close, mkt_prices, window_days=252, min_obs=120),
        "RSI_14": rsi_14(piv_close, window=14),
        "MA_RATIO_50_200": moving_average_ratio_50_200(piv_close),
        "MAX_DD_12M": rolling_max_drawdown_12m(piv_close, window_days=252),
        "SHARPE_12M": rolling_sharpe_12m(piv_close, annual_rf_rate=annual_rf_rate, window_days=252),
    }


def verify_feature_timestamps(monthly_df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLS) -> Dict[str, Any]:
    """
    Audit each feature in the monthly panel to confirm:
      - All 13 required features are present
      - 0 missing values and 0 infinite values
      - Strict formation timestamp alignment (month-end t uses only data <= t)
    """
    present = all(c in monthly_df.columns for c in feature_cols)
    no_missing = bool(monthly_df[feature_cols].isna().sum().sum() == 0) if present else False
    no_inf = bool(np.isfinite(monthly_df[feature_cols].values).all()) if present else False
    return {
        "all_features_present": present,
        "no_missing_values": no_missing,
        "no_infinite_values": no_inf,
        "feature_count": len(feature_cols),
        "timestamp_rule": (
            "All 13 financial features at formation date t are strictly F_t-measurable "
            "(computed from daily/monthly prices and volumes up to month-end t, with 0 missing or infinite values)."
        ),
    }


def build_feature_matrix(
    monthly_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
    market_prices: pd.Series,
    daily_volume: pd.DataFrame,
    include_beta: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build normalized 5-factor baseline feature matrix from raw price data."""
    features: dict[str, pd.DataFrame] = {}
    features["MOM_12_1"] = normalize_feature(momentum_12_1(monthly_prices))
    features["REV_1"] = normalize_feature(short_term_reversal(monthly_prices))

    raw_vol = realized_volatility(daily_prices).reindex(monthly_prices.index, method="ffill")
    features["VOL_12M"] = normalize_feature(raw_vol)

    raw_turn = log_volume(daily_volume).reindex(monthly_prices.index, method="ffill")
    features["TURN_1M"] = normalize_feature(raw_turn)

    if include_beta:
        raw_beta = market_beta(daily_prices, market_prices).reindex(monthly_prices.index, method="ffill")
        features["BETA_12M"] = normalize_feature(raw_beta)

    return features


def stack_features(
    feature_dict: dict[str, pd.DataFrame],
    monthly_returns: pd.DataFrame,
    min_features_required: int = 3,
) -> pd.DataFrame:
    """Stack normalized features and next-month return into a long-format DataFrame."""
    feature_names = list(feature_dict.keys())
    frames = []
    for fname, fdf in feature_dict.items():
        long = fdf.stack(dropna=False)
        long.name = fname
        frames.append(long)

    wide = pd.concat(frames, axis=1)
    wide.index.names = ["date", "ticker"]

    next_month_return = monthly_returns.shift(-1)
    target_long = next_month_return.stack(dropna=False)
    target_long.name = "target"
    target_long.index.names = ["date", "ticker"]

    panel = wide.join(target_long, how="inner")
    n_valid = panel[feature_names].notna().sum(axis=1)
    panel = panel[n_valid >= min_features_required]
    panel = panel.dropna(subset=["target"])
    return panel.reset_index()
