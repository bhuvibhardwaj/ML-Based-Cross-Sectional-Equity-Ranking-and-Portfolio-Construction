"""
Portfolio Construction and Backtesting Module (`src/backtest/portfolio.py`).

Implements Top-5 cross-sectional equity portfolio construction and monthly rebalancing
backtesting for ML ranking models and financial benchmarks:
  - ML Top-5 Portfolios:
      1. Ridge Regression (Top-5 equal-weighted)
      2. XGBoost Regression (Top-5 equal-weighted)
      3. XGBoost Learning-to-Rank (Top-5 equal-weighted)
  - Benchmarks:
      4. Momentum-Ranked Top-5 (ranked by MOM_12_1 each month, equal-weighted)
      5. Equal-Weighted Full Universe (1/N across all 19 NSE stocks)
      6. Nifty 50 total-return series only when explicitly present; otherwise a
         liquidity-weighted 19-stock proxy with a fixed dividend uplift.

Evaluates:
  - Monthly rebalancing & constituent tracking
  - Monthly turnover (both drifted-weight L1 turnover and constituent-replacement turnover)
  - Transaction costs (default 10 bps one-way / 20 bps round-trip)
  - Gross and Net annualized returns & CAGR
  - Gross and Net Sharpe ratios (excess over RBI 91-day T-bill yield)
  - Maximum Drawdown
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    annualized_return,
    cagr_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
    hit_rate,
    cumulative_returns,
)

# Historical RBI 91-day Treasury Bill monthly yields (annualized % divided by 12)
# Spanning the out-of-sample test window (Jul 2022 - Nov 2024, mean 6.66% p.a.)
DEFAULT_ANNUAL_RF = 0.0666


def construct_top_k_weights(
    df_eval: pd.DataFrame,
    score_col: str,
    top_k: int = 5,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Construct equal-weighted Top-K portfolio weights (1/K for top K ranked stocks, 0 otherwise)
    at each formation month t.

    Parameters
    ----------
    df_eval : pd.DataFrame
        Evaluation panel containing [date_col, ticker_col, score_col].
    score_col : str
        Column containing model predicted score or benchmark signal (higher = better).
    top_k : int
        Number of top-ranked stocks to hold in equal weights (default 5).

    Returns
    -------
    pd.DataFrame
        Wide weight matrix (index = formation date, columns = tickers) where each row sums to 1.0.
    """
    tickers = sorted(df_eval[ticker_col].unique())
    dates = sorted(df_eval[date_col].unique())
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)

    for dt, grp in df_eval.groupby(date_col, sort=True):
        # Deterministic tie-breaking by ticker
        ranked = grp.sort_values([score_col, ticker_col], ascending=[False, True])
        selected = ranked.head(top_k)[ticker_col].tolist()
        if len(selected) > 0:
            weights.loc[dt, selected] = 1.0 / len(selected)

    weights.index.name = date_col
    return weights


def construct_equal_weight_universe_weights(
    df_eval: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Construct 1/N equal-weighted portfolio weights across all available stocks each month.
    """
    tickers = sorted(df_eval[ticker_col].unique())
    dates = sorted(df_eval[date_col].unique())
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)

    for dt, grp in df_eval.groupby(date_col, sort=True):
        avail = grp[ticker_col].unique().tolist()
        if len(avail) > 0:
            weights.loc[dt, avail] = 1.0 / len(avail)

    weights.index.name = date_col
    return weights


def simulate_rebalanced_portfolio(
    weights_df: pd.DataFrame,
    df_eval: pd.DataFrame,
    strategy_name: str,
    one_way_cost_bps: float = 10.0,
    annual_rf: float = DEFAULT_ANNUAL_RF,
    date_col: str = "date",
    ticker_col: str = "ticker",
    excess_ret_col: str = "NEXT_MONTH_EXCESS_RETURN",
    total_ret_col: str = "NEXT_MONTH_RETURN",
    rf_col: str = "RF_MONTHLY",
) -> Dict[str, object]:
    """
    Simulate a monthly rebalanced long-only equity portfolio with drifted-weight
    turnover and transaction cost deduction.

    At formation month t:
      - Target weights w_t are formed using signals available at month t.
      - Portfolio earns next-month total return R_{p, t+1} = sum_i w_{i,t} * R_{i, t+1}.
      - End-of-month drifted weight before rebalancing at t+1:
          w_{i, t}^{drift} = w_{i, t} * (1 + R_{i, t+1}) / (1 + R_{p, t+1})
      - One-way turnover at rebalancing:
          TO_t = 0.5 * sum_i |w_{i, t} - w_{i, t-1}^{drift}|
      - Transaction cost deduction:
          TC_t = 2 * one_way_cost * TO_t  (where 2 * 10 bps = 20 bps per unit one-way turnover)

    Returns
    -------
    dict with:
      - 'summary': dict of portfolio metrics (Gross/Net Ann Return, CAGR, Sharpe, MaxDD, Turnover, Cost)
      - 'monthly_series': pd.DataFrame with monthly gross/net returns, turnover, costs, cumulative wealth
      - 'weights': pd.DataFrame of monthly target weights
    """
    dates = sorted(weights_df.index.unique())
    tickers = list(weights_df.columns)
    monthly_rf_default = (1.0 + annual_rf) ** (1.0 / 12.0) - 1.0

    # Pivot excess returns and total returns
    excess_piv = df_eval.pivot(index=date_col, columns=ticker_col, values=excess_ret_col).reindex(
        index=dates, columns=tickers
    ).fillna(0.0)

    if total_ret_col in df_eval.columns:
        total_piv = df_eval.pivot(index=date_col, columns=ticker_col, values=total_ret_col).reindex(
            index=dates, columns=tickers
        ).fillna(0.0)
    else:
        total_piv = excess_piv + monthly_rf_default

    if rf_col in df_eval.columns:
        rf_series = df_eval.groupby(date_col)[rf_col].first().reindex(dates).fillna(monthly_rf_default)
    else:
        rf_series = pd.Series(monthly_rf_default, index=dates)

    # Round-trip transaction cost per unit of one-way turnover (10 bps buy + 10 bps sell = 20 bps)
    round_trip_cost_rate = 2.0 * (one_way_cost_bps / 10000.0)

    gross_total_rets: List[float] = []
    net_total_rets: List[float] = []
    gross_excess_rets: List[float] = []
    net_excess_rets: List[float] = []
    drifted_turnovers: List[float] = []
    set_turnovers: List[float] = []
    cost_deductions: List[float] = []
    holdings_strings: List[str] = []

    prev_drifted_w: Optional[np.ndarray] = None
    prev_target_w: Optional[np.ndarray] = None

    for dt in dates:
        w_t = weights_df.loc[dt].values.astype(float)
        r_total_t = total_piv.loc[dt].values.astype(float)
        r_excess_t = excess_piv.loc[dt].values.astype(float)

        # Record top holdings
        active_tickers = [tickers[i] for i, val in enumerate(w_t) if val > 1e-6]
        holdings_strings.append(", ".join([t.replace(".NS", "") for t in active_tickers]))

        # Turnover calculation
        if prev_drifted_w is None:
            # After initial formation, track ongoing monthly rebalancing turnover
            to_drift = 0.0
            to_set = 0.0
        else:
            to_drift = 0.5 * float(np.sum(np.abs(w_t - prev_drifted_w)))
            to_set = 0.5 * float(np.sum(np.abs(w_t - prev_target_w)))

        tc_t = to_drift * round_trip_cost_rate

        port_gross_total = float(np.dot(w_t, r_total_t))
        port_net_total = port_gross_total - tc_t
        port_gross_excess = float(np.dot(w_t, r_excess_t))
        port_net_excess = port_gross_excess - tc_t

        gross_total_rets.append(port_gross_total)
        net_total_rets.append(port_net_total)
        gross_excess_rets.append(port_gross_excess)
        net_excess_rets.append(port_net_excess)
        drifted_turnovers.append(to_drift)
        set_turnovers.append(to_set)
        cost_deductions.append(tc_t)

        # Update drifted weights at end of month before next rebalance
        denom = 1.0 + port_gross_total
        if abs(denom) > 1e-9:
            prev_drifted_w = w_t * (1.0 + r_total_t) / denom
        else:
            prev_drifted_w = w_t.copy()
        prev_target_w = w_t.copy()

    monthly_df = pd.DataFrame(
        {
            "date": dates,
            "strategy": strategy_name,
            "gross_return": gross_total_rets,
            "net_return": net_total_rets,
            "gross_excess_return": gross_excess_rets,
            "net_excess_return": net_excess_rets,
            "rf_monthly": rf_series.values,
            "turnover_drifted": drifted_turnovers,
            "turnover_set": set_turnovers,
            "transaction_cost": cost_deductions,
            "holdings": holdings_strings,
        }
    )
    monthly_df["gross_wealth"] = (1.0 + monthly_df["gross_return"]).cumprod()
    monthly_df["net_wealth"] = (1.0 + monthly_df["net_return"]).cumprod()

    gross_s = pd.Series(gross_total_rets, index=dates)
    net_s = pd.Series(net_total_rets, index=dates)

    # Mean turnover over rebalancing months (t >= 2)
    avg_turnover_drifted = float(np.mean(drifted_turnovers[1:])) if len(drifted_turnovers) > 1 else 0.0
    avg_turnover_set = float(np.mean(set_turnovers[1:])) if len(set_turnovers) > 1 else 0.0
    ann_cost_drag = float(annualized_return(gross_s) - annualized_return(net_s))

    summary = {
        "Strategy": strategy_name,
        "Gross_Ann_Return_Pct": round(float(annualized_return(gross_s) * 100.0), 2),
        "Net_Ann_Return_Pct": round(float(annualized_return(net_s) * 100.0), 2),
        "Gross_CAGR_Pct": round(float(cagr_return(gross_s) * 100.0), 2),
        "Net_CAGR_Pct": round(float(cagr_return(net_s) * 100.0), 2),
        "Ann_Volatility_Pct": round(float(annualized_volatility(net_s) * 100.0), 2),
        "Gross_Sharpe": round(float(sharpe_ratio(gross_s, risk_free_annual=annual_rf)), 3),
        "Net_Sharpe": round(float(sharpe_ratio(net_s, risk_free_annual=annual_rf)), 3),
        "Max_Drawdown_Pct": round(float(max_drawdown(net_s) * 100.0), 2),
        "Gross_Max_Drawdown_Pct": round(float(max_drawdown(gross_s) * 100.0), 2),
        "Calmar_Ratio": round(float(calmar_ratio(net_s)), 3),
        "Monthly_Turnover_Pct": round(avg_turnover_drifted * 100.0, 2),
        "Set_Turnover_Pct": round(avg_turnover_set * 100.0, 2),
        "Annual_Cost_Drag_Pct": round(ann_cost_drag * 100.0, 2),
        "Hit_Rate_Pct": round(float(hit_rate(net_s) * 100.0), 1),
        "Terminal_Wealth_Gross": round(float(monthly_df["gross_wealth"].iloc[-1]), 4),
        "Terminal_Wealth_Net": round(float(monthly_df["net_wealth"].iloc[-1]), 4),
    }

    return {
        "summary": summary,
        "monthly_series": monthly_df,
        "weights": weights_df,
    }


def evaluate_nifty_benchmark(
    daily_df: pd.DataFrame,
    eval_dates: List[pd.Timestamp],
    annual_rf: float = DEFAULT_ANNUAL_RF,
) -> Dict[str, object]:
    """
    Compute the benchmark return series aligned to formation dates. The CSV used by this
    project has no external index series, so the active fallback is only a
    liquidity-weighted 19-stock proxy with a fixed 1.2% annual dividend uplift.
    """
    dates = sorted(pd.to_datetime(eval_dates))
    daily_sorted = daily_df.sort_values("date").copy()
    daily_sorted["date"] = pd.to_datetime(daily_sorted["date"])

    if "nifty_tri" in daily_sorted.columns:
        strategy_name = "Nifty 50 TRI (provided series)"
        holdings_label = "NIFTY 50 TRI SERIES"
        daily_idx = daily_sorted.groupby("date")["nifty_tri"].first().sort_index()
        monthly_nifty = daily_idx.resample("ME").last()
        next_month_nifty = monthly_nifty.pct_change().shift(-1)
    else:
        strategy_name = "19-Stock Liquidity-Weighted Proxy (+1.2% dividend uplift)"
        holdings_label = "19-STOCK LIQUIDITY-WEIGHTED PROXY"
        # This is not a Nifty index: it uses only the project's 19-stock universe.
        piv_close = daily_sorted.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        piv_vol = daily_sorted.pivot(index="date", columns="ticker", values="volume").sort_index()
        rupee_turn = (piv_close * piv_vol).rolling(126, min_periods=21).mean().resample("ME").last()
        cap_weights = rupee_turn.div(rupee_turn.sum(axis=1), axis=0).fillna(1.0 / piv_close.shape[1])
        monthly_rets = piv_close.resample("ME").last().pct_change()
        next_month_stock_rets = monthly_rets.shift(-1)
        # Include ~1.20% p.a. (0.0010/mo) Nifty 50 dividend reinvestment total-return adjustment
        next_month_nifty = (cap_weights * next_month_stock_rets).sum(axis=1) + 0.0010

    aligned_rets = next_month_nifty.reindex(dates).fillna(0.0)

    monthly_rf_default = (1.0 + annual_rf) ** (1.0 / 12.0) - 1.0
    monthly_df = pd.DataFrame(
        {
            "date": dates,
            "strategy": strategy_name,
            "gross_return": aligned_rets.values,
            "net_return": aligned_rets.values,
            "gross_excess_return": aligned_rets.values - monthly_rf_default,
            "net_excess_return": aligned_rets.values - monthly_rf_default,
            "rf_monthly": monthly_rf_default,
            "turnover_drifted": 0.0,
            "turnover_set": 0.0,
            "transaction_cost": 0.0,
            "holdings": holdings_label,
        }
    )
    monthly_df["gross_wealth"] = (1.0 + monthly_df["gross_return"]).cumprod()
    monthly_df["net_wealth"] = (1.0 + monthly_df["net_return"]).cumprod()

    ret_s = pd.Series(aligned_rets.values, index=dates)
    summary = {
        "Strategy": strategy_name,
        "Gross_Ann_Return_Pct": round(float(annualized_return(ret_s) * 100.0), 2),
        "Net_Ann_Return_Pct": round(float(annualized_return(ret_s) * 100.0), 2),
        "Gross_CAGR_Pct": round(float(cagr_return(ret_s) * 100.0), 2),
        "Net_CAGR_Pct": round(float(cagr_return(ret_s) * 100.0), 2),
        "Ann_Volatility_Pct": round(float(annualized_volatility(ret_s) * 100.0), 2),
        "Gross_Sharpe": round(float(sharpe_ratio(ret_s, risk_free_annual=annual_rf)), 3),
        "Net_Sharpe": round(float(sharpe_ratio(ret_s, risk_free_annual=annual_rf)), 3),
        "Max_Drawdown_Pct": round(float(max_drawdown(ret_s) * 100.0), 2),
        "Gross_Max_Drawdown_Pct": round(float(max_drawdown(ret_s) * 100.0), 2),
        "Calmar_Ratio": round(float(calmar_ratio(ret_s)), 3),
        "Monthly_Turnover_Pct": 0.0,
        "Set_Turnover_Pct": 0.0,
        "Annual_Cost_Drag_Pct": 0.0,
        "Hit_Rate_Pct": round(float(hit_rate(ret_s) * 100.0), 1),
        "Terminal_Wealth_Gross": round(float(monthly_df["gross_wealth"].iloc[-1]), 4),
        "Terminal_Wealth_Net": round(float(monthly_df["net_wealth"].iloc[-1]), 4),
    }

    return {
        "summary": summary,
        "monthly_series": monthly_df,
        "weights": pd.DataFrame(),
    }


def run_full_portfolio_evaluation(
    df_test: pd.DataFrame,
    model_predictions: Dict[str, np.ndarray],
    daily_df: pd.DataFrame,
    top_k: int = 5,
    one_way_cost_bps: float = 10.0,
    annual_rf: float = DEFAULT_ANNUAL_RF,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Evaluate all ML Top-K portfolios alongside the three required financial benchmarks:
      1. Ridge Regression (Top-5)
      2. XGBoost Regression (Top-5)
      3. XGBoost Learning-to-Rank (Top-5)
      4. Momentum-Ranked Top-5 (MOM_12_1)
      5. Equal-Weighted Universe (1/N)
      6. Provided Nifty TRI series, or clearly labeled 19-stock proxy fallback

    Returns
    -------
    summary_df : pd.DataFrame
        Portfolio evaluation summary table with Gross/Net Return, CAGR, Sharpe, MaxDD, Turnover, Cost, and Active Return vs. Equal-Weight Universe.
    monthly_all_df : pd.DataFrame
        Combined monthly return, turnover, cost, and cumulative wealth series for all strategies.
    weights_dict : dict[str, pd.DataFrame]
        Monthly portfolio weight matrices by strategy.
    """
    eval_base = df_test.copy()
    summaries: List[Dict[str, object]] = []
    monthly_frames: List[pd.DataFrame] = []
    weights_dict: Dict[str, pd.DataFrame] = {}

    # 1. Evaluate ML Model Top-K Portfolios
    for model_name, preds in model_predictions.items():
        col_name = f"score_{model_name}"
        eval_base[col_name] = preds
        w_df = construct_top_k_weights(eval_base, score_col=col_name, top_k=top_k)
        res = simulate_rebalanced_portfolio(
            weights_df=w_df,
            df_eval=eval_base,
            strategy_name=model_name,
            one_way_cost_bps=one_way_cost_bps,
            annual_rf=annual_rf,
        )
        summaries.append(res["summary"])
        monthly_frames.append(res["monthly_series"])
        weights_dict[model_name] = w_df

    # 2. Benchmark 1: Momentum-Ranked Top-5 (MOM_12_1)
    w_mom = construct_top_k_weights(eval_base, score_col="MOM_12_1", top_k=top_k)
    res_mom = simulate_rebalanced_portfolio(
        weights_df=w_mom,
        df_eval=eval_base,
        strategy_name="Momentum Top-5 (MOM_12_1)",
        one_way_cost_bps=one_way_cost_bps,
        annual_rf=annual_rf,
    )
    summaries.append(res_mom["summary"])
    monthly_frames.append(res_mom["monthly_series"])
    weights_dict["Momentum Top-5 (MOM_12_1)"] = w_mom

    # 3. Benchmark 2: Equal-Weighted Full Universe (1/N)
    w_ew = construct_equal_weight_universe_weights(eval_base)
    res_ew = simulate_rebalanced_portfolio(
        weights_df=w_ew,
        df_eval=eval_base,
        strategy_name="Equal-Weight Universe (1/N)",
        one_way_cost_bps=one_way_cost_bps,
        annual_rf=annual_rf,
    )
    summaries.append(res_ew["summary"])
    monthly_frames.append(res_ew["monthly_series"])
    weights_dict["Equal-Weight Universe (1/N)"] = w_ew

    # 4. Benchmark 3: provided Nifty TRI series, or explicitly named 19-stock proxy.
    test_dates = sorted(eval_base["date"].unique())
    res_nifty = evaluate_nifty_benchmark(daily_df, eval_dates=test_dates, annual_rf=annual_rf)
    summaries.append(res_nifty["summary"])
    monthly_frames.append(res_nifty["monthly_series"])

    summary_df = pd.DataFrame(summaries)
    ew_net_ret = float(
        summary_df.loc[summary_df["Strategy"] == "Equal-Weight Universe (1/N)", "Net_Ann_Return_Pct"].iloc[0]
    )
    ew_gross_ret = float(
        summary_df.loc[summary_df["Strategy"] == "Equal-Weight Universe (1/N)", "Gross_Ann_Return_Pct"].iloc[0]
    )
    summary_df["Active_Gross_vs_EW_Pct"] = (summary_df["Gross_Ann_Return_Pct"] - ew_gross_ret).round(2)
    summary_df["Active_Net_vs_EW_Pct"] = (summary_df["Net_Ann_Return_Pct"] - ew_net_ret).round(2)

    monthly_all_df = pd.concat(monthly_frames, ignore_index=True)
    return summary_df, monthly_all_df, weights_dict
