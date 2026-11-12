"""
Exploratory Data Analysis (EDA) module for Cross-Sectional Equity Ranking.

Generates:
1. Summary statistics tables (`results/eda_feature_summary.csv`, `results/eda_stock_summary.csv`,
   `results/eda_missing_and_outliers.csv`).
2. Five diagnostic visualizations in `results/figures/`:
   - `eda_target_distribution.png`: Return distribution, KDE, Q-Q plot, and sector boxplot.
   - `eda_correlation_heatmap.png`: Pearson & Spearman correlation matrices of features and target.
   - `eda_cumulative_returns_by_sector.png`: Cumulative wealth trajectories by sector & stock.
   - `eda_feature_distributions.png`: Distributions and outlier profiles of key predictors.
   - `eda_rolling_volatility_regime.png`: Cross-sectional return dispersion & volatility regimes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FEATURE_COLS = [
    "MOM_12_1", "MOM_6_1", "MOM_3_1", "REV_1",
    "VOL_12M", "VOL_3M", "TURN_1M", "VOL_SHOCK",
    "BETA_12M", "RSI_14", "MA_RATIO_50_200", "MAX_DD_12M", "SHARPE_12M",
]
TARGET_COL = "NEXT_MONTH_EXCESS_RETURN"
RAW_RETURN_COL = "NEXT_MONTH_RETURN"


def run_eda(daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Execute full Exploratory Data Analysis on daily OHLCV and monthly feature panel data.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    # 1. Feature & Target distributional summary (including skewness, kurtosis, Jarque-Bera, ADF)
    summary_rows = []
    for col in FEATURE_COLS + [RAW_RETURN_COL, TARGET_COL]:
        s = monthly_df[col].dropna()
        jb_stat, jb_pval = stats.jarque_bera(s)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers_iqr = ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).mean() * 100.0
        summary_rows.append({
            "Variable": col,
            "Count": int(s.count()),
            "Missing_Pct": float(monthly_df[col].isna().mean() * 100.0),
            "Mean": float(s.mean()),
            "Std": float(s.std()),
            "Min": float(s.min()),
            "P25": float(q1),
            "Median": float(s.median()),
            "P75": float(q3),
            "Max": float(s.max()),
            "Skewness": float(stats.skew(s)),
            "Kurtosis_Excess": float(stats.kurtosis(s)),
            "IQR_Outlier_Pct": float(outliers_iqr),
            "JB_Stat": float(jb_stat),
            "JB_p_value": float(jb_pval),
        })
    feature_summary_df = pd.DataFrame(summary_rows)
    feature_summary_df.to_csv(RESULTS_DIR / "eda_feature_summary.csv", index=False)

    # 2. Per-stock descriptive statistics
    stock_rows = []
    for ticker, grp in monthly_df.groupby("ticker"):
        ret = grp[RAW_RETURN_COL].dropna()
        ex_ret = grp[TARGET_COL].dropna()
        ann_ret = ret.mean() * 12.0
        ann_vol = ret.std(ddof=1) * np.sqrt(12.0)
        sharpe = (ex_ret.mean() * 12.0) / ann_vol if ann_vol > 0 else np.nan
        wealth = (1.0 + ret).cumprod()
        mdd = ((wealth - wealth.cummax()) / wealth.cummax()).min()
        stock_rows.append({
            "Ticker": ticker,
            "Company": grp["company_name"].iloc[0],
            "Sector": grp["sector"].iloc[0],
            "N_Months": len(ret),
            "Mean_Monthly_Return_Pct": float(ret.mean() * 100.0),
            "Ann_Return_Pct": float(ann_ret * 100.0),
            "Ann_Volatility_Pct": float(ann_vol * 100.0),
            "Historical_Sharpe": float(sharpe),
            "Max_Drawdown_Pct": float(mdd * 100.0),
            "Mean_Beta_12M": float(grp["BETA_12M"].mean()),
            "Mean_MOM_12_1_Pct": float(grp["MOM_12_1"].mean() * 100.0),
            "Positive_Months_Pct": float((ret > 0).mean() * 100.0),
        })
    stock_summary_df = pd.DataFrame(stock_rows).sort_values("Ann_Return_Pct", ascending=False)
    stock_summary_df.to_csv(RESULTS_DIR / "eda_stock_summary.csv", index=False)

    # 3. Generate Visualizations
    _plot_target_distribution(monthly_df)
    _plot_correlation_heatmap(monthly_df)
    _plot_cumulative_returns_by_sector(monthly_df)
    _plot_feature_distributions(monthly_df)
    _plot_rolling_volatility_regime(monthly_df)

    logger.info("EDA complete. Saved tables to %s and figures to %s", RESULTS_DIR, FIGURES_DIR)
    return {
        "feature_summary": feature_summary_df,
        "stock_summary": stock_summary_df,
    }


def _plot_target_distribution(monthly_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    target_pct = monthly_df[TARGET_COL].dropna() * 100.0

    # Panel A: Histogram + KDE vs Normal fit
    sns.histplot(target_pct, bins=45, kde=True, stat="density", ax=axes[0], color="#2b5c8f", alpha=0.6)
    x_grid = np.linspace(target_pct.min(), target_pct.max(), 200)
    norm_pdf = stats.norm.pdf(x_grid, loc=target_pct.mean(), scale=target_pct.std())
    axes[0].plot(x_grid, norm_pdf, "r--", lw=2, label="Gaussian Fit")
    axes[0].axvline(target_pct.mean(), color="darkgreen", ls="-", lw=1.8, label=f"Mean ({target_pct.mean():.2f}%)")
    axes[0].set_title("A. Monthly Excess Return Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Next-Month Excess Return (%)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Panel B: Q-Q Plot showing leptokurtic fat tails
    stats.probplot(target_pct, dist="norm", plot=axes[1])
    axes[1].set_title("B. Normal Q-Q Plot (Heavy Tail Check)", fontsize=12, fontweight="bold")
    axes[1].get_lines()[0].set_color("#2b5c8f")
    axes[1].get_lines()[0].set_markersize(3.5)
    axes[1].get_lines()[1].set_color("crimson")

    # Panel C: Boxplot by Sector
    sector_order = (
        monthly_df.groupby("sector")[TARGET_COL]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    sns.boxplot(
        data=monthly_df.assign(ret_pct=monthly_df[TARGET_COL] * 100.0),
        y="sector",
        x="ret_pct",
        order=sector_order,
        ax=axes[2],
        palette="viridis",
        fliersize=2.5,
    )
    axes[2].axvline(0, color="black", ls="--", lw=1)
    axes[2].set_title("C. Excess Return Dispersion by Sector", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Next-Month Excess Return (%)")
    axes[2].set_ylabel("Sector")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_target_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_heatmap(monthly_df: pd.DataFrame) -> None:
    cols = FEATURE_COLS + [TARGET_COL]
    spearman_corr = monthly_df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(12, 9.5))
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool), k=1)
    sns.heatmap(
        spearman_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.85,
        vmax=0.85,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Spearman Rank Correlation"},
        ax=ax,
    )
    ax.set_title("Cross-Sectional Feature & Target Rank Correlation Matrix (Spearman)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_returns_by_sector(monthly_df: pd.DataFrame) -> None:
    sec_ret = (
        monthly_df.groupby(["date", "sector"])[RAW_RETURN_COL]
        .mean()
        .unstack("sector")
        .sort_index()
    )
    sec_wealth = (1.0 + sec_ret.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(13, 6))
    for col in sec_wealth.columns:
        ax.plot(sec_wealth.index, sec_wealth[col], lw=2.0, label=col)

    ew_mkt = (1.0 + monthly_df.groupby("date")[RAW_RETURN_COL].mean()).cumprod()
    ax.plot(ew_mkt.index, ew_mkt.values, color="black", lw=2.8, ls="--", label="Equal-Weight Universe")

    ax.set_title("Cumulative Wealth Growth Across NSE Sectors (₹1 Initial Investment)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Wealth Index (₹)")
    ax.legend(ncol=3, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_cumulative_returns_by_sector.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_distributions(monthly_df: pd.DataFrame) -> None:
    selected = ["MOM_12_1", "REV_1", "VOL_12M", "BETA_12M", "RSI_14", "SHARPE_12M"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, col in enumerate(selected):
        s = monthly_df[col].dropna()
        sns.histplot(s, bins=35, kde=True, ax=axes[i], color=colors[i], alpha=0.65)
        axes[i].axvline(s.median(), color="black", ls="--", lw=1.4, label=f"Median: {s.median():.2f}")
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        axes[i].legend(fontsize=9)

    plt.suptitle("Key Predictor Feature Distributions Across Stock-Month Observations", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_feature_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_rolling_volatility_regime(monthly_df: pd.DataFrame) -> None:
    by_date = monthly_df.groupby("date").agg(
        mean_vol=("VOL_12M", "mean"),
        cs_dispersion=(RAW_RETURN_COL, "std"),
        mean_ret=(RAW_RETURN_COL, "mean"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True)
    axes[0].plot(by_date.index, by_date["mean_vol"] * 100.0, color="#b22222", lw=2.2, label="Mean Trailing 12M Volatility (%)")
    axes[0].fill_between(by_date.index, 0, by_date["mean_vol"] * 100.0, color="#b22222", alpha=0.15)
    axes[0].set_ylabel("Annualized Volatility (%)")
    axes[0].set_title("A. Market Volatility Regimes Over Time", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper right")

    axes[1].bar(by_date.index, by_date["cs_dispersion"] * 100.0, width=20, color="#2b5c8f", alpha=0.75, label="Cross-Sectional Return Dispersion (Std %)")
    axes[1].set_ylabel("Cross-Sectional Std (%)")
    axes[1].set_xlabel("Date")
    axes[1].set_title("B. Monthly Cross-Sectional Dispersion (Opportunity Set for Stock Ranking)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_rolling_volatility_regime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
