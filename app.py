"""
Interactive Streamlit Application for ML-Based Cross-Sectional Equity Ranking (`app.py`).

Launch with:
    streamlit run app.py

Implements the 7 required interactive pages:
  1. Project Overview
  2. Dataset Explorer
  3. Feature Analysis
  4. Model Training and Comparison
  5. Cross-Sectional Stock Ranking (Central Interface)
  6. Ranking Performance
  7. Portfolio Evaluation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import EQUITY_UNIVERSE_NSE, DATASET_LIMITATIONS, validate_monthly_panel
from src.features.engineering import FEATURE_COLS, FEATURE_METADATA, verify_feature_timestamps
from src.labels.targets import TARGET_COL, build_pairwise_preferences_for_month

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(
    page_title="ML-Based Cross-Sectional Equity Ranking | NSE Universe",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Risk-free / cost conventions actually used by the pipeline (see src/backtest/portfolio.py)
SHARPE_RF_ANNUAL = 0.0666  # DEFAULT_ANNUAL_RF: mean RBI 91-day T-bill yield over Jul 2022 – Nov 2024
ONE_WAY_COST_BPS = 10.0  # applied per side -> cost = one-way drifted turnover x 20 bps

# Human-readable labels for raw result-column names (used in tables and chart legends)
READABLE = {
    "Test_Mean_Rank_IC": "Test Mean Rank IC",
    "Val_Mean_Rank_IC": "Val Mean Rank IC",
    "Val_NDCG_at_5": "Val NDCG@5",
    "Test_Rank_IC_Std": "Test Rank IC Std",
    "Test_ICIR": "Test ICIR",
    "Test_NW_IC_t_stat": "NW HAC t-stat (IC)",
    "Test_Pct_Positive_IC": "% Months IC > 0",
    "Test_NDCG_at_5": "NDCG@5",
    "Test_NDCG_at_10": "NDCG@10",
    "Top5_Ann_Excess_Pct": "Top-5 Excess (% p.a.)",
    "Bottom5_Ann_Excess_Pct": "Bottom-5 Excess (% p.a.)",
    "Top5_Minus_Bottom5_Ann_Spread_Pct": "Top-5 − Bottom-5 (% p.a.)",
    "Top5_Minus_Bottom5_Monthly_Pct": "Top-5 − Bottom-5 (%/mo)",
    "Val_RMSE": "Val RMSE",
    "Test_RMSE": "Test RMSE",
    "Test_MAE": "Test MAE",
    "Test_R2_OOS": "Test R² (OOS)",
    "Output_Type": "Output Type",
    "Gross_Ann_Return_Pct": "Gross Return (% p.a.)",
    "Net_Ann_Return_Pct": "Net Return (% p.a.)",
    "Gross_CAGR_Pct": "Gross CAGR (%)",
    "Net_CAGR_Pct": "Net CAGR (%)",
    "Ann_Volatility_Pct": "Volatility (% p.a.)",
    "Gross_Sharpe": "Gross Sharpe",
    "Net_Sharpe": "Net Sharpe",
    "Max_Drawdown_Pct": "Max DD Net (%)",
    "Gross_Max_Drawdown_Pct": "Max DD Gross (%)",
    "Calmar_Ratio": "Calmar",
    "Monthly_Turnover_Pct": "Drifted Turnover (%/mo)",
    "Set_Turnover_Pct": "Set Turnover (%/mo)",
    "Annual_Cost_Drag_Pct": "Cost Drag (% p.a.)",
    "Hit_Rate_Pct": "Hit Rate (%)",
    "Terminal_Wealth_Gross": "Terminal ₹ (Gross)",
    "Terminal_Wealth_Net": "Terminal ₹ (Net)",
    "Active_Gross_vs_EW_Pct": "Active Gross vs 1/N (%)",
    "Active_Net_vs_EW_Pct": "Active Net vs 1/N (%)",
    "Ridge_Normalized_Importance": "Ridge |coef| (normalized)",
    "XGB_Regression_Gain_Importance": "XGBoost Regression importance",
    "XGB_LTR_Pairwise_Gain_Importance": "XGBoost LTR importance",
    "Monthly_Turnover_Pct_bar": "Drifted Turnover (%/mo)",
}


def show_table(df: pd.DataFrame, height: int | str | None = None, rename: bool = True) -> None:
    """
    Render a dataframe at full width without index, with readable headers,
    wide text columns (no truncation of long strings) and an auto height that
    shows every row of small tables without an inner scrollbar.
    """
    out = df.rename(columns=READABLE) if rename else df
    col_cfg = {}
    for c in out.columns:
        if out[c].dtype == object or str(out[c].dtype).startswith("str"):
            max_len = int(out[c].astype(str).str.len().max()) if len(out) else 0
            if max_len > 28:
                col_cfg[c] = st.column_config.TextColumn(c, width="large")
    if height is None:
        height = "auto" if len(out) <= 30 else 560
    st.dataframe(out, width="stretch", hide_index=True, column_config=col_cfg, height=height)


def style_fig(fig: go.Figure, height: int = 460, legend_below: bool = True, angle_x: bool = False) -> go.Figure:
    """Consistent, non-clipping layout: smaller titles, room for legends and long tick labels."""
    fig.update_layout(
        height=height,
        title=dict(font=dict(size=14), x=0.0, xanchor="left"),
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title_text="",
    )
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0))
    if angle_x:
        fig.update_xaxes(tickangle=-25, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def readable_legend(fig: go.Figure) -> go.Figure:
    """Map raw column names used as trace names to readable labels."""
    fig.for_each_trace(lambda t: t.update(name=READABLE.get(t.name, t.name)))
    return fig


def melt_existing(
    df: pd.DataFrame,
    id_vars: list[str],
    value_vars: list[str],
    var_name: str,
    value_name: str,
) -> pd.DataFrame:
    """pandas.DataFrame.melt using only columns that actually exist on `df`."""
    present_id = [c for c in id_vars if c in df.columns]
    present_val = [c for c in value_vars if c in df.columns]
    missing_val = [c for c in value_vars if c not in df.columns]
    if not present_val:
        raise KeyError(
            f"melt value_vars not found. requested={value_vars} available={list(df.columns)}"
        )
    if missing_val:
        raise KeyError(
            f"melt value_vars missing from DataFrame: {missing_val}. available={list(df.columns)}"
        )
    kwargs: Dict[str, Any] = {
        "value_vars": present_val,
        "var_name": var_name,
        "value_name": value_name,
    }
    if present_id:
        kwargs["id_vars"] = present_id
    return df.melt(**kwargs)


def plot_chart(
    fig: go.Figure,
    height: int = 460,
    legend_below: bool = True,
    angle_x: bool = False,
) -> None:
    st.plotly_chart(
        readable_legend(style_fig(fig, height=height, legend_below=legend_below, angle_x=angle_x)),
        width="stretch",
    )


@st.cache_data
def load_all_artifacts() -> Dict[str, Any]:
    """Load dataset and precomputed pipeline results (or trigger pipeline if missing)."""
    if not (RESULTS_DIR / "monthly_stock_rankings.csv").exists():
        from run_pipeline import main as run_full_pipeline
        run_full_pipeline()

    monthly_df = pd.read_csv(DATA_DIR / "stocks_monthly_features.csv", parse_dates=["date"])
    daily_df = pd.read_csv(DATA_DIR / "stocks_daily_ohlcv.csv", parse_dates=["date"])
    rankings_df = pd.read_csv(RESULTS_DIR / "monthly_stock_rankings.csv", parse_dates=["date"])
    ranking_metrics_df = pd.read_csv(RESULTS_DIR / "ranking_metrics_comparison.csv")
    portfolio_summary_df = pd.read_csv(RESULTS_DIR / "portfolio_evaluation.csv")
    portfolio_monthly_df = pd.read_csv(RESULTS_DIR / "portfolio_monthly_returns.csv", parse_dates=["date"])
    feature_importance_df = pd.read_csv(RESULTS_DIR / "feature_importance_comparison.csv")
    stability_df = pd.read_csv(RESULTS_DIR / "ranking_stability.csv")
    pred_dist_df = pd.read_csv(RESULTS_DIR / "prediction_distributions.csv")
    monthly_ic_df = pd.read_csv(RESULTS_DIR / "monthly_rank_ic_series.csv", parse_dates=["date"])
    monthly_spread_df = pd.read_csv(RESULTS_DIR / "monthly_top_bottom_spread.csv", parse_dates=["date"])
    val_ridge_df = pd.read_csv(RESULTS_DIR / "validation_search_ridge.csv")
    val_xgb_reg_df = pd.read_csv(RESULTS_DIR / "validation_search_xgb_reg.csv")
    val_ltr_df = pd.read_csv(RESULTS_DIR / "label_scheme_validation.csv")
    target_summary_df = pd.read_csv(RESULTS_DIR / "target_construction_summary.csv")

    with open(RESULTS_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        model_metadata = json.load(f)

    return {
        "monthly_df": monthly_df,
        "daily_df": daily_df,
        "rankings_df": rankings_df,
        "ranking_metrics": ranking_metrics_df,
        "portfolio_summary": portfolio_summary_df,
        "portfolio_monthly": portfolio_monthly_df,
        "feature_importance": feature_importance_df,
        "stability": stability_df,
        "pred_dist": pred_dist_df,
        "monthly_ic": monthly_ic_df,
        "monthly_spread": monthly_spread_df,
        "val_ridge": val_ridge_df,
        "val_xgb_reg": val_xgb_reg_df,
        "val_ltr": val_ltr_df,
        "target_summary": target_summary_df,
        "model_metadata": model_metadata,
    }


def render_page_1_overview(data: Dict[str, Any]) -> None:
    st.title("📈 ML-Based Cross-Sectional Equity Ranking System")
    st.caption(
        "Undergraduate MLBD Mini-Project · 19 NSE Large-Cap Equities · Chronological Evaluation (Feb 2016 – Nov 2024)"
    )

    st.info(
        "**Historical Backtest Disclaimer**: All predictions, cross-sectional rankings, and portfolio simulations in this "
        "application are historical out-of-sample research evaluations over **July 2022 – November 2024** (plus validation "
        "months **January 2021 – June 2022**). They are **not** live investment forecasts or trading recommendations."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stock Universe", "19 NSE Equities", "9 Sectors (Balanced Panel)")
    col2.metric("Monthly Observations", "2,014 Stock-Months", "106 Formation Months")
    col3.metric("Financial Features", "13 Lagged Signals", "0 Look-Ahead Leakage")
    col4.metric("ML Ranking Models", "3 Architectures", "Ridge · XGBRegressor · XGBRanker")

    st.markdown("### 1. Problem Formulation: Cross-Sectional Equity Ranking")
    st.markdown(
        """
        In empirical asset pricing and quantitative equity selection, the primary decision at each monthly formation date $t$
        is **relative ordering** across the cross-section of $N = 19$ stocks rather than predicting exact point return magnitudes.
        Given a feature vector $\\mathbf{x}_{i,t} \\in \\mathbb{R}^{13}$ computed strictly from information available up to month $t$,
        our system produces a cross-sectional score $s_{i,t} = f(\\mathbf{x}_{i,t})$ and ranks all 19 equities from **Rank 1 (Highest Predicted Relative Performance)**
        to **Rank 19 (Lowest Predicted Relative Performance)** over month $t+1$.
        """
    )

    st.markdown("### 2. Three Implemented Ranking Approaches")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("#### Model 1: Ridge Regression")
        st.markdown(
            """
            - **Type**: Pointwise Regularized Linear Model (`sklearn.linear_model.Ridge`)
            - **Objective**: Minimizes L2-penalized squared error on continuous next-month excess returns:
              $$\\min_{\\mathbf{w}} \\sum_{i,t} (y_{i,t+1} - \\mathbf{w}^\\top \\mathbf{x}_{i,t})^2 + \\alpha \\|\\mathbf{w}\\|_2^2$$
            - **Ranking Mechanism**: Sorts predicted monthly excess returns $\\hat{y}_{i,t+1}$ within each formation month $t$.
            """
        )
    with m_col2:
        st.markdown("#### Model 2: XGBoost Regression")
        st.markdown(
            """
            - **Type**: Pointwise Gradient-Boosted Decision Trees (`XGBRegressor`)
            - **Objective**: `reg:squarederror` with chronological validation early stopping and L2 leaf regularization.
            - **Ranking Mechanism**: Captures non-linear feature interactions to predict $\\hat{y}_{i,t+1}$, then ranks stocks cross-sectionally each month.
            """
        )
    with m_col3:
        st.markdown("#### Model 3: XGBoost Learning-to-Rank")
        st.markdown(
            """
            - **Type**: Pairwise Ranking Boosted Trees (`XGBRanker`)
            - **Objective**: `rank:pairwise` grouped by formation month (`qid` = month index, 19 stocks per query group).
            - **Ranking Mechanism**: Directly optimizes pairwise preference probabilities $P(i \\succ_t j) = \\sigma(s_{i,t} - s_{j,t})$ over relevance grades derived strictly from training-period information.
            """
        )

    st.markdown("### 3. Chronological Train / Validation / Test Partition")
    split_df = pd.DataFrame(
        [
            {
                "Split": "Training Set",
                "Date Range": "February 2016 – December 2020",
                "Formation Months": 59,
                "Stock-Month Rows": 1121,
                "Role in Pipeline": "Fit feature StandardScaler, fit training relevance quantile bins, fit model parameters",
            },
            {
                "Split": "Validation Set",
                "Date Range": "January 2021 – June 2022",
                "Formation Months": 18,
                "Stock-Month Rows": 342,
                "Role in Pipeline": "Select Ridge alpha, XGBoost tree depth/regularization, and LTR relevance label scheme",
            },
            {
                "Split": "Out-of-Sample Test Set",
                "Date Range": "July 2022 – November 2024",
                "Formation Months": 29,
                "Stock-Month Rows": 551,
                "Role in Pipeline": "Strictly held-out evaluation of Ranking IC, NDCG@5/10, and Top-5 Portfolio Backtests",
            },
        ]
    )
    show_table(split_df, rename=False)

    st.markdown("### 4. Executive Summary of Out-of-Sample Results (Jul 2022 – Nov 2024)")
    left, right = st.columns(2)
    with left:
        st.markdown("**Ranking & Regression Performance Summary**")
        st.caption("Ranking IC / NDCG / Top-5−Bottom-5 use next-month **excess** returns (same figures as Page 6).")
        show_table(
            data["ranking_metrics"][
                [
                    "Model",
                    "Test_Mean_Rank_IC",
                    "Test_ICIR",
                    "Test_NDCG_at_5",
                    "Top5_Minus_Bottom5_Ann_Spread_Pct",
                    "Test_RMSE",
                ]
            ]
        )
    with right:
        st.markdown("**Top-5 Portfolio vs. Benchmarks Summary (Net of 20 bps Round-Trip Costs)**")
        st.caption(
            "Portfolio returns are **total** returns (not excess). Cost = drifted one-way turnover × 20 bps "
            f"(Sharpe uses rf = {SHARPE_RF_ANNUAL:.2%} p.a.). Same figures as Page 7."
        )
        show_table(
            data["portfolio_summary"][
                [
                    "Strategy",
                    "Gross_Ann_Return_Pct",
                    "Net_Ann_Return_Pct",
                    "Net_Sharpe",
                    "Max_Drawdown_Pct",
                    "Monthly_Turnover_Pct",
                ]
            ]
        )


def render_page_2_dataset_explorer(data: Dict[str, Any]) -> None:
    st.title("🗂️ Dataset Explorer & Validation")
    monthly_df = data["monthly_df"]
    daily_df = data["daily_df"]
    val_summary = validate_monthly_panel(monthly_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily OHLCV Rows", f"{len(daily_df):,}", "Jan 2015 – Dec 2024")
    c2.metric("Monthly Feature Panel", f"{val_summary['total_rows']:,} rows", f"{val_summary['num_months']} months × 19 stocks")
    c3.metric("Missing Feature Cells", f"{val_summary['missing_values_total']}", "Balanced Panel Verified")
    c4.metric("Duplicate (Date, Ticker)", f"{val_summary['duplicate_date_ticker']}", "Strict Integrity Check")

    st.markdown("### 1. NSE 19-Stock Equity Universe & Sector Breakdown")
    univ_rows = []
    for ticker, info in sorted(EQUITY_UNIVERSE_NSE.items()):
        sub = monthly_df[monthly_df["ticker"] == ticker]
        ann_ret = ((1.0 + sub["NEXT_MONTH_RETURN"].mean()) ** 12 - 1.0) * 100.0
        ann_vol = sub["NEXT_MONTH_RETURN"].std() * np.sqrt(12) * 100.0
        univ_rows.append(
            {
                "Ticker": ticker,
                "Company Name": info["name"],
                "Sector": info["sector"],
                "Observations": len(sub),
                "Start Month": str(sub["date"].min().date()),
                "End Month": str(sub["date"].max().date()),
                "Mean Ann. Return (%)": round(ann_ret, 2),
                "Ann. Volatility (%)": round(ann_vol, 2),
            }
        )
    univ_df = pd.DataFrame(univ_rows)
    show_table(univ_df, rename=False)

    col_a, col_b = st.columns(2)
    with col_a:
        sector_counts = univ_df.groupby("Sector", as_index=False)["Ticker"].count().rename(columns={"Ticker": "Stock Count"})
        fig_sec = px.bar(
            sector_counts,
            x="Sector",
            y="Stock Count",
            color="Sector",
            title="Universe Composition by Sector (19 NSE Large-Cap Equities)",
            text_auto=True,
        )
        plot_chart(fig_sec, height=420, legend_below=False, angle_x=True)

    with col_b:
        fig_ret = px.histogram(
            monthly_df,
            x=TARGET_COL,
            nbins=55,
            color="sector",
            title="Distribution of Next-Month Excess Returns (2,014 Stock-Months)",
            labels={TARGET_COL: "Next-Month Excess Return"},
        )
        plot_chart(fig_ret, height=420, legend_below=True)

    st.markdown("### 2. Cumulative Sector Wealth Trajectories (2016 – 2024)")
    sec_monthly = (
        monthly_df.groupby(["date", "sector"], as_index=False)["NEXT_MONTH_RETURN"].mean()
        .sort_values(["sector", "date"])
    )
    sec_monthly["Cumulative_Wealth"] = sec_monthly.groupby("sector")["NEXT_MONTH_RETURN"].transform(
        lambda s: (1.0 + s).cumprod()
    )
    fig_cum_sec = px.line(
        sec_monthly,
        x="date",
        y="Cumulative_Wealth",
        color="sector",
        title="Equal-Weighted Sector Cumulative Growth of ₹1 (Feb 2016 – Nov 2024)",
        labels={"Cumulative_Wealth": "Growth of ₹1", "date": "Formation Month"},
    )
    plot_chart(fig_cum_sec, height=440, legend_below=True)

    st.markdown("### 3. Documented Dataset Limitations")
    for idx, lim in enumerate(DATASET_LIMITATIONS, start=1):
        st.warning(f"**Limitation {idx}**: {lim}")


def render_page_3_feature_analysis(data: Dict[str, Any]) -> None:
    st.title("🔬 Financial Feature Engineering & Target Construction")
    monthly_df = data["monthly_df"]

    st.markdown("### 1. Feature Specification & Look-Ahead Verification (13 Lagged Signals)")
    ts_audit = verify_feature_timestamps(monthly_df)
    meta_rows = []
    for col in FEATURE_COLS:
        m = FEATURE_METADATA[col]
        meta_rows.append(
            {
                "Feature": col,
                "Category": m["category"],
                "Mathematical Definition": m["formula"],
                "Look-Back Window": m["window"],
                "Mean": round(float(monthly_df[col].mean()), 4),
                "Std": round(float(monthly_df[col].std()), 4),
                "P5": round(float(monthly_df[col].quantile(0.05)), 4),
                "Median": round(float(monthly_df[col].median()), 4),
                "P95": round(float(monthly_df[col].quantile(0.95)), 4),
                "Missing": int(monthly_df[col].isna().sum()),
            }
        )
    show_table(pd.DataFrame(meta_rows), rename=False)
    st.success(f"**Timestamp & Leakage Audit**: {ts_audit['timestamp_rule']}")

    st.markdown("### 2. Interactive Feature Distribution & Cross-Sectional Relationship")
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        selected_feat = st.selectbox("Select Financial Feature to Inspect", FEATURE_COLS, index=0)
        f_info = FEATURE_METADATA[selected_feat]
        st.markdown(f"**Category**: `{f_info['category']}`")
        st.markdown(f"**Formula**: `{f_info['formula']}`")
        st.markdown(f"**Look-back Window**: `{f_info['window']}`")
        rank_corr = monthly_df.groupby("date").apply(
            lambda g: g[selected_feat].corr(g[TARGET_COL], method="spearman"),
            include_groups=False,
        )
        st.metric("Univariate Mean Monthly Rank IC", f"{rank_corr.mean():+.4f}", f"Std: {rank_corr.std():.4f}")

    with f_col2:
        fig_box = px.box(
            monthly_df,
            x="sector",
            y=selected_feat,
            color="sector",
            title=f"Cross-Sector Distribution of {selected_feat}",
        )
        plot_chart(fig_box, height=420, legend_below=False, angle_x=True)

    st.markdown("### 3. Cross-Feature Spearman Correlation Matrix")
    corr_mat = monthly_df[FEATURE_COLS].corr(method="spearman").round(2)
    fig_corr = px.imshow(
        corr_mat,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Spearman Rank Correlation Across the 13 Financial Features",
        aspect="auto",
    )
    plot_chart(fig_corr, height=560, legend_below=False)

    st.markdown("### 4. Target & Ranking Label Construction (`src/labels/targets.py`)")
    st.markdown(
        """
        Target construction is strictly separated from feature engineering (`src/labels/targets.py`).
        For each formation month $t$, we construct four complementary target representations:
        1. **Continuous Excess Return (`NEXT_MONTH_EXCESS_RETURN`)**: $y_{i,t+1} = R_{i,t+1} - R_{f,t+1}$, used by **Ridge Regression** and **XGBoost Regression**.
        2. **Cross-Sectional Percentile Rank (`CS_PERCENTILE_RANK`)**: $p_{i,t+1} \\in [0, 1]$ within month $t$ ($1.0$ = top performer).
        3. **Relevance Grades for Learning-to-Rank (`0..4` or `0..18`)**: Discrete integer grades where higher values strictly represent better future stock performance (`Spearman rho > 0.90` with excess return).
        4. **Pairwise Preference Labels ($P(i \\succ_t j)$)**: All $\\binom{19}{2} = 171$ ordered pairs per month used by `rank:pairwise`.
        """
    )
    show_table(data["target_summary"], rename=False)


def render_page_4_model_training_comparison(data: Dict[str, Any]) -> None:
    st.title("⚙️ Model Training, Validation Selection & Feature Interpretation")

    st.markdown("### 1. Chronological Validation Hyperparameter & Label-Scheme Selection (`Jan 2021 – Jun 2022`)")
    st.markdown(
        "All hyperparameters and Learning-to-Rank label schemes were selected strictly using the **18-month Validation Split (Jan 2021 – Jun 2022)** without inspecting the Out-of-Sample Test Set."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Model 1: Ridge Validation Search",
            "Model 2: XGBoost Regression Validation Search",
            "Model 3: XGBoost LTR Label-Scheme & Hyperparameter Investigation",
        ]
    )
    with tab1:
        st.markdown("#### Ridge Regression Regularization Grid (`alpha`)")
        show_table(data["val_ridge"], rename=False)
        st.json(data["model_metadata"]["Ridge Regression"])
    with tab2:
        st.markdown("#### XGBoost Regression Validation Grid (`reg:squarederror`)")
        show_table(data["val_xgb_reg"], rename=False)
        st.json(data["model_metadata"]["XGBoost Regression"])
    with tab3:
        st.markdown(
            """
            #### XGBoost Learning-to-Rank (`rank:pairwise`) Relevance Label Investigation
            We evaluated three relevance label constructions on the chronological validation split (`Jan 2021 – Jun 2022`):
            - **`train_quantile_5`**: 5 integer grades (`0..4`) using strictly training-period (`Feb 2016 – Dec 2020`) excess-return quantile cutoffs.
            - **`cs_quintile_5`**: 5 integer grades (`0..4`) representing within-month cross-sectional quintiles (`4` = Top Quintile).
            - **`cs_ordinal_19`**: 19 exact ordinal grades (`0..18`) within each formation month (`18` = Best Stock).
            """
        )
        show_table(data["val_ltr"], rename=False)
        st.json(data["model_metadata"]["XGBoost Learning-to-Rank"])

    st.markdown("### 2. Feature Importance & Standardized Coefficient Comparison")
    feat_df = data["feature_importance"]
    # Actual columns in results/feature_importance_comparison.csv:
    # Feature, Category, Formula, Window, Ridge_Std_Coefficient, Ridge_Normalized_Importance,
    # XGB_Regression_Gain_Importance, XGB_LTR_Pairwise_Gain_Importance, Mean_Importance_Across_Models
    col_l, col_r = st.columns(2)
    with col_l:
        fig_ridge = px.bar(
            feat_df.sort_values("Ridge_Std_Coefficient"),
            x="Ridge_Std_Coefficient",
            y="Feature",
            color="Category",
            orientation="h",
            title="Model 1: Ridge Standardized Linear Coefficients",
        )
        plot_chart(fig_ridge, height=520, legend_below=True)

    with col_r:
        imp_melt = melt_existing(
            feat_df,
            id_vars=["Feature", "Category"],
            value_vars=[
                "Ridge_Normalized_Importance",
                "XGB_Regression_Gain_Importance",
                "XGB_LTR_Pairwise_Gain_Importance",
            ],
            var_name="Model_Importance_Type",
            value_name="Normalized_Importance",
        )
        fig_imp = px.bar(
            imp_melt,
            x="Normalized_Importance",
            y="Feature",
            color="Model_Importance_Type",
            barmode="group",
            orientation="h",
            title="Normalized Feature Importance: Ridge vs. XGBRegressor vs. XGBRanker",
        )
        plot_chart(fig_imp, height=520, legend_below=True)

    show_table(feat_df)

    st.markdown("### 3. Out-of-Sample Prediction Distributions & Cross-Sectional Dispersion")
    pred = data["pred_dist"].set_index("Model")
    ridge_p = pred.loc["Ridge Regression"]
    xgb_p = pred.loc["XGBoost Regression"]
    ltr_p = pred.loc["XGBoost Learning-to-Rank"]
    st.markdown(
        f"""
        Notice the fundamental difference in output scale across the three models (figures from `prediction_distributions.csv`):
        - **Ridge Regression** and **XGBoost Regression** output pointwise predicted monthly excess returns
          (Ridge mean `{ridge_p['Mean']:.5f}`, XGB mean `{xgb_p['Mean']:.5f}`).
          **XGBoost Regression** compresses the monthly cross-section
          (`Mean_CrossSectional_Std = {xgb_p['Mean_CrossSectional_Std']:.5f}`,
          `{xgb_p['Mean_CrossSectional_Std'] * 1e4:.0f}` bps/month) relative to Ridge
          (`{ridge_p['Mean_CrossSectional_Std']:.5f}`).
        - **XGBoost Learning-to-Rank (`rank:pairwise`)** outputs dimensionless pairwise margin scores
          centered near zero (`Mean = {ltr_p['Mean']:.5f}`, `Std = {ltr_p['Std']:.5f}`),
          engineered purely to order securities within each monthly query group.
        """
    )
    show_table(data["pred_dist"], rename=False)

    st.markdown("### 4. Ranking Stability Across Consecutive Months")
    show_table(data["stability"], rename=False)


def render_page_5_cross_sectional_stock_ranking(data: Dict[str, Any]) -> None:
    st.title("🏆 Cross-Sectional Stock Ranking Inspector (Central Interface)")
    st.markdown(
        """
        Select any historical formation month $t$ and ranking model below to inspect the complete cross-sectional ranking of all
        **19 NSE equities**, their predicted scores, key financial features at month $t$, and realized subsequent-month returns ($t+1$).
        """
    )
    st.caption(
        "⚠️ Note: All dates represent historical formation months in the Validation (Jan 2021 – Jun 2022) or Out-of-Sample Test (Jul 2022 – Nov 2024) windows. These are historical test evaluations, not live forecasts."
    )

    rankings_df = data["rankings_df"].copy()
    rankings_df["month_str"] = rankings_df["date"].dt.strftime("%Y-%m-%d")

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.4, 1.4])
    with ctrl1:
        split_options = ["Test (Jul 2022 - Nov 2024)", "Validation (Jan 2021 - Jun 2022)"]
        selected_split = st.selectbox("1. Select Evaluation Period", split_options, index=0)

    sub_split = rankings_df[rankings_df["split"] == selected_split]
    available_months = sorted(sub_split["month_str"].unique(), reverse=True)

    with ctrl2:
        selected_month = st.selectbox("2. Select Formation Month (t)", available_months, index=0)

    with ctrl3:
        selected_model = st.selectbox(
            "3. Select Ranking Model",
            ["XGBoost Learning-to-Rank", "Ridge Regression", "XGBoost Regression"],
            index=0,
        )

    month_df = sub_split[
        (sub_split["month_str"] == selected_month) & (sub_split["model"] == selected_model)
    ].sort_values("predicted_rank").reset_index(drop=True)

    # Compute month-specific ranking diagnostics
    month_ic = float(month_df["predicted_score"].corr(month_df[TARGET_COL], method="spearman"))
    top5_df = month_df[month_df["is_top5"]]
    bot5_df = month_df[month_df["is_bottom5"]]
    top5_ret_pct = float(top5_df[TARGET_COL].mean() * 100.0)
    bot5_ret_pct = float(bot5_df[TARGET_COL].mean() * 100.0)
    spread_pct = top5_ret_pct - bot5_ret_pct
    ew_ret_pct = float(month_df[TARGET_COL].mean() * 100.0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Selected Model", selected_model)
    m2.metric(f"Month ({selected_month}) Rank IC", f"{month_ic:+.4f}", "Spearman Rank Correlation")
    m3.metric("Top-5 Realized Excess Return", f"{top5_ret_pct:+.2f}%", f"vs EW Universe: {top5_ret_pct - ew_ret_pct:+.2f}%")
    m4.metric("Bottom-5 Realized Excess Return", f"{bot5_ret_pct:+.2f}%")
    m5.metric("Top-5 minus Bottom-5 Spread", f"{spread_pct:+.2f}%", "Long-Short Monthly Spread")

    # Model metadata banner
    meta_info = data["model_metadata"].get(selected_model, {})
    st.markdown(
        f"**Model Architecture & Objective**: `{meta_info.get('objective', '')}` · "
        f"**Validation-Selected Parameters**: `{json.dumps(meta_info.get('best_params', {}))}`"
    )

    st.markdown(f"### Complete Cross-Sectional Ranking Table — Formation Month `{selected_month}` ({selected_model})")

    display_table = month_df[
        [
            "predicted_rank",
            "ticker",
            "sector",
            "bucket",
            "predicted_score",
            "predicted_percentile",
            TARGET_COL,
            "NEXT_MONTH_RETURN",
            "realized_rank",
            "MOM_12_1",
            "MOM_6_1",
            "REV_1",
            "VOL_12M",
            "RSI_14",
            "SHARPE_12M",
        ]
    ].copy()
    display_table.columns = [
        "Predicted Rank (1=Best)",
        "Stock Symbol",
        "Sector",
        "Portfolio Bucket",
        "Predicted Score",
        "Percentile Rank",
        "Realized Excess Return (t+1)",
        "Realized Total Return (t+1)",
        "Realized Rank (1=Best)",
        "MOM_12_1",
        "MOM_6_1",
        "REV_1",
        "VOL_12M",
        "RSI_14",
        "SHARPE_12M",
    ]
    display_table["Predicted Score"] = display_table["Predicted Score"].round(5)
    display_table["Realized Excess Return (t+1)"] = (display_table["Realized Excess Return (t+1)"] * 100.0).round(2)
    display_table["Realized Total Return (t+1)"] = (display_table["Realized Total Return (t+1)"] * 100.0).round(2)
    for f_c in ["MOM_12_1", "MOM_6_1", "REV_1", "VOL_12M", "RSI_14", "SHARPE_12M"]:
        display_table[f_c] = display_table[f_c].round(3)

    show_table(display_table, height=680, rename=False)

    c_left, c_right = st.columns(2)
    with c_left:
        fig_rank_bar = px.bar(
            month_df.sort_values("predicted_rank", ascending=False),
            x="predicted_score",
            y="ticker",
            color="bucket",
            orientation="h",
            title=f"Predicted Cross-Sectional Scores by Stock ({selected_month})",
            color_discrete_map={
                "Top-5 (Selected)": "#16a34a",
                "Middle-9": "#64748b",
                "Bottom-5 (Lowest Ranked)": "#dc2626",
            },
            labels={"predicted_score": "Model Predicted Score", "ticker": "Stock Symbol"},
        )
        plot_chart(fig_rank_bar, height=560, legend_below=True)

    with c_right:
        month_df["Realized_Excess_Pct"] = month_df[TARGET_COL] * 100.0
        fig_real_bar = px.bar(
            month_df.sort_values("predicted_rank", ascending=False),
            x="Realized_Excess_Pct",
            y="ticker",
            color="bucket",
            orientation="h",
            title=f"Subsequent-Month Realized Excess Returns (%) Ordered by Predicted Rank ({selected_month})",
            color_discrete_map={
                "Top-5 (Selected)": "#16a34a",
                "Middle-9": "#64748b",
                "Bottom-5 (Lowest Ranked)": "#dc2626",
            },
            labels={"Realized_Excess_Pct": "Realized Excess Return (%)", "ticker": "Stock Symbol"},
        )
        plot_chart(fig_real_bar, height=560, legend_below=True)

    st.markdown(f"### Side-by-Side Rank Agreement Across All 3 Models in `{selected_month}`")
    all_models_month = sub_split[sub_split["month_str"] == selected_month]
    piv_ranks = all_models_month.pivot(index=["ticker", "sector", "realized_rank"], columns="model", values="predicted_rank").reset_index()
    piv_ranks = piv_ranks.sort_values("XGBoost Learning-to-Rank")
    show_table(piv_ranks, rename=False)


def render_page_6_ranking_performance(data: Dict[str, Any]) -> None:
    st.title("📊 Ranking vs. Regression Evaluation (Out-of-Sample Test Set)")
    ranking_df = data["ranking_metrics"]
    monthly_ic_df = data["monthly_ic"]
    monthly_spread_df = data["monthly_spread"]

    st.markdown(
        """
        ### 1. Why Ranking Metrics and Pointwise Regression Metrics Differ
        In high-noise monthly equity panels, minimizing pointwise squared error (`RMSE`) encourages models to shrink forecasts toward the
        cross-sectional mean (`~0.8%` per month). However, a model with small `RMSE` can still produce poor stock orderings if its tiny
        variations are noisy, whereas a model optimizing pairwise preferences (`XGBoost Learning-to-Rank`) outputs scale-free utility scores
        that directly target relative ordering (`Spearman Rank IC`, `NDCG@5`, `NDCG@10`, and `Top-5 minus Bottom-5 Spread`).
        """
    )

    st.markdown("#### A. Cross-Sectional Ranking Metrics Table (Jul 2022 – Nov 2024, 29 Test Months)")
    st.caption("All ranking metrics use next-month **excess** returns. These match Page 1 (left table) and the monthly IC/NDCG series below.")
    show_table(
        ranking_df[
            [
                "Model",
                "Objective",
                "Val_Mean_Rank_IC",
                "Test_Mean_Rank_IC",
                "Test_Rank_IC_Std",
                "Test_ICIR",
                "Test_NW_IC_t_stat",
                "Test_Pct_Positive_IC",
                "Test_NDCG_at_5",
                "Test_NDCG_at_10",
                "Top5_Ann_Excess_Pct",
                "Bottom5_Ann_Excess_Pct",
                "Top5_Minus_Bottom5_Ann_Spread_Pct",
            ]
        ]
    )

    st.markdown("#### B. Pointwise Regression Metrics Table (Strictly Separated from Ranking Metrics)")
    st.caption("RMSE / MAE / R² apply only to pointwise return models. Learning-to-Rank and the momentum heuristic leave these blank because they do not emit calibrated return forecasts.")
    show_table(
        ranking_df[
            [
                "Model",
                "Output_Type",
                "Val_RMSE",
                "Test_RMSE",
                "Test_MAE",
                "Test_R2_OOS",
            ]
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        ndcg_long = melt_existing(
            ranking_df,
            id_vars=["Model"],
            value_vars=["Test_NDCG_at_5", "Test_NDCG_at_10"],
            var_name="Metric",
            value_name="Score",
        )
        fig_ndcg = px.bar(
            ndcg_long,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Out-of-Sample NDCG@5 and NDCG@10 by Model",
            text_auto=".4f",
        )
        # Observed Test NDCG@5/10 in results is ~0.31–0.53. A [0.5, 0.85] axis hid every bar.
        fig_ndcg.update_yaxes(range=[0.0, 1.0], title_text="NDCG")
        plot_chart(fig_ndcg, height=480, angle_x=True)

    with col2:
        fig_tb = px.bar(
            melt_existing(
                ranking_df,
                id_vars=["Model"],
                value_vars=["Top5_Ann_Excess_Pct", "Bottom5_Ann_Excess_Pct", "Top5_Minus_Bottom5_Ann_Spread_Pct"],
                var_name="Bucket_Metric",
                value_name="Annualized_Return_Pct",
            ),
            x="Model",
            y="Annualized_Return_Pct",
            color="Bucket_Metric",
            barmode="group",
            title="Top-5 vs. Bottom-5 Annualized Realized Excess Return (%)",
            text_auto=".2f",
        )
        plot_chart(fig_tb, height=480, angle_x=True)

    st.markdown("### 2. Monthly Spearman Rank IC Time Series (Jul 2022 – Nov 2024)")
    fig_ic_ts = px.line(
        monthly_ic_df,
        x="date",
        y="Rank_IC",
        color="Model",
        markers=True,
        title="Monthly Cross-Sectional Spearman Rank IC Across 29 Out-of-Sample Test Months",
        labels={"Rank_IC": "Monthly Spearman Rank IC", "date": "Formation Month"},
    )
    fig_ic_ts.add_hline(y=0.0, line_dash="dash", line_color="gray")
    plot_chart(fig_ic_ts, height=440)

    st.markdown("### 3. Cumulative Top-5 Minus Bottom-5 Long-Short Spread")
    monthly_spread_df = monthly_spread_df.sort_values(["Model", "date"]).copy()
    monthly_spread_df["Cum_Spread"] = monthly_spread_df.groupby("Model")["spread"].transform(
        lambda s: (1.0 + s).cumprod() - 1.0
    ) * 100.0
    fig_sp = px.line(
        monthly_spread_df,
        x="date",
        y="Cum_Spread",
        color="Model",
        markers=True,
        title="Cumulative Top-5 Minus Bottom-5 Return Spread (%) Over the Test Window",
        labels={"Cum_Spread": "Cumulative Long-Short Spread (%)", "date": "Formation Month"},
    )
    fig_sp.add_hline(y=0.0, line_dash="dash", line_color="gray")
    plot_chart(fig_sp, height=440)


def render_page_7_portfolio_evaluation(data: Dict[str, Any]) -> None:
    st.title("💼 Portfolio Evaluation & Benchmark Comparison")
    portfolio_summary = data["portfolio_summary"]
    portfolio_monthly = data["portfolio_monthly"]

    st.markdown(
        """
        ### 1. Portfolio Construction Rules & Benchmarks
        At the close of each formation month $t$ across the **29-month Out-of-Sample Test Window (July 2022 – November 2024)**:
        - **ML Top-5 Portfolios (`Ridge Regression`, `XGBoost Regression`, `XGBoost Learning-to-Rank`)**: Allocate equal weights ($w_{i,t} = 20\\%$) to the **Top-5 highest-ranked stocks** and $0\\%$ to the remaining 14 stocks.
        - **Benchmark 1 — `Momentum Top-5 (MOM_12_1)`**: Equal-weighted Top-5 stocks ranked by 12M-1M price momentum (`MOM_12_1`).
        - **Benchmark 2 — `Equal-Weight Universe (1/N)`**: Equal weights ($1/19 \\approx 5.26\\%$) across all 19 NSE stocks, rebalanced monthly.
        - **Benchmark 3 — 19-stock liquidity-weighted proxy**: Calculated from the same 19-stock universe with a fixed 0.10% monthly dividend uplift. This is not the Nifty 50 price index or TRI; no external index series is present in the checked-in daily data.
        - **Transaction Cost Convention**: `10 bps` one-way (`20 bps` round-trip) applied to monthly drifted-weight turnover $TO_t = \\frac{1}{2}\\sum_i |w_{i,t} - w_{i,t-1}^{\\text{drift}}|$.
        - **Sharpe**: $(12\\bar{r} - r_f)/(\\sqrt{12}\\sigma)$ with $r_f = 6.66\\%$ p.a. (mean RBI 91-day T-bill over the test window). Page 6 Top-5 figures are **excess** returns of the selected names; this page reports **total** portfolio returns.
        """
    )

    show_table(portfolio_summary)

    c1, c2 = st.columns(2)
    with c1:
        fig_net_wealth = px.line(
            portfolio_monthly,
            x="date",
            y="net_wealth",
            color="strategy",
            title="Out-of-Sample Cumulative Wealth of ₹1 (Net of 20 bps Round-Trip Transaction Costs)",
            labels={"net_wealth": "Cumulative Net Wealth (₹)", "date": "Formation Month"},
        )
        plot_chart(fig_net_wealth, height=440)

    with c2:
        fig_gross_wealth = px.line(
            portfolio_monthly,
            x="date",
            y="gross_wealth",
            color="strategy",
            title="Out-of-Sample Cumulative Wealth of ₹1 (Gross Before Transaction Costs)",
            labels={"gross_wealth": "Cumulative Gross Wealth (₹)", "date": "Formation Month"},
        )
        plot_chart(fig_gross_wealth, height=440)

    st.markdown("### 2. Gross vs. Net Return, Turnover & Transaction Cost Drag")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_ret_comp = px.bar(
            melt_existing(
                portfolio_summary,
                id_vars=["Strategy"],
                value_vars=["Gross_Ann_Return_Pct", "Net_Ann_Return_Pct", "Net_CAGR_Pct"],
                var_name="Return_Metric",
                value_name="Annualized_Pct",
            ),
            x="Strategy",
            y="Annualized_Pct",
            color="Return_Metric",
            barmode="group",
            title="Gross Annualized Return, Net Annualized Return & Net CAGR (%)",
            text_auto=".2f",
        )
        plot_chart(fig_ret_comp, height=480, angle_x=True)

    with col_b:
        fig_to = px.bar(
            melt_existing(
                portfolio_summary,
                id_vars=["Strategy"],
                value_vars=["Monthly_Turnover_Pct", "Annual_Cost_Drag_Pct"],
                var_name="Turnover_Metric",
                value_name="Percent",
            ),
            x="Strategy",
            y="Percent",
            color="Turnover_Metric",
            barmode="group",
            title="Average Monthly Drifted Turnover (%) & Annualized Transaction Cost Drag (%)",
            text_auto=".2f",
        )
        plot_chart(fig_to, height=480, angle_x=True)

    st.markdown("### 3. Monthly Top-5 Holdings Log by Strategy")
    selected_strat = st.selectbox(
        "Select Strategy to Inspect Monthly Rebalanced Holdings & Returns",
        [
            "XGBoost Learning-to-Rank",
            "Ridge Regression",
            "XGBoost Regression",
            "Momentum Top-5 (MOM_12_1)",
        ],
    )
    strat_log = portfolio_monthly[portfolio_monthly["strategy"] == selected_strat][
        [
            "date",
            "holdings",
            "gross_return",
            "net_return",
            "turnover_drifted",
            "transaction_cost",
            "net_wealth",
        ]
    ].copy()
    strat_log["date"] = strat_log["date"].dt.strftime("%Y-%m-%d")
    strat_log["gross_return"] = (strat_log["gross_return"] * 100.0).round(2)
    strat_log["net_return"] = (strat_log["net_return"] * 100.0).round(2)
    strat_log["turnover_drifted"] = (strat_log["turnover_drifted"] * 100.0).round(2)
    strat_log["transaction_cost"] = (strat_log["transaction_cost"] * 10000.0).round(2)
    strat_log["net_wealth"] = strat_log["net_wealth"].round(4)
    strat_log.columns = [
        "Formation Month",
        "Top-5 Equal-Weighted Holdings",
        "Gross Monthly Return (%)",
        "Net Monthly Return (%)",
        "Monthly Drifted Turnover (%)",
        "Transaction Cost (bps)",
        "Cumulative Net Wealth (₹)",
    ]
    show_table(strat_log, rename=False)


def main() -> None:
    data = load_all_artifacts()

    st.sidebar.title("🧭 Navigation")
    st.sidebar.caption("ML-Based Cross-Sectional Equity Ranking")
    page = st.sidebar.radio(
        "Select View",
        [
            "1. Project Overview",
            "2. Dataset Explorer",
            "3. Feature Analysis",
            "4. Model Training & Comparison",
            "5. Cross-Sectional Stock Ranking",
            "6. Ranking Performance",
            "7. Portfolio Evaluation",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Chronological Split Summary**")
    st.sidebar.markdown(
        "- **Train**: Feb 2016 – Dec 2020 (59m)\n"
        "- **Validation**: Jan 2021 – Jun 2022 (18m)\n"
        "- **Test**: Jul 2022 – Nov 2024 (29m)\n"
        "- **Universe**: 19 NSE Stocks (2,014 rows)"
    )

    if page.startswith("1."):
        render_page_1_overview(data)
    elif page.startswith("2."):
        render_page_2_dataset_explorer(data)
    elif page.startswith("3."):
        render_page_3_feature_analysis(data)
    elif page.startswith("4."):
        render_page_4_model_training_comparison(data)
    elif page.startswith("5."):
        render_page_5_cross_sectional_stock_ranking(data)
    elif page.startswith("6."):
        render_page_6_ranking_performance(data)
    elif page.startswith("7."):
        render_page_7_portfolio_evaluation(data)


if __name__ == "__main__":
    main()
