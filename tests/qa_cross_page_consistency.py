"""
Cross-page consistency checks for the Streamlit app's data sources.

Verifies that figures shown on different pages agree with each other and with
the underlying data (no hardcoded numbers drifting from the exported results).

Run:  .venv/bin/python tests/qa_cross_page_consistency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def main() -> int:
    ok = True
    monthly = pd.read_csv(ROOT / "data" / "stocks_monthly_features.csv", parse_dates=["date"])
    rank = pd.read_csv(R / "ranking_metrics_comparison.csv").set_index("Model")
    port = pd.read_csv(R / "portfolio_evaluation.csv").set_index("Strategy")
    pm = pd.read_csv(R / "portfolio_monthly_returns.csv", parse_dates=["date"])
    spread = pd.read_csv(R / "monthly_top_bottom_spread.csv", parse_dates=["date"])
    ic = pd.read_csv(R / "monthly_rank_ic_series.csv", parse_dates=["date"])

    # 1. Implied risk-free rate embedded in the dataset's excess-return target
    rf_implied = (monthly["NEXT_MONTH_RETURN"] - monthly["NEXT_MONTH_EXCESS_RETURN"])
    print(f"Dataset target rf (monthly): min={rf_implied.min():.6f} max={rf_implied.max():.6f} "
          f"-> {rf_implied.mean() * 12:.4%} p.a.")

    # 2. Page 6 Top-5 excess (selection by score) == Page 7 portfolio gross excess series
    for model in ["Ridge Regression", "XGBoost Regression", "XGBoost Learning-to-Rank"]:
        s = spread[spread.Model == model].set_index("date")["top_k_return"]
        p = pm[pm.strategy == model].set_index("date")["gross_excess_return"]
        diff = float((s - p).abs().max())
        same = diff < 1e-12
        ok &= same
        print(f"[{'OK' if same else 'MISMATCH'}] {model}: Page6 Top-5 excess vs Page7 gross excess max|diff|={diff:.2e}")
    s = spread[spread.Model == "Momentum Baseline (MOM_12_1)"].set_index("date")["top_k_return"]
    p = pm[pm.strategy == "Momentum Top-5 (MOM_12_1)"].set_index("date")["gross_excess_return"]
    diff = float((s - p).abs().max())
    ok &= diff < 1e-12
    print(f"[{'OK' if diff < 1e-12 else 'MISMATCH'}] Momentum: Page6 vs Page7 max|diff|={diff:.2e}")

    # 3. Summary table values recompute from monthly series
    for strat, grp in pm.groupby("strategy"):
        gross = grp["gross_return"].mean() * 12 * 100
        net = grp["net_return"].mean() * 12 * 100
        tc_check = np.allclose(grp["gross_return"] - grp["net_return"], grp["transaction_cost"])
        g_ok = abs(gross - port.loc[strat, "Gross_Ann_Return_Pct"]) < 0.006
        n_ok = abs(net - port.loc[strat, "Net_Ann_Return_Pct"]) < 0.006
        ok &= g_ok and n_ok and tc_check
        print(f"[{'OK' if g_ok and n_ok and tc_check else 'MISMATCH'}] {strat}: gross {gross:.2f} vs {port.loc[strat, 'Gross_Ann_Return_Pct']}, "
              f"net {net:.2f} vs {port.loc[strat, 'Net_Ann_Return_Pct']}, gross-net==tc: {tc_check}")

    # 4. Transaction cost convention: cost == turnover * 20 bps
    ml = pm[pm.strategy.isin(["Ridge Regression", "XGBoost Regression", "XGBoost Learning-to-Rank"])]
    conv = np.allclose(ml["transaction_cost"], ml["turnover_drifted"] * 0.0020)
    ok &= conv
    print(f"[{'OK' if conv else 'MISMATCH'}] Transaction cost = one-way drifted turnover x 20 bps (10 bps/side)")

    # 5. Sharpe convention: (12*mean total - 6.66%) / (sqrt(12)*sd)
    for strat, grp in pm.groupby("strategy"):
        r = grp["net_return"]
        sh = (r.mean() * 12 - 0.0666) / (r.std(ddof=1) * np.sqrt(12))
        good = abs(sh - port.loc[strat, "Net_Sharpe"]) < 0.0006
        ok &= good
        print(f"[{'OK' if good else 'MISMATCH'}] {strat}: Net Sharpe recomputed {sh:.3f} vs table {port.loc[strat, 'Net_Sharpe']}")

    # 6. Mean of monthly IC / NDCG series == summary table
    for model, grp in ic.groupby("Model"):
        for col, tcol in [("Rank_IC", "Test_Mean_Rank_IC"), ("NDCG_at_5", "Test_NDCG_at_5"), ("NDCG_at_10", "Test_NDCG_at_10")]:
            good = abs(grp[col].mean() - rank.loc[model, tcol]) < 6e-5
            ok &= good
            if not good:
                print(f"[MISMATCH] {model} {col}: {grp[col].mean():.4f} vs {rank.loc[model, tcol]}")
    print("[OK] Monthly IC/NDCG series means match ranking summary table" if ok else "")

    # 7. NDCG value range (to pick a chart axis that actually shows the bars)
    nd = rank[["Test_NDCG_at_5", "Test_NDCG_at_10"]]
    print(f"NDCG range in results: {nd.min().min():.4f} .. {nd.max().max():.4f}")

    # 8. Universe facts used in page headers
    print(f"Universe: {monthly.ticker.nunique()} stocks, {monthly.sector.nunique()} sectors, "
          f"{monthly.date.nunique()} months, {len(monthly)} rows")

    print("\nRESULT:", "CONSISTENT" if ok else "INCONSISTENCIES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
