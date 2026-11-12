"""
Master Execution Pipeline for ML-Based Cross-Sectional Equity Ranking (`run_pipeline.py`).

Executes all 11 project steps end-to-end:
  1. Loads and validates the 19-stock NSE dataset (2,014 stock-months, Feb 2016 – Nov 2024).
  2. Verifies the 13 lagged financial features and confirms zero look-ahead leakage.
  3. Constructs continuous excess-return targets, cross-sectional percentile ranks,
     training-quantile relevance grades, query groups, and pairwise preference labels.
  4. Performs chronological train / validation / test partitioning:
       - Training:   February 2016 – December 2020 (59 months, 1,121 rows)
       - Validation: January 2021 – June 2022     (18 months,   342 rows)
       - Testing:    July 2022 – November 2024    (29 months,   551 rows)
  5. Trains Model 1: Ridge Regression (`sklearn.linear_model.Ridge`).
  6. Trains Model 2: XGBoost Regression (`xgboost.XGBRegressor`, `reg:squarederror`).
  7. Trains Model 3: XGBoost Learning-to-Rank (`xgboost.XGBRanker`, `rank:pairwise`)
     grouped by formation month (`qid`), comparing relevance label schemes on validation.
  8. Evaluates ranking metrics (Monthly Spearman Rank IC, Mean Rank IC, ICIR,
     NDCG@5, NDCG@10, Top-5 vs. Bottom-5 spread) alongside regression metrics (RMSE, MAE, OOS R²).
    9. Evaluates Top-5 monthly rebalanced portfolios against Momentum Top-5 (`MOM_12_1`),
       Equal-Weighted Universe (1/N), and an explicit index/proxy comparator.
 10. Extracts feature importances, Ridge coefficients, prediction distributions, and ranking stability.
 11. Exports all CSV tables to `results/` for the interactive Streamlit application (`app.py`).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.data.generate_dataset import acquire_or_generate_data
from src.data.loader import validate_monthly_panel, DATASET_LIMITATIONS
from src.features.engineering import FEATURE_COLS, verify_feature_timestamps
from src.evaluation.eda import run_eda
from src.models.ranking_models import (
    chronological_split_and_preprocess,
    train_ridge_ranking_model,
    train_xgboost_regression_model,
    train_xgboost_ltr_model,
    evaluate_ranking_and_regression_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== STEP 1: Loading & Validating NSE Stock Dataset ===")
    daily_df, monthly_df = acquire_or_generate_data(force_regenerate=False)
    validation_summary = validate_monthly_panel(monthly_df)
    with open(RESULTS_DIR / "dataset_validation.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "validation": validation_summary,
                "limitations": DATASET_LIMITATIONS,
            },
            f,
            indent=2,
        )
    logger.info(
        "Validated NSE dataset: %d rows, %d stocks, %d months (%s to %s)",
        validation_summary["total_rows"],
        validation_summary["num_stocks"],
        validation_summary["num_months"],
        validation_summary["start_date"],
        validation_summary["end_date"],
    )

    logger.info("=== STEP 2: Verifying Financial Feature Timestamps & Running EDA ===")
    ts_audit = verify_feature_timestamps(monthly_df)
    assert ts_audit["no_missing_values"] and ts_audit["no_infinite_values"], "Feature validation failed!"
    run_eda(daily_df, monthly_df)

    logger.info("=== STEP 3: Chronological Split & Target / Query-Group Construction ===")
    splits = chronological_split_and_preprocess(
        monthly_df,
        train_end="2020-12-31",
        val_end="2022-06-30",
    )
    logger.info(
        "Splits ready: Train=%d rows (%d months), Val=%d rows (%d months), Test=%d rows (%d months)",
        len(splits["df_train"]),
        len(splits["group_train"]),
        len(splits["df_val"]),
        len(splits["group_val"]),
        len(splits["df_test"]),
        len(splits["group_test"]),
    )

    logger.info("=== STEP 4: Training Model 1 — Ridge Regression ===")
    ridge_out = train_ridge_ranking_model(splits)
    logger.info("Ridge selected params: %s", ridge_out["best_params"])

    logger.info("=== STEP 5: Training Model 2 — XGBoost Regression (reg:squarederror) ===")
    xgb_reg_out = train_xgboost_regression_model(splits)
    logger.info("XGBoost Regression selected params: %s", xgb_reg_out["best_params"])

    logger.info("=== STEP 6: Training Model 3 — XGBoost Learning-to-Rank (rank:pairwise) ===")
    xgb_ltr_out = train_xgboost_ltr_model(splits)
    logger.info("XGBoost LTR selected params: %s", xgb_ltr_out["best_params"])

    models_dict = {
        "Ridge Regression": ridge_out,
        "XGBoost Regression": xgb_reg_out,
        "XGBoost Learning-to-Rank": xgb_ltr_out,
    }

    # Save selected model metadata for the Streamlit application
    model_metadata = {
        name: {
            "objective": out["objective"],
            "best_params": out["best_params"],
        }
        for name, out in models_dict.items()
    }
    with open(RESULTS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    logger.info("=== STEP 7: Evaluating Ranking Metrics, Regression Metrics & Top-5 Portfolios ===")
    eval_outputs = evaluate_ranking_and_regression_models(
        splits=splits,
        models_dict=models_dict,
        daily_df=daily_df,
        results_dir=RESULTS_DIR,
    )

    ranking_df = eval_outputs["ranking_metrics"]
    portfolio_df = eval_outputs["portfolio_summary"]
    feat_df = eval_outputs["feature_importance"]
    stab_df = eval_outputs["stability"]

    print("\n" + "=" * 110)
    print("1. RANKING VS. REGRESSION EVALUATION (OUT-OF-SAMPLE TEST SET: JUL 2022 – NOV 2024, 29 MONTHS)")
    print("=" * 110)
    rank_cols = [
        "Model",
        "Val_Mean_Rank_IC",
        "Test_Mean_Rank_IC",
        "Test_ICIR",
        "Test_NDCG_at_5",
        "Test_NDCG_at_10",
        "Top5_Minus_Bottom5_Monthly_Pct",
        "Top5_Minus_Bottom5_Ann_Spread_Pct",
        "Test_RMSE",
        "Test_MAE",
        "Test_R2_OOS",
    ]
    print(ranking_df[rank_cols].to_string(index=False))

    print("\n" + "=" * 110)
    print("2. PORTFOLIO EVALUATION (TOP-5 MONTHLY REBALANCED VS. BENCHMARKS, 20 BPS ROUND-TRIP COST)")
    print("=" * 110)
    port_cols = [
        "Strategy",
        "Gross_Ann_Return_Pct",
        "Net_Ann_Return_Pct",
        "Net_CAGR_Pct",
        "Gross_Sharpe",
        "Net_Sharpe",
        "Max_Drawdown_Pct",
        "Monthly_Turnover_Pct",
        "Annual_Cost_Drag_Pct",
        "Active_Net_vs_EW_Pct",
    ]
    print(portfolio_df[port_cols].to_string(index=False))

    print("\n" + "=" * 110)
    print("3. FEATURE INTERPRETATION (STANDARDIZED RIDGE COEFFICIENTS & XGBOOST GAIN IMPORTANCE)")
    print("=" * 110)
    print(
        feat_df[
            [
                "Feature",
                "Category",
                "Ridge_Std_Coefficient",
                "Ridge_Normalized_Importance",
                "XGB_Regression_Gain_Importance",
                "XGB_LTR_Pairwise_Gain_Importance",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("4. CROSS-SECTIONAL RANKING STABILITY ACROSS TEST MONTHS")
    print("=" * 110)
    print(stab_df.to_string(index=False))


if __name__ == "__main__":
    main()
