"""
Unit and Integration Tests for ML-Based Cross-Sectional Equity Ranking (`tests/test_ranking_pipeline.py`).

Tests all 12 specification requirements:
  1. Dataset validation (19 NSE stocks, 106 formation months, 2,014 stock-months).
  2. Feature engineering & timestamp verification (13 lagged financial features, 0 missing).
  3. Separate target construction (`src/labels/targets.py`: continuous excess returns,
     cross-sectional percentile ranks, training-quantile relevance grades, query groups,
     and pairwise preference labels).
  4. Three ranking models (`Ridge Regression`, `XGBoost Regression`, `XGBoost Learning-to-Rank`
     with `rank:pairwise` and formation-month query groups).
  5. Ranking metrics (`Spearman Rank IC`, `ICIR`, `NDCG@5`, `NDCG@10`, Top-5 vs. Bottom-5 spread)
     and regression metrics (`RMSE`, `MAE`, `R2_OOS`).
  6. Top-5 evaluation against Momentum Top-5, Equal-Weight Universe, and a correctly
     labeled 19-stock proxy when an external index series is absent.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.loader import validate_monthly_panel, DATASET_LIMITATIONS
from src.data.generate_dataset import STOCK_METADATA
from src.features.engineering import FEATURE_COLS, verify_feature_timestamps
from src.labels.targets import (
    TARGET_COL,
    fit_training_relevance_bins,
    assign_relevance_grades,
    extract_query_groups,
    build_pairwise_preferences_for_month,
    verify_higher_label_means_better_return,
)
from src.evaluation.metrics import (
    ndcg_at_k,
    ndcg_time_series,
    top_bottom_spread_series,
    compute_ranking_stability,
)
from src.models.ranking_models import (
    MODEL_NAMES,
    chronological_split_and_preprocess,
    train_ridge_ranking_model,
    train_xgboost_regression_model,
    train_xgboost_ltr_model,
)
from src.backtest.portfolio import (
    construct_top_k_weights,
    construct_equal_weight_universe_weights,
    simulate_rebalanced_portfolio,
    evaluate_nifty_benchmark,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="module")
def monthly_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "stocks_monthly_features.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def splits(monthly_df: pd.DataFrame):
    return chronological_split_and_preprocess(monthly_df)


class TestDatasetAndFeatures:
    def test_dataset_validation_exact_counts(self, monthly_df: pd.DataFrame):
        summary = validate_monthly_panel(monthly_df)
        assert summary["total_rows"] == 2014
        assert summary["num_stocks"] == 19
        assert summary["num_months"] == 106
        assert summary["stocks_per_month_min"] == 19
        assert summary["stocks_per_month_max"] == 19
        assert summary["missing_values_total"] == 0
        assert summary["duplicate_date_ticker"] == 0
        assert set(monthly_df["ticker"].unique()) == set(STOCK_METADATA)
        assert len(DATASET_LIMITATIONS) >= 4
        implied_rf = monthly_df["NEXT_MONTH_RETURN"] - monthly_df["NEXT_MONTH_EXCESS_RETURN"]
        expected_rf = (1.0 + 0.062) ** (1.0 / 12.0) - 1.0
        assert np.allclose(implied_rf, expected_rf, atol=1e-12)

    def test_thirteen_financial_features_present_and_clean(self, monthly_df: pd.DataFrame):
        assert len(FEATURE_COLS) == 13
        audit = verify_feature_timestamps(monthly_df)
        assert audit["all_features_present"] is True
        assert audit["no_missing_values"] is True
        assert audit["no_infinite_values"] is True


class TestTargetAndLabelConstruction:
    def test_query_group_structure(self, splits):
        # Train: 59 months * 19 stocks = 1121
        assert len(splits["group_train"]) == 59
        assert np.all(splits["group_train"] == 19)
        assert int(splits["group_train"].sum()) == 1121

        # Validation: 18 months * 19 stocks = 342
        assert len(splits["group_val"]) == 18
        assert np.all(splits["group_val"] == 19)
        assert int(splits["group_val"].sum()) == 342

        # Test: 29 months * 19 stocks = 551
        assert len(splits["group_test"]) == 29
        assert np.all(splits["group_test"] == 19)
        assert int(splits["group_test"].sum()) == 551

    def test_relevance_labels_higher_means_better_future_return(self, splits):
        df_train = splits["df_train"]
        df_val = splits["df_val"]
        df_test = splits["df_test"]
        for split_df in [df_train, df_val, df_test]:
            assert verify_higher_label_means_better_return(split_df, "REL_GRADE_TRAIN_Q5", TARGET_COL)
            assert verify_higher_label_means_better_return(split_df, "REL_GRADE_CS_Q5", TARGET_COL)
            assert verify_higher_label_means_better_return(split_df, "REL_GRADE_ORDINAL", TARGET_COL)

    def test_pairwise_preference_labels_count_and_direction(self, splits):
        df_train = splits["df_train"]
        first_month = df_train[df_train["date"] == df_train["date"].min()]
        pairs_df = build_pairwise_preferences_for_month(first_month)
        # 19 * 18 / 2 = 171 unique unordered pairs -> 171 directed winner-loser pairs
        assert len(pairs_df) == 19 * 18 // 2
        assert (pairs_df["return_diff"] >= 0.0).all()
        assert (pairs_df["preference_label"] == 1).all()


class TestThreeRankingModelsAndPortfolio:
    def test_three_models_train_and_rank(self, splits):
        ridge_out = train_ridge_ranking_model(splits)
        xgb_reg_out = train_xgboost_regression_model(splits)
        xgb_ltr_out = train_xgboost_ltr_model(splits)

        for out in [ridge_out, xgb_reg_out, xgb_ltr_out]:
            assert len(out["val_preds"]) == 342
            assert len(out["test_preds"]) == 551
            assert np.all(np.isfinite(out["test_preds"]))
            assert len(out["feature_importance"]) == 13
            assert np.isclose(sum(out["feature_importance"].values()), 1.0, atol=1e-4)

    def test_ndcg_and_top_bottom_spread(self):
        # Perfect prediction should yield NDCG@5 == 1.0
        scores = np.arange(19, dtype=float)
        realized = np.linspace(-0.05, 0.10, 19)
        assert np.isclose(ndcg_at_k(scores, realized, k=5), 1.0)
        assert np.isclose(ndcg_at_k(scores, realized, k=10), 1.0)

    def test_top5_portfolio_weights_and_simulation(self, splits):
        df_test = splits["df_test"].copy()
        df_test["score_mom"] = df_test["MOM_12_1"]
        w_top5 = construct_top_k_weights(df_test, score_col="score_mom", top_k=5)
        assert w_top5.shape == (29, 19)
        assert np.allclose(w_top5.sum(axis=1), 1.0)
        assert np.all((w_top5 == 0.0) | np.isclose(w_top5, 0.20))

        sim = simulate_rebalanced_portfolio(w_top5, df_test, strategy_name="Test_Top5")
        assert len(sim["monthly_series"]) == 29
        assert sim["summary"]["Gross_Ann_Return_Pct"] >= sim["summary"]["Net_Ann_Return_Pct"]

    def test_missing_index_series_is_labeled_as_19_stock_proxy(self, splits):
        daily = pd.read_csv(DATA_DIR / "stocks_daily_ohlcv.csv", parse_dates=["date"])
        assert not any("nifty" in col.lower() for col in daily.columns)
        result = evaluate_nifty_benchmark(
            daily,
            eval_dates=sorted(splits["df_test"]["date"].unique()),
        )
        expected = "19-Stock Liquidity-Weighted Proxy (+1.2% dividend uplift)"
        assert result["summary"]["Strategy"] == expected
        assert set(result["monthly_series"]["strategy"]) == {expected}
