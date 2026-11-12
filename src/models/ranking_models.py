"""
Cross-Sectional Equity Ranking Models (`src/models/ranking_models.py`).

Implements and compares three machine learning approaches for monthly cross-sectional
equity ranking over the 19-stock NSE universe:
  1. Model 1: Ridge Regression (`sklearn.linear_model.Ridge`)
     Pointwise L2-regularized linear regression predicting next-month excess returns.
  2. Model 2: XGBoost Regression (`xgboost.XGBRegressor`, `objective="reg:squarederror"`)
     Pointwise gradient-boosted regression trees selected via chronological validation.
  3. Model 3: XGBoost Learning-to-Rank (`xgboost.XGBRanker`, `objective="rank:pairwise"`)
     Listwise/pairwise ranking model grouped by formation month (`qid` / `group`),
     optimizing pairwise preference orderings within each monthly cross-section.

Strictly enforces chronological train / validation / test partitioning:
  - Training:   February 2016 – December 2020 (59 formation months, 1,121 stock-months)
  - Validation: January 2021 – June 2022     (18 formation months,   342 stock-months)
  - Testing:    July 2022 – November 2024    (29 formation months,   551 stock-months)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.features.engineering import FEATURE_COLS, FEATURE_METADATA
from src.labels.targets import (
    TARGET_COL,
    construct_all_targets,
    extract_query_groups,
    fit_training_relevance_bins,
    assign_relevance_grades,
    verify_higher_label_means_better_return,
    summarize_target_constructions,
)
from src.evaluation.metrics import (
    rank_ic,
    ic_summary,
    ndcg_time_series,
    top_bottom_spread_series,
    compute_ranking_stability,
    compute_regression_metrics,
)
from src.backtest.portfolio import run_full_portfolio_evaluation

logger = logging.getLogger(__name__)

MODEL_NAMES = [
    "Ridge Regression",
    "XGBoost Regression",
    "XGBoost Learning-to-Rank",
]


def chronological_split_and_preprocess(
    df: pd.DataFrame,
    train_end: str = "2020-12-31",
    val_end: str = "2022-06-30",
    n_relevance_bins: int = 5,
) -> Dict[str, object]:
    """
    Partition the monthly stock-feature panel chronologically into Train, Validation,
    and Test sets, standardize features using ONLY training-period statistics, and
    construct target labels and query-group structures for Learning-to-Rank.

    Chronological Split:
      - Training:   February 2016 to December 2020 (59 months, 1,121 observations)
      - Validation: January 2021 to June 2022      (18 months, 342 observations)
      - Testing:    July 2022 to November 2024     (29 months, 551 observations)
    """
    panel = df.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    train_mask = panel["date"] <= pd.Timestamp(train_end)
    val_mask = (panel["date"] > pd.Timestamp(train_end)) & (panel["date"] <= pd.Timestamp(val_end))
    test_mask = panel["date"] > pd.Timestamp(val_end)

    # Fit training-only relevance bin edges on training split excess returns
    df_train_raw = panel.loc[train_mask].copy().reset_index(drop=True)
    train_bin_edges = fit_training_relevance_bins(
        df_train_raw, target_col=TARGET_COL, n_bins=n_relevance_bins
    )

    # Construct targets across all splits using ONLY training bin edges
    panel_with_targets, _ = construct_all_targets(
        panel,
        train_end=train_end,
        target_col=TARGET_COL,
        n_relevance_bins=n_relevance_bins,
    )

    df_train = panel_with_targets.loc[train_mask].copy().reset_index(drop=True)
    df_val = panel_with_targets.loc[val_mask].copy().reset_index(drop=True)
    df_test = panel_with_targets.loc[test_mask].copy().reset_index(drop=True)

    # Verify higher relevance grades strictly represent better future stock performance
    assert verify_higher_label_means_better_return(df_train, "REL_GRADE_TRAIN_Q5", TARGET_COL)
    assert verify_higher_label_means_better_return(df_train, "REL_GRADE_CS_Q5", TARGET_COL)
    assert verify_higher_label_means_better_return(df_train, "REL_GRADE_ORDINAL", TARGET_COL)

    # Standardize features using ONLY training set statistics (preventing look-ahead leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS].values.astype(np.float64))
    X_val = scaler.transform(df_val[FEATURE_COLS].values.astype(np.float64))
    X_test = scaler.transform(df_test[FEATURE_COLS].values.astype(np.float64))

    y_train = df_train[TARGET_COL].values.astype(np.float64)
    y_val = df_val[TARGET_COL].values.astype(np.float64)
    y_test = df_test[TARGET_COL].values.astype(np.float64)

    # Extract query groups (number of stocks per formation month) for XGBoost rank:pairwise
    group_train, qid_train, dates_train = extract_query_groups(df_train)
    group_val, qid_val, dates_val = extract_query_groups(df_val)
    group_test, qid_test, dates_test = extract_query_groups(df_test)

    return {
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "group_train": group_train,
        "group_val": group_val,
        "group_test": group_test,
        "qid_train": qid_train,
        "qid_val": qid_val,
        "qid_test": qid_test,
        "dates_train": dates_train,
        "dates_val": dates_val,
        "dates_test": dates_test,
        "scaler": scaler,
        "train_bin_edges": train_bin_edges,
        "feature_cols": list(FEATURE_COLS),
    }


def _compute_validation_rank_ic(df_val: pd.DataFrame, preds: np.ndarray) -> float:
    """Compute Mean Monthly Spearman Rank IC on the validation split."""
    temp = df_val[["date", "ticker", TARGET_COL]].copy()
    temp["pred"] = preds
    pred_piv = temp.pivot(index="date", columns="ticker", values="pred")
    real_piv = temp.pivot(index="date", columns="ticker", values=TARGET_COL)
    ic_s = rank_ic(pred_piv, real_piv)
    return float(ic_s.mean()) if len(ic_s) > 0 else 0.0


def _compute_validation_ndcg5(df_val: pd.DataFrame, preds: np.ndarray) -> float:
    """Compute Mean Monthly NDCG@5 on the validation split."""
    temp = df_val[["date", "ticker", TARGET_COL]].copy()
    temp["pred"] = preds
    pred_piv = temp.pivot(index="date", columns="ticker", values="pred")
    real_piv = temp.pivot(index="date", columns="ticker", values=TARGET_COL)
    ndcg_s = ndcg_time_series(pred_piv, real_piv, k=5)
    return float(ndcg_s.mean()) if len(ndcg_s) > 0 else 0.0


# ===========================================================================
# Model 1: Ridge Regression
# ===========================================================================

def train_ridge_ranking_model(splits: Dict[str, object]) -> Dict[str, object]:
    """
    Train Model 1: Ridge Regression (`sklearn.linear_model.Ridge`).

    Tunes L2 regularization strength `alpha` across a logarithmic grid using
    chronological validation (`Jan 2021 – Jun 2022`).
    """
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    df_val = splits["df_val"]

    alpha_grid = [0.1, 1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    validation_records: List[Dict[str, object]] = []

    best_alpha = 100.0
    best_val_ic = -np.inf
    best_model: Optional[Ridge] = None

    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_rmse = float(np.sqrt(np.mean((y_val - val_preds) ** 2)))
        val_ic = _compute_validation_rank_ic(df_val, val_preds)
        val_ndcg5 = _compute_validation_ndcg5(df_val, val_preds)

        validation_records.append(
            {
                "Model": "Ridge Regression",
                "Config": f"alpha={alpha}",
                "alpha": alpha,
                "Val_RMSE": round(val_rmse, 6),
                "Val_Mean_Rank_IC": round(val_ic, 4),
                "Val_NDCG_at_5": round(val_ndcg5, 4),
            }
        )

        # Select alpha that maximizes validation Rank IC (with tie-break toward stronger regularization)
        if val_ic > best_val_ic + 1e-5:
            best_val_ic = val_ic
            best_alpha = alpha
            best_model = model

    assert best_model is not None
    train_preds = best_model.predict(X_train)
    val_preds = best_model.predict(X_val)
    test_preds = best_model.predict(X_test)

    coefs = best_model.coef_.astype(float)
    abs_coefs = np.abs(coefs)
    norm_importance = abs_coefs / np.sum(abs_coefs) if np.sum(abs_coefs) > 0 else abs_coefs

    return {
        "name": "Ridge Regression",
        "objective": "L2 Penalized Squared Error (Pointwise Regression)",
        "model": best_model,
        "best_params": {"alpha": best_alpha},
        "train_preds": train_preds,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "coefficients": dict(zip(FEATURE_COLS, coefs)),
        "feature_importance": dict(zip(FEATURE_COLS, norm_importance)),
        "validation_search": pd.DataFrame(validation_records),
    }


# ===========================================================================
# Model 2: XGBoost Regression
# ===========================================================================

def train_xgboost_regression_model(splits: Dict[str, object]) -> Dict[str, object]:
    """
    Train Model 2: XGBoost Regression (`xgboost.XGBRegressor` with `reg:squarederror`).

    Uses chronological validation (`Jan 2021 – Jun 2022`) to select tree depth,
    learning rate, L2 regularization, and number of boosting rounds via early stopping.
    """
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    df_val = splits["df_val"]

    param_grid = [
        {"max_depth": 2, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0, "min_child_weight": 5},
        {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0, "min_child_weight": 5},
        {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 15.0, "min_child_weight": 10},
        {"max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 20.0, "min_child_weight": 10},
        {"max_depth": 4, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 10.0, "min_child_weight": 8},
    ]

    validation_records: List[Dict[str, object]] = []
    best_score = -np.inf
    best_model: Optional[xgb.XGBRegressor] = None
    best_params: Dict[str, object] = {}

    for idx, params in enumerate(param_grid):
        reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            min_child_weight=params["min_child_weight"],
            early_stopping_rounds=25,
            eval_metric="rmse",
            random_state=42,
            n_jobs=1,
        )
        reg.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_preds = reg.predict(X_val)
        val_rmse = float(np.sqrt(np.mean((y_val - val_preds) ** 2)))
        val_ic = _compute_validation_rank_ic(df_val, val_preds)
        val_ndcg5 = _compute_validation_ndcg5(df_val, val_preds)

        validation_records.append(
            {
                "Model": "XGBoost Regression",
                "Config": (
                    f"depth={params['max_depth']}, lr={params['learning_rate']}, "
                    f"lambda={params['reg_lambda']}, mcw={params['min_child_weight']}"
                ),
                **params,
                "best_iteration": int(getattr(reg, "best_iteration", 50)),
                "Val_RMSE": round(val_rmse, 6),
                "Val_Mean_Rank_IC": round(val_ic, 4),
                "Val_NDCG_at_5": round(val_ndcg5, 4),
            }
        )

        # Select configuration balancing validation Rank IC and negative RMSE on validation set
        composite_val = val_ic - 0.5 * val_rmse
        if composite_val > best_score:
            best_score = composite_val
            best_model = reg
            best_params = {**params, "best_iteration": int(getattr(reg, "best_iteration", 50))}

    assert best_model is not None
    train_preds = best_model.predict(X_train)
    val_preds = best_model.predict(X_val)
    test_preds = best_model.predict(X_test)

    imp = best_model.feature_importances_.astype(float)
    norm_imp = imp / np.sum(imp) if np.sum(imp) > 0 else imp

    return {
        "name": "XGBoost Regression",
        "objective": "reg:squarederror (Pointwise Gradient Boosted Trees)",
        "model": best_model,
        "best_params": best_params,
        "train_preds": train_preds,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "feature_importance": dict(zip(FEATURE_COLS, norm_imp)),
        "validation_search": pd.DataFrame(validation_records),
    }


# ===========================================================================
# Model 3: XGBoost Learning-to-Rank (rank:pairwise)
# ===========================================================================

def train_xgboost_ltr_model(splits: Dict[str, object]) -> Dict[str, object]:
    """
    Train Model 3: XGBoost Learning-to-Rank (`xgboost.XGBRanker` with `objective="rank:pairwise"`).

    Key Design Requirements (Section 5, Model 3):
      - Constructs ranking query groups using formation-month identifiers (`qid` / `group`),
        where each query group corresponds to the 19 securities available in that month.
      - Investigates three relevance label schemes where higher labels strictly represent
        better future stock performance:
          1. `train_quantile_5`: 5 relevance grades (0..4) using strictly training-period
             excess return quantile cutoffs (`fit_training_relevance_bins`).
          2. `cs_quintile_5`: 5 cross-sectional quintile grades (0..4) per formation month.
          3. `cs_ordinal_19`: 19 ordinal relevance grades (0..18) per formation month.
      - Selects the label scheme and hyperparameters strictly on the chronological
        validation set (`Jan 2021 – Jun 2022`) without touching the test set.
    """
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]
    y_train_cont = splits["y_train"]
    df_train = splits["df_train"]
    df_val = splits["df_val"]
    qid_train = splits["qid_train"]
    qid_val = splits["qid_val"]
    train_bin_edges = splits["train_bin_edges"]

    label_schemes = {
        "train_quantile_5": (
            assign_relevance_grades(df_train, scheme="train_quantile_5", train_bin_edges=train_bin_edges),
            assign_relevance_grades(df_val, scheme="train_quantile_5", train_bin_edges=train_bin_edges),
            "Training-Period Excess-Return Quantile Grades (0..4)",
        ),
        "cs_quintile_5": (
            assign_relevance_grades(df_train, scheme="cs_quintile_5"),
            assign_relevance_grades(df_val, scheme="cs_quintile_5"),
            "Cross-Sectional Monthly Quintile Grades (0..4)",
        ),
        "cs_ordinal_19": (
            assign_relevance_grades(df_train, scheme="cs_ordinal_19"),
            assign_relevance_grades(df_val, scheme="cs_ordinal_19"),
            "Cross-Sectional Exact Ordinal Grades (0..18)",
        ),
    }

    hp_candidates = [
        {"n_estimators": 60, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0, "min_child_weight": 5},
        {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0, "min_child_weight": 5},
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 15.0, "min_child_weight": 8},
        {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 20.0, "min_child_weight": 10},
    ]

    validation_records: List[Dict[str, object]] = []
    best_val_metric = -np.inf
    best_ranker: Optional[xgb.XGBRanker] = None
    best_config: Dict[str, object] = {}

    for scheme_key, (y_tr_rel, y_val_rel, scheme_desc) in label_schemes.items():
        # Verify monotonic alignment between relevance grades and realized excess returns within each month
        tmp_check = df_train[["date", TARGET_COL]].copy()
        tmp_check["rel"] = y_tr_rel
        monthly_rhos = [
            float(spearmanr(g["rel"].values, g[TARGET_COL].values)[0])
            for _, g in tmp_check.groupby("date")
            if g["rel"].nunique() > 1
        ]
        rho_train = float(np.mean(monthly_rhos))
        assert rho_train > 0.85, f"Label scheme {scheme_key} failed positive monotonicity check: {rho_train}"

        for hp in hp_candidates:
            ranker = xgb.XGBRanker(
                objective="rank:pairwise",
                eval_metric="ndcg@5",
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                learning_rate=hp["learning_rate"],
                subsample=hp["subsample"],
                colsample_bytree=hp["colsample_bytree"],
                reg_lambda=hp["reg_lambda"],
                min_child_weight=hp["min_child_weight"],
                random_state=42,
                n_jobs=1,
            )
            ranker.fit(
                X_train,
                y_tr_rel,
                qid=qid_train,
                eval_set=[(X_val, y_val_rel)],
                eval_qid=[qid_val],
                verbose=False,
            )

            val_scores = ranker.predict(X_val)
            val_ic = _compute_validation_rank_ic(df_val, val_scores)
            val_ndcg5 = _compute_validation_ndcg5(df_val, val_scores)

            # Compute validation Top-5 minus Bottom-5 monthly spread
            temp_val = df_val[["date", "ticker", TARGET_COL]].copy()
            temp_val["score"] = val_scores
            s_piv = temp_val.pivot(index="date", columns="ticker", values="score")
            r_piv = temp_val.pivot(index="date", columns="ticker", values=TARGET_COL)
            spread_df = top_bottom_spread_series(s_piv, r_piv, k=5)
            val_spread_bps = float(spread_df["spread"].mean() * 10000.0)

            validation_records.append(
                {
                    "Label_Scheme": scheme_key,
                    "Label_Description": scheme_desc,
                    "Train_Label_Return_Spearman": round(float(rho_train), 4),
                    "n_estimators": hp["n_estimators"],
                    "max_depth": hp["max_depth"],
                    "learning_rate": hp["learning_rate"],
                    "reg_lambda": hp["reg_lambda"],
                    "min_child_weight": hp["min_child_weight"],
                    "Val_Mean_Rank_IC": round(val_ic, 4),
                    "Val_NDCG_at_5": round(val_ndcg5, 4),
                    "Val_Top5_Minus_Bottom5_Bps": round(val_spread_bps, 2),
                }
            )

            # Selection criterion strictly on validation set (Rank IC + NDCG@5)
            selection_score = val_ic + 0.25 * val_ndcg5
            if selection_score > best_val_metric:
                best_val_metric = selection_score
                best_ranker = ranker
                best_config = {
                    "label_scheme": scheme_key,
                    "label_description": scheme_desc,
                    **hp,
                    "Val_Mean_Rank_IC": round(val_ic, 4),
                    "Val_NDCG_at_5": round(val_ndcg5, 4),
                }

    assert best_ranker is not None
    train_scores = best_ranker.predict(X_train)
    val_scores = best_ranker.predict(X_val)
    test_scores = best_ranker.predict(X_test)

    # Also fit a monotonic linear calibration from training LTR scores -> training excess return
    # purely for diagnostic regression comparison (does not change cross-sectional rankings!)
    score_std = float(np.std(train_scores))
    if score_std > 1e-9:
        slope = float(np.cov(train_scores, y_train_cont, ddof=0)[0, 1] / (score_std ** 2))
        intercept = float(np.mean(y_train_cont) - slope * np.mean(train_scores))
    else:
        slope, intercept = 0.0, float(np.mean(y_train_cont))

    calibrated_val_return = intercept + slope * val_scores
    calibrated_test_return = intercept + slope * test_scores

    imp = best_ranker.feature_importances_.astype(float)
    norm_imp = imp / np.sum(imp) if np.sum(imp) > 0 else imp

    return {
        "name": "XGBoost Learning-to-Rank",
        "objective": "rank:pairwise (Grouped by Formation Month qid)",
        "model": best_ranker,
        "best_params": best_config,
        "train_preds": train_scores,
        "val_preds": val_scores,
        "test_preds": test_scores,
        "calibrated_val_return": calibrated_val_return,
        "calibrated_test_return": calibrated_test_return,
        "feature_importance": dict(zip(FEATURE_COLS, norm_imp)),
        "validation_search": pd.DataFrame(validation_records),
    }


# ===========================================================================
# Comprehensive Evaluation & Export Pipeline
# ===========================================================================

def evaluate_ranking_and_regression_models(
    splits: Dict[str, object],
    models_dict: Dict[str, Dict[str, object]],
    daily_df: pd.DataFrame,
    results_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Evaluate all three ML ranking models + Momentum baseline across:
      1. Ranking metrics (Monthly Spearman Rank IC, Mean Rank IC, ICIR, NW t-stat,
         NDCG@5, NDCG@10, Top-5 return, Bottom-5 return, Top-5 minus Bottom-5 spread).
      2. Regression metrics (RMSE, MAE, OOS R^2) with clear distinction between pointwise
         return predictions and pairwise LTR utility scores.
      3. Portfolio evaluation (Top-5 equal-weighted portfolios vs. Momentum Top-5,
         Equal-Weight Universe 1/N, and the explicit index/proxy comparator).
      4. Feature importance, Ridge coefficients, prediction distributions, and ranking stability.
      5. Monthly stock rankings table for both Validation and Out-of-Sample Test periods.
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_train = splits["df_train"]
    df_val = splits["df_val"]
    df_test = splits["df_test"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    y_test = splits["y_test"]
    y_train_mean = float(np.mean(y_train))

    realized_test_piv = df_test.pivot(index="date", columns="ticker", values=TARGET_COL)
    realized_val_piv = df_val.pivot(index="date", columns="ticker", values=TARGET_COL)

    ranking_metrics_rows: List[Dict[str, object]] = []
    stability_rows: List[Dict[str, object]] = []
    pred_dist_rows: List[Dict[str, object]] = []
    monthly_ic_records: List[pd.DataFrame] = []
    monthly_spread_records: List[pd.DataFrame] = []

    # Include the 3 ML models + Momentum (MOM_12_1) signal for ranking comparison
    eval_signals: List[Tuple[str, str, np.ndarray, np.ndarray, bool]] = []
    for m_name in MODEL_NAMES:
        m_out = models_dict[m_name]
        is_ltr = m_name == "XGBoost Learning-to-Rank"
        eval_signals.append(
            (
                m_name,
                str(m_out["objective"]),
                np.asarray(m_out["val_preds"], dtype=float),
                np.asarray(m_out["test_preds"], dtype=float),
                is_ltr,
            )
        )

    # Also evaluate Momentum (MOM_12_1) as a transparent ranking baseline
    eval_signals.append(
        (
            "Momentum Baseline (MOM_12_1)",
            "12M-1M Return Signal (Heuristic Ranking Rule)",
            df_val["MOM_12_1"].values.astype(float),
            df_test["MOM_12_1"].values.astype(float),
            True,
        )
    )

    for name, objective_desc, val_scores, test_scores, is_rank_only in eval_signals:
        # Pivot predictions on validation and test sets
        tmp_val = df_val[["date", "ticker"]].copy()
        tmp_val["score"] = val_scores
        val_score_piv = tmp_val.pivot(index="date", columns="ticker", values="score")

        tmp_test = df_test[["date", "ticker"]].copy()
        tmp_test["score"] = test_scores
        test_score_piv = tmp_test.pivot(index="date", columns="ticker", values="score")

        # 1. Ranking Metrics on Validation & Test
        val_ic_s = rank_ic(val_score_piv, realized_val_piv)
        test_ic_s = rank_ic(test_score_piv, realized_test_piv)
        test_ic_stats = ic_summary(test_ic_s)

        val_ndcg5_s = ndcg_time_series(val_score_piv, realized_val_piv, k=5)
        test_ndcg5_s = ndcg_time_series(test_score_piv, realized_test_piv, k=5)
        test_ndcg10_s = ndcg_time_series(test_score_piv, realized_test_piv, k=10)

        spread_df = top_bottom_spread_series(test_score_piv, realized_test_piv, k=5)
        top5_mean_mo = float(spread_df["top_k_return"].mean())
        bot5_mean_mo = float(spread_df["bottom_k_return"].mean())
        spread_mean_mo = float(spread_df["spread"].mean())

        top5_ann_pct = ((1.0 + top5_mean_mo) ** 12 - 1.0) * 100.0
        bot5_ann_pct = ((1.0 + bot5_mean_mo) ** 12 - 1.0) * 100.0
        spread_ann_pct = top5_ann_pct - bot5_ann_pct

        # 2. Regression Metrics (only directly meaningful for pointwise return predictors)
        if not is_rank_only:
            reg_val = compute_regression_metrics(y_val, val_scores, y_train_mean)
            reg_test = compute_regression_metrics(y_test, test_scores, y_train_mean)
            val_rmse = round(reg_val["RMSE"], 5)
            test_rmse = round(reg_test["RMSE"], 5)
            test_mae = round(reg_test["MAE"], 5)
            test_r2_oos = round(reg_test["R2_OOS"], 4)
            output_scale_note = "Predicted Monthly Excess Return"
        elif name == "XGBoost Learning-to-Rank":
            # Raw LTR output is a dimensionless pairwise utility score; also report calibrated return RMSE in notes
            cal_test = models_dict[name]["calibrated_test_return"]
            cal_reg = compute_regression_metrics(y_test, cal_test, y_train_mean)
            val_rmse = np.nan
            test_rmse = np.nan
            test_mae = np.nan
            test_r2_oos = np.nan
            output_scale_note = (
                f"Pairwise Margin Score (Calibrated Return RMSE={cal_reg['RMSE']:.4f})"
            )
        else:
            val_rmse = np.nan
            test_rmse = np.nan
            test_mae = np.nan
            test_r2_oos = np.nan
            output_scale_note = "Raw 12M-1M Momentum Return Signal"

        ranking_metrics_rows.append(
            {
                "Model": name,
                "Objective": objective_desc,
                "Output_Type": output_scale_note,
                "Val_Mean_Rank_IC": round(float(val_ic_s.mean()), 4),
                "Val_NDCG_at_5": round(float(val_ndcg5_s.mean()), 4),
                "Test_Mean_Rank_IC": round(float(test_ic_stats["mean_ic"]), 4),
                "Test_Rank_IC_Std": round(float(test_ic_stats["std_ic"]), 4),
                "Test_ICIR": round(float(test_ic_stats["icir"]), 3),
                "Test_NW_IC_t_stat": round(float(test_ic_stats["nw_t_stat"]), 3),
                "Test_Pct_Positive_IC": round(float(test_ic_stats["pct_positive"] * 100.0), 1),
                "Test_NDCG_at_5": round(float(test_ndcg5_s.mean()), 4),
                "Test_NDCG_at_10": round(float(test_ndcg10_s.mean()), 4),
                "Top5_Avg_Monthly_Excess_Pct": round(top5_mean_mo * 100.0, 3),
                "Bottom5_Avg_Monthly_Excess_Pct": round(bot5_mean_mo * 100.0, 3),
                "Top5_Minus_Bottom5_Monthly_Pct": round(spread_mean_mo * 100.0, 3),
                "Top5_Ann_Excess_Pct": round(top5_ann_pct, 2),
                "Bottom5_Ann_Excess_Pct": round(bot5_ann_pct, 2),
                "Top5_Minus_Bottom5_Ann_Spread_Pct": round(spread_ann_pct, 2),
                "Val_RMSE": val_rmse,
                "Test_RMSE": test_rmse,
                "Test_MAE": test_mae,
                "Test_R2_OOS": test_r2_oos,
            }
        )

        # Store monthly IC and spread series
        ic_df = pd.DataFrame(
            {
                "date": test_ic_s.index,
                "Model": name,
                "Rank_IC": test_ic_s.values,
                "NDCG_at_5": test_ndcg5_s.reindex(test_ic_s.index).values,
                "NDCG_at_10": test_ndcg10_s.reindex(test_ic_s.index).values,
            }
        )
        monthly_ic_records.append(ic_df)

        sp_df = spread_df.reset_index().rename(columns={"index": "date"})
        sp_df["Model"] = name
        monthly_spread_records.append(sp_df)

        # 3. Ranking stability across test months
        stab = compute_ranking_stability(test_score_piv, top_k=5)
        stability_rows.append(
            {
                "Model": name,
                "Mean_Rank_Autocorr": round(stab["Mean_Rank_Autocorr"], 4),
                "Mean_Top5_Overlap_Fraction": round(stab["Mean_Top_K_Overlap"], 4),
                "Mean_Top5_Retained_Count": round(stab["Mean_Top_K_Overlap"] * 5.0, 2),
                "Implied_Monthly_Set_Turnover_Pct": round(stab["Implied_Monthly_Set_Turnover"] * 100.0, 2),
            }
        )

        # 4. Prediction distribution statistics on Test set
        cs_std = float(test_score_piv.std(axis=1, ddof=0).mean())
        pred_dist_rows.append(
            {
                "Model": name,
                "Output_Type": output_scale_note,
                "Mean": round(float(np.mean(test_scores)), 5),
                "Std": round(float(np.std(test_scores)), 5),
                "Mean_CrossSectional_Std": round(cs_std, 5),
                "Min": round(float(np.min(test_scores)), 5),
                "P10": round(float(np.percentile(test_scores, 10)), 5),
                "Median": round(float(np.median(test_scores)), 5),
                "P90": round(float(np.percentile(test_scores, 90)), 5),
                "Max": round(float(np.max(test_scores)), 5),
            }
        )

    ranking_metrics_df = pd.DataFrame(ranking_metrics_rows)
    stability_df = pd.DataFrame(stability_rows)
    pred_dist_df = pd.DataFrame(pred_dist_rows)
    monthly_ic_df = pd.concat(monthly_ic_records, ignore_index=True)
    monthly_spread_df = pd.concat(monthly_spread_records, ignore_index=True)

    # 5. Run Full Portfolio Evaluation (Top-5 ML models + Momentum Top-5 + Equal-Weight 1/N + index/proxy comparator)
    model_test_preds = {m_name: np.asarray(models_dict[m_name]["test_preds"], dtype=float) for m_name in MODEL_NAMES}
    portfolio_summary_df, portfolio_monthly_df, weights_dict = run_full_portfolio_evaluation(
        df_test=df_test,
        model_predictions=model_test_preds,
        daily_df=daily_df,
        top_k=5,
        one_way_cost_bps=10.0,
    )

    # 6. Feature Importance & Ridge Coefficients Table
    feat_rows: List[Dict[str, object]] = []
    ridge_coefs = models_dict["Ridge Regression"]["coefficients"]
    ridge_imp = models_dict["Ridge Regression"]["feature_importance"]
    xgb_reg_imp = models_dict["XGBoost Regression"]["feature_importance"]
    xgb_ltr_imp = models_dict["XGBoost Learning-to-Rank"]["feature_importance"]

    for col in FEATURE_COLS:
        meta = FEATURE_METADATA.get(col, {})
        feat_rows.append(
            {
                "Feature": col,
                "Category": meta.get("category", ""),
                "Formula": meta.get("formula", ""),
                "Window": meta.get("window", ""),
                "Ridge_Std_Coefficient": round(float(ridge_coefs[col]), 5),
                "Ridge_Normalized_Importance": round(float(ridge_imp[col]), 4),
                "XGB_Regression_Gain_Importance": round(float(xgb_reg_imp[col]), 4),
                "XGB_LTR_Pairwise_Gain_Importance": round(float(xgb_ltr_imp[col]), 4),
                "Mean_Importance_Across_Models": round(
                    float(np.mean([ridge_imp[col], xgb_reg_imp[col], xgb_ltr_imp[col]])), 4
                ),
            }
        )
    feature_importance_df = pd.DataFrame(feat_rows).sort_values(
        "Mean_Importance_Across_Models", ascending=False
    ).reset_index(drop=True)

    # 7. Export Monthly Stock Rankings Table (Validation + Out-of-Sample Test)
    monthly_rankings_df = build_monthly_stock_rankings_table(splits, models_dict)

    # Save all CSV outputs to results/
    ranking_metrics_df.to_csv(results_dir / "ranking_metrics_comparison.csv", index=False)
    portfolio_summary_df.to_csv(results_dir / "portfolio_evaluation.csv", index=False)
    portfolio_monthly_df.to_csv(results_dir / "portfolio_monthly_returns.csv", index=False)
    feature_importance_df.to_csv(results_dir / "feature_importance_comparison.csv", index=False)
    stability_df.to_csv(results_dir / "ranking_stability.csv", index=False)
    pred_dist_df.to_csv(results_dir / "prediction_distributions.csv", index=False)
    monthly_ic_df.to_csv(results_dir / "monthly_rank_ic_series.csv", index=False)
    monthly_spread_df.to_csv(results_dir / "monthly_top_bottom_spread.csv", index=False)
    monthly_rankings_df.to_csv(results_dir / "monthly_stock_rankings.csv", index=False)

    # Save validation hyperparameter & label scheme logs
    models_dict["Ridge Regression"]["validation_search"].to_csv(
        results_dir / "validation_search_ridge.csv", index=False
    )
    models_dict["XGBoost Regression"]["validation_search"].to_csv(
        results_dir / "validation_search_xgb_reg.csv", index=False
    )
    models_dict["XGBoost Learning-to-Rank"]["validation_search"].to_csv(
        results_dir / "label_scheme_validation.csv", index=False
    )

    # Save target construction summary
    full_panel = pd.concat([df_train, df_val, df_test], ignore_index=True)
    target_summary_df = summarize_target_constructions(full_panel)
    target_summary_df.to_csv(results_dir / "target_construction_summary.csv", index=False)

    return {
        "ranking_metrics": ranking_metrics_df,
        "portfolio_summary": portfolio_summary_df,
        "portfolio_monthly": portfolio_monthly_df,
        "feature_importance": feature_importance_df,
        "stability": stability_df,
        "prediction_distributions": pred_dist_df,
        "monthly_ic": monthly_ic_df,
        "monthly_spread": monthly_spread_df,
        "monthly_rankings": monthly_rankings_df,
        "weights_dict": weights_dict,
    }


def build_monthly_stock_rankings_table(
    splits: Dict[str, object],
    models_dict: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    """
    Build a comprehensive stock-month ranking table across Validation and Test splits
    so the user can select any formation month and model in the Streamlit app and inspect:
      - Stock symbol and sector
      - Predicted score
      - Cross-sectional rank (1 = highest predicted rank, 19 = lowest)
      - Cross-sectional percentile
      - Portfolio bucket (Top-5 Long, Middle-9 Neutral, Bottom-5 Avoid)
      - Realized next-month excess return, total return, and realized rank (1..19)
      - All 13 financial features
    """
    frames: List[pd.DataFrame] = []

    for split_label, df_key, pred_key in [
        ("Validation (Jan 2021 - Jun 2022)", "df_val", "val_preds"),
        ("Test (Jul 2022 - Nov 2024)", "df_test", "test_preds"),
    ]:
        base_df = splits[df_key].copy()
        base_df["split"] = split_label

        # Realized cross-sectional rank (1 = highest realized return in that month)
        base_df["realized_rank"] = (
            base_df.groupby("date")[TARGET_COL]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        for model_name in MODEL_NAMES:
            m_df = base_df[
                ["date", "split", "ticker", "sector", TARGET_COL, "NEXT_MONTH_RETURN", "realized_rank"]
                + list(FEATURE_COLS)
            ].copy()
            m_df["model"] = model_name
            m_df["predicted_score"] = np.asarray(models_dict[model_name][pred_key], dtype=float)

            # Rank 1 = highest predicted score within each formation month
            m_df["predicted_rank"] = (
                m_df.groupby("date")["predicted_score"]
                .rank(ascending=False, method="first")
                .astype(int)
            )
            n_per_month = m_df.groupby("date")["predicted_rank"].transform("count")
            m_df["predicted_percentile"] = (
                (n_per_month - m_df["predicted_rank"]) / np.maximum(n_per_month - 1, 1)
            ).round(4)

            m_df["is_top5"] = m_df["predicted_rank"] <= 5
            m_df["is_bottom5"] = m_df["predicted_rank"] > (n_per_month - 5)
            m_df["bucket"] = np.where(
                m_df["is_top5"],
                "Top-5 (Selected)",
                np.where(m_df["is_bottom5"], "Bottom-5 (Lowest Ranked)", "Middle-9"),
            )
            frames.append(m_df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "model", "predicted_rank"]).reset_index(drop=True)
    return combined
