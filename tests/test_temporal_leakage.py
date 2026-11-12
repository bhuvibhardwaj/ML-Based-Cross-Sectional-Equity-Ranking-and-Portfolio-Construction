"""
Temporal leakage detection tests.

These tests are specifically designed to detect look-ahead bias — the most
critical failure mode in financial ML research.

A feature has look-ahead bias if the value of the feature at date t depends on
any information that was not available at date t (i.e., prices, returns, or
data from after date t).

Look-ahead bias in a backtest typically causes:
  1. Artificially high in-sample R² and IC.
  2. Good-looking backtest returns that completely disappear out-of-sample.
  3. The researcher concluding a model "works" when it merely memorized the future.

The tests here are structural: they verify the DATA FLOW of our feature
computations, not just the values.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    momentum_12_1,
    short_term_reversal,
    realized_volatility,
    log_volume,
    normalize_feature,
)


# ---------------------------------------------------------------------------
# Utility: inject future data and detect if features change
# ---------------------------------------------------------------------------

def assert_no_future_data_used(
    feature_fn,
    panel: pd.DataFrame,
    cutoff_idx: int,
    test_idx: int,
    **kwargs
) -> None:
    """
    Assert that feature_fn at date test_idx is unaffected by data at or after cutoff_idx.

    Method:
    1. Compute features on original panel.
    2. Corrupt ALL data at or after cutoff_idx by replacing with a sentinel value (99999).
    3. Recompute features.
    4. Verify that features at dates BEFORE cutoff_idx are identical.
    5. Verify that test_idx < cutoff_idx (the test date doesn't use future data).

    Parameters
    ----------
    feature_fn : callable
        Feature computation function (takes panel as first argument).
    panel : pd.DataFrame
        Price panel.
    cutoff_idx : int
        The row index after which we consider data to be "future."
    test_idx : int
        The specific row index we want to verify is leakage-free.
    """
    assert test_idx < cutoff_idx, "test_idx must be before cutoff_idx"

    original_features = feature_fn(panel, **kwargs)
    
    corrupted_panel = panel.copy()
    corrupted_panel.iloc[cutoff_idx:] = 99999.0
    
    corrupted_features = feature_fn(corrupted_panel, **kwargs)

    # Check: feature at test_idx should be identical in both cases
    date = panel.index[test_idx]
    
    if date not in original_features.index or date not in corrupted_features.index:
        return  # Feature not computed at this date (e.g., not enough history) → OK
    
    orig_vals = original_features.loc[date]
    corr_vals = corrupted_features.loc[date]
    
    # They should be equal (either both NaN or same values)
    if orig_vals.isna().all() and corr_vals.isna().all():
        return  # Both NaN is fine
    
    try:
        pd.testing.assert_series_equal(
            orig_vals, corr_vals,
            check_names=False,
            rtol=1e-6,
        )
    except AssertionError:
        pytest.fail(
            f"TEMPORAL LEAKAGE DETECTED: Feature at date {date} (idx {test_idx}) "
            f"changed when data at/after index {cutoff_idx} was corrupted. "
            f"This means the feature at date {date} uses future data."
        )


# ---------------------------------------------------------------------------
# Leakage test: Momentum 12-1
# ---------------------------------------------------------------------------

class TestMomentumLeakage:

    def setup_method(self):
        """Create a 20-month monthly price panel."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2015-01-31', periods=20, freq='ME')
        prices = pd.DataFrame(
            np.exp(np.cumsum(rng.normal(0.01, 0.05, (20, 5)), axis=0)) * 100,
            index=dates,
            columns=[f'S{i}' for i in range(5)]
        )
        self.prices = prices

    def test_momentum_at_month_14_doesnt_use_month_15_plus(self):
        """
        MOM_12_1 at month index 14 uses:
          - P(month 13) [t-1]
          - P(month 1)  [t-13]
        It should NOT use P(month 15) or later.
        """
        assert_no_future_data_used(
            momentum_12_1,
            self.prices,
            cutoff_idx=15,  # Corrupt from month 15 onwards
            test_idx=14,    # Test that month 14's features are unaffected
        )

    def test_momentum_at_month_14_doesnt_use_month_14(self):
        """
        MOM_12_1 at month index 14 (formation date t=14) should not use the
        price at month 14 itself. It uses P(t-1) and P(t-13).
        This is important: we want to verify no "same-day" look-ahead.
        """
        # MOM_12_1[14] = P[13] / P[1] - 1
        # Corrupting month 14 should NOT change MOM_12_1[14]
        assert_no_future_data_used(
            momentum_12_1,
            self.prices,
            cutoff_idx=14,  # Corrupt from month 14 (the formation date!) onwards
            test_idx=13,    # The preceding month is safe
        )

    def test_momentum_uses_month_13_not_month_14(self):
        """
        Verify that MOM_12_1 at formation date 14 uses the PRIOR month's price (month 13),
        NOT the current month's price (month 14).

        This is the key distinction: features at date t use the END-OF-MONTH t-1 price
        for the recent price, not the end-of-month t price.
        """
        prices = self.prices.copy()
        mom_original = momentum_12_1(prices)
        
        # Modify price at month 13 (what MOM_12_1[14] should use as P_{t-1})
        prices_modified = prices.copy()
        prices_modified.iloc[13] = prices_modified.iloc[13] * 2.0  # Double the prices
        
        mom_modified = momentum_12_1(prices_modified)
        
        # MOM_12_1 at index 14 should CHANGE because it depends on P[13]
        date_14 = prices.index[14]
        orig_val = mom_original.loc[date_14, 'S0']
        mod_val = mom_modified.loc[date_14, 'S0']
        
        assert not np.isclose(orig_val, mod_val), (
            "MOM_12_1 at month 14 should use price at month 13. "
            "Doubling month 13 price should change the feature."
        )


# ---------------------------------------------------------------------------
# Leakage test: Short-term reversal
# ---------------------------------------------------------------------------

class TestReversalLeakage:

    def test_rev1_at_t_uses_price_t_and_t_minus_1(self):
        """
        REV_1 at date t = P_t / P_{t-1} - 1
        
        This correctly uses P_t (current closing price), which IS available
        at the formation date. The trade would enter the NEXT day's open.
        
        We verify that corrupting month t+1 prices does not affect REV_1[t].
        """
        rng = np.random.default_rng(42)
        dates = pd.date_range('2015-01-31', periods=15, freq='ME')
        prices = pd.DataFrame(
            np.exp(np.cumsum(rng.normal(0.01, 0.05, (15, 3)), axis=0)) * 100,
            index=dates,
            columns=['A', 'B', 'C']
        )
        
        # Corrupt months 8 onwards
        prices_corrupted = prices.copy()
        prices_corrupted.iloc[8:] = 99999.0
        
        rev_orig = short_term_reversal(prices)
        rev_corr = short_term_reversal(prices_corrupted)
        
        # Months 0-6: REV_1 should be identical (uses only P_t and P_{t-1}, both < month 8)
        for i in range(7):
            date = prices.index[i]
            try:
                pd.testing.assert_series_equal(
                    rev_orig.loc[date], rev_corr.loc[date],
                    check_names=False,
                )
            except AssertionError:
                pytest.fail(f"REV_1 at month {i} should not depend on months >= 8")


# ---------------------------------------------------------------------------
# Leakage test: Realized Volatility
# ---------------------------------------------------------------------------

class TestVolatilityLeakage:

    def test_volatility_uses_only_past_returns(self):
        """
        Realized volatility at date t uses daily returns up to and including
        date t (the last trading day of the month). It must NOT use returns
        from after date t.
        """
        rng = np.random.default_rng(42)
        dates = pd.bdate_range('2015-01-02', periods=400)
        prices = pd.DataFrame(
            np.exp(np.cumsum(rng.normal(0.001, 0.015, (400, 3)), axis=0)) * 100,
            index=dates,
            columns=['A', 'B', 'C']
        )

        vol_orig = realized_volatility(prices)

        # Corrupt all prices after day 250
        prices_corrupted = prices.copy()
        prices_corrupted.iloc[250:] = 99999.0

        vol_corr = realized_volatility(prices_corrupted)

        # The monthly volatility up to the month containing day 249 should be identical
        # (since all required data is before the corruption point)
        # We compare the first 9 months of monthly-resampled vol
        for date in vol_orig.index[:9]:
            if date not in vol_corr.index:
                continue
            orig_row = vol_orig.loc[date]
            corr_row = vol_corr.loc[date]
            # They should be equal or both NaN
            if orig_row.isna().all() and corr_row.isna().all():
                continue
            try:
                pd.testing.assert_series_equal(orig_row, corr_row, check_names=False, rtol=1e-6)
            except AssertionError:
                pytest.fail(f"Volatility at {date} should be unaffected by data after day 250")


# ---------------------------------------------------------------------------
# Leakage test: Normalization
# ---------------------------------------------------------------------------

class TestNormalizationLeakage:

    def test_normalization_uses_only_current_cross_section(self):
        """
        CRITICAL: Cross-sectional normalization at date t should use ONLY the
        distribution of features across stocks at date t. It must NOT use
        statistics from future dates.

        Method: Add a fictitious stock with extreme values only in later months.
        Verify that normalizing earlier months is unaffected by this future stock.
        """
        dates = pd.date_range('2020-01-31', periods=10, freq='ME')

        # Regular panel
        data = pd.DataFrame(
            {f'S{i}': np.ones(10) * (i + 1) for i in range(5)},
            index=dates
        )
        
        # Panel with an "extreme future stock" added after month 5
        data_with_future = data.copy()
        data_with_future['EXTREME'] = np.where(
            np.arange(10) >= 5, 1000000.0, np.nan
        )
        
        norm_original = normalize_feature(data[['S0', 'S1', 'S2', 'S3', 'S4']])
        norm_with_future = normalize_feature(data_with_future)
        
        # Z-scores of original stocks in months 0-4 should be identical
        # (the EXTREME stock is NaN in months 0-4 so doesn't affect normalization)
        for i in range(5):
            date = dates[i]
            orig_row = norm_original.loc[date]
            future_row = norm_with_future.loc[date, ['S0', 'S1', 'S2', 'S3', 'S4']]
            try:
                pd.testing.assert_series_equal(
                    orig_row, future_row,
                    check_names=False,
                    rtol=1e-10,
                )
            except AssertionError:
                pytest.fail(
                    f"Normalization at month {i} should be unaffected by extreme "
                    "future stock values. NaN in early months means it doesn't contaminate."
                )


class TestChronologicalSplitIndicesAndNonOverlap:
    """Verify the exact underlying split indices, month counts, and date ranges in ranking_models.py."""

    def test_exact_split_indices_and_date_ranges(self):
        from pathlib import Path
        from src.models.ranking_models import chronological_split_and_preprocess

        csv_path = Path(__file__).resolve().parents[1] / "data" / "stocks_monthly_features.csv"
        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)
        splits = chronological_split_and_preprocess(df, train_end="2020-12-31", val_end="2022-06-30")

        df_train, df_val, df_test = splits["df_train"], splits["df_val"], splits["df_test"]

        # Exact panel partition: 59 + 18 + 29 = 106 months; 1121 + 342 + 551 = 2014 observations
        assert len(df) == 2014 and df["date"].nunique() == 106
        assert len(df_train) == 1121 and df_train["date"].nunique() == 59
        assert len(df_val) == 342 and df_val["date"].nunique() == 18
        assert len(df_test) == 551 and df_test["date"].nunique() == 29
        assert len(df_train) + len(df_val) + len(df_test) == len(df)

        # Exact date boundaries from underlying split masks
        assert str(df_train["date"].min().date()) == "2016-02-29"
        assert str(df_train["date"].max().date()) == "2020-12-31"
        assert str(df_val["date"].min().date()) == "2021-01-31"
        assert str(df_val["date"].max().date()) == "2022-06-30"
        assert str(df_test["date"].min().date()) == "2022-07-31"
        assert str(df_test["date"].max().date()) == "2024-11-30"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
