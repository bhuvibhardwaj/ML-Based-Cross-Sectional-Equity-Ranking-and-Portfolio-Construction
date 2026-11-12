"""
Tests for feature engineering.

These tests verify:
1. Mathematical correctness of each feature formula using hand-computed examples.
2. Temporal correctness: features at date t use no information from after date t.
3. Edge cases: missing data, short histories, all-missing cross-sections.

Run with: pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    momentum_12_1,
    short_term_reversal,
    realized_volatility,
    winsorize_cross_sectional,
    zscore_cross_sectional,
    normalize_feature,
)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

def make_price_panel(
    n_stocks: int = 5,
    n_months: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic monthly price panel for testing."""
    rng = np.random.default_rng(seed)
    # Generate random monthly log returns and cumulate into prices
    monthly_log_returns = rng.normal(0.01, 0.05, size=(n_months, n_stocks))
    log_prices = np.cumsum(monthly_log_returns, axis=0)
    prices = np.exp(log_prices) * 100  # Start at 100

    dates = pd.date_range(start='2015-01-31', periods=n_months, freq='ME')
    tickers = [f'STOCK{i:02d}' for i in range(n_stocks)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def make_daily_price_panel(
    n_stocks: int = 5,
    n_days: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic daily price panel for testing."""
    rng = np.random.default_rng(seed)
    daily_log_returns = rng.normal(0.0003, 0.015, size=(n_days, n_stocks))
    log_prices = np.cumsum(daily_log_returns, axis=0)
    prices = np.exp(log_prices) * 100

    dates = pd.bdate_range(start='2015-01-02', periods=n_days)
    tickers = [f'STOCK{i:02d}' for i in range(n_stocks)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Test 1: Momentum 12-1 correctness
# ---------------------------------------------------------------------------

class TestMomentum121:

    def test_hand_computed_single_stock(self):
        """
        Verify MOM_12_1 against a hand-computed example.

        MOM_12_1 at date t = price(t-1) / price(t-13) - 1
        """
        # Create a simple price series where prices double every month
        dates = pd.date_range(start='2015-01-31', periods=20, freq='ME')
        prices = pd.DataFrame(
            {'A': [100.0 * (2.0 ** i) for i in range(20)]},
            index=dates,
        )

        mom = momentum_12_1(prices)

        # At month index 13 (14th month, date = 2016-02-29):
        # t-1 = month 12, price = 100 * 2^12 = 409600
        # t-13 = month 0, price = 100 * 2^0 = 100
        # MOM_12_1 = 409600/100 - 1 = 4095
        # BUT: shifted by 13 means first valid at row 13
        # At row 13: shift(1) gives row 12, shift(13) gives row 0
        row_13_date = dates[13]
        expected = (100.0 * 2.0 ** 12) / (100.0 * 2.0 ** 0) - 1  # = 4095.0
        computed = mom.loc[row_13_date, 'A']
        assert abs(computed - expected) < 1e-6, f"Expected {expected:.4f}, got {computed:.4f}"

    def test_first_13_rows_are_nan(self):
        """MOM_12_1 requires 13 months of history; first 13 rows must be NaN."""
        prices = make_price_panel(n_stocks=3, n_months=20)
        mom = momentum_12_1(prices)
        assert mom.iloc[:13].isna().all().all(), "First 13 rows of MOM_12_1 must be NaN"

    def test_row_13_is_not_nan(self):
        """Row at index 13 (the 14th row) should have valid momentum values."""
        prices = make_price_panel(n_stocks=3, n_months=20)
        mom = momentum_12_1(prices)
        # Row 13 should be valid (has both P_{t-1} at row 12 and P_{t-13} at row 0)
        assert mom.iloc[13].notna().all(), "Row 13 of MOM_12_1 should be valid for all stocks"

    def test_temporal_correctness(self):
        """
        CRITICAL: MOM_12_1 at date t must use prices from BEFORE date t.

        Specifically: MOM_12_1[t] = f(prices[t-1], prices[t-13]).
        It must NOT use prices[t] or anything after.

        We verify by replacing prices after a cutoff date with NaN and
        checking that MOM_12_1 values before the cutoff are unchanged.
        """
        prices = make_price_panel(n_months=20)
        mom_original = momentum_12_1(prices)

        # Corrupt prices on and after month 15 by setting them to a very wrong value
        prices_corrupted = prices.copy()
        prices_corrupted.iloc[15:] = 99999.0

        mom_corrupted = momentum_12_1(prices_corrupted)

        # Months 0-13 (first 14 rows): MOM_12_1 does NOT use prices from rows 14+
        # Only months with t-1 >= 14 would use corrupted data.
        # Row 14 has t-1 = row 13 and t-13 = row 1 → safe (neither is in corrupted zone row 15+)
        # Row 16 has t-1 = row 15 (corrupted!) → should differ
        for i in range(14):  # rows 0-13
            row_date = prices.index[i]
            orig_vals = mom_original.loc[row_date]
            corr_vals = mom_corrupted.loc[row_date]
            # These should be identical because row i < 13 → all NaN
            pd.testing.assert_series_equal(orig_vals, corr_vals, check_names=False)

        # Row 16 (index 16) should be different because it uses corrupted price at row 15
        # t-1 for row 16 = row 15, which IS corrupted
        row16_date = prices.index[16]
        assert not np.allclose(
            mom_original.loc[row16_date].values,
            mom_corrupted.loc[row16_date].values,
            equal_nan=True
        ), "MOM_12_1 at row 16 should differ when row 15 prices are corrupted"


# ---------------------------------------------------------------------------
# Test 2: Short-term reversal
# ---------------------------------------------------------------------------

class TestShortTermReversal:

    def test_hand_computed(self):
        """REV_1 at date t = P_t / P_{t-1} - 1 (simple 1-month return)."""
        dates = pd.date_range('2020-01-31', periods=4, freq='ME')
        prices = pd.DataFrame({'A': [100, 110, 99, 121]}, index=dates)
        rev = short_term_reversal(prices)

        # Month 1: 110/100 - 1 = 0.10
        assert abs(rev.iloc[1, 0] - 0.10) < 1e-10, f"Got {rev.iloc[1, 0]}"
        # Month 2: 99/110 - 1 = -0.0909...
        expected = 99 / 110 - 1
        assert abs(rev.iloc[2, 0] - expected) < 1e-10

    def test_first_row_is_nan(self):
        """First row has no prior month, must be NaN."""
        prices = make_price_panel(n_months=5)
        rev = short_term_reversal(prices)
        assert rev.iloc[0].isna().all(), "First row of REV_1 must be NaN"


# ---------------------------------------------------------------------------
# Test 3: Cross-sectional normalization
# ---------------------------------------------------------------------------

class TestCrossSectionalNormalization:

    def test_zscore_has_zero_mean_unit_std(self):
        """
        After z-scoring, each row (date) should have mean ~0 and std ~1.
        """
        prices = make_price_panel(n_stocks=10, n_months=20)
        mom = momentum_12_1(prices).dropna(how='all')

        z = zscore_cross_sectional(mom)
        valid_rows = z.dropna(how='all')

        row_means = valid_rows.mean(axis=1)
        row_stds = valid_rows.std(axis=1, ddof=1)

        assert (row_means.abs() < 1e-10).all(), (
            f"Z-scored rows must have mean ~0. Max absolute mean: {row_means.abs().max():.2e}"
        )
        assert ((row_stds - 1.0).abs() < 1e-10).all(), (
            f"Z-scored rows must have std ~1. Max deviation: {(row_stds - 1.0).abs().max():.2e}"
        )

    def test_winsorize_clips_outliers(self):
        """
        After winsorizing at [1%, 99%], no value should exceed the 99th percentile
        of the original cross-sectional distribution.
        """
        # Create a panel with extreme outliers
        dates = pd.date_range('2020-01-31', periods=5, freq='ME')
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 1000],  # extreme outlier at date 5
            'B': [1, 2, 3, 4, 5],
            'C': [1, 2, 3, 4, 5],
            'D': [1, 2, 3, 4, 5],
            'E': [1, 2, 3, 4, 5],
        }, index=dates)

        winsorized = winsorize_cross_sectional(data, lower_pct=0.01, upper_pct=0.99)

        # On the last date, the outlier (1000) should be clipped
        last_row = winsorized.iloc[-1]
        # The maximum value on this date should be <= original 99th percentile
        original_last_row = data.iloc[-1]
        original_p99 = original_last_row.quantile(0.99)
        assert last_row['A'] <= original_p99 + 1e-6, (
            f"Winsorized value {last_row['A']} exceeds original 99th percentile {original_p99}"
        )

    def test_zscore_uses_only_cross_section(self):
        """
        CRITICAL: Z-scoring must use only the statistics of the CURRENT cross-section.
        It must NOT use any future cross-section statistics.

        We verify by checking that the z-score of a feature at date t depends
        ONLY on the values of that feature at date t, not at date t+1.
        """
        # Create a panel where the last date has very different values
        dates = pd.date_range('2020-01-31', periods=4, freq='ME')
        data = pd.DataFrame({
            'A': [1.0, 1.0, 1.0, 100.0],
            'B': [2.0, 2.0, 2.0, 200.0],
            'C': [3.0, 3.0, 3.0, 300.0],
        }, index=dates)

        z = zscore_cross_sectional(data)

        # On dates 0-2, the values are [1, 2, 3] → z-scores are [-1, 0, +1] (if exactly 3 stocks)
        # The last date (100, 200, 300) should NOT affect z-scores of earlier dates.
        # Check date 0: values [1, 2, 3]
        row0_mean = data.iloc[0].mean()  # = 2.0
        row0_std = data.iloc[0].std(ddof=0)  # = sqrt(2/3) ≈ 0.816
        expected_A_date0 = (1.0 - row0_mean) / row0_std

        # Note: ddof=1 is used in implementation; recompute
        row0_std_ddof1 = data.iloc[0].std(ddof=1)
        expected_A_date0_ddof1 = (1.0 - row0_mean) / row0_std_ddof1

        computed = z.iloc[0, 0]
        assert abs(computed - expected_A_date0_ddof1) < 1e-10, (
            f"Z-score at date 0 should be {expected_A_date0_ddof1:.6f}, got {computed:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Factor score (sanity)
# ---------------------------------------------------------------------------

class TestFactorScore:

    def test_higher_momentum_gets_higher_score(self):
        """
        With only MOM_12_1 and weight +1, stocks with higher momentum should
        receive higher composite factor scores.
        """
        from src.features.engineering import normalize_feature
        from src.portfolio.factor_score import compute_factor_scores, FACTOR_WEIGHTS

        prices = make_price_panel(n_stocks=5, n_months=20, seed=0)
        mom_raw = momentum_12_1(prices)
        mom_norm = normalize_feature(mom_raw)

        # Use only momentum with weight +1
        feature_dict = {'MOM_12_1': mom_norm}
        weights = {'MOM_12_1': +1.0}

        scores = compute_factor_scores(feature_dict, weights=weights)

        # At every date with valid scores, the stock with highest MOM_12_1 z-score
        # should also have the highest composite score
        for date, row in scores.iterrows():
            valid_scores = row.dropna()
            valid_mom = mom_norm.loc[date].dropna()
            if len(valid_scores) < 2 or len(valid_mom) < 2:
                continue
            common = valid_scores.index.intersection(valid_mom.index)
            # Rank correlation between score and momentum should be 1.0
            score_ranks = valid_scores[common].rank()
            mom_ranks = valid_mom[common].rank()
            corr = score_ranks.corr(mom_ranks, method='spearman')
            assert abs(corr - 1.0) < 1e-6, (
                f"At date {date}, score and momentum should be perfectly correlated. Got {corr:.4f}"
            )


# ---------------------------------------------------------------------------
# Test 5: Evaluation metrics mathematical correctness
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:

    def test_ic_is_spearman_correlation(self):
        """IC should equal scipy's Spearman correlation."""
        from scipy.stats import spearmanr
        from src.evaluation.metrics import information_coefficient

        rng = np.random.default_rng(42)
        n = 50
        pred = pd.Series(rng.normal(0, 1, n))
        real = pd.Series(rng.normal(0, 1, n) + 0.3 * pred)  # Some correlation

        scipy_corr, _ = spearmanr(pred.values, real.values)
        our_ic = information_coefficient(pred, real)

        assert abs(our_ic - scipy_corr) < 1e-10, (
            f"IC should equal scipy Spearman. scipy: {scipy_corr:.6f}, ours: {our_ic:.6f}"
        )

    def test_ic_perfect_ranking_is_1(self):
        """When predicted scores perfectly rank stocks, IC = 1."""
        from src.evaluation.metrics import information_coefficient

        n = 20
        real = pd.Series(range(n), dtype=float)   # returns = 0, 1, ..., 19
        pred = pd.Series(range(n), dtype=float)   # predictions = same order

        ic = information_coefficient(pred, real)
        assert abs(ic - 1.0) < 1e-10, f"Perfect ranking should give IC=1, got {ic}"

    def test_ic_reversed_ranking_is_minus_1(self):
        """When predicted scores perfectly reverse-rank stocks, IC = -1."""
        from src.evaluation.metrics import information_coefficient

        n = 20
        real = pd.Series(range(n), dtype=float)
        pred = pd.Series(range(n - 1, -1, -1), dtype=float)  # reversed order

        ic = information_coefficient(pred, real)
        assert abs(ic + 1.0) < 1e-10, f"Reversed ranking should give IC=-1, got {ic}"

    def test_maximum_drawdown_simple_case(self):
        """Verify MDD against a hand-computed example."""
        from src.evaluation.metrics import maximum_drawdown

        # Prices go: 100 → 110 → 90 → 95
        # Returns: NaN, 10%, -18.18%, 5.56%
        # Wealth index: 1 → 1.10 → 0.90 → 0.9455...
        # Peak at month 1: 1.10
        # Trough at month 2: 0.90
        # Drawdown = (1.10 - 0.90) / 1.10 = 0.1818...
        returns = pd.Series([0.10, -0.1818181818, 0.05556])
        mdd = maximum_drawdown(returns)
        expected = (1.10 - 0.90) / 1.10  # ≈ 0.1818...
        assert abs(mdd - expected) < 1e-4, f"MDD should be ~{expected:.4f}, got {mdd:.4f}"

    def test_annualized_return_multiplied_by_12(self):
        """Annualized return = mean monthly return * 12."""
        from src.evaluation.metrics import annualized_return

        monthly_rets = pd.Series([0.01, 0.02, -0.005, 0.03])
        expected = monthly_rets.mean() * 12
        computed = annualized_return(monthly_rets)
        assert abs(computed - expected) < 1e-12

    def test_sharpe_ratio_sign(self):
        """Sharpe ratio should be positive for returns > risk-free rate."""
        from src.evaluation.metrics import sharpe_ratio

        n = 36
        rf = pd.Series([0.004] * n)  # ~4.8% annual
        # Portfolio earns well above risk-free rate with realistic non-zero variance
        rng = np.random.default_rng(42)
        ret = pd.Series(0.015 + rng.normal(0, 0.002, n))

        sr = sharpe_ratio(ret, rf)
        assert sr > 0, f"Sharpe should be positive when returns exceed risk-free rate. Got {sr}"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
