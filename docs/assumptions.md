# Assumptions and Limitations

These statements describe the current implementation and its stored artifacts.

## Data and universe

- The evaluated panel contains 19 stocks, 106 months and 2,014 rows. It is a fixed
  subset, not a Nifty 100 universe or a point-in-time index membership sample.
- The data generation code attempts Yahoo Finance acquisition and falls back to seeded
  synthetic data. Existing CSV files have no provenance field, so their source cannot
  be determined from repository evidence.
- The available features are OHLCV technical and risk measures. No accounting
  fundamentals or historical index constituent membership are included.
- Because selection history and data provenance are absent, survivorship/selection bias
  and market realism cannot be quantified.

## Target, split and model assumptions

- Formation month-end features predict the close-to-close simple return in the following
  month. The target subtracts the geometric monthly equivalent of a constant 6.2%
  annual rate.
- The fixed chronological split is 59 training months, 18 validation months and 29 test
  months. There is no rolling retraining or purging of adjacent month boundaries.
- Only Ridge, XGBoost regression and XGBoost pairwise ranking are implemented.
- The portfolio holds five equally weighted names and rebalances monthly.

## Performance conventions

- Portfolio Sharpe uses total monthly returns, a constant 6.66% annual rate, arithmetic
  annualization and sample standard deviation. The target's 6.2% assumption is separate.
- Trading costs are simplified: 10 bps one-way is doubled against half-L1 drifted
  turnover; the first test month has no starting-position cost.
- The market comparator falls back to a liquidity-weighted return proxy from the same
  19 stocks plus a fixed monthly dividend uplift. It is not the Nifty 50 price index or
  TRI. No external index series is stored.
- Backtest results are historical calculations and are not live forecasts.
