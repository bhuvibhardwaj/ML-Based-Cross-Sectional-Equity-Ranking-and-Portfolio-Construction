# Executable Methodology

This document follows `run_pipeline.py` and the code under `src/`. It replaces the
earlier proposed Nifty 100 / Top-20 specification, which is not implemented.

## Panel and features

The checked-in monthly panel contains 19 tickers, 106 monthly formation dates, and
2,014 rows (February 2016–November 2024). It uses 13 OHLCV-derived features calculated
from information available through formation month-end. The evaluated universe is this
19-stock subset; it is not the Nifty 100.

At formation month *t*, the target is
`NEXT_MONTH_EXCESS_RETURN = NEXT_MONTH_RETURN − ((1 + 0.062) ** (1/12) − 1)`. The next-month total return is
the simple return from the month-*t* close to the month-*t+1* close. Relevance bins are
fit on training targets only. Cross-sectional percentile ranks and ordinal labels are
also constructed for diagnostics and learning-to-rank.

## Models and chronological split

The executable pipeline trains only:

1. Ridge regression (`sklearn.linear_model.Ridge`).
2. XGBoost regression (`reg:squarederror`).
3. XGBoost Learning-to-Rank (`rank:pairwise`), grouped by formation month.

| Split | Period | Months | Stock-month rows |
|---|---|---:|---:|
| Train | Feb 2016–Dec 2020 | 59 | 1,121 |
| Validation | Jan 2021–Jun 2022 | 18 | 342 |
| Test | Jul 2022–Nov 2024 | 29 | 551 |

Features at *t* use no future observations; each target is shifted to *t+1*. The current
split does not purge adjacent dates, so the realized month between the final training
target and first validation feature date is shared as the natural boundary between
adjacent examples. Validation selects model settings; the test split supplies reported
out-of-sample results.

## Ranking and portfolio evaluation

Ranking diagnostics include monthly Spearman Rank IC, ICIR, NDCG@5 and NDCG@10. The
portfolio strategy selects the five highest-scored stocks each month and weights them
equally. Comparators are a MOM_12_1 Top-5 strategy and the equal-weighted 19-stock panel.
The fallback market comparator is a liquidity-weighted return proxy built from those
same 19 stocks with a fixed 0.10% monthly dividend uplift. It is not a Nifty 50 index or
TRI. Results are exported to `results/` and used by `app.py`.

## Risk-free rate, costs, and limits

The target uses the geometric monthly equivalent of a constant 6.2% annual rate.
Portfolio Sharpe uses total
portfolio returns, annualized as `(12 × mean monthly return − 6.66%) / (sqrt(12) ×
sample standard deviation)`. These are separate fixed conventions; the stored panel
does not carry a time-varying risk-free series.

Portfolio turnover is half the L1 distance from prior drifted weights to current target
weights. The code charges twice the configured 10 bps one-way rate per turnover unit.
No initial test-period turnover is charged because no pre-test weights are supplied.
This is a simplified, assumed cost model.

Data generation attempts a Yahoo Finance download and can use a seeded synthetic fallback.
The checked-in CSV files have no provenance flag, so the producing path is unknown from
repository evidence. The 19-stock fixed subset, possible survivorship bias, technical
features only, fixed splits, simplified costs and proxy benchmark constrain conclusions.
