# Implementation Review: ML-Based Cross-Sectional Equity Ranking

This report describes the code and checked-in artifacts in this repository. It is an
implementation-focused academic mini-project, not a claim of a full-scale market study.

## Implemented system

`run_pipeline.py` runs three models: Ridge regression, XGBoost regression
(`reg:squarederror`), and XGBoost Learning-to-Rank (`rank:pairwise`). The input panel has
13 OHLCV-derived features. At formation month *t*, the target is the next month's simple
stock return less the fixed monthly risk-free rate. Features use prices and volumes
through *t*; the target uses the close at *t+1*. Portfolio selection is equal-weighted
Top-5, rebalanced monthly.

The data artifacts contain 2,014 stock-month observations, 19 tickers and 106 formation
months from February 2016 to November 2024. This is the evaluated subset, not a Nifty 100
backtest. The loader defines a larger candidate ticker list, but the stored monthly panel
is the authority for the evaluated universe.

## Evaluation design

| Split | Formation dates | Months | Rows |
|---|---|---:|---:|
| Train | Feb 2016–Dec 2020 | 59 | 1,121 |
| Validation | Jan 2021–Jun 2022 | 18 | 342 |
| Test | Jul 2022–Nov 2024 | 29 | 551 |

Features are formed at month-end and predict the following month's return. The code does
not purge boundary months; therefore the final train target and first validation feature
date share the Jan 2021 realization date, as expected for adjacent monthly observations.
Model selection uses validation; final ranking and portfolio metrics use the test split.

Ranking results are in `results/ranking_metrics_comparison.csv`; portfolio results are
in `results/portfolio_evaluation.csv`. These CSVs, rather than older narrative tables,
are the source for reported metrics. NDCG@5/@10 are ranking diagnostics; portfolio
holdings use Top-5. No Top-20 strategy is implemented.

## Return and risk conventions

- Target excess returns subtract the geometric monthly equivalent of a constant 6.2%
  annual rate, `(1 + 0.062) ** (1/12) - 1` (about 0.5025% per month). The checked-in
  target series confirms this conversion. It is a fixed modeling assumption, not a
  historical monthly RBI series.
- Portfolio Sharpe uses total monthly portfolio returns, arithmetic annualization
  (`12 × mean`), annualized sample volatility (`sqrt(12) × standard deviation`), and a
  constant 6.66% annual risk-free rate (`src/backtest/portfolio.py`). This is the code's
  Sharpe convention. It differs from the target rate and should not be described as a
  time-varying risk-free series.
- Portfolio turnover is `0.5 × sum(abs(target weights − prior drifted weights))`.
  The code deducts `2 × one_way_cost_bps × turnover`, with a default of 10 bps one-way.
  The first test month has zero turnover/cost because there is no prior test holding
  state. Reported costs are the implementation's assumed cost, not brokerage quotes.

## Benchmark and data provenance

The daily CSV contains no Nifty index or TRI series. In its absence, the backtest
constructs a trailing rupee-volume-weighted return from the same 19-stock universe and
adds a fixed 0.10% monthly dividend uplift. This is a project-universe proxy; it is not
the Nifty 50 price index or Nifty 50 TRI. Dashboard and CSV labels must call it a
19-stock liquidity-weighted proxy. No external index comparison can be verified from
these artifacts.

The generator attempts to download Yahoo Finance data and falls back to seeded synthetic
data when download is unavailable. The checked-in CSVs contain no provenance marker, so
repository evidence cannot establish which path produced them. Treat the data source as
unverified; do not present the generated or cached values as independently validated
market observations.

## Limitations

- Narrow, fixed 19-stock universe; possible survivorship and selection bias.
- OHLCV technical features only; no point-in-time fundamentals.
- Fixed chronological split and no rolling retraining.
- Fixed risk-free assumptions and simplified transaction costs.
- Benchmark is the proxy described above, not an official index series.
- Existing performance artifacts are historical outputs, not live forecasts or
  investment recommendations.

## Reproduction

Run `./.venv/bin/python run_pipeline.py` to regenerate outputs and
`./.venv/bin/python -m pytest -q` to run tests. Regeneration may attempt a live download;
to preserve exact data provenance, record whether the download or synthetic fallback was
used when producing new artifacts.
