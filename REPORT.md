# ML-Based Cross-Sectional Equity Ranking

This repository implements an academic mini-project using a fixed 19-stock NSE
evaluation subset. The executable models are Ridge regression, XGBoost regression, and
XGBoost Learning-to-Rank. The strategy selects an equal-weighted Top-5 portfolio each
month. This is not a Nifty 100 evaluation and does not implement Keras, fastai, TabNet,
or Lightning Flash.

The validated panel contains 2,014 rows across 106 formation months (February 2016 to
November 2024). The chronological split is 1,121 train rows over 59 months (through
December 2020), 342 validation rows over 18 months (January 2021–June 2022), and 551
test rows over 29 months (July 2022–November 2024).

`run_pipeline.py` generates the result CSVs consumed by the seven-page Streamlit app.
Use `results/ranking_metrics_comparison.csv` and `results/portfolio_evaluation.csv` as
the authoritative numeric result tables. Older narrative values have been removed so
that results are not duplicated inconsistently.

## Conventions and limits

- The target subtracts the geometric monthly equivalent of a constant 6.2% annual
  risk-free assumption (about 0.5025% monthly; verified from the stored target series).
- Portfolio Sharpe uses total returns and a constant 6.66% annual rate, with arithmetic
  annualization and sample volatility. It is not a time-varying RBI-rate calculation.
- Transaction costs use the pipeline's assumed 10 bps one-way input, doubled per unit
  of half-L1 drifted-weight turnover. No starting test holdings are supplied, so the
  first test-month cost is zero.
- No index series is present in the daily CSV. The reported market comparator is a
  liquidity-weighted return proxy from the same 19 stocks with a fixed dividend uplift,
  not Nifty 50 price index or TRI.
- The generator can download Yahoo Finance data or fall back to seeded synthetic data.
  The checked-in CSVs do not record which source path produced them.
- See [`docs/final_report.md`](docs/final_report.md) for the implementation review and
  [`docs/methodology.md`](docs/methodology.md) for executable methodology.

## Run

```bash
./.venv/bin/python run_pipeline.py
./.venv/bin/python -m pytest -q
./.venv/bin/python tests/qa_cross_page_consistency.py
./.venv/bin/python tests/qa_app_pages.py
```
