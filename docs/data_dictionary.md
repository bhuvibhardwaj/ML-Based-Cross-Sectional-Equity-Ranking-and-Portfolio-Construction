# Data Dictionary

This dictionary follows the checked-in files and current pipeline. The monthly CSV is
the authority for the evaluated subset: 19 tickers, 106 dates and 2,014 rows. This is
not a Nifty 100 panel.

## Input files

### `data/stocks_daily_ohlcv.csv`

| Field | Meaning |
|---|---|
| `date` | Daily observation date |
| `ticker` | NSE symbol with `.NS` suffix |
| `company_name`, `sector` | Descriptive metadata |
| `open`, `high`, `low`, `close` | Daily OHLC price fields |
| `adj_close` | Adjusted close used for price features and returns |
| `volume` | Daily traded share volume |

The checked-in file has 46,854 rows across 19 tickers and no index series. The generator
attempts Yahoo Finance acquisition and may use seeded synthetic data as a fallback. The
CSV has no provenance marker, so its source cannot be established from this file.

### `data/stocks_monthly_features.csv`

One row per ticker and formation month. The file contains 2,014 observations from
February 2016 through November 2024, with 19 tickers at each of 106 dates.

| Field | Meaning |
|---|---|
| `date`, `ticker`, `company_name`, `sector` | Formation date and stock identifiers |
| `CLOSE_PRICE` | Adjusted month-end close |
| `MOM_12_1`, `MOM_6_1`, `MOM_3_1` | Price momentum, excluding the latest month |
| `REV_1` | Latest one-month return signal |
| `VOL_12M`, `VOL_3M` | Annualized trailing realized volatility |
| `TURN_1M`, `VOL_SHOCK` | Log-volume liquidity/activity features |
| `BETA_12M` | Rolling beta against the market return series used by feature generation |
| `RSI_14`, `MA_RATIO_50_200` | Technical indicators at month-end |
| `MAX_DD_12M`, `SHARPE_12M` | Trailing drawdown and risk-adjusted return features |
| `NEXT_MONTH_RETURN` | Simple return from this month-end close to next month-end close |
| `NEXT_MONTH_EXCESS_RETURN` | Next-month return less `(1 + 0.062) ** (1/12) - 1` |

The 13 model features are the fields above excluding identifiers, `CLOSE_PRICE`, and
the two forward return target fields.

## Results

`results/` contains model metrics, rankings, portfolio summaries and monthly series used
by `app.py`. The portfolio market comparator is a rolling liquidity-weighted proxy
from the same 19 stocks with a fixed 0.10% monthly dividend uplift when an explicit
`nifty_tri` series is unavailable. It must not be described as the Nifty 50 price index
or Nifty 50 TRI.

## Risk-free conventions

Target construction uses the geometric monthly equivalent of 6.2% per year. Portfolio Sharpe uses
6.66% per year as a constant annual rate with arithmetic annualization. The stored panel
does not include a historical monthly RBI rate series.
