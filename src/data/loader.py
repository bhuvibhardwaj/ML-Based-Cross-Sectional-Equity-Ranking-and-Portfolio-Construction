"""
Data loader and dataset validator for the ML-Based Cross-Sectional Equity Ranking system.

Design philosophy:
  - Separate acquisition (download/generate) from loading (read cached CSV data).
  - Comprehensive validation of daily OHLCV and monthly cross-sectional feature panels.
  - Explicit documentation of dataset scope and structural limitations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Candidate-symbol catalog only; it is not the evaluated model universe.
NIFTY100_CANDIDATE_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFOSYS.NS", "SBIN.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "WIPRO.NS",
    "HCLTECH.NS", "TITAN.NS", "ADANIENT.NS", "ONGC.NS", "NTPC.NS",
    "POWERGRID.NS", "COALINDIA.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "PIDILITIND.NS", "DABUR.NS",
    "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "EMAMILTD.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "M&M.NS", "EICHERMOT.NS",
    "TATAPOWER.NS", "ADANIPORTS.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "HINDALCO.NS",
    "VEDL.NS", "SAIL.NS", "NMDC.NS", "BPCL.NS", "GAIL.NS",
    "IOC.NS", "HPCL.NS", "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS",
    "MUTHOOTFIN.NS", "BAJAJFINSV.NS", "SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS",
    "TECHM.NS", "MPHASIS.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS",
    "DIVISLAB.NS", "BIOCON.NS", "AUROPHARMA.NS", "TORNTPHARM.NS", "LUPIN.NS",
    "HAVELLS.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "BLUESTARCO.NS", "CUMMINSIND.NS",
    "ABB.NS", "SIEMENS.NS", "BEL.NS", "HAL.NS", "BDL.NS",
    "IRCTC.NS", "CONCOR.NS", "PIIND.NS", "UPL.NS", "CHAMBLFERT.NS",
    "INDIGO.NS", "INTERGLOBE.NS", "DMART.NS", "ZOMATO.NS", "NYKAA.NS",
    "PAYTM.NS", "POLICYBZR.NS", "NAUKRI.NS", "JUSTDIAL.NS", "MAKEMYTRIP.NS",
    "ULTRACEMCO.NS", "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS", "RAMCOCEM.NS",
    "IDFCFIRSTB.NS", "AUBANK.NS", "RBLBANK.NS",
]

NIFTY50_TICKER = "^NSEI"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

EQUITY_UNIVERSE_NSE: Dict[str, Dict[str, str]] = {
    # Must match the 19 tickers / sectors present in data/stocks_monthly_features.csv
    "ASIANPAINT.NS": {"name": "Asian Paints", "sector": "Materials"},
    "AXISBANK.NS": {"name": "Axis Bank", "sector": "Financials"},
    "BAJFINANCE.NS": {"name": "Bajaj Finance", "sector": "Financials"},
    "HCLTECH.NS": {"name": "HCL Technologies", "sector": "IT"},
    "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Financials"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "sector": "Consumer Staples"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "sector": "Financials"},
    "INFY.NS": {"name": "Infosys Limited", "sector": "IT"},
    "ITC.NS": {"name": "ITC Limited", "sector": "Consumer Staples"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank", "sector": "Financials"},
    "LT.NS": {"name": "Larsen & Toubro", "sector": "Industrials"},
    "MARUTI.NS": {"name": "Maruti Suzuki India", "sector": "Consumer Discretionary"},
    "NTPC.NS": {"name": "NTPC Limited", "sector": "Utilities"},
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy"},
    "SBIN.NS": {"name": "State Bank of India", "sector": "Financials"},
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical", "sector": "Healthcare"},
    "TATASTEEL.NS": {"name": "Tata Steel", "sector": "Materials"},
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "IT"},
    "TITAN.NS": {"name": "Titan Company", "sector": "Consumer Discretionary"},
}

DATASET_LIMITATIONS: List[str] = [
    "Narrow Cross-Sectional Breadth (N = 19 Stocks): Each monthly cross-section has 19 stocks. Under H0: IC_t = 0, a single month's Spearman Rank IC has approximate standard error 1/sqrt(19-2) = 0.2425. The Top-5 portfolio holds 5 of the 19 stocks.",
    "Fixed-Universe Selection: The evaluated panel contains 19 stocks, not a point-in-time Nifty 100 universe. Selection history is not recorded in the data artifacts, so survivorship and selection bias cannot be quantified.",
    "Price and Volume Technical Scope Only: Features are derived exclusively from daily OHLCV market data (momentum, reversal, volatility, turnover, beta, drawdown, RSI, MA ratio, Sharpe) without quarterly fundamental accounting ratios.",
    "Fixed Chronological Split vs. Post-2022 Regime Rotation: Models trained on Feb 2016 – Dec 2020 and validated on Jan 2021 – Jun 2022 face a macroeconomic regime shift in the Jul 2022 – Nov 2024 test window (post-COVID industrial/value/PSU cyclical leadership).",
]


# ---------------------------------------------------------------------------
# Download & Panel Helpers
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download daily OHLCV price data for a list of tickers."""
    import yfinance as yf

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_date = datetime.now().strftime("%Y%m%d")
    cache_file = raw_dir / f"prices_{download_date}.parquet"

    if cache_file.exists() and not force_refresh:
        logger.info("Loading cached price data from %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Downloading price data for %d tickers from %s to %s", len(tickers), start, end)
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {e}") from e

    if raw.empty:
        raise ValueError("yfinance returned empty DataFrame. Check tickers and date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns.names = ["field", "ticker"]
        field_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "adj_close", "Volume": "volume",
        }
        raw = raw.rename(columns=field_map, level=0)
    else:
        raw.columns = pd.MultiIndex.from_tuples(
            [(c.lower(), tickers[0]) for c in raw.columns], names=["field", "ticker"]
        )

    raw.index.name = "date"
    raw.to_parquet(cache_file)
    return raw


def get_adj_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Extract the adjusted close price panel from a multi-level price DataFrame."""
    if isinstance(prices.columns, pd.MultiIndex):
        return prices["adj_close"].copy()
    return prices.copy()


def get_volume(prices: pd.DataFrame) -> pd.DataFrame:
    """Extract the volume panel from a multi-level price DataFrame."""
    if isinstance(prices.columns, pd.MultiIndex):
        return prices["volume"].copy()
    raise ValueError("Expected MultiIndex price DataFrame with 'volume' level.")


def validate_prices(adj_close: pd.DataFrame) -> dict:
    """Run data quality checks on the adjusted close price panel."""
    results: dict = {}

    missing_pct = adj_close.isna().mean()
    results["missing_pct"] = missing_pct

    monthly_returns = adj_close.resample("ME").last().pct_change()
    large_moves = (monthly_returns.abs() > 0.50).stack()
    large_moves = large_moves[large_moves].reset_index()
    large_moves.columns = ["date", "ticker", "flag"]
    results["large_moves"] = large_moves[["date", "ticker"]]

    flat_mask = adj_close.diff().fillna(0).abs() < 1e-8
    flat_runs = flat_mask.rolling(10).sum()
    flat_tickers = (flat_runs >= 10).any()
    results["flat_prices"] = flat_tickers[flat_tickers]

    results["start_dates"] = adj_close.apply(lambda s: s.first_valid_index())
    results["end_dates"] = adj_close.apply(lambda s: s.last_valid_index())

    zero_prices = (adj_close <= 0).any()
    zero_tickers = zero_prices[zero_prices]
    results["zero_prices"] = zero_tickers if not zero_tickers.empty else pd.Series(dtype=bool)

    return results


def validate_monthly_panel(
    monthly_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate the historical NSE monthly stock-feature dataset and return a structured audit dictionary.

    Verifies:
      - Exact stock count (19 NSE stocks) and formation month count (106 months: Feb 2016 – Nov 2024)
      - Balanced cross-sectional structure (19 stocks per formation month = 2,014 observations)
      - Absence of missing values in engineered features or targets
      - Absence of duplicate (date, ticker) pairs
    """
    if feature_cols is None:
        feature_cols = [
            "MOM_12_1", "MOM_6_1", "MOM_3_1", "REV_1",
            "VOL_12M", "VOL_3M", "TURN_1M", "VOL_SHOCK",
            "BETA_12M", "RSI_14", "MA_RATIO_50_200", "MAX_DD_12M", "SHARPE_12M",
        ]
    df = monthly_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    n_stocks = int(df["ticker"].nunique())
    n_months = int(df["date"].nunique())
    n_obs = int(len(df))
    counts_per_month = df.groupby("date")["ticker"].count()
    min_stocks_per_month = int(counts_per_month.min())
    max_stocks_per_month = int(counts_per_month.max())

    check_cols = [c for c in feature_cols + ["NEXT_MONTH_RETURN", "NEXT_MONTH_EXCESS_RETURN"] if c in df.columns]
    total_missing = int(df[check_cols].isna().sum().sum())
    duplicate_pairs = int(df.duplicated(subset=["date", "ticker"]).sum())

    return {
        "total_rows": n_obs,
        "num_stocks": n_stocks,
        "num_months": n_months,
        "stocks_per_month_min": min_stocks_per_month,
        "stocks_per_month_max": max_stocks_per_month,
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
        "missing_values_total": total_missing,
        "duplicate_date_ticker": duplicate_pairs,
    }


def build_monthly_panel(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Resample daily adjusted closes to month-end prices."""
    monthly = adj_close.resample("ME").last()
    logger.info("Built monthly price panel: %d months × %d tickers", *monthly.shape)
    return monthly


def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple arithmetic monthly returns from month-end prices."""
    return monthly_prices.pct_change()
