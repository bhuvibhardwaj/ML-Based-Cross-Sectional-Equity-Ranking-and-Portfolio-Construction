# Decision Log
## Design Decisions and Their Rationale

> Historical planning record. Early proposals below (including the Nifty 100 universe,
> Top-20 portfolio and time-varying RBI rate) were not all implemented and are superseded
> by the executable specification in [`methodology.md`](methodology.md). Do not use old
> proposals as descriptions of the evaluated pipeline.

Each entry records a decision, the alternatives considered, the rationale, and
what evidence would cause us to reconsider the decision.

---

## Entry 001 — 2026-10-27: Universe Selection

**Decision:** Use Nifty 100 as the working investment universe rather than Nifty 200 or Nifty 500.

**Alternatives considered:**
- Nifty 500: Broader, more stocks, better diversification of the ranking exercise, 
  but higher survivorship bias because smaller stocks have higher delisting rates.
- Nifty 200: Middle ground, but still requires reliable historical constituent data.
- All NSE-listed stocks: Maximum coverage but survivorship bias is most severe and 
  data quality is inconsistent for smaller stocks.

**Rationale:**
1. Nifty 100 (large-cap) has lower constituent turnover than broader indices →
   survivorship bias is smaller (not zero, but smaller).
2. Nifty 100 stocks have better data quality, longer price histories, and more 
   consistent coverage in yfinance.
3. For a long-only portfolio of 20 stocks, selecting from 100 candidates is reasonable;
   selecting from 500 would require more data infrastructure than a first prototype needs.
4. All comparisons between models (factor vs. Ridge vs. GBRT) are made on the same 
   biased universe, so the bias affects all methods equally. Comparisons remain valid
   even if absolute levels are biased.

**Label:** All results must be labeled "Exploratory — Nifty 100 survivors, survivorship-
bias-affected. Absolute return levels may be upward-biased. Model comparisons are
internally valid."

**What would cause reconsideration:**
- If we obtain reliable historical constituent data (e.g., from NSE archives or a 
  paid data vendor), we should switch to a point-in-time universe.
- If Nifty 100 survivorship bias turns out to be less than 0.5% per year (estimated
  by checking how many 2015 Nifty 100 constituents survive to 2024), we can retain
  the label "conservative exploratory."

---

## Entry 002 — 2026-10-27: Risk-Free Rate Proxy

**Decision:** Use the 91-day India T-bill rate (RBI data) as the risk-free rate for
excess return calculation and performance metrics.

**Alternatives considered:**
- Nifty 50 index return: Better captures the investable benchmark but conflates
  risk-free rate with market return. Using it to compute "excess return" changes
  the interpretation.
- MIBOR overnight rate: More theoretically precise (replicates money-market 
  opportunity cost) but less commonly used in academic literature and harder to
  obtain historically.
- Fixed constant (e.g., 6% per year): Simple but ignores rate changes (India's 
  repo rate has varied from 4% to 6.5% over 2015–2024).

**Rationale:**
- 91-day T-bill rate is the standard risk-free proxy in Indian academic finance.
- RBI publishes historical rates on its website (rbi.org.in).
- It corresponds to the holding period of approximately one month (actually 91 days 
  is 3 months; we will use the monthly equivalent: (1 + annual rate)^(1/12) - 1).
- Consistent with how GKX (2020) uses the 1-month T-bill rate for US data.

**What would cause reconsideration:**
- If we find that RBI 91-day T-bill rate is missing for significant historical periods,
  switch to RBI repo rate series (widely available).

---

## Entry 003 — 2026-10-27: Return Convention

**Decision:** Use simple (arithmetic) returns for feature computation and for 
monthly portfolio return calculation. Use log returns only for volatility calculations
(where their additive property over time is needed).

**Mathematical note:**
- Simple return: R_{i,t} = (P_{i,t} - P_{i,t-1}) / P_{i,t-1}
- Log return: r_{i,t} = log(P_{i,t} / P_{i,t-1}) ≈ R_{i,t} for small returns
- Portfolio return is the weighted average of simple returns; portfolio log return
  ≠ weighted average of log returns (Jensen's inequality).
- For monthly rebalancing with monthly returns, the difference is small but not zero.

**What would cause reconsideration:**
- If we move to intraday or high-frequency data, log returns become more important.

---

## Entry 004 — 2026-10-27: Adjusted vs. Unadjusted Prices

**Decision:** Use split-adjusted and dividend-adjusted (total return) prices from 
yfinance, which provides the "Adj Close" column adjusted for splits and dividends.

**What this means:**
- yfinance returns historical adjusted prices that account for stock splits, bonuses,
  and dividends (approximately — yfinance uses a backward-adjustment method).
- Total return prices allow momentum and return calculations to reflect actual
  investor wealth changes, including dividends.

**Limitation:**
- yfinance's adjustment for Indian stocks is imperfect for:
  - Bonus shares (treated as splits in some cases)
  - Rights issues (complex adjustment)
  - Mergers and name changes
- This is a known data quality issue. We will add a manual validation step for
  outlier price movements (> ±50% in a single month) to detect unadjusted splits.

**What would cause reconsideration:**
- Access to a properly adjusted data provider (e.g., Refinitiv, Bloomberg) for research-
  grade results.

---

## Entry 005 — 2026-10-27: Monthly Rebalancing Calendar

**Decision:** Rebalance on the last trading day of each calendar month.

**Rationale:**
- Standard in academic literature (GKX 2020 uses monthly rebalancing).
- Aligns with earnings reporting calendar.
- Practical: avoids quarter-end anomalies (window dressing by funds at March/June/
  September/December).

**Implementation note:**
- Formation date t = last trading day of month t.
- Features computed as of close on day t.
- Portfolio positions entered at OPEN of the next trading day (day t+1, first day 
  of month t+1). This is a more conservative assumption than using close prices of 
  day t, because it avoids overnight price impact.
- Target return measured from open of first day of month t+1 to open of first day 
  of month t+2. (Alternative: close-to-close; we choose this for simplicity and 
  will document it.)

**Practical simplification for Phase 1:**
- Use month-end close to month-end close returns as an approximation.
  This slightly overstates returns (we assume we can trade at the closing price of 
  the last day, which is often impossible for large positions). Document this.

---

## Entry 006 — 2026-10-27: Transaction Cost Model (Phase 3 onwards)

**Decision:** Use a transparent, parametric cost model with the following components.

**Round-trip cost per trade (equity delivery, India):**
```
c_roundtrip = STT_buy + STT_sell + exchange_buy + exchange_sell + stamp_buy
              + SEBI_buy + SEBI_sell + GST_component + slippage_assumption

Where:
  STT_buy = 0.1% × trade_value
  STT_sell = 0.1% × trade_value
  exchange_buy = 0.00325% × trade_value
  exchange_sell = 0.00325% × trade_value
  stamp_buy = 0.015% × trade_value
  SEBI = 0.0001% × trade_value (each side, negligible)
  GST_component ≈ 18% × (exchange_charges) ≈ 0.00117% × trade_value
  Statutory total ≈ 0.23% of round-trip traded value

  slippage_assumption = [0.10%, 0.25%, 0.50%] tested as three scenarios
  (low, mid, high)
```

**Turnover calculation:**
- At each rebalance, turnover = sum(|new_weight - old_weight|) / 2
- Each unit of two-way turnover costs `c_roundtrip` per trade.
- DP charge: ₹15 per scrip sold (negligible at fund scale but documented).

**What would cause reconsideration:**
- If we have actual execution data or market-impact models based on volume.

---

*All future major design decisions should be added here with a date stamp.*
