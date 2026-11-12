# Research Log
## ML-Based Cross-Sectional Equity Ranking — Indian Market

This log records research questions as they arise, key findings, data source discoveries,
unresolved uncertainties, and decisions made during the project. Entries are chronological.

---

## Entry 001 — 2026-10-27: Phase 0 Initiated

### Context
Repository initialized from empty state. Phase 0 objective: research literature, assess data
availability, specify the initial experiment mathematically, and document design decisions.

### Literature findings summary
- GKX (2020) is the primary benchmark. Their key result (ML R² ~0.4–0.6%, Sharpe ~2.0–2.5x)
  applies to US 1987–2016. We need to assess whether comparable results are achievable
  for India with restricted data.
- Indian equity factor premia (momentum, value, quality) are documented in practitioner
  and index literature but less rigorously in peer-reviewed academic work. This is a data
  limitation of the Indian market, not evidence that the premia don't exist.
- Learning-to-rank approaches (LambdaMART) are promising but the peer-reviewed evidence
  in finance is thinner than for regression-based ML. Treat as a research question, not
  an established improvement.

### Data source assessment
**Critical finding: Survivorship bias is a material risk for Indian data.**

- `yfinance` can retrieve price histories for currently-listed NSE stocks (ticker + ".NS")
  but does NOT provide historical index constituents or delisted stock data.
- NSE publishes semi-annual index rebalancing notices (IndexInclExcl files) but maintaining
  a complete historical constituent database requires either (a) manual assembly from
  archived NSE announcements, or (b) a paid data provider.
- For a university project using free data:
  - We can download NSE's current EQUITY_L.csv (all listed securities) for prices.
  - We CANNOT cleanly reconstruct point-in-time Nifty 200/500 membership.
  - We WILL have survivorship bias: today's Nifty 200 survivors look better than the 
    true historical universe.

**Decision made (see decision_log.md Entry 001):**
- Use Nifty 100 large-cap universe as a proxy (lower turnover → less survivorship bias)
- Label all backtests "exploratory — potentially upward-biased due to survivorship"
- Use consistent methodology throughout so comparisons between models remain valid
  even if absolute return levels are biased
- Document this limitation prominently in every result

### Transaction cost research
Indian equity delivery costs (verified 2026-10-27):
- STT: 0.1% per side (buy and sell) → 0.2% round-trip
- Exchange charges: ~0.003% per side
- SEBI: 0.0001% per side
- Stamp duty: 0.015% on buys only
- Bid-ask + slippage: 0.1–0.4% per trade (assumption; market-dependent)
- Brokerage: Assumed ₹0 for delivery (discount broker assumption)
- Total round-trip statutory (ex-slippage): ~0.23%
- Total round-trip including mid-range slippage: ~0.35–0.55%
- For monthly rebalancing with 20-stock portfolio and ~30% monthly turnover:
  cost drag ≈ 0.10–0.17% per month ≈ 1.2–2.0% per year
  This is material and will materially affect net Sharpe ratios.

### Unresolved questions (to be answered before Phase 3)
1. Can we obtain reliable historical Nifty 100 constituent data from NSE archives? Check
   NSE India's historical data downloads section.
2. What is the typical bid-ask spread for Nifty 100 stocks? Need empirical data or
   reliable proxy.
3. Are there any free sources for point-in-time fundamental data (P/B ratio, ROE, 
   earnings yield) for Indian stocks? Screener.in? Tijori Finance?
4. What publication lag should we assume for quarterly earnings announcements in India?
   SEBI mandates quarterly results within 45 days of quarter end; use 60-day lag
   conservatively.
5. Which specific peer-reviewed papers document momentum in Indian equities?
   Candidate: Sehgal & Jain (2011), Ansari & Khan (2012).

---

## Entry 002 — 2026-10-27: Mathematical Specification Started

### Panel data structure defined
- Panel indexed by (stock i, formation date t) at monthly frequency.
- X_{i,t}: feature vector available at end of month t.
- y_{i,t+1}: excess return of stock i from end-of-month t to end-of-month t+1.
- Excess return defined as: R_{i,t+1} - R_{f,t+1} where R_{f,t+1} is the 
  91-day T-bill return or Nifty 50 index return (TBD; see assumptions.md).

### Feature set (initial, price-only — no fundamental data required)
For Phase 1/2 prototype:
1. MOM_12_1: 12-month return skipping most recent month = (P_{t-1} / P_{t-13}) - 1
   Note: t-1 means previous month-end price; t-13 means 13 months ago.
   Financial interpretation: Captures intermediate-horizon momentum.
   Evidence: Jegadeesh & Titman (1993), documented globally.

2. MOM_1: 1-month return = (P_t / P_{t-1}) - 1
   Note: Used with negative sign (short-term reversal). High last-month return
   predicts negative next-month return (bid-ask bounce, microstructure).
   
3. VOL_12: Annualized realized volatility over past 12 months using daily returns.
   Mathematical definition: σ_{i,t} = std(r_{i,d}) × sqrt(252) over trailing 252 trading days.
   Financial interpretation: Low volatility predicts higher risk-adjusted returns.
   
4. TURNOVER_1M: Average daily traded volume (shares) over past month.
   Used as illiquidity proxy (higher turnover = more liquid).
   Alternative: Amihud (2002) illiquidity ratio = |R| / Volume_value (add in Phase 3).

5. BETA_12M: Market beta estimated via OLS regression of daily returns on Nifty 50
   daily returns over past 252 days.
   Financial interpretation: Low-beta factor (BAB). Used with negative sign.

6. RETURN_VOLATILITY_RATIO (optional): Sortino or Sharpe-like signal using trailing returns
   and downside volatility.

### Fundamental features (requires Phase 2 data investigation)
- BEME: Book-to-market equity = book value of equity / market cap
  Requires annual report data with appropriate publication lag
- EARNINGS_YIELD: Earnings per share / price = P/E inverted
  Quarterly data with 60-day publication lag
- ROE: Return on equity = net income / book equity
  Annual or quarterly with publication lag

### Key temporal constraint (critical)
- Feature X_{i,t} uses data from end of month t or earlier.
- Target y_{i,t+1} is return from end of month t to end of month t+1.
- Execution: portfolio is formed at close of month-end t; positions held until close of month t+1.
- NO feature should use prices or data from after date t.
- Momentum MOM_12_1 uses P_{t-1} / P_{t-13}: uses prices from previous month, correct.
- Volatility VOL_12 uses daily returns up to and including day t: correct.
- Common leakage error: using prices that include the first day of month t+1. Must test.

---

*This log is updated whenever a material research decision is made or a significant
finding is recorded. Minor code decisions go to decision_log.md.*
# Historical Research Log

> Planning notes below record proposed questions, not completed implementation. The
> current scope is the 19-stock, three-model, Top-5 pipeline described in
> [`methodology.md`](methodology.md).

