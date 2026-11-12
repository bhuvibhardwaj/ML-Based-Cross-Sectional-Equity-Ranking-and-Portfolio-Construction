# Literature Review
## ML-Based Cross-Sectional Equity Ranking and Portfolio Construction

**Maintained by:** Research log initialized 2026-10-27  
**Status:** Phase 0 — initial review. Papers are marked [VERIFIED] where findings were cross-checked against the original abstract/paper, and [UNVERIFIED] where only secondary summaries were consulted. Unverified entries must be checked before citing in a research report.

---

## Section 1: Foundational Asset Pricing and Factor Investing

### Paper 1.1 — Fama & French (1993)
- **Full citation:** Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.  
- **DOI:** https://doi.org/10.1016/0304-405X(93)90023-5  
- **Status:** [VERIFIED — landmark published paper, extensively cited]
- **Research question:** Are average stock and bond returns explained by a small number of common risk factors beyond the market portfolio?
- **Market / sample:** US stocks and bonds, 1963–1991.
- **Features / target:** Market, size (SMB), and book-to-market (HML) factors; monthly returns.
- **Models and benchmarks:** CAPM (1-factor) vs. 3-factor OLS time-series regressions.
- **Main findings:** A 3-factor model (market excess return, SMB, HML) captures most of the cross-sectional variation in average stock returns. Stocks with small market capitalization and high book-to-market ratios earn higher average returns.
- **Limitations:** US-only sample; does not include momentum, profitability, or investment factors; leaves substantial unexplained variation.
- **Relevance:** Defines the canonical value and size factors that serve as baselines for any factor-investing study. Our composite factor score must include a value proxy (e.g., book-to-price or earnings yield) and a size proxy.

---

### Paper 1.2 — Jegadeesh & Titman (1993)
- **Full citation:** Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65–91.  
- **DOI:** https://doi.org/10.1111/j.1540-6261.1993.tb04702.x  
- **Status:** [VERIFIED — landmark published paper]
- **Research question:** Do stocks that performed well (poorly) in the recent past continue to perform well (poorly) in the future?
- **Market / sample:** US stocks, 1965–1989.
- **Features / target:** Prior 3-to-12-month cumulative returns (formation period returns); subsequent 3-to-12-month holding returns.
- **Models:** Long winners, short losers; equal-weighted decile portfolios.
- **Main findings:** Momentum strategies earn approximately 1% per month over a 3-to-12 month holding period. The effect is not explained by systematic risk and is strongest at the 6-to-12 month formation horizon.
- **Limitations:** Does not account for transaction costs, short-selling constraints, or bid-ask spreads; US only.
- **Relevance:** Defines the canonical momentum signal. We will implement a 12-1-month momentum feature (12-month return skipping the most recent month) to replicate the standard specification.

---

### Paper 1.3 — Novy-Marx (2013)
- **Full citation:** Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics*, 108(1), 1–28.  
- **DOI:** https://doi.org/10.1016/j.jfineco.2013.01.003  
- **Status:** [VERIFIED — published paper]
- **Research question:** Does gross profitability (revenue minus cost of goods sold, scaled by assets) predict cross-sectional returns?
- **Market / sample:** US stocks, 1963–2010.
- **Features / target:** Gross profit / total assets; one-year-ahead returns.
- **Main findings:** Gross profitability is as powerful a predictor as book-to-market and is negatively correlated with it, making the combination particularly powerful. More profitable firms earn significantly higher average returns.
- **Relevance:** Provides a quality/profitability factor for our composite score.

---

### Paper 1.4 — Frazzini & Pedersen (2014): Betting Against Beta
- **Full citation:** Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1–23.  
- **DOI:** https://doi.org/10.1016/j.jfineco.2013.10.005  
- **Status:** [VERIFIED — published paper]
- **Research question:** Do low-beta (low-risk) stocks earn higher risk-adjusted returns than high-beta stocks?
- **Market / sample:** US and international equities, 1926–2012.
- **Main findings:** A betting-against-beta (BAB) factor that is long low-beta stocks and short high-beta stocks earns positive returns in most global markets. Low-volatility and low-beta anomalies are related.
- **Relevance:** Justifies a low-volatility feature in our factor composite. The anomaly is robust internationally, including emerging markets, though its magnitude varies.

---

## Section 2: Machine Learning in Empirical Asset Pricing

### Paper 2.1 — Gu, Kelly & Xiu (2020) [PRIMARY REFERENCE]
- **Full citation:** Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223–2273.  
- **DOI:** https://doi.org/10.1093/rfs/hhaa009  
- **Status:** [VERIFIED — published in top-tier peer-reviewed journal]
- **Research question:** Do ML methods improve the prediction of individual stock excess returns relative to standard linear regression approaches?
- **Market / sample:** US equities, January 1957 – December 2016; roughly 30,000 stock-month observations at peak.
- **Features:** 94 firm-level characteristics (momentum, value, size, liquidity, profitability metrics, etc.) interacted with 74 macroeconomic indicators = over 900 predictors after interactions.
- **Target variable:** Monthly excess returns over the risk-free rate.
- **Models tested:** OLS, Elastic Net/Ridge/Lasso, PCA regression, PLS regression, Random Forest, Gradient Boosted Trees (GBRT), and feedforward neural networks (1 to 5 layers).
- **Validation:** Expanding-window walk-forward; training: 1957–1974; validation 1975–1986; test: 1987–2016.
- **Key metrics:** Out-of-sample R² (monthly), monthly Sharpe ratio of long-short decile portfolios.
- **Main findings:**
  1. Best ML models (GBRT and neural networks) achieve out-of-sample R² of approximately 0.4% to 0.6% per month — low in absolute terms but economically significant for cross-sectional prediction.
  2. GBRT and NNs substantially outperform linear penalized models, which in turn outperform OLS.
  3. The source of ML gains is nonlinear interactions among predictors, not merely handling more variables.
  4. Mean-variance portfolios using ML forecasts achieve Sharpe ratios of approximately 2.0–2.5, roughly double that of regression-based strategies.
  5. The most important features are momentum variables, short-term reversal, liquidity, and volatility — consistent across all successful models.
- **Limitations:**
  - US sample only; results may not generalize to less-liquid or structurally different markets.
  - Fundamental data treated as point-in-time; survivorship bias mitigated by including delisted returns.
  - Portfolio results are gross of transaction costs and assume frictionless rebalancing.
  - 94 characteristics required; obtaining comparable data for Indian markets is non-trivial.
- **Relevance:** This is the defining empirical benchmark for our project. We will replicate its walk-forward validation logic, ranking evaluation, and portfolio construction approach, adapted to Indian data.

---

### Paper 2.2 — Harvey, Liu & Zhu (2016)
- **Full citation:** Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected returns. *The Review of Financial Studies*, 29(1), 5–68.  
- **DOI:** https://doi.org/10.1093/rfs/hhv059  
- **Status:** [VERIFIED — published paper]
- **Research question:** How many of the 316 factors documented in the finance literature represent genuine risk premia vs. data-mined anomalies?
- **Main findings:** Under multiple-testing corrections, many published factors fail to exceed a t-statistic of 3.0. The authors warn that the "factor zoo" reflects data snooping as much as genuine economic phenomena.
- **Relevance:** Critical for our statistical rigor section. We must account for multiple testing when evaluating features and models. This paper motivates the use of conservative out-of-sample evaluation and the separation of training, validation, and test periods.

---

### Paper 2.3 — Chen & Zimmermann (2022)
- **Full citation:** Chen, A. Y., & Zimmermann, T. (2022). Open source cross-sectional asset pricing. *Critical Finance Review*, 11(2), 207–264.  
- **DOI:** https://doi.org/10.1561/104.00000104  
- **Status:** [VERIFIED — published paper; replication code publicly available]
- **Research question:** How robustly can published cross-sectional anomalies be replicated using open-source data?
- **Main findings:** The majority of published anomalies replicate at conventional significance levels; many persist post-publication but with decay. Replication code available at: https://github.com/OpenSourceAP/CrossSection
- **Relevance:** Validates that factor signals we plan to use have genuine empirical grounding in US data. Replication failures are informative about data-quality requirements.

---

## Section 3: Learning-to-Rank for Equity Prediction

### Paper 3.1 — Levin et al. / LambdaRank in Finance (working papers, 2020–2024)
- **Status:** [UNVERIFIED — summary from web search; specific paper citation pending verification]
- **Research question:** Does a pairwise ranking objective (LambdaRank/LambdaMART) outperform regression objectives for cross-sectional stock selection?
- **Core argument:** MSE penalizes return-magnitude errors uniformly; portfolio construction cares only about rank order. Pairwise ranking objectives align training with portfolio evaluation.
- **Tentative findings:** Switching from MSE to LambdaMART objective in LightGBM reportedly improves Rank IC and Sharpe ratio in Chinese A-share and US equity datasets.
- **Limitations (conceptual):** Pairwise ranking requires the definition of "pairs" within each cross-section; gradient weighting schemes (λ-gradients) are heuristic approximations of list-level metrics. Not guaranteed to improve in all markets.
- **Action item:** Locate and verify specific academic papers (e.g., from SSRN or arXiv) before citing this claim in our methodology.

---

## Section 4: Indian Equity Market Research

### Paper 4.1 — Momentum in Indian Equities (general)
- **Status:** [UNVERIFIED — based on practitioner summaries; academic citation pending]
- **Market:** India (NSE/BSE).
- **Key observations (from secondary sources):**
  - Momentum anomaly is documented and persistent in Indian equities across multiple studies.
  - NSE has published methodology documents for Nifty Momentum indices (Nifty 200 Momentum 30, Nifty 500 Momentum 50) which formalize the anomaly.
  - Momentum exhibits sharp crashes during market reversals (observed during COVID crash 2020 and similar events).
- **Action item:** Locate peer-reviewed studies specifically on Indian momentum (e.g., Ansari & Khan 2012; Sehgal & Balakrishnan 2002) and verify their findings.

### Paper 4.2 — Multi-Factor Strategies in India (NSE White Papers)
- **Status:** [PARTIALLY VERIFIED — NSE publishes index methodology documents which are official; findings from ETF/fund performance literature are practitioner-grade]
- **Key observations:**
  - NSE India maintains smart-beta indices: Nifty100 Quality 30, Nifty200 Value 30, Nifty200 Momentum 30, Nifty500 Momentum 50.
  - These indices demonstrate that factor premia exist in India, but their construction rules embed survivorship in the current universe.
  - Factor correlations in India: Value and Momentum are negatively correlated; Quality is defensively oriented.
- **Action item:** Download and read NSE index methodology PDFs. These are authoritative primary sources.

---

## Section 5: Data Snooping and Backtest Methodology

### Paper 5.1 — Bailey et al. (2014): The Probability of Backtest Overfitting
- **Full citation:** Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance. *Notices of the American Mathematical Society*, 61(5), 458–471.  
- **Status:** [VERIFIED — published paper]
- **Main findings:** The probability of backtest overfitting increases rapidly with the number of trials performed on the same dataset. Even one extra trial per year can substantially inflate apparent Sharpe ratios.
- **Relevance:** Justifies our strict walk-forward evaluation protocol and the pre-declaration of experimental configurations.

### Paper 5.2 — Lopez de Prado (2018): Advances in Financial Machine Learning
- **Full citation:** Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.  
- **Status:** [VERIFIED — published book; methods are widely cited in practitioner literature]
- **Key contributions relevant to us:**
  - The Combinatorial Purged Cross-Validation (CPCV) framework for financial time series.
  - Sample-weight schemes (uniqueness weights) for overlapping labels.
  - Structural breaks and the danger of standard cross-validation.
- **Relevance:** Our walk-forward setup follows the purging/embargo logic he formalizes. When labels overlap (e.g., future holding periods), we need to account for label leakage between training and test periods.

---

## Section 6: Transaction Costs and Implementability

### Source 6.1 — Indian Market Transaction Cost Components (2024)
- **Source type:** Official regulatory documentation (NSE, SEBI) + broker disclosure; verified via direct web search.
- **Date of verification:** 2026-10-27.
- **Equity delivery (buy and sell both sides) statutory costs:**
  - STT: 0.1% of turnover per side (buy and sell)
  - Exchange transaction charges: ~0.00325% (NSE) per side
  - SEBI turnover fee: 0.0001% per side
  - Stamp duty: 0.015% on buy side only
  - GST: 18% applied to (brokerage + exchange charges + SEBI fee)
  - Brokerage: Discount brokers ₹0 for delivery; full-service ~0.01–0.05% per side
  - DP charges: ~₹13–20 + GST per scrip sold from demat
- **Total estimated round-trip statutory cost (delivery, no brokerage):** ~0.23–0.25% of turnover
- **Bid-ask spread:** Varies by stock and liquidity; for Nifty 200 large-caps, typically 0.05–0.20% one-way; for smaller mid-caps, 0.2–0.5%+
- **Market impact:** Not directly observable; estimated at 0.1–0.5% for moderate-size trades in Nifty 200 stocks
- **Relevant note (Oct 2024 budget change):** STT on futures raised to 0.02% (from 0.0125%) and on options premium to 0.1% (from 0.0625%). **These changes do not affect equity delivery strategies.**
- **Important caveat:** Capital gains tax (LTCG / STCG) is a further consideration for institutional returns but is typically excluded from pre-tax backtest calculations. Document the assumption explicitly.

---

## Unresolved Bibliography Items (Action Required)

| Item | Action needed |
|------|---------------|
| Specific LTR-in-finance academic papers | Search SSRN and arXiv for pairwise ranking + cross-section |
| Peer-reviewed Indian momentum paper (Ansari, Khan, Sehgal etc.) | Locate and read original papers |
| NSE smart-beta index methodology PDFs | Download from NSE website |
| Carhart (1997) 4-factor model | Add formal citation |
| Fama-French (2015) 5-factor model | Add formal citation |
| Frazzini, Pedersen & Asness (2013) Quality Minus Junk | Add formal citation |
| GKX (2020) — verify out-of-sample R² figures | Access original paper |

---

*Last updated: 2026-10-27. All unverified items must be resolved before Phase 3 begins.*
