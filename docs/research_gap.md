# Research Gap Analysis
## ML-Based Cross-Sectional Equity Ranking — Indian Market

**Status:** Phase 0 preliminary. Claims here are hypotheses and candidate directions,
not confirmed research gaps. Each must be verified against the literature before use
as a novelty claim.

---

## 1. What Existing Research Has Established

### In US and Global Markets

1. **ML outperforms linear regression for return prediction (US equities).**  
   Gu, Kelly & Xiu (2020) document this comprehensively for US equities over 1987–2016. The improvement is concentrated in nonlinear interactions among predictors, not merely the volume of features.

2. **Momentum, liquidity, and volatility are the most consistently important predictors across ML models.**  
   This finding is robust in GKX (2020) and has been confirmed in multiple replication studies.

3. **Factor premia in India exist for momentum, value, and quality.**  
   NSE smart-beta index data and practitioner research confirm these anomalies in Indian equities, though robust peer-reviewed evidence with proper point-in-time testing is sparser than for US markets.

4. **Learning-to-Rank objectives can improve portfolio-relevant ranking metrics.**  
   Practitioner research and working papers suggest pairwise ranking losses (LambdaMART) improve Rank IC relative to MSE, particularly in noisy cross-sections. This remains less established in peer-reviewed literature than regression-based methods.

5. **Transaction costs, particularly STT and slippage in India, substantially reduce gross returns.**  
   This is documented qualitatively in practitioner research but rarely modeled explicitly with Indian-specific statutory rates in academic studies.

---

## 2. What Has NOT Been Established (Candidate Research Questions)

**Important disclaimer:** These are candidate gaps identified through preliminary research. Before claiming novelty, we must verify that these questions have not already been answered in published work we have not yet reviewed.

### Gap 2.1 — Replication of GKX methodology on Indian equities with honest data constraints
GKX (2020) used 94 characteristics over a 30-year US sample. No comparable published study applies an equivalent framework to Indian equities with:
- A point-in-time survivorship-bias-free universe
- Comparable feature richness
- The same walk-forward evaluation protocol
- Net-of-cost performance assessment using Indian statutory costs

**What we can investigate:** Whether the ML advantage documented by GKX extends to Indian equities, given India's different microstructure, liquidity profile, and data availability constraints.

**Important caveat:** If our data has survivorship bias or missing fundamental data, the comparison with GKX will be compromised and must be labeled exploratory.

### Gap 2.2 — Separating linear vs. nonlinear contributions in Indian cross-section
GKX (2020) documents that nonlinear ML models outperform linear ones in the US. For India:
- Is any ML improvement over a factor composite driven by learning *new* linear combinations, or by *nonlinear interactions*?
- The distinction matters because a well-chosen linear multi-factor model with learned weights (Ridge regression) already captures many improvements.

**Experimental design:** Compare (a) composite factor score with fixed weights, (b) Ridge regression with learned weights, (c) GBRT. If (b) ≈ (c), the improvement is linear, not from ML-specific nonlinearity.

### Gap 2.3 — Return regression vs. pairwise ranking objective in Indian market
No peer-reviewed study (that we have found) systematically compares MSE-based regression with LambdaMART on Indian equities using identical features, validation protocol, and portfolio construction rules.

**Experimental design:** Train GBRT with MSE objective vs. LambdaMART objective. Hold all other choices constant. Evaluate using both Rank IC and net portfolio Sharpe.

### Gap 2.4 — Net-of-cost performance for monthly-rebalanced ML portfolio in India
Monthly rebalancing with a top-20 long-only portfolio generates meaningful turnover. After applying:
- STT (0.1% buy + 0.1% sell = 0.2% round-trip on traded value)
- Exchange charges, stamp duty, SEBI fees (~0.03–0.05% additional)
- Bid-ask spread and slippage (0.1–0.4% depending on stock)

...the gross-to-net gap could exceed 1–2% per rebalance for strategies with high turnover. No existing research we have identified documents this explicitly for Indian monthly-rebalance ML strategies.

### Gap 2.5 — Stability across market regimes in India
India has experienced distinct market regimes: post-GFC recovery (2009–2012), demonetization shock (2016), IL&FS credit crisis (2018), COVID crash and recovery (2020–2021), post-COVID inflation regime (2022–2023). Whether ML rankings outperform factor composites consistently across these regimes, or only in certain conditions, is an empirically open question.

---

## 3. What We Are NOT Claiming as Novel

- Applying machine learning to stock returns prediction — GKX (2020) has thoroughly explored this.
- Demonstrating that momentum or quality exists in India — NSE smart-beta indices and multiple practitioner papers have documented this.
- Using Ridge regression with financial features — well-established and widely practiced.
- Implementing a walk-forward backtest — standard practice, not a contribution.

---

## 4. Realistic Scope of Contribution

Given the data constraints of a university research project (primarily free/open data, no Bloomberg terminal, no point-in-time fundamental data), our realistic contribution is:

1. **A reproducible, documented, honest replication** of GKX-style evaluation adapted to Indian data with all limitations explicitly stated.
2. **An empirical comparison** of factor scoring vs. Ridge vs. GBRT under identical conditions.
3. **A cost-aware evaluation** using documented Indian statutory rates.
4. **A documented investigation** of whether pairwise ranking objectives improve portfolio metrics.

The value of this project is in its methodological rigor and transparency, not in claiming to beat professional hedge funds with free data.

---

*Last updated: 2026-10-27. Revise as literature review is completed.*
# Research Gap (Planning Note)

> This document records the original proposal, including an unimplemented Top-20
> strategy. It is not the methodology of the current implementation; see
> [`methodology.md`](methodology.md).

