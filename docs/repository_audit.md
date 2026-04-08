# Repository Audit

## Top-Level Inventory

| Path | Type | Purpose | Portfolio Assessment |
| --- | --- | --- | --- |
| `README.md` | Documentation | Top-level portfolio overview | Rewritten to frame the repository as a targeted quant portfolio |
| `projects/` | Documentation | Recruiter-facing project index and polished project pages | Added to make the repo browse like a real portfolio |
| `streamlit_app.py` | App | Interactive dashboard for crypto volatility research | Relevant and useful as a demo surface |
| `crypto_volatility/` | Project | Crypto volatility forecasting, risk metrics, and dashboard | Strong project, upgraded with strategy analytics and richer outputs |
| `factor_risk_model/` | Project | Originally a light factor-regression utility project | Weak before; rebuilt into a cross-sectional factor research engine |
| `options_pricing/` | Project | New derivatives pricing toolkit | Added to close the portfolio gap in options and numerical finance |
| `requirements*.txt` | Infra | Environment setup and CI dependencies | Modernized for current Python compatibility |

## Existing Project Evaluation

| Project | Relevance to Quant / Data Roles | Technical Depth | Real-World Applicability | Resume Value | Decision |
| --- | --- | --- | --- | --- | --- |
| `crypto_volatility` | High | Medium-High | High | High | Keep and upgrade |
| `factor_risk_model` before rewrite | Medium | Low-Medium | Medium | Medium-Low | Rebuild into factor research project |
| `options_pricing` | High | Medium-High | High | High | Added as a new standout project |

## Upgrade Decisions

### `crypto_volatility`

Before:

- Strong GARCH core
- Useful Streamlit presentation layer
- Limited trading relevance because it stopped near forecast evaluation

After:

- Added walk-forward strategy analytics
- Added volatility-targeted position sizing with turnover costs
- Added reusable performance metric utilities
- Added sample outputs and a dedicated project README

### `factor_risk_model`

Before:

- Mostly regression utilities
- Simplified factor construction with little portfolio evidence
- Not enough on its own for quant recruiting

After:

- Added cross-sectional signal construction
- Added long-short backtester with quarterly rebalancing
- Added information coefficient tracking, turnover, and factor attribution
- Added a live value-momentum-quality screener
- Added a scikit-learn classifier for next-month outperformance prediction
- Added CLI pipeline and committed sample outputs

### `options_pricing`

Added because the repository was missing a pure derivatives / numerical methods project.

Implemented:

- Black-Scholes pricing
- Greeks
- Implied volatility solver
- Monte Carlo pricing with confidence intervals
- American option binomial tree

## Outcome

The repository now has coverage across:

1. Time-series modeling and market risk
2. Cross-sectional alpha research, factor screening, and portfolio construction
3. Derivatives pricing and calibration

That mix is materially stronger for quant finance, data science, and software engineering recruiting than a single-project repository.
