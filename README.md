# Quantitative Investment & Financial Engineering Suite

A dual-component quantitative finance ecosystem developed for portfolio optimization, high-frequency execution simulation, natural language financial sentiment extraction, and automated intrinsic equity valuation.

---

## Suite Architecture & Projects

```
quant_finance_suite/
├── quant_trading_optimizer/          # Component 1: High-Frequency Algorithmic Trading & Portfolio Optimizer
│   ├── src/
│   │   ├── data_engine.py            # Market data ingestion & Merton Jump Diffusion tick stream generator
│   │   ├── optimizer.py              # Markowitz MPT, SLSQP Tangency & Minimum Volatility solver
│   │   ├── backtester.py             # Institutional risk engine (Sharpe, Sortino, Drawdown, VaR/CVaR, CAPM)
│   │   ├── execution_engine.py       # Order execution simulator & Transaction Cost Analysis (TCA)
│   │   └── utils.py                  # Plotly visualizations & tabular ASCII performance reporting
│   ├── tests/
│   │   ├── test_optimizer.py         # Mathematical constraint tests (sum(w)=1, variance > 0)
│   │   └── test_backtester.py        # Compounding, Drawdown, and VaR/CVaR validation
│   ├── app.py                        # Interactive Streamlit dashboard with Efficient Frontier & live ticks
│   ├── main.py                       # CLI runner with automated summary & JSON export
│   ├── requirements.txt              # Core dependencies
│   └── README.md                     # Component technical documentation
│
└── equity_sentiment_valuation/       # Component 2: Automated Equity Valuation & Financial Sentiment Pipeline
    ├── data/filings/                 # Sample SEC 10-K filings (AAPL, NVDA, BAC)
    ├── src/
    │   ├── document_parser.py        # SEC 10-K & earnings call transcript segmenter (MD&A, Risk Factors)
    │   ├── sentiment_analyzer.py     # FinBERT transformer NLP & Loughran-McDonald lexicon fallback
    │   ├── dcf_model.py              # Multi-stage DCF model (WACC, FCFF, Gordon Growth TV, Sensitivity)
    │   ├── sentiment_dcf_integrator.py # Dynamic growth & risk premium modulation via sentiment index
    │   └── report_generator.py       # Publication-grade equity research report generator
    ├── tests/
    │   ├── test_dcf.py               # WACC, FCFF discounting, and Gordon Growth constraints
    │   └── test_sentiment.py         # Polarity scoring, section extraction, and tokenization tests
    ├── app.py                        # Streamlit Equity Research & Valuation workbench
    ├── main.py                       # CLI runner with research memo synthesis & JSON export
    ├── requirements.txt              # Core dependencies
    └── README.md                     # Component technical documentation
```

---

## 1. High-Frequency Algorithmic Trading & Portfolio Optimizer

* **Mathematical Foundations:**
  - Markowitz Mean-Variance Portfolio Optimization:
    $$\max_{\mathbf{w}} \frac{\mathbf{w}^T \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}} \quad \text{subject to} \quad \sum_{i=1}^N w_i = 1, \quad w_i \ge 0$$
  - Sub-second High-Frequency Tick Simulation with Merton Jump Diffusion:
    $$dS_t = \mu S_t dt + \sigma S_t dW_t + J_t S_t dN_t$$
  - Risk Metrics: Annualized Sharpe, Sortino Ratio, Calmar Ratio, Historical/Parametric VaR (95%, 99%), Conditional VaR (CVaR / Expected Shortfall), Jensen's Alpha, Beta, Tracking Error, and Maximum Drawdown.

* **Quickstart:**
  ```bash
  cd quant_trading_optimizer
  python main.py --tickers AAPL MSFT NVDA JPM BAC --start 2021-01-01 --rebalance monthly
  streamlit run app.py
  ```

---

## 2. Automated Equity Valuation & Financial Sentiment Pipeline

* **Mathematical Foundations:**
  - Weighted Average Cost of Capital (WACC):
    $$WACC = \left(\frac{E}{V} \times K_e\right) + \left(\frac{D}{V} \times K_d \times (1 - t)\right), \quad K_e = R_f + \beta \times ERP$$
  - Free Cash Flow to Firm (FCFF):
    $$FCFF_t = \text{EBIT}_t \times (1 - t) + \text{D\&A}_t - \text{CapEx}_t - \Delta \text{NWC}_t$$
  - Gordon Growth Terminal Value & Enterprise Value:
    $$TV = \frac{FCFF_N \times (1 + g)}{WACC - g}, \quad EV = \sum_{t=1}^N \frac{FCFF_t}{(1 + WACC)^t} + \frac{TV}{(1 + WACC)^N}$$
  - Dynamic Sentiment Coupling: FinBERT confidence and compound scores modulate multi-year revenue growth trajectories, operating margins, and cost of capital risk premiums across Base, Bull, Bear, and Sentiment-Informed scenarios.

* **Quickstart:**
  ```bash
  cd equity_sentiment_valuation
  python main.py --ticker AAPL
  streamlit run app.py
  ```

---

## Test Execution

Both projects feature comprehensive test suites:

```bash
# Test Portfolio Optimizer & Backtester
cd quant_trading_optimizer
python -m pytest tests/ -v

# Test DCF Valuation & FinBERT Sentiment Engine
cd ../equity_sentiment_valuation
python -m pytest tests/ -v
```
