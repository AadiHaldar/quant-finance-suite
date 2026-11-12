# High-Frequency Algorithmic Trading & Portfolio Optimizer

A production-ready quantitative finance framework implementing **Markowitz Modern Portfolio Theory (MPT)**, non-linear **SLSQP numerical optimization**, institutional **risk analytics & backtesting**, and a **Merton Jump Diffusion high-frequency tick emulator** with transaction cost analysis (TCA).

---

## Key Features

1. **Markowitz Mean-Variance Optimization (`src/optimizer.py`)**:
   - Computes the **Maximum Sharpe Ratio (Tangency Portfolio)**:
     $$\max_{\mathbf{w}} \frac{\mathbf{w}^T \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}} \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad w_i \ge 0$$
   - Solves for the **Global Minimum Variance (GMV)** portfolio.
   - Generates the continuous **Efficient Frontier** curve with the **Capital Allocation Line (CAL)**.
   - Computes benchmark allocations: **Inverse Volatility (Risk Parity)** and **Equal Weight (1/N)**.

2. **Market Ingestion & HFT Tick Stream Generator (`src/data_engine.py`)**:
   - Ingests real-time/historical OHLCV data using `yfinance` with automated synthetic fallback.
   - Emulates sub-second Level 1/Level 2 order book streams using **Merton Jump Diffusion**:
     $$dS_t = \mu S_t dt + \sigma S_t dW_t + J_t S_t dN_t$$

3. **Institutional Backtester & Risk Engine (`src/backtester.py`)**:
   - Evaluates strategies across historical periods with customizable rebalancing frequencies (*daily, weekly, monthly, quarterly, static*).
   - Computes metrics: **CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Maximum Drawdown (MDD), Parametric/Historical VaR (95%, 99%), Conditional VaR (CVaR/Expected Shortfall), Jensen's Alpha, Beta, R-Squared, Tracking Error, Information Ratio, and Win Rate**.

4. **Order Execution Simulator & TCA (`src/execution_engine.py`)**:
   - Models market microstructure slippage, bid-ask spread crossing, broker commissions, and temporary/permanent price impact.
   - Exports post-trade transaction cost analysis (TCA) logs and implementation shortfall.

5. **Interactive Streamlit Dashboard (`app.py`)**:
   - Interactive frontier explorer, donut allocation diagrams, underwater drawdown charts, and real-time tick emulator.

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run CLI Backtest & Optimization
```bash
python main.py --tickers AAPL MSFT GOOGL AMZN NVDA JPM BAC --start 2021-01-01 --rf 0.045 --rebalance monthly
```

### 3. Launch Interactive Streamlit UI
```bash
streamlit run app.py
```

### 4. Run Test Suite
```bash
pytest tests/ -v
```
