"""
Interactive Quantitative Portfolio Optimizer & High-Frequency Trading Dashboard.
Built with Streamlit and Plotly for real-time risk analytics and execution visualization.
"""
import datetime
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure module path is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_engine import MarketDataEngine, TickStreamGenerator
from src.optimizer import PortfolioOptimizer
from src.backtester import PortfolioBacktester
from src.execution_engine import ExecutionSimulator
from src.utils import format_currency, format_percent, create_plotly_efficient_frontier

# Configure Streamlit Page
st.set_page_config(
    page_title="Quant Portfolio Optimizer & Execution Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 High-Frequency Algorithmic Trading & Portfolio Optimizer")
st.caption("Bank of America Quantitative Investment & Financial Engineering Suite")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "BAC"]
    selected_tickers = st.multiselect(
        "Universe Tickers",
        options=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS", "WMT", "V", "UNH"],
        default=default_tickers,
    )
    if len(selected_tickers) < 2:
        st.warning("Please select at least 2 tickers for optimization.")
        selected_tickers = default_tickers

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", datetime.date(2021, 1, 1))
    with col_d2:
        end_date = st.date_input("End Date", datetime.date.today())

    st.subheader("Risk & Execution Parameters")
    risk_free_rate = st.slider("Risk-Free Rate (Rf)", min_value=0.0, max_value=0.10, value=0.045, step=0.005, format="%.3f")
    rebalance_freq = st.selectbox("Rebalancing Frequency", ["monthly", "weekly", "daily", "quarterly", "static"], index=0)
    initial_capital = st.number_input("Initial Portfolio Value ($)", min_value=10000.0, value=100000.0, step=10000.0)
    tx_cost_bps = st.slider("Transaction Cost (bps)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)

    run_btn = st.button("🚀 Run Portfolio Optimization", type="primary", use_container_width=True)

# --- DATA & CACHE ---
@st.cache_data(show_spinner=False)
def load_market_data(tickers, start, end, rf):
    engine = MarketDataEngine(
        tickers=tickers,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        risk_free_rate=rf,
    )
    engine.fetch_data(benchmark="SPY")
    mu, sigma = engine.get_annualized_stats()
    return engine, mu, sigma

with st.spinner("Fetching Market Prices & Calculating Covariance Matrix..."):
    engine, mu, sigma = load_market_data(selected_tickers, start_date, end_date, risk_free_rate)

optimizer = PortfolioOptimizer(expected_returns=mu, cov_matrix=sigma, risk_free_rate=risk_free_rate)
max_sharpe = optimizer.max_sharpe_portfolio()
min_vol = optimizer.min_variance_portfolio()
eq_weight = optimizer.equal_weight_portfolio()
inv_vol = optimizer.inverse_volatility_portfolio()
frontier_df = optimizer.generate_efficient_frontier(n_points=40)
asset_stats = optimizer.get_individual_assets_summary()

# --- BACKTEST ---
backtester = PortfolioBacktester(
    asset_returns=engine.returns,
    benchmark_returns=engine.benchmark_returns,
    risk_free_rate=risk_free_rate,
    transaction_cost_bps=tx_cost_bps,
    initial_capital=initial_capital,
)
metrics_sharpe, ts_sharpe = backtester.run_backtest(max_sharpe.weights, rebalance_freq=rebalance_freq)
metrics_minvol, ts_minvol = backtester.run_backtest(min_vol.weights, rebalance_freq=rebalance_freq)
metrics_eq, ts_eq = backtester.run_backtest(eq_weight.weights, rebalance_freq=rebalance_freq)

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs([
    "🎯 Efficient Frontier & Allocations",
    "📊 Backtesting & Risk Analytics",
    "⚡ High-Frequency Execution & TCA",
])

# --- TAB 1: EFFICIENT FRONTIER ---
with tab1:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Max Sharpe Ratio", f"{max_sharpe.sharpe_ratio:.2f}", help="Tangency Portfolio")
    with col_m2:
        st.metric("Tangency Exp. Return", format_percent(max_sharpe.expected_return), f"Vol: {format_percent(max_sharpe.volatility)}")
    with col_m3:
        st.metric("Min Volatility", format_percent(min_vol.volatility), f"Return: {format_percent(min_vol.expected_return)}")
    with col_m4:
        st.metric("Equal Weight Sharpe", f"{eq_weight.sharpe_ratio:.2f}", f"Return: {format_percent(eq_weight.expected_return)}")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns([3, 2])
    with col_chart1:
        st.subheader("Markowitz Efficient Frontier")
        ef_fig = create_plotly_efficient_frontier(frontier_df, max_sharpe, min_vol, asset_stats, risk_free_rate)
        if ef_fig:
            st.plotly_chart(ef_fig, use_container_width=True)

    with col_chart2:
        st.subheader("Optimal Portfolio Weights")
        strategy_choice = st.radio("Select Allocation Strategy", ["Max Sharpe", "Min Volatility", "Inverse Volatility", "Equal Weight"], horizontal=True)
        if strategy_choice == "Max Sharpe":
            active_weights = max_sharpe.weights
        elif strategy_choice == "Min Volatility":
            active_weights = min_vol.weights
        elif strategy_choice == "Inverse Volatility":
            active_weights = inv_vol.weights
        else:
            active_weights = eq_weight.weights

        w_df = pd.DataFrame(list(active_weights.items()), columns=["Asset", "Weight"])
        w_df = w_df[w_df["Weight"] > 0.001]

        donut_fig = px.pie(
            w_df,
            values="Weight",
            names="Asset",
            hole=0.45,
            title=f"{strategy_choice} Allocation Breakdown",
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        donut_fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(donut_fig, use_container_width=True)

    # Weights table
    st.subheader("Comparative Asset Allocation Table")
    all_w_df = pd.DataFrame({
        "Asset": selected_tickers,
        "Max Sharpe Weight": [f"{max_sharpe.weights.get(t, 0.0):.1%}" for t in selected_tickers],
        "Min Vol Weight": [f"{min_vol.weights.get(t, 0.0):.1%}" for t in selected_tickers],
        "Inverse Vol Weight": [f"{inv_vol.weights.get(t, 0.0):.1%}" for t in selected_tickers],
        "Equal Weight": [f"{eq_weight.weights.get(t, 0.0):.1%}" for t in selected_tickers],
    })
    st.dataframe(all_w_df, use_container_width=True)

# --- TAB 2: BACKTESTING & RISK ---
with tab2:
    st.subheader("Cumulative Portfolio Performance vs S&P 500 Benchmark")

    cum_fig = go.Figure()
    cum_fig.add_trace(go.Scatter(x=ts_sharpe.index, y=ts_sharpe["cumulative_return"], mode="lines", name="Max Sharpe Strategy", line=dict(color="#1f77b4", width=2.5)))
    cum_fig.add_trace(go.Scatter(x=ts_minvol.index, y=ts_minvol["cumulative_return"], mode="lines", name="Min Volatility Strategy", line=dict(color="#9467bd", width=2)))
    cum_fig.add_trace(go.Scatter(x=ts_eq.index, y=ts_eq["cumulative_return"], mode="lines", name="Equal Weight (1/N)", line=dict(color="#ff7f0e", width=1.5, dash="dot")))
    cum_fig.add_trace(go.Scatter(x=ts_sharpe.index, y=ts_sharpe["benchmark_cum_return"], mode="lines", name="S&P 500 (SPY)", line=dict(color="#7f7f7f", width=2, dash="dash")))

    cum_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".1%",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(cum_fig, use_container_width=True)

    # Underwater Drawdown Plot
    st.subheader("Historical Underwater Drawdown (MDD)")
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=ts_sharpe.index, y=ts_sharpe["drawdown"], fill="tozeroy", mode="lines", name="Max Sharpe Drawdown", line=dict(color="#d62728", width=1.5)))
    dd_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".1%",
        template="plotly_white",
        margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(dd_fig, use_container_width=True)

    # Risk scorecard
    st.subheader("Institutional Risk & Return Scorecard")
    metrics_comp = pd.DataFrame({
        "Metric": [
            "Total Cumulative Return",
            "CAGR (Annualized)",
            "Annualized Volatility (σ)",
            "Sharpe Ratio (Rf=4.5%)",
            "Sortino Ratio",
            "Calmar Ratio",
            "Maximum Drawdown (MDD)",
            "Max Drawdown Duration",
            "Value at Risk (VaR 95%)",
            "Conditional VaR (CVaR 95%)",
            "Jensen's Alpha (α vs SPY)",
            "Market Beta (β vs SPY)",
            "R-Squared (R²)",
            "Tracking Error",
            "Information Ratio",
            "Daily Win Rate",
            "Profit Factor",
        ],
        "Max Sharpe Strategy": [
            format_percent(metrics_sharpe.total_return),
            format_percent(metrics_sharpe.cagr),
            format_percent(metrics_sharpe.annualized_volatility),
            f"{metrics_sharpe.sharpe_ratio:.2f}",
            f"{metrics_sharpe.sortino_ratio:.2f}",
            f"{metrics_sharpe.calmar_ratio:.2f}",
            format_percent(metrics_sharpe.max_drawdown),
            f"{metrics_sharpe.max_drawdown_duration_days} days",
            format_percent(metrics_sharpe.var_95),
            format_percent(metrics_sharpe.cvar_95),
            format_percent(metrics_sharpe.alpha),
            f"{metrics_sharpe.beta:.2f}",
            f"{metrics_sharpe.r_squared:.2f}",
            format_percent(metrics_sharpe.tracking_error),
            f"{metrics_sharpe.information_ratio:.2f}",
            format_percent(metrics_sharpe.win_rate),
            f"{metrics_sharpe.profit_factor:.2f}",
        ],
        "Min Volatility Strategy": [
            format_percent(metrics_minvol.total_return),
            format_percent(metrics_minvol.cagr),
            format_percent(metrics_minvol.annualized_volatility),
            f"{metrics_minvol.sharpe_ratio:.2f}",
            f"{metrics_minvol.sortino_ratio:.2f}",
            f"{metrics_minvol.calmar_ratio:.2f}",
            format_percent(metrics_minvol.max_drawdown),
            f"{metrics_minvol.max_drawdown_duration_days} days",
            format_percent(metrics_minvol.var_95),
            format_percent(metrics_minvol.cvar_95),
            format_percent(metrics_minvol.alpha),
            f"{metrics_minvol.beta:.2f}",
            f"{metrics_minvol.r_squared:.2f}",
            format_percent(metrics_minvol.tracking_error),
            f"{metrics_minvol.information_ratio:.2f}",
            format_percent(metrics_minvol.win_rate),
            f"{metrics_minvol.profit_factor:.2f}",
        ],
        "Equal Weight (1/N)": [
            format_percent(metrics_eq.total_return),
            format_percent(metrics_eq.cagr),
            format_percent(metrics_eq.annualized_volatility),
            f"{metrics_eq.sharpe_ratio:.2f}",
            f"{metrics_eq.sortino_ratio:.2f}",
            f"{metrics_eq.calmar_ratio:.2f}",
            format_percent(metrics_eq.max_drawdown),
            f"{metrics_eq.max_drawdown_duration_days} days",
            format_percent(metrics_eq.var_95),
            format_percent(metrics_eq.cvar_95),
            format_percent(metrics_eq.alpha),
            f"{metrics_eq.beta:.2f}",
            f"{metrics_eq.r_squared:.2f}",
            format_percent(metrics_eq.tracking_error),
            f"{metrics_eq.information_ratio:.2f}",
            format_percent(metrics_eq.win_rate),
            f"{metrics_eq.profit_factor:.2f}",
        ],
    })
    st.dataframe(metrics_comp, use_container_width=True)

# --- TAB 3: HFT & EXECUTION ---
with tab3:
    st.subheader("⚡ Real-Time High-Frequency Tick Data Stream Emulator")
    st.caption("Simulated using Merton Jump Diffusion (GBM + Poisson Jumps) with Microstructure Spreads")

    col_hft1, col_hft2 = st.columns([1, 3])
    with col_hft1:
        hft_sym = st.selectbox("Select Asset for Live Feed", selected_tickers, index=0)
        n_ticks = st.slider("Number of Ticks", 20, 200, 50)
        jump_lambda = st.slider("Jump Intensity (λ)", 0.0, 10.0, 3.0, 0.5)
        gen_ticks_btn = st.button("⚡ Generate Live Tick Stream")

    latest_px = float(engine.prices[hft_sym].iloc[-1]) if hft_sym in engine.prices.columns else 150.0
    tick_gen = TickStreamGenerator(
        symbol=hft_sym,
        initial_price=latest_px,
        jump_intensity=jump_lambda,
    )
    ticks_df = tick_gen.generate_tick_stream(n_ticks=n_ticks)

    with col_hft2:
        tick_plot = go.Figure()
        tick_plot.add_trace(go.Scatter(x=list(range(len(ticks_df))), y=ticks_df["price"], mode="lines+markers", name=f"{hft_sym} Tick Price", line=dict(color="#00cc96", width=2)))
        tick_plot.add_trace(go.Scatter(x=list(range(len(ticks_df))), y=ticks_df["bid"], mode="lines", name="Bid", line=dict(color="#ef553b", dash="dot")))
        tick_plot.add_trace(go.Scatter(x=list(range(len(ticks_df))), y=ticks_df["ask"], mode="lines", name="Ask", line=dict(color="#636efa", dash="dot")))
        tick_plot.update_layout(title=f"Sub-Second Microstructure Stream - {hft_sym}", xaxis_title="Tick Sequence", yaxis_title="Price ($)", template="plotly_white", margin=dict(l=40, r=40, t=40, b=30))
        st.plotly_chart(tick_plot, use_container_width=True)

    st.subheader("Simulated Order Execution & Transaction Cost Analysis (TCA)")
    exec_sim = ExecutionSimulator(base_fee_per_share=0.005, fixed_slippage_bps=1.5)
    latest_prices = engine.prices.iloc[-1].to_dict()
    orders = exec_sim.rebalance_portfolio_orders(
        current_holdings={},
        target_weights=max_sharpe.weights,
        current_prices=latest_prices,
        total_portfolio_value=initial_capital,
    )
    tca_df = exec_sim.get_trade_logs_df()
    if not tca_df.empty:
        st.dataframe(tca_df, use_container_width=True)
