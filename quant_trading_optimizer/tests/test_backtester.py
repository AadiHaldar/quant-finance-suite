"""
Unit tests for PortfolioBacktester and ExecutionSimulator.
Verifies return compounding, drawdown calculation, VaR/CVaR, and TCA logs.
"""
import pytest
import numpy as np
import pandas as pd
from src.backtester import PortfolioBacktester
from src.execution_engine import ExecutionSimulator


@pytest.fixture
def sample_returns_data():
    """Creates synthetic multi-asset daily return timeseries."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2023-01-01", freq="B")
    n = len(dates)
    
    asset_rets = pd.DataFrame({
        "AAPL": np.random.normal(0.0008, 0.015, n),
        "MSFT": np.random.normal(0.0007, 0.014, n),
        "NVDA": np.random.normal(0.0012, 0.025, n),
    }, index=dates)
    
    bench_rets = pd.Series(np.random.normal(0.0004, 0.010, n), index=dates)
    return asset_rets, bench_rets


def test_backtest_capital_compounding(sample_returns_data):
    asset_rets, bench_rets = sample_returns_data
    backtester = PortfolioBacktester(asset_rets, bench_rets, initial_capital=100000.0)
    
    weights = {"AAPL": 0.4, "MSFT": 0.3, "NVDA": 0.3}
    metrics, ts_df = backtester.run_backtest(weights, rebalance_freq="monthly")
    
    assert metrics.initial_capital == 100000.0
    assert metrics.final_capital > 0
    expected_final = metrics.initial_capital * (1.0 + metrics.total_return)
    assert pytest.approx(metrics.final_capital, rel=1e-2) == expected_final


def test_drawdown_bounds(sample_returns_data):
    asset_rets, bench_rets = sample_returns_data
    backtester = PortfolioBacktester(asset_rets, bench_rets)
    
    weights = {"AAPL": 0.5, "MSFT": 0.5, "NVDA": 0.0}
    metrics, ts_df = backtester.run_backtest(weights)
    
    assert metrics.max_drawdown <= 0.0
    assert metrics.max_drawdown >= -1.0
    assert (ts_df["drawdown"] <= 0.0).all()


def test_var_and_cvar_relationship(sample_returns_data):
    asset_rets, bench_rets = sample_returns_data
    backtester = PortfolioBacktester(asset_rets, bench_rets)
    
    weights = {"AAPL": 0.34, "MSFT": 0.33, "NVDA": 0.33}
    metrics, _ = backtester.run_backtest(weights)
    
    # CVaR (expected shortfall) should be as severe as or more severe than VaR
    assert metrics.cvar_95 <= metrics.var_95 + 1e-6


def test_execution_simulator_orders():
    exec_sim = ExecutionSimulator(base_fee_per_share=0.005, fixed_slippage_bps=2.0)
    order = exec_sim.execute_order(
        symbol="AAPL",
        side="BUY",
        quantity=100,
        market_mid_price=150.0,
        bid_price=149.95,
        ask_price=150.05,
    )
    
    assert order.status == "FILLED"
    assert order.filled_qty == 100
    assert order.filled_price >= 150.0  # Buy price includes half-spread and slippage
    assert order.fee == 0.50  # 100 * $0.005
    assert len(exec_sim.trade_logs) == 1
