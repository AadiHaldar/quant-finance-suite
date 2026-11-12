"""
Unit tests for PortfolioOptimizer (Markowitz MPT).
Verifies weight constraints, positive variance, and Sharpe ratio optimality.
"""
import pytest
import numpy as np
import pandas as pd
from src.optimizer import PortfolioOptimizer


@pytest.fixture
def sample_market_stats():
    """Generates synthetic annualized returns and covariance matrix."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    mu = pd.Series([0.20, 0.18, 0.15, 0.22], index=tickers)
    
    # Positive semi-definite covariance matrix
    corr = np.array([
        [1.0, 0.6, 0.5, 0.4],
        [0.6, 1.0, 0.55, 0.45],
        [0.5, 0.55, 1.0, 0.5],
        [0.4, 0.45, 0.5, 1.0],
    ])
    vols = np.array([0.25, 0.22, 0.28, 0.30])
    cov = np.outer(vols, vols) * corr
    sigma = pd.DataFrame(cov, index=tickers, columns=tickers)
    return mu, sigma


def test_max_sharpe_weights_sum_to_one(sample_market_stats):
    mu, sigma = sample_market_stats
    opt = PortfolioOptimizer(mu, sigma, risk_free_rate=0.04)
    res = opt.max_sharpe_portfolio()
    
    total_w = sum(res.weights.values())
    assert pytest.approx(total_w, rel=1e-3) == 1.0
    assert res.success is True
    assert res.sharpe_ratio > 0


def test_min_volatility_weights_sum_to_one(sample_market_stats):
    mu, sigma = sample_market_stats
    opt = PortfolioOptimizer(mu, sigma, risk_free_rate=0.04)
    res = opt.min_variance_portfolio()
    
    total_w = sum(res.weights.values())
    assert pytest.approx(total_w, rel=1e-3) == 1.0
    assert res.success is True
    assert res.volatility > 0


def test_positive_portfolio_variance(sample_market_stats):
    mu, sigma = sample_market_stats
    opt = PortfolioOptimizer(mu, sigma, risk_free_rate=0.04)
    res_sharpe = opt.max_sharpe_portfolio()
    res_minvol = opt.min_variance_portfolio()
    
    assert res_sharpe.volatility > 0
    assert res_minvol.volatility > 0
    assert res_minvol.volatility <= res_sharpe.volatility + 1e-4


def test_max_sharpe_dominance_over_equal_weight(sample_market_stats):
    mu, sigma = sample_market_stats
    opt = PortfolioOptimizer(mu, sigma, risk_free_rate=0.04)
    res_sharpe = opt.max_sharpe_portfolio()
    res_eq = opt.equal_weight_portfolio()
    
    assert res_sharpe.sharpe_ratio >= res_eq.sharpe_ratio - 1e-4


def test_efficient_frontier_generation(sample_market_stats):
    mu, sigma = sample_market_stats
    opt = PortfolioOptimizer(mu, sigma, risk_free_rate=0.04)
    frontier_df = opt.generate_efficient_frontier(n_points=10)
    
    assert len(frontier_df) > 0
    assert "return" in frontier_df.columns
    assert "volatility" in frontier_df.columns
    assert (frontier_df["volatility"] > 0).all()
