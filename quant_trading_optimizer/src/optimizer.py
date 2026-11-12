"""
Markowitz Modern Portfolio Theory (MPT) Optimization Engine.
Solves Max Sharpe Ratio, Minimum Volatility, and traces the Efficient Frontier
using Sequential Least Squares Programming (SLSQP) via scipy.optimize.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    """Structured container for portfolio optimization results."""
    strategy_name: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str


class PortfolioOptimizer:
    """
    Mean-Variance Portfolio Optimizer implementing Markowitz (1952) framework
    with quadratic programming and nonlinear constraint solvers.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.045,
        allow_short: bool = False,
        weight_bounds: Optional[Tuple[float, float]] = None,
    ):
        self.tickers = list(expected_returns.index)
        self.n_assets = len(self.tickers)
        self.mu = expected_returns.values
        self.Sigma = cov_matrix.values
        self.risk_free_rate = risk_free_rate
        self.allow_short = allow_short

        if weight_bounds is not None:
            self.bounds = [weight_bounds for _ in range(self.n_assets)]
        elif allow_short:
            self.bounds = [(-1.0, 1.0) for _ in range(self.n_assets)]
        else:
            self.bounds = [(0.0, 1.0) for _ in range(self.n_assets)]

    def _portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculates expected return, volatility, and Sharpe ratio for a weight vector."""
        ret = np.dot(weights, self.mu)
        var = np.dot(weights.T, np.dot(self.Sigma, weights))
        vol = np.sqrt(max(var, 1e-10))
        sharpe = (ret - self.risk_free_rate) / vol
        return ret, vol, sharpe

    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """Objective function for Max Sharpe."""
        ret, vol, _ = self._portfolio_performance(weights)
        return -(ret - self.risk_free_rate) / vol

    def _portfolio_variance(self, weights: np.ndarray) -> float:
        """Objective function for Minimum Variance."""
        return float(np.dot(weights.T, np.dot(self.Sigma, weights)))

    def max_sharpe_portfolio(self) -> OptimizationResult:
        """
        Solves for the Tangency Portfolio (Maximum Sharpe Ratio).
        max (w^T mu - r_f) / sqrt(w^T Sigma w) subject to sum(w) = 1.
        """
        init_weights = np.ones(self.n_assets) / self.n_assets
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        opt = minimize(
            fun=self._negative_sharpe,
            x0=init_weights,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        weights = opt.x if opt.success else init_weights
        weights = weights / np.sum(weights)  # strict normalization
        ret, vol, sharpe = self._portfolio_performance(weights)
        weight_dict = {ticker: round(float(w), 4) for ticker, w in zip(self.tickers, weights)}

        return OptimizationResult(
            strategy_name="Maximum Sharpe Ratio",
            weights=weight_dict,
            expected_return=float(ret),
            volatility=float(vol),
            sharpe_ratio=float(sharpe),
            success=opt.success,
            message=opt.message if hasattr(opt, "message") else "Solved",
        )

    def min_variance_portfolio(self) -> OptimizationResult:
        """
        Solves for the Global Minimum Variance (GMV) Portfolio.
        min w^T Sigma w subject to sum(w) = 1.
        """
        init_weights = np.ones(self.n_assets) / self.n_assets
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        opt = minimize(
            fun=self._portfolio_variance,
            x0=init_weights,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        weights = opt.x if opt.success else init_weights
        weights = weights / np.sum(weights)
        ret, vol, sharpe = self._portfolio_performance(weights)
        weight_dict = {ticker: round(float(w), 4) for ticker, w in zip(self.tickers, weights)}

        return OptimizationResult(
            strategy_name="Minimum Volatility",
            weights=weight_dict,
            expected_return=float(ret),
            volatility=float(vol),
            sharpe_ratio=float(sharpe),
            success=opt.success,
            message=opt.message if hasattr(opt, "message") else "Solved",
        )

    def efficient_return_portfolio(self, target_return: float) -> OptimizationResult:
        """
        Finds the minimum variance portfolio for a specified target return.
        min w^T Sigma w subject to w^T mu = target_return, sum(w) = 1.
        """
        init_weights = np.ones(self.n_assets) / self.n_assets
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: np.dot(w, self.mu) - target_return},
        ]

        opt = minimize(
            fun=self._portfolio_variance,
            x0=init_weights,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        weights = opt.x if opt.success else init_weights
        weights = weights / np.sum(weights)
        ret, vol, sharpe = self._portfolio_performance(weights)
        weight_dict = {ticker: round(float(w), 4) for ticker, w in zip(self.tickers, weights)}

        return OptimizationResult(
            strategy_name=f"Efficient Target Ret ({target_return:.1%})",
            weights=weight_dict,
            expected_return=float(ret),
            volatility=float(vol),
            sharpe_ratio=float(sharpe),
            success=opt.success,
            message=opt.message if hasattr(opt, "message") else "Solved",
        )

    def generate_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Constructs the Efficient Frontier curve by solving optimal risk for return steps.
        Returns a DataFrame of (return, volatility, sharpe_ratio, weights).
        """
        min_var_res = self.min_variance_portfolio()
        min_ret = min_var_res.expected_return
        max_ret = max(float(np.max(self.mu)), min_ret + 0.05)

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_points = []

        for target_ret in target_returns:
            res = self.efficient_return_portfolio(float(target_ret))
            if res.success:
                row = {
                    "return": res.expected_return,
                    "volatility": res.volatility,
                    "sharpe_ratio": res.sharpe_ratio,
                }
                row.update({f"w_{k}": v for k, v in res.weights.items()})
                frontier_points.append(row)

        return pd.DataFrame(frontier_points)

    def equal_weight_portfolio(self) -> OptimizationResult:
        """Benchmark 1/N allocation."""
        weights = np.ones(self.n_assets) / self.n_assets
        ret, vol, sharpe = self._portfolio_performance(weights)
        weight_dict = {ticker: round(float(w), 4) for ticker, w in zip(self.tickers, weights)}

        return OptimizationResult(
            strategy_name="Equal Weight (1/N)",
            weights=weight_dict,
            expected_return=float(ret),
            volatility=float(vol),
            sharpe_ratio=float(sharpe),
            success=True,
            message="Analytical equal weight",
        )

    def inverse_volatility_portfolio(self) -> OptimizationResult:
        """Risk parity heuristic based on asset standard deviations."""
        asset_vols = np.sqrt(np.diag(self.Sigma))
        inv_vols = 1.0 / np.maximum(asset_vols, 1e-6)
        weights = inv_vols / np.sum(inv_vols)
        ret, vol, sharpe = self._portfolio_performance(weights)
        weight_dict = {ticker: round(float(w), 4) for ticker, w in zip(self.tickers, weights)}

        return OptimizationResult(
            strategy_name="Inverse Volatility",
            weights=weight_dict,
            expected_return=float(ret),
            volatility=float(vol),
            sharpe_ratio=float(sharpe),
            success=True,
            message="Inverse volatility allocation",
        )

    def get_individual_assets_summary(self) -> pd.DataFrame:
        """Returns stats for individual component tickers."""
        stats = []
        for i, ticker in enumerate(self.tickers):
            vol = np.sqrt(self.Sigma[i, i])
            ret = self.mu[i]
            sharpe = (ret - self.risk_free_rate) / vol
            stats.append({
                "Ticker": ticker,
                "Expected Return": round(float(ret), 4),
                "Volatility": round(float(vol), 4),
                "Sharpe Ratio": round(float(sharpe), 4),
            })
        return pd.DataFrame(stats)
