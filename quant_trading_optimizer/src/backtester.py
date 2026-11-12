"""
Portfolio Backtesting & Risk Analytics Engine.
Calculates CAGR, Sharpe, Sortino, Max Drawdown, VaR/CVaR,
and CAPM Alpha/Beta against benchmark.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Comprehensive performance and risk metrics report."""
    initial_capital: float
    final_capital: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    var_95: float
    cvar_95: float
    var_99: float
    alpha: float
    beta: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    win_rate: float
    profit_factor: float


class PortfolioBacktester:
    """
    Backtests asset allocation strategies with rebalancing,
    transaction costs, and calculates institutional-grade risk metrics.
    """

    def __init__(
        self,
        asset_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.045,
        transaction_cost_bps: float = 5.0,  # 5 basis points per rebalance
        initial_capital: float = 100000.0,
    ):
        self.asset_returns = asset_returns.copy()
        self.benchmark_returns = benchmark_returns.reindex(asset_returns.index).fillna(0.0)
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.initial_capital = initial_capital

    def run_backtest(
        self,
        weights: Dict[str, float],
        rebalance_freq: str = "monthly",  # 'daily', 'weekly', 'monthly', 'quarterly', 'static'
    ) -> Tuple[BacktestMetrics, pd.DataFrame]:
        """
        Executes historical backtest over the returns dataframe.
        Returns BacktestMetrics object and daily timeseries DataFrame.
        """
        tickers = [t for t in weights.keys() if t in self.asset_returns.columns]
        w_vec = np.array([weights[t] for t in tickers])
        w_vec = w_vec / np.sum(w_vec) if np.sum(w_vec) != 0 else w_vec

        df_rets = self.asset_returns[tickers]
        dates = df_rets.index
        n_days = len(dates)

        portfolio_daily_rets = np.zeros(n_days)
        portfolio_values = np.zeros(n_days)
        current_val = self.initial_capital
        portfolio_values[0] = current_val

        # Tracking asset holdings in currency
        current_holdings = current_val * w_vec

        for t in range(n_days):
            day_asset_ret = df_rets.iloc[t].values
            
            # Asset values before day's close
            current_holdings = current_holdings * (1.0 + day_asset_ret)
            end_val = np.sum(current_holdings)

            # Check if rebalancing is due
            is_rebalance = False
            if rebalance_freq == "daily":
                is_rebalance = True
            elif rebalance_freq == "weekly" and dates[t].weekday() == 4:
                is_rebalance = True
            elif rebalance_freq == "monthly" and (t == n_days - 1 or dates[t].month != dates[min(t + 1, n_days - 1)].month):
                is_rebalance = True
            elif rebalance_freq == "quarterly" and (t == n_days - 1 or dates[t].quarter != dates[min(t + 1, n_days - 1)].quarter):
                is_rebalance = True

            if is_rebalance and t < n_days - 1:
                target_holdings = end_val * w_vec
                turnover = np.sum(np.abs(target_holdings - current_holdings))
                rebalance_fee = turnover * self.transaction_cost
                end_val -= rebalance_fee
                current_holdings = end_val * w_vec

            day_ret = (end_val - current_val) / current_val if current_val > 0 else 0.0
            portfolio_daily_rets[t] = day_ret
            current_val = end_val
            portfolio_values[t] = current_val

        # Create timeseries DataFrame
        ts_df = pd.DataFrame(
            {
                "portfolio_value": portfolio_values,
                "daily_return": portfolio_daily_rets,
                "benchmark_return": self.benchmark_returns.values,
            },
            index=dates,
        )

        ts_df["cumulative_return"] = (1.0 + ts_df["daily_return"]).cumprod() - 1.0
        ts_df["benchmark_cum_return"] = (1.0 + ts_df["benchmark_return"]).cumprod() - 1.0
        ts_df["peak"] = ts_df["portfolio_value"].cummax()
        ts_df["drawdown"] = (ts_df["portfolio_value"] - ts_df["peak"]) / ts_df["peak"]

        # Calculate institutional metrics
        metrics = self._calculate_metrics(ts_df)
        return metrics, ts_df

    def _calculate_metrics(self, ts_df: pd.DataFrame) -> BacktestMetrics:
        """Computes risk and return metrics."""
        returns = ts_df["daily_return"].values
        bench_returns = ts_df["benchmark_return"].values
        n_days = len(returns)
        years = max(n_days / 252.0, 1.0 / 252.0)

        initial_val = self.initial_capital
        final_val = ts_df["portfolio_value"].iloc[-1]
        total_ret = (final_val - initial_val) / initial_val
        cagr = (final_val / initial_val) ** (1.0 / years) - 1.0

        ann_vol = float(np.std(returns) * np.sqrt(252))

        # Sharpe Ratio
        excess_returns = returns - self.daily_rf
        mean_excess = np.mean(excess_returns) * 252
        sharpe = float(mean_excess / ann_vol) if ann_vol > 0 else 0.0

        # Sortino Ratio (downside deviation with MAR = risk_free_rate)
        downside_diff = np.minimum(0, returns - self.daily_rf)
        downside_dev = float(np.sqrt(np.mean(downside_diff**2)) * np.sqrt(252))
        sortino = float(mean_excess / downside_dev) if downside_dev > 0 else 0.0

        # Maximum Drawdown & Duration
        mdd = float(ts_df["drawdown"].min())
        calmar = float(cagr / abs(mdd)) if abs(mdd) > 0 else 0.0

        # Drawdown duration
        is_in_dd = ts_df["drawdown"] < 0
        dd_runs = (~is_in_dd).cumsum()
        max_dd_duration = int(is_in_dd.groupby(dd_runs).sum().max()) if is_in_dd.any() else 0

        # Value at Risk (Historical & Parametric)
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))
        cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

        # CAPM Alpha and Beta
        covariance_matrix = np.cov(returns, bench_returns)
        bench_variance = covariance_matrix[1, 1]
        cov_p_m = covariance_matrix[0, 1]

        if bench_variance > 1e-8:
            beta = float(cov_p_m / bench_variance)
            ann_bench_ret = float(np.mean(bench_returns) * 252)
            ann_p_ret = float(np.mean(returns) * 252)
            alpha = float(ann_p_ret - (self.risk_free_rate + beta * (ann_bench_ret - self.risk_free_rate)))
            corr = covariance_matrix[0, 1] / (np.std(returns) * np.std(bench_returns) + 1e-8)
            r_squared = float(corr**2)
        else:
            beta, alpha, r_squared = 1.0, 0.0, 1.0

        # Tracking Error & Information Ratio
        diff_returns = returns - bench_returns
        tracking_error = float(np.std(diff_returns) * np.sqrt(252))
        info_ratio = float((np.mean(diff_returns) * 252) / tracking_error) if tracking_error > 0 else 0.0

        # Win Rate and Profit Factor
        positive_days = returns[returns > 0]
        negative_days = returns[returns < 0]
        win_rate = float(len(positive_days) / n_days) if n_days > 0 else 0.0
        gross_profit = float(np.sum(positive_days)) if len(positive_days) > 0 else 0.0
        gross_loss = float(abs(np.sum(negative_days))) if len(negative_days) > 0 else 1e-6
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0

        return BacktestMetrics(
            initial_capital=initial_val,
            final_capital=round(final_val, 2),
            total_return=round(total_ret, 4),
            cagr=round(cagr, 4),
            annualized_volatility=round(ann_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
            max_drawdown=round(mdd, 4),
            max_drawdown_duration_days=max_dd_duration,
            var_95=round(var_95, 4),
            cvar_95=round(cvar_95, 4),
            var_99=round(var_99, 4),
            alpha=round(alpha, 4),
            beta=round(beta, 4),
            r_squared=round(r_squared, 4),
            tracking_error=round(tracking_error, 4),
            information_ratio=round(info_ratio, 4),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
        )
