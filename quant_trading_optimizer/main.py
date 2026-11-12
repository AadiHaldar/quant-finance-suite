"""
Quantitative Portfolio Optimizer & Algorithmic Trading Framework CLI Runner.
Executes Markowitz MPT, historical backtesting, order execution simulation,
and Merton Jump Diffusion HFT tick streams.
"""
import argparse
import datetime
import json
import os
import sys
import pandas as pd

# Add module to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_engine import MarketDataEngine, TickStreamGenerator
from src.optimizer import PortfolioOptimizer
from src.backtester import PortfolioBacktester
from src.execution_engine import ExecutionSimulator
from src.utils import print_metrics_table, format_percent, format_currency


def run_pipeline(
    tickers: list,
    start_date: str,
    end_date: str,
    risk_free_rate: float,
    rebalance_freq: str,
    initial_capital: float,
    export_dir: str = "output",
    simulate_ticks: bool = True,
):
    print("=" * 70)
    print(" QUANTITATIVE PORTFOLIO OPTIMIZATION & TRADING SUITE ")
    print(" Bank of America Algorithmic Strategy Implementation ")
    print("=" * 70)
    print(f"[*] Universe Tickers : {', '.join(tickers)}")
    print(f"[*] Date Range       : {start_date} to {end_date}")
    print(f"[*] Risk-Free Rate   : {risk_free_rate:.2%}")
    print(f"[*] Initial Capital  : {format_currency(initial_capital)}")
    print(f"[*] Rebalancing Freq : {rebalance_freq.capitalize()}")
    print("-" * 70)

    # 1. Market Data Ingestion
    print("\n[1/5] Ingesting Market Data & Estimating Parameters...")
    engine = MarketDataEngine(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
    )
    prices = engine.fetch_data(benchmark="SPY")
    mu, sigma = engine.get_annualized_stats()
    print(f"      Successfully loaded {len(prices)} trading days for {len(engine.prices.columns)} assets.")

    # 2. Portfolio Optimization
    print("\n[2/5] Running Markowitz Mean-Variance Optimization (SLSQP)...")
    optimizer = PortfolioOptimizer(
        expected_returns=mu,
        cov_matrix=sigma,
        risk_free_rate=risk_free_rate,
    )

    max_sharpe = optimizer.max_sharpe_portfolio()
    min_vol = optimizer.min_variance_portfolio()
    eq_weight = optimizer.equal_weight_portfolio()
    inv_vol = optimizer.inverse_volatility_portfolio()

    print("\n" + "=" * 55)
    print(" OPTIMAL PORTFOLIO WEIGHT ALLOCATIONS ")
    print("=" * 55)
    w_df = pd.DataFrame({
        "Max Sharpe": max_sharpe.weights,
        "Min Volatility": min_vol.weights,
        "Inverse Vol": inv_vol.weights,
        "Equal Weight": eq_weight.weights,
    })
    for ticker, row in w_df.iterrows():
        print(f"  {ticker:<8} | Max Sharpe: {row['Max Sharpe']:>6.1%} | Min Vol: {row['Min Volatility']:>6.1%} | Inv Vol: {row['Inverse Vol']:>6.1%}")
    print("-" * 55)
    print(f"  Max Sharpe Exp Return : {format_percent(max_sharpe.expected_return)} (Vol: {format_percent(max_sharpe.volatility)}, Sharpe: {max_sharpe.sharpe_ratio:.2f})")
    print(f"  Min Vol Exp Return    : {format_percent(min_vol.expected_return)} (Vol: {format_percent(min_vol.volatility)}, Sharpe: {min_vol.sharpe_ratio:.2f})")

    # 3. Backtesting
    print("\n[3/5] Backtesting Max Sharpe Strategy vs S&P 500 Benchmark...")
    backtester = PortfolioBacktester(
        asset_returns=engine.returns,
        benchmark_returns=engine.benchmark_returns,
        risk_free_rate=risk_free_rate,
        initial_capital=initial_capital,
    )

    metrics_max_sharpe, ts_sharpe = backtester.run_backtest(
        weights=max_sharpe.weights,
        rebalance_freq=rebalance_freq,
    )
    metrics_bench, ts_bench = backtester.run_backtest(
        weights={t: 1.0 / len(tickers) for t in tickers},
        rebalance_freq=rebalance_freq,
    )

    report_str = print_metrics_table(metrics_max_sharpe, title="Max Sharpe Portfolio Backtest vs SPY")
    print(report_str)

    # 4. Execution Simulator & TCA
    print("\n[4/5] Simulating Order Execution & Microstructure Slippage...")
    exec_sim = ExecutionSimulator()
    latest_prices = engine.prices.iloc[-1].to_dict()
    orders = exec_sim.rebalance_portfolio_orders(
        current_holdings={},
        target_weights=max_sharpe.weights,
        current_prices=latest_prices,
        total_portfolio_value=initial_capital,
    )
    print(f"      Generated & filled {len(orders)} initial allocation orders.")
    for o in orders:
        print(f"      [{o.order_id}] {o.side:<4} {o.quantity:>5} shs {o.symbol:<5} @ ${o.filled_price:<7.2f} (Slippage: ${o.slippage_cost:.2f}, Fee: ${o.fee:.2f})")

    # 5. High-Frequency Tick Generation
    if simulate_ticks:
        print("\n[5/5] Emulating Real-time Merton Jump Diffusion Tick Stream for NVDA...")
        tick_gen = TickStreamGenerator(symbol="NVDA", initial_price=latest_prices.get("NVDA", 140.0))
        ticks = tick_gen.generate_tick_stream(n_ticks=10)
        print("      Sample Live Order Book & Trade Ticks:")
        for _, t in ticks.head(5).iterrows():
            print(f"      {t['timestamp']} | {t['symbol']} | {t['event']:<5} {t['size']:>4} @ ${t['price']:.2f} [Bid: ${t['bid']:.2f} x Ask: ${t['ask']:.2f}]")

    # Exporting
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        summary_path = os.path.join(export_dir, "portfolio_summary.json")
        summary_data = {
            "tickers": tickers,
            "max_sharpe_weights": max_sharpe.weights,
            "min_vol_weights": min_vol.weights,
            "metrics": vars(metrics_max_sharpe),
            "generated_at": datetime.datetime.now().isoformat(),
        }
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        ts_sharpe.to_csv(os.path.join(export_dir, "portfolio_timeseries.csv"))
        print(f"\n[OK] Performance artifacts exported to: {export_dir}/")

    print("\n" + "=" * 70)
    print(" EXECUTION COMPLETE. Launch Streamlit UI with: streamlit run app.py")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Quantitative Portfolio Optimizer & Algorithmic Trading Suite")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "BAC"], help="Portfolio tickers")
    parser.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.date.today().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--rf", type=float, default=0.045, help="Annualized risk-free rate")
    parser.add_argument("--rebalance", default="monthly", choices=["daily", "weekly", "monthly", "quarterly", "static"], help="Rebalance frequency")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital in USD")
    parser.add_argument("--export-dir", default="output", help="Directory to export results")
    parser.add_argument("--no-ticks", action="store_true", help="Disable tick simulation")

    args = parser.parse_args()
    run_pipeline(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        risk_free_rate=args.rf,
        rebalance_freq=args.rebalance,
        initial_capital=args.capital,
        export_dir=args.export_dir,
        simulate_ticks=not args.no_ticks,
    )


if __name__ == "__main__":
    main()
