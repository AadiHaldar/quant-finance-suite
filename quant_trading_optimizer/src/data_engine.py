"""
Market Data Ingestion & Synthetic High-Frequency Tick Generation Engine.
Supports yfinance historical fetching, offline synthetic data fallback,
and Merton Jump Diffusion HFT tick simulations.
"""
from typing import Dict, List, Optional, Tuple
import datetime
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


class MarketDataEngine:
    """
    Engine to ingest market prices for portfolio optimization & backtesting.
    Includes offline fallback to guarantee reliability under rate-limits.
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        start_date: str = "2021-01-01",
        end_date: Optional[str] = None,
        risk_free_rate: float = 0.045,
    ):
        self.tickers = tickers or ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "BAC"]
        self.start_date = start_date
        self.end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")
        self.risk_free_rate = risk_free_rate
        self.prices: pd.DataFrame = pd.DataFrame()
        self.returns: pd.DataFrame = pd.DataFrame()
        self.benchmark_prices: pd.Series = pd.Series(dtype=float)
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)

    def fetch_data(self, benchmark: str = "SPY") -> pd.DataFrame:
        """
        Fetches adjusted close prices for portfolio tickers and benchmark.
        Falls back to realistic synthetic data generator if offline or rate-limited.
        """
        all_symbols = list(set(self.tickers + [benchmark]))
        prices_df = pd.DataFrame()

        if yf is not None:
            try:
                raw = yf.download(
                    tickers=all_symbols,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    if "Close" in raw.columns.levels[0]:
                        prices_df = raw["Close"]
                    else:
                        prices_df = raw.xs(raw.columns.levels[0][0], axis=1, level=0)
                elif "Close" in raw.columns:
                    prices_df = raw[["Close"]]
                else:
                    prices_df = raw
            except Exception as e:
                print(f"[Warning] yfinance download failed: {e}. Falling back to synthetic market data.")
                prices_df = pd.DataFrame()

        # Check if enough valid columns were downloaded
        valid_cols = [c for c in all_symbols if c in prices_df.columns and not prices_df[c].dropna().empty]
        if len(valid_cols) < len(self.tickers):
            prices_df = self._generate_synthetic_historical(all_symbols)

        prices_df = prices_df.ffill().bfill().dropna(axis=0)

        # Separate portfolio assets and benchmark
        available_tickers = [t for t in self.tickers if t in prices_df.columns]
        self.prices = prices_df[available_tickers]
        self.returns = self.prices.pct_change().dropna()

        if benchmark in prices_df.columns:
            self.benchmark_prices = prices_df[benchmark]
            self.benchmark_returns = self.benchmark_prices.pct_change().dropna()
        else:
            self.benchmark_prices = self.prices.mean(axis=1)
            self.benchmark_returns = self.benchmark_prices.pct_change().dropna()

        return self.prices

    def _generate_synthetic_historical(self, symbols: List[str]) -> pd.DataFrame:
        """
        Generates realistic multi-asset daily price paths using correlated geometric Brownian motion.
        """
        np.random.seed(42)
        date_index = pd.date_range(start=self.start_date, end=self.end_date, freq="B")
        n_days = len(date_index)
        n_assets = len(symbols)

        # Baseline parameters per asset class
        base_annual_returns = {
            "AAPL": 0.22, "MSFT": 0.20, "GOOGL": 0.18, "AMZN": 0.19,
            "NVDA": 0.35, "JPM": 0.14, "BAC": 0.12, "SPY": 0.15,
        }
        base_annual_vols = {
            "AAPL": 0.26, "MSFT": 0.24, "GOOGL": 0.27, "AMZN": 0.29,
            "NVDA": 0.42, "JPM": 0.22, "BAC": 0.25, "SPY": 0.16,
        }
        base_start_prices = {
            "AAPL": 130.0, "MSFT": 220.0, "GOOGL": 88.0, "AMZN": 160.0,
            "NVDA": 135.0, "JPM": 125.0, "BAC": 30.0, "SPY": 375.0,
        }

        dt = 1.0 / 252.0
        # Correlation matrix construction
        corr = np.full((n_assets, n_assets), 0.45)
        np.fill_diagonal(corr, 1.0)
        # Tech cluster higher correlation
        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                if s1 in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"] and s2 in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]:
                    if i != j:
                        corr[i, j] = 0.68
                elif s1 in ["JPM", "BAC"] and s2 in ["JPM", "BAC"]:
                    if i != j:
                        corr[i, j] = 0.80

        L = np.linalg.cholesky(corr)
        std_normals = np.random.normal(0, 1, size=(n_days, n_assets))
        correlated_shocks = std_normals @ L.T

        price_matrix = np.zeros((n_days, n_assets))
        for j, sym in enumerate(symbols):
            mu = base_annual_returns.get(sym, 0.15)
            sigma = base_annual_vols.get(sym, 0.25)
            s0 = base_start_prices.get(sym, 100.0)

            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * correlated_shocks[:, j]
            log_returns = drift + diffusion
            log_returns[0] = 0.0
            price_matrix[:, j] = s0 * np.exp(np.cumsum(log_returns))

        return pd.DataFrame(price_matrix, index=date_index, columns=symbols)

    def get_annualized_stats(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Returns annualized expected returns (vector mu) and annualized covariance matrix (Sigma).
        """
        if self.returns.empty:
            self.fetch_data()
        mu = self.returns.mean() * 252
        sigma = self.returns.cov() * 252
        return mu, sigma


class TickStreamGenerator:
    """
    Simulates high-frequency tick streams using Merton Jump Diffusion:
    dS_t = mu*S_t*dt + sigma*S_t*dW_t + J_t*S_t*dN_t
    Generates realistic Level 1/2 microsecond quotes, bid-ask spreads, and volume.
    """

    def __init__(
        self,
        symbol: str = "NVDA",
        initial_price: float = 140.0,
        drift: float = 0.05,
        volatility: float = 0.35,
        jump_intensity: float = 2.0,  # lambda jumps per year
        jump_mean: float = -0.01,
        jump_std: float = 0.03,
        spread_bps: float = 2.5,
    ):
        self.symbol = symbol
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.spread_bps = spread_bps

    def generate_tick_stream(
        self,
        n_ticks: int = 500,
        interval_ms: int = 100,
        start_time: Optional[datetime.datetime] = None,
    ) -> pd.DataFrame:
        """
        Generates simulated real-time tick messages.
        """
        if start_time is None:
            start_time = datetime.datetime.now()

        # Time delta in years per tick
        dt_years = (interval_ms / 1000.0) / (252 * 6.5 * 3600)  # 6.5 trading hours/day

        ticks = []
        current_price = self.initial_price
        k = np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1

        for i in range(n_ticks):
            curr_ts = start_time + datetime.timedelta(milliseconds=i * interval_ms)

            # Continuous Brownian diffusion
            dW = np.random.normal(0, np.sqrt(dt_years))
            drift_term = (self.drift - 0.5 * self.volatility**2 - self.jump_intensity * k) * dt_years
            diff_term = self.volatility * dW

            # Poisson Jump occurrence
            n_jumps = np.random.poisson(self.jump_intensity * dt_years)
            jump_term = 0.0
            if n_jumps > 0:
                jump_magnitudes = np.random.normal(self.jump_mean, self.jump_std, size=n_jumps)
                jump_term = np.sum(jump_magnitudes)

            log_return = drift_term + diff_term + jump_term
            current_price = float(current_price * np.exp(log_return))
            current_price = max(0.01, round(current_price, 4))

            # Bid / Ask spread simulation
            half_spread = current_price * (self.spread_bps / 10000.0) / 2.0
            bid_price = round(current_price - half_spread, 2)
            ask_price = round(current_price + half_spread, 2)
            if ask_price <= bid_price:
                ask_price = bid_price + 0.01

            # Microstructure trade simulation
            trade_side = np.random.choice(["BUY", "SELL", "QUOTE"], p=[0.35, 0.35, 0.30])
            trade_size = int(np.random.lognormal(mean=4.0, sigma=1.0) * 10)
            trade_size = max(10, min(trade_size, 5000))

            if trade_side == "BUY":
                trade_price = ask_price
            elif trade_side == "SELL":
                trade_price = bid_price
            else:
                trade_price = round((bid_price + ask_price) / 2.0, 2)

            ticks.append({
                "timestamp": curr_ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "symbol": self.symbol,
                "event": trade_side,
                "price": trade_price,
                "bid": bid_price,
                "ask": ask_price,
                "size": trade_size,
                "spread_cents": round((ask_price - bid_price) * 100, 2),
            })

        return pd.DataFrame(ticks)
