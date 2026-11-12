"""
High-Frequency Order Execution Simulator & Transaction Cost Analysis (TCA).
Models linear market impact, bid-ask spread crossing, and transaction slippage.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str = "MARKET"  # 'MARKET', 'LIMIT'
    limit_price: Optional[float] = None
    status: str = "PENDING"  # 'PENDING', 'FILLED', 'REJECTED'
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    filled_price: float = 0.0
    filled_qty: int = 0
    slippage_cost: float = 0.0
    fee: float = 0.0


@dataclass
class TradeLog:
    order_id: str
    timestamp: str
    symbol: str
    side: str
    quantity: int
    benchmark_price: float
    executed_price: float
    slippage_bps: float
    total_cost: float
    implementation_shortfall: float


class ExecutionSimulator:
    """
    Simulates high-speed algorithmic execution with realistic microstructure slippage.
    """

    def __init__(
        self,
        base_fee_per_share: float = 0.005,  # $0.005 / share broker fee
        fixed_slippage_bps: float = 1.5,
        impact_factor_gamma: float = 0.1,    # Market impact coefficient
        daily_volume_estimate: int = 25000000,
    ):
        self.base_fee_per_share = base_fee_per_share
        self.fixed_slippage_bps = fixed_slippage_bps
        self.gamma = impact_factor_gamma
        self.adv = daily_volume_estimate
        self.order_history: List[Order] = []
        self.trade_logs: List[TradeLog] = []

    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        market_mid_price: float,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
    ) -> Order:
        """
        Executes a simulated single order with market impact and slippage.
        """
        order_id = f"ORD-{len(self.order_history) + 1:05d}"
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            order_type=order_type.upper(),
            limit_price=limit_price,
        )

        if quantity <= 0 or market_mid_price <= 0:
            order.status = "REJECTED"
            self.order_history.append(order)
            return order

        # Reference prices
        half_spread_bps = 2.0
        if bid_price is not None and ask_price is not None and ask_price > bid_price:
            half_spread_pct = ((ask_price - bid_price) / market_mid_price) / 2.0
        else:
            half_spread_pct = (half_spread_bps / 10000.0)

        # Market impact model: Fixed spread + Temporary linear impact
        participation_rate = quantity / max(self.adv, 100000)
        impact_pct = (self.fixed_slippage_bps / 10000.0) + (self.gamma * np.sqrt(participation_rate))

        if side.upper() == "BUY":
            effective_price = market_mid_price * (1.0 + half_spread_pct + impact_pct)
            if order_type.upper() == "LIMIT" and limit_price is not None and effective_price > limit_price:
                order.status = "REJECTED"
                self.order_history.append(order)
                return order
        else:  # SELL
            effective_price = market_mid_price * (1.0 - half_spread_pct - impact_pct)
            if order_type.upper() == "LIMIT" and limit_price is not None and effective_price < limit_price:
                order.status = "REJECTED"
                self.order_history.append(order)
                return order

        effective_price = max(0.01, round(effective_price, 4))
        slippage_cost = abs(effective_price - market_mid_price) * quantity
        fee = round(quantity * self.base_fee_per_share, 4)

        order.status = "FILLED"
        order.filled_price = effective_price
        order.filled_qty = quantity
        order.slippage_cost = round(slippage_cost, 4)
        order.fee = fee
        self.order_history.append(order)

        slippage_bps = abs(effective_price - market_mid_price) / market_mid_price * 10000.0
        total_cost = (effective_price * quantity) + fee

        trade_log = TradeLog(
            order_id=order_id,
            timestamp=order.timestamp,
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            benchmark_price=market_mid_price,
            executed_price=effective_price,
            slippage_bps=round(slippage_bps, 2),
            total_cost=round(total_cost, 2),
            implementation_shortfall=round(slippage_cost + fee, 4),
        )
        self.trade_logs.append(trade_log)

        return order

    def rebalance_portfolio_orders(
        self,
        current_holdings: Dict[str, int],
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        total_portfolio_value: float,
    ) -> List[Order]:
        """
        Computes rebalance orders to match target weights and simulates execution.
        """
        executed_orders = []
        for symbol, target_w in target_weights.items():
            price = current_prices.get(symbol, 0.0)
            if price <= 0:
                continue

            target_val = total_portfolio_value * target_w
            target_shares = int(target_val / price)
            curr_shares = current_holdings.get(symbol, 0)
            diff_shares = target_shares - curr_shares

            if diff_shares > 0:
                order = self.execute_order(symbol, "BUY", diff_shares, price)
                executed_orders.append(order)
            elif diff_shares < 0:
                order = self.execute_order(symbol, "SELL", abs(diff_shares), price)
                executed_orders.append(order)

        return executed_orders

    def get_trade_logs_df(self) -> pd.DataFrame:
        """Returns trade log dataframe for TCA reporting."""
        if not self.trade_logs:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self.trade_logs])
