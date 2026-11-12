"""
Discounted Cash Flow (DCF) Equity Valuation Engine.
Computes WACC, multi-stage Free Cash Flows (FCFF), Terminal Value (Gordon Growth & Exit Multiple),
Implied Intrinsic Share Price, and 2D Sensitivity Matrices.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class DCFInputs:
    """Standardized fundamental and macroeconomic inputs for DCF modeling."""
    ticker: str = "AAPL"
    current_share_price: float = 220.0
    shares_outstanding: float = 15300.0  # in millions
    revenue_base: float = 385000.0       # in millions (e.g. FY revenue)
    revenue_growth_rates: List[float] = field(default_factory=lambda: [0.08, 0.07, 0.06, 0.05, 0.04])
    operating_margin: float = 0.30       # EBIT / Revenue
    tax_rate: float = 0.21               # Corporate tax rate
    d_and_a_rate: float = 0.035          # D&A as % of Revenue
    capex_rate: float = 0.030            # CapEx as % of Revenue
    nwc_rate: float = 0.020              # ΔNWC as % of ΔRevenue
    risk_free_rate: float = 0.045        # 10-Yr US Treasury Yield
    market_risk_premium: float = 0.055   # Equity Risk Premium (ERP)
    beta: float = 1.15                   # Systematic risk
    cost_of_debt: float = 0.048          # Pre-tax cost of debt
    total_debt: float = 105000.0         # in millions
    cash_and_equivalents: float = 65000.0  # in millions
    terminal_growth_rate: float = 0.025  # Long-term GDP perpetual growth g
    exit_multiple_ev_ebitda: float = 18.0


@dataclass
class DCFValuationResult:
    """Complete institutional DCF output with valuation metrics."""
    ticker: str
    cost_of_equity: float
    after_tax_cost_of_debt: float
    weight_of_equity: float
    weight_of_debt: float
    wacc: float
    projected_years: List[int]
    projected_revenues: List[float]
    projected_ebit: List[float]
    projected_fcf: List[float]
    discounted_fcf: List[float]
    sum_pv_fcf: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    equity_value: float
    implied_share_price: float
    current_share_price: float
    margin_of_safety: float  # (Implied - Current) / Current
    recommendation: str     # 'STRONG BUY', 'BUY', 'HOLD', 'SELL'
    sensitivity_table: Optional[pd.DataFrame] = None


class DCFValuationEngine:
    """
    Automated DCF model calculating enterprise & equity value with WACC discounting.
    """

    def __init__(self, inputs: Optional[DCFInputs] = None):
        self.inputs = inputs or DCFInputs()

    def calculate_wacc(self) -> Tuple[float, float, float, float, float]:
        """
        Calculates Weighted Average Cost of Capital (WACC):
        WACC = (E/V * Ke) + (D/V * Kd * (1 - t))
        Where Ke = Rf + Beta * ERP (CAPM)
        """
        inp = self.inputs
        cost_of_equity = inp.risk_free_rate + (inp.beta * inp.market_risk_premium)
        after_tax_debt = inp.cost_of_debt * (1.0 - inp.tax_rate)

        market_equity_val = inp.current_share_price * inp.shares_outstanding
        total_val = market_equity_val + inp.total_debt

        if total_val <= 0:
            w_e = 1.0
            w_d = 0.0
        else:
            w_e = market_equity_val / total_val
            w_d = inp.total_debt / total_val

        wacc = (w_e * cost_of_equity) + (w_d * after_tax_debt)
        return float(wacc), float(cost_of_equity), float(after_tax_debt), float(w_e), float(w_d)

    def run_valuation(self) -> DCFValuationResult:
        """
        Projects Free Cash Flows to Firm (FCFF), discounts to present, and calculates target price.
        """
        inp = self.inputs
        wacc, cost_of_equity, after_tax_debt, w_e, w_d = self.calculate_wacc()

        n_years = len(inp.revenue_growth_rates)
        projected_revenues = []
        projected_ebit = []
        projected_fcf = []
        discounted_fcf = []
        years = list(range(1, n_years + 1))

        current_rev = inp.revenue_base
        prev_rev = current_rev

        for t, g in enumerate(inp.revenue_growth_rates, 1):
            rev_t = current_rev * (1.0 + g)
            ebit_t = rev_t * inp.operating_margin
            nopat_t = ebit_t * (1.0 - inp.tax_rate)
            da_t = rev_t * inp.d_and_a_rate
            capex_t = rev_t * inp.capex_rate
            delta_nwc_t = (rev_t - prev_rev) * inp.nwc_rate

            # FCFF = NOPAT + D&A - CapEx - ΔNWC
            fcff_t = nopat_t + da_t - capex_t - delta_nwc_t
            pv_fcff_t = fcff_t / ((1.0 + wacc) ** t)

            projected_revenues.append(rev_t)
            projected_ebit.append(ebit_t)
            projected_fcf.append(fcff_t)
            discounted_fcf.append(pv_fcff_t)

            prev_rev = rev_t
            current_rev = rev_t

        sum_pv_fcf = float(np.sum(discounted_fcf))

        # Terminal Value via Gordon Growth Model: TV = FCFF_n * (1 + g) / (WACC - g)
        g_term = inp.terminal_growth_rate
        if wacc <= g_term:
            g_term = wacc - 0.005  # Prevent mathematical singularity

        last_fcf = projected_fcf[-1]
        terminal_value = (last_fcf * (1.0 + g_term)) / (wacc - g_term)
        pv_terminal_value = terminal_value / ((1.0 + wacc) ** n_years)

        # Enterprise Value = Sum(PV of FCFF) + PV of Terminal Value
        enterprise_value = sum_pv_fcf + pv_terminal_value

        # Equity Value = Enterprise Value + Cash - Debt
        equity_value = enterprise_value + inp.cash_and_equivalents - inp.total_debt
        implied_share_price = equity_value / max(inp.shares_outstanding, 1.0)

        margin_of_safety = (implied_share_price - inp.current_share_price) / inp.current_share_price

        # Investment Recommendation
        if margin_of_safety >= 0.20:
            rec = "STRONG BUY"
        elif margin_of_safety >= 0.05:
            rec = "BUY"
        elif margin_of_safety >= -0.10:
            rec = "HOLD"
        else:
            rec = "SELL"

        # Generate 2D Sensitivity Table (WACC vs Terminal Growth)
        sensitivity_df = self._generate_sensitivity_table(sum_pv_fcf, last_fcf, n_years, wacc, g_term)

        return DCFValuationResult(
            ticker=inp.ticker,
            cost_of_equity=round(cost_of_equity, 4),
            after_tax_cost_of_debt=round(after_tax_debt, 4),
            weight_of_equity=round(w_e, 4),
            weight_of_debt=round(w_d, 4),
            wacc=round(wacc, 4),
            projected_years=years,
            projected_revenues=[round(r, 2) for r in projected_revenues],
            projected_ebit=[round(e, 2) for e in projected_ebit],
            projected_fcf=[round(f, 2) for f in projected_fcf],
            discounted_fcf=[round(d, 2) for d in discounted_fcf],
            sum_pv_fcf=round(sum_pv_fcf, 2),
            terminal_value=round(terminal_value, 2),
            pv_terminal_value=round(pv_terminal_value, 2),
            enterprise_value=round(enterprise_value, 2),
            equity_value=round(equity_value, 2),
            implied_share_price=round(implied_share_price, 2),
            current_share_price=round(inp.current_share_price, 2),
            margin_of_safety=round(margin_of_safety, 4),
            recommendation=rec,
            sensitivity_table=sensitivity_df,
        )

    def _generate_sensitivity_table(
        self,
        sum_pv_fcf: float,
        last_fcf: float,
        n_years: int,
        base_wacc: float,
        base_g: float,
    ) -> pd.DataFrame:
        """
        Creates a 2D matrix of implied share prices across varying WACC and Terminal Growth rates.
        """
        inp = self.inputs
        wacc_steps = [base_wacc - 0.015, base_wacc - 0.0075, base_wacc, base_wacc + 0.0075, base_wacc + 0.015]
        g_steps = [base_g - 0.010, base_g - 0.005, base_g, base_g + 0.005, base_g + 0.010]

        matrix = {}
        for g in g_steps:
            col_label = f"g = {g:.1%}"
            matrix[col_label] = []
            for w in wacc_steps:
                if w <= g:
                    matrix[col_label].append(np.nan)
                    continue
                tv = (last_fcf * (1.0 + g)) / (w - g)
                pv_tv = tv / ((1.0 + w) ** n_years)
                ev = sum_pv_fcf + pv_tv
                eq_val = ev + inp.cash_and_equivalents - inp.total_debt
                px = eq_val / inp.shares_outstanding
                matrix[col_label].append(round(px, 2))

        index_labels = [f"WACC = {w:.2%}" for w in wacc_steps]
        return pd.DataFrame(matrix, index=index_labels)
