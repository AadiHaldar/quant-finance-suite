"""
Unit tests for DCFValuationEngine and SentimentDCFIntegrator.
Verifies WACC formulas, Gordon Growth constraints, FCFF discounting, and price sensitivity.
"""
import pytest
import numpy as np
import pandas as pd
from src.dcf_model import DCFInputs, DCFValuationEngine
from src.sentiment_analyzer import SentimentScore
from src.sentiment_dcf_integrator import SentimentDCFIntegrator


@pytest.fixture
def sample_dcf_inputs():
    return DCFInputs(
        ticker="TEST",
        current_share_price=100.0,
        shares_outstanding=1000.0,
        revenue_base=10000.0,
        revenue_growth_rates=[0.10, 0.08, 0.06, 0.05, 0.04],
        operating_margin=0.25,
        tax_rate=0.20,
        d_and_a_rate=0.03,
        capex_rate=0.03,
        nwc_rate=0.02,
        risk_free_rate=0.04,
        market_risk_premium=0.06,
        beta=1.20,
        cost_of_debt=0.05,
        total_debt=20000.0,
        cash_and_equivalents=5000.0,
        terminal_growth_rate=0.025,
    )


def test_wacc_calculation(sample_dcf_inputs):
    engine = DCFValuationEngine(sample_dcf_inputs)
    wacc, ke, kd_tax, we, wd = engine.calculate_wacc()
    
    expected_ke = 0.04 + (1.20 * 0.06)  # 11.2%
    expected_kd_tax = 0.05 * (1.0 - 0.20)  # 4.0%
    
    assert pytest.approx(ke, rel=1e-3) == expected_ke
    assert pytest.approx(kd_tax, rel=1e-3) == expected_kd_tax
    assert pytest.approx(we + wd, rel=1e-3) == 1.0
    assert wacc > kd_tax
    assert wacc < ke


def test_dcf_positive_valuation(sample_dcf_inputs):
    engine = DCFValuationEngine(sample_dcf_inputs)
    res = engine.run_valuation()
    
    assert res.enterprise_value > 0
    assert res.equity_value > 0
    assert res.implied_share_price > 0
    expected_price = res.equity_value / sample_dcf_inputs.shares_outstanding
    assert pytest.approx(res.implied_share_price, rel=1e-2) == expected_price


def test_gordon_growth_constraint(sample_dcf_inputs):
    # Edge case: set high terminal growth near WACC
    sample_dcf_inputs.terminal_growth_rate = 0.15  # Greater than WACC
    engine = DCFValuationEngine(sample_dcf_inputs)
    res = engine.run_valuation()
    
    assert res.terminal_value > 0
    assert not np.isnan(res.implied_share_price)


def test_sentiment_growth_adjustment(sample_dcf_inputs):
    integrator = SentimentDCFIntegrator()
    
    pos_sentiment = SentimentScore("Positive", 0.8, 0.1, 0.1, compound_score=0.7, confidence=0.85)
    neg_sentiment = SentimentScore("Negative", 0.1, 0.8, 0.1, compound_score=-0.7, confidence=0.85)
    
    summary_pos = integrator.evaluate_all_scenarios(sample_dcf_inputs, pos_sentiment)
    summary_neg = integrator.evaluate_all_scenarios(sample_dcf_inputs, neg_sentiment)
    
    base_price = summary_pos.base_dcf.implied_share_price
    pos_price = summary_pos.sentiment_adjusted_dcf.implied_share_price
    neg_price = summary_neg.sentiment_adjusted_dcf.implied_share_price
    
    assert pos_price > base_price
    assert neg_price < base_price
    assert summary_pos.bull_case_dcf.implied_share_price > summary_pos.bear_case_dcf.implied_share_price


def test_sensitivity_table_monotonicity(sample_dcf_inputs):
    engine = DCFValuationEngine(sample_dcf_inputs)
    res = engine.run_valuation()
    
    sens_df = res.sensitivity_table
    assert sens_df is not None
    # Higher WACC down the rows should decrease valuation
    for col in sens_df.columns:
        col_vals = sens_df[col].dropna().values
        assert (np.diff(col_vals) <= 0).all()
