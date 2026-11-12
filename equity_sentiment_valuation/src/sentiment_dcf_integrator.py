"""
Sentiment-DCF Integrator Engine.
Dynamically bridges NLP financial sentiment polarity into fundamental valuation parameters
(Revenue Growth, Operating Margins, Cost of Capital Risk Premiums, and Price Targets).
"""
import copy
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

from .sentiment_analyzer import SentimentScore, FinBERTSentimentAnalyzer
from .dcf_model import DCFInputs, DCFValuationEngine, DCFValuationResult


@dataclass
class ValuationSummary:
    """Consolidated institutional equity research output."""
    ticker: str
    sentiment_score: SentimentScore
    base_dcf: DCFValuationResult
    sentiment_adjusted_dcf: DCFValuationResult
    bull_case_dcf: DCFValuationResult
    bear_case_dcf: DCFValuationResult
    price_delta_sentiment: float
    executive_summary: str


class SentimentDCFIntegrator:
    """
    Orchestrates end-to-end sentiment-adjusted equity valuation.
    Adjusts growth rates and risk premiums based on FinBERT confidence and scores.
    """

    def __init__(
        self,
        growth_sensitivity: float = 0.25,  # +/- 25% growth modulation on max sentiment (+1.0)
        margin_sensitivity: float = 0.10,  # +/- 10% operating margin modulation
        risk_sensitivity: float = 0.15,    # +/- 15% ERP risk premium adjustment
    ):
        self.growth_sensitivity = growth_sensitivity
        self.margin_sensitivity = margin_sensitivity
        self.risk_sensitivity = risk_sensitivity

    def adjust_dcf_inputs(
        self,
        base_inputs: DCFInputs,
        sentiment: SentimentScore,
    ) -> DCFInputs:
        """
        Applies sentiment multipliers to DCF input parameters.
        S in [-1.0, 1.0]:
        - Positive sentiment increases growth & margins, lowers ERP.
        - Negative sentiment suppresses growth & margins, elevates ERP.
        """
        adj = copy.deepcopy(base_inputs)
        s = sentiment.compound_score
        conf = sentiment.confidence

        # Effective sentiment factor scaled by confidence
        factor = s * conf

        # 1. Adjust revenue growth rates
        adj.revenue_growth_rates = [
            round(g * (1.0 + self.growth_sensitivity * factor), 4)
            for g in base_inputs.revenue_growth_rates
        ]

        # 2. Adjust operating margin
        adj.operating_margin = round(
            base_inputs.operating_margin * (1.0 + self.margin_sensitivity * factor), 4
        )

        # 3. Adjust Equity Risk Premium (ERP)
        # Higher positive sentiment -> lower risk premium -> lower WACC -> higher valuation
        adj.market_risk_premium = round(
            base_inputs.market_risk_premium * (1.0 - self.risk_sensitivity * factor), 4
        )

        return adj

    def evaluate_all_scenarios(
        self,
        base_inputs: DCFInputs,
        sentiment: SentimentScore,
    ) -> ValuationSummary:
        """
        Runs Base, Sentiment-Adjusted, Bull, and Bear DCF valuations.
        """
        # 1. Base Case
        base_engine = DCFValuationEngine(base_inputs)
        base_res = base_engine.run_valuation()

        # 2. Sentiment-Adjusted Case
        adj_inputs = self.adjust_dcf_inputs(base_inputs, sentiment)
        adj_engine = DCFValuationEngine(adj_inputs)
        adj_res = adj_engine.run_valuation()

        # 3. Bull Case (+20% Growth, +200bps Margin, -50bps ERP)
        bull_inputs = copy.deepcopy(base_inputs)
        bull_inputs.revenue_growth_rates = [g * 1.20 for g in bull_inputs.revenue_growth_rates]
        bull_inputs.operating_margin = min(0.60, bull_inputs.operating_margin + 0.02)
        bull_inputs.market_risk_premium = max(0.035, bull_inputs.market_risk_premium - 0.005)
        bull_engine = DCFValuationEngine(bull_inputs)
        bull_res = bull_engine.run_valuation()

        # 4. Bear Case (-20% Growth, -200bps Margin, +50bps ERP)
        bear_inputs = copy.deepcopy(base_inputs)
        bear_inputs.revenue_growth_rates = [g * 0.80 for g in bear_inputs.revenue_growth_rates]
        bear_inputs.operating_margin = max(0.05, bear_inputs.operating_margin - 0.02)
        bear_inputs.market_risk_premium = bull_inputs.market_risk_premium + 0.010
        bear_engine = DCFValuationEngine(bear_inputs)
        bear_res = bear_engine.run_valuation()

        delta_px = adj_res.implied_share_price - base_res.implied_share_price

        # Narrative generation
        summary_text = (
            f"Fundamental analysis for {base_inputs.ticker} yields a Base DCF target of "
            f"${base_res.implied_share_price:.2f} (WACC: {base_res.wacc:.2%}). "
            f"NLP sentiment analysis detects {sentiment.label.upper()} tone "
            f"(Score: {sentiment.compound_score:+.2f}, Conf: {sentiment.confidence:.1%}). "
            f"Incorporating sentiment-driven growth adjustments yields an intrinsic price target of "
            f"${adj_res.implied_share_price:.2f} (a {delta_px:+.2f} per share sentiment impact), "
            f"representing a Margin of Safety of {adj_res.margin_of_safety:+.1%} relative to current market price "
            f"(${base_inputs.current_share_price:.2f}). Recommendation: {adj_res.recommendation}."
        )

        return ValuationSummary(
            ticker=base_inputs.ticker,
            sentiment_score=sentiment,
            base_dcf=base_res,
            sentiment_adjusted_dcf=adj_res,
            bull_case_dcf=bull_res,
            bear_case_dcf=bear_res,
            price_delta_sentiment=round(delta_px, 2),
            executive_summary=summary_text,
        )
