"""
Equity Research Report Generator & Visualizer.
Generates institutional markdown research memos, JSON exports,
and Plotly valuation breakdown charts.
"""
from datetime import datetime
import json
import os
from typing import Dict, Optional
import pandas as pd
import numpy as np

from .sentiment_dcf_integrator import ValuationSummary


class EquityResearchReportGenerator:
    """
    Produces publication-grade equity research memos and interactive charts.
    """

    @staticmethod
    def generate_markdown_memo(summary: ValuationSummary) -> str:
        """Constructs an executive equity research memo in GitHub-flavored Markdown."""
        base = summary.base_dcf
        adj = summary.sentiment_adjusted_dcf
        bull = summary.bull_case_dcf
        bear = summary.bear_case_dcf
        s = summary.sentiment_score

        memo = f"""# EQUITY RESEARCH VALUATION MEMO: {summary.ticker}
**Date:** {datetime.now().strftime('%B %d, %Y')} | **Analyst:** Antigravity Quant Suite | **Rating:** **[{adj.recommendation}]**

---

## Executive Investment Summary

> {summary.executive_summary}

---

## 1. Intrinsic Price Targets & Scenario Analysis

| Scenario | Implied Target Price | Implied Upside / Downside | WACC | Terminal Growth ($g$) |
| :--- | :---: | :---: | :---: | :---: |
| **Current Market Price** | **${base.current_share_price:.2f}** | - | - | - |
| **Bear Case (-20% Growth)** | **${bear.implied_share_price:.2f}** | **{bear.margin_of_safety:+.1%}** | {bear.wacc:.2%} | 2.5% |
| **Base Case DCF** | **${base.implied_share_price:.2f}** | **{base.margin_of_safety:+.1%}** | {base.wacc:.2%} | 2.5% |
| **Sentiment-Adjusted DCF** | **${adj.implied_share_price:.2f}** | **{adj.margin_of_safety:+.1%}** | {adj.wacc:.2%} | 2.5% |
| **Bull Case (+20% Growth)** | **${bull.implied_share_price:.2f}** | **{bull.margin_of_safety:+.1%}** | {bull.wacc:.2%} | 2.5% |

* **Sentiment Delta Impact:** {summary.price_delta_sentiment:+.2f} per share based on {s.label} sentiment score ({s.compound_score:+.2f}).

---

## 2. FinBERT Natural Language Sentiment Breakdown

* **Classification Polarity:** **{s.label.upper()}**
* **Confidence Level:** {s.confidence:.1%}
* **Positive Probability:** {s.positive_prob:.1%}
* **Neutral Probability:** {s.neutral_prob:.1%}
* **Negative Probability:** {s.negative_prob:.1%}
* **Compound NLP Score ($S \\in [-1, 1]$):** {s.compound_score:+.3f}

---

## 3. Projected Free Cash Flow Schedule (Base Case)

| Metric ($ Millions) | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Projected Revenue** | {base.projected_revenues[0]:,.0f} | {base.projected_revenues[1]:,.0f} | {base.projected_revenues[2]:,.0f} | {base.projected_revenues[3]:,.0f} | {base.projected_revenues[4]:,.0f} |
| **EBIT (Operating Income)** | {base.projected_ebit[0]:,.0f} | {base.projected_ebit[1]:,.0f} | {base.projected_ebit[2]:,.0f} | {base.projected_ebit[3]:,.0f} | {base.projected_ebit[4]:,.0f} |
| **Free Cash Flow (FCFF)** | {base.projected_fcf[0]:,.0f} | {base.projected_fcf[1]:,.0f} | {base.projected_fcf[2]:,.0f} | {base.projected_fcf[3]:,.0f} | {base.projected_fcf[4]:,.0f} |
| **PV of FCFF (WACC = {base.wacc:.2%})** | {base.discounted_fcf[0]:,.0f} | {base.discounted_fcf[1]:,.0f} | {base.discounted_fcf[2]:,.0f} | {base.discounted_fcf[3]:,.0f} | {base.discounted_fcf[4]:,.0f} |

* **Cumulative PV of 5-Yr FCFF:** ${base.sum_pv_fcf:,.0f}M
* **Terminal Value (Gordon Growth):** ${base.terminal_value:,.0f}M (PV: ${base.pv_terminal_value:,.0f}M)
* **Implied Enterprise Value (EV):** ${base.enterprise_value:,.0f}M
* **Implied Equity Value:** ${base.equity_value:,.0f}M

---

## 4. Cost of Capital (WACC) Formulation

* **Cost of Equity ($K_e = R_f + \\beta \\times ERP$):** {base.cost_of_equity:.2%}
* **After-Tax Cost of Debt ($K_d \\times (1 - t)$):** {base.after_tax_cost_of_debt:.2%}
* **Equity Weight ($E/V$):** {base.weight_of_equity:.1%} | **Debt Weight ($D/V$):** {base.weight_of_debt:.1%}
* **Weighted Average Cost of Capital (WACC):** **{base.wacc:.2%}**
"""
        return memo

    @staticmethod
    def export_summary_json(summary: ValuationSummary, filepath: str):
        """Exports valuation summary to JSON."""
        data = {
            "ticker": summary.ticker,
            "generated_at": datetime.now().isoformat(),
            "sentiment": {
                "label": summary.sentiment_score.label,
                "compound_score": summary.sentiment_score.compound_score,
                "confidence": summary.sentiment_score.confidence,
                "positive_prob": summary.sentiment_score.positive_prob,
                "negative_prob": summary.sentiment_score.negative_prob,
                "neutral_prob": summary.sentiment_score.neutral_prob,
            },
            "scenarios": {
                "current_market_price": summary.base_dcf.current_share_price,
                "base_dcf_price": summary.base_dcf.implied_share_price,
                "sentiment_adjusted_price": summary.sentiment_adjusted_dcf.implied_share_price,
                "bull_case_price": summary.bull_case_dcf.implied_share_price,
                "bear_case_price": summary.bear_case_dcf.implied_share_price,
                "sentiment_delta": summary.price_delta_sentiment,
            },
            "base_dcf_details": {
                "wacc": summary.base_dcf.wacc,
                "enterprise_value": summary.base_dcf.enterprise_value,
                "equity_value": summary.base_dcf.equity_value,
                "margin_of_safety": summary.base_dcf.margin_of_safety,
                "recommendation": summary.base_dcf.recommendation,
            },
            "executive_summary": summary.executive_summary,
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def create_plotly_scenario_chart(summary: ValuationSummary):
        """Builds an interactive Plotly bar chart comparing scenario price targets."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        base = summary.base_dcf
        adj = summary.sentiment_adjusted_dcf
        bull = summary.bull_case_dcf
        bear = summary.bear_case_dcf

        scenarios = ["Current Market", "Bear Case", "Base DCF", "Sentiment-Adjusted", "Bull Case"]
        prices = [
            base.current_share_price,
            bear.implied_share_price,
            base.implied_share_price,
            adj.implied_share_price,
            bull.implied_share_price,
        ]
        colors = ["#7f7f7f", "#d62728", "#1f77b4", "#00cc96", "#2ca02c"]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=scenarios,
                    y=prices,
                    marker_color=colors,
                    text=[f"${p:.2f}" for p in prices],
                    textposition="auto",
                )
            ]
        )

        fig.add_hline(
            y=base.current_share_price,
            line_dash="dash",
            line_color="black",
            annotation_text="Current Market Price",
            annotation_position="bottom right",
        )

        fig.update_layout(
            title=f"<b>{summary.ticker} Valuation Target Price by Scenario</b>",
            xaxis_title="Valuation Scenario",
            yaxis_title="Implied Share Price ($ USD)",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        return fig
