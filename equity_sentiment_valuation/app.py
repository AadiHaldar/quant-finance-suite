"""
Interactive Automated Equity Valuation & Financial Sentiment Workbench.
Built with Streamlit and Plotly for real-time DCF modeling and FinBERT sentiment NLP.
"""
import copy
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure module path is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.document_parser import FinancialDocumentParser
from src.sentiment_analyzer import FinBERTSentimentAnalyzer
from src.dcf_model import DCFInputs, DCFValuationEngine
from src.sentiment_dcf_integrator import SentimentDCFIntegrator
from src.report_generator import EquityResearchReportGenerator
from main import PRESET_EQUITY_DATA

st.set_page_config(
    page_title="Equity Valuation & Financial Sentiment Suite",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💼 Automated Equity Valuation & Financial Sentiment Pipeline")
st.caption("Bank of America Equity Research & Quantitative Valuation Engine")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🏢 Target Equity & Financials")
    ticker_choice = st.selectbox("Select Equity Ticker", ["AAPL", "NVDA", "BAC", "MSFT", "CUSTOM"], index=0)

    if ticker_choice in PRESET_EQUITY_DATA:
        preset = copy.deepcopy(PRESET_EQUITY_DATA[ticker_choice])
    else:
        preset = DCFInputs(ticker="CUSTOM")

    current_price = st.number_input("Current Share Price ($)", min_value=1.0, value=float(preset.current_share_price), step=1.0)
    shares_out = st.number_input("Shares Outstanding (M)", min_value=10.0, value=float(preset.shares_outstanding), step=100.0)
    rev_base = st.number_input("Base Revenue ($M)", min_value=100.0, value=float(preset.revenue_base), step=1000.0)
    op_margin = st.slider("Operating Margin (EBIT %)", min_value=0.05, max_value=0.80, value=float(preset.operating_margin), step=0.01, format="%.2f")
    tax_rate = st.slider("Tax Rate", min_value=0.10, max_value=0.35, value=float(preset.tax_rate), step=0.01, format="%.2f")

    st.subheader("Cost of Capital & Terminal Parameters")
    beta = st.slider("Stock Beta (β)", min_value=0.5, max_value=3.0, value=float(preset.beta), step=0.05)
    rf_rate = st.slider("Risk-Free Rate (Rf)", min_value=0.01, max_value=0.08, value=float(preset.risk_free_rate), step=0.005, format="%.3f")
    erp = st.slider("Equity Risk Premium (ERP)", min_value=0.03, max_value=0.08, value=float(preset.market_risk_premium), step=0.005, format="%.3f")
    term_g = st.slider("Terminal Growth Rate (g)", min_value=0.01, max_value=0.04, value=float(preset.terminal_growth_rate), step=0.0025, format="%.3f")

    st.subheader("Financial Document / 10-K Input")
    parser = FinancialDocumentParser()
    sample_text = parser.load_sample_filing(ticker_choice) or ""
    filing_text = st.text_area("Paste SEC 10-K / MD&A or Earnings Transcript", value=sample_text, height=180)

# Build custom inputs
custom_inputs = DCFInputs(
    ticker=ticker_choice,
    current_share_price=current_price,
    shares_outstanding=shares_out,
    revenue_base=rev_base,
    revenue_growth_rates=preset.revenue_growth_rates,
    operating_margin=op_margin,
    tax_rate=tax_rate,
    d_and_a_rate=preset.d_and_a_rate,
    capex_rate=preset.capex_rate,
    nwc_rate=preset.nwc_rate,
    risk_free_rate=rf_rate,
    market_risk_premium=erp,
    beta=beta,
    cost_of_debt=preset.cost_of_debt,
    total_debt=preset.total_debt,
    cash_and_equivalents=preset.cash_and_equivalents,
    terminal_growth_rate=term_g,
)

# --- NLP SENTIMENT PROCESSING ---
@st.cache_resource(show_spinner=False)
def get_sentiment_engine():
    return FinBERTSentimentAnalyzer(use_transformers=True)

with st.spinner("Analyzing Financial Filing with FinBERT NLP Engine..."):
    sentiment_engine = get_sentiment_engine()
    parsed_doc = parser.parse_sec_10k(filing_text, ticker=ticker_choice)
    section_scores = sentiment_engine.analyze_document_sections(parsed_doc.sections)
    aggregate_sentiment = sentiment_engine.compute_aggregate_score(list(section_scores.values()))

# --- INTEGRATED VALUATION ---
integrator = SentimentDCFIntegrator()
summary = integrator.evaluate_all_scenarios(custom_inputs, aggregate_sentiment)
base_dcf = summary.base_dcf
adj_dcf = summary.sentiment_adjusted_dcf

# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Valuation Summary & Target Prices",
    "🧠 FinBERT Sentiment Diagnostics",
    "📊 Financial DCF Schedule & Sensitivity",
    "📝 Executive Research Memo",
])

# --- TAB 1: VALUATION SUMMARY ---
with tab1:
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        st.metric("Current Market Price", f"${base_dcf.current_share_price:.2f}")
    with col_v2:
        st.metric("Base DCF Fair Value", f"${base_dcf.implied_share_price:.2f}", f"WACC: {base_dcf.wacc:.2%}")
    with col_v3:
        delta_str = f"{summary.price_delta_sentiment:+.2f} ({adj_dcf.margin_of_safety:+.1%})"
        st.metric("Sentiment-Adjusted Target", f"${adj_dcf.implied_share_price:.2f}", delta_str)
    with col_v4:
        rec_color = "normal"
        st.metric("Rating Recommendation", adj_dcf.recommendation, f"Margin of Safety: {adj_dcf.margin_of_safety:+.1%}")

    st.markdown("---")

    col_chart, col_narrative = st.columns([3, 2])
    with col_chart:
        st.subheader("Valuation Scenarios vs Market Price")
        chart_fig = EquityResearchReportGenerator.create_plotly_scenario_chart(summary)
        if chart_fig:
            st.plotly_chart(chart_fig, use_container_width=True)

    with col_narrative:
        st.subheader("Executive Analyst Takeaway")
        st.info(summary.executive_summary)
        st.markdown(f"""
        * **Enterprise Value:** ${adj_dcf.enterprise_value:,.0f}M
        * **Implied Equity Value:** ${adj_dcf.equity_value:,.0f}M
        * **Cost of Equity ($K_e$):** {adj_dcf.cost_of_equity:.2%}
        * **WACC Discount Rate:** {adj_dcf.wacc:.2%}
        * **Terminal Growth Rate ($g$):** {custom_inputs.terminal_growth_rate:.2%}
        """)

# --- TAB 2: SENTIMENT DIAGNOSTICS ---
with tab2:
    st.subheader("FinBERT Natural Language Polarity Breakdown")
    s = summary.sentiment_score
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Sentiment Classification", s.label.upper(), f"Confidence: {s.confidence:.1%}")
    with col_s2:
        st.metric("Positive Probability", f"{s.positive_prob:.1%}")
    with col_s3:
        st.metric("Neutral Probability", f"{s.neutral_prob:.1%}")
    with col_s4:
        st.metric("Negative Probability", f"{s.negative_prob:.1%}")

    st.markdown("---")
    st.subheader("Section-by-Section Tone Analysis")
    sec_data = []
    for sec_name, sc in section_scores.items():
        sec_data.append({
            "Filing Section": sec_name,
            "Tone Polarity": sc.label,
            "Compound Score (S)": f"{sc.compound_score:+.3f}",
            "Positive Prob": f"{sc.positive_prob:.1%}",
            "Negative Prob": f"{sc.negative_prob:.1%}",
            "Neutral Prob": f"{sc.neutral_prob:.1%}",
            "Confidence": f"{sc.confidence:.1%}",
        })
    st.dataframe(pd.DataFrame(sec_data), use_container_width=True)

    with st.expander("📄 View Parsed Document Text & Sections"):
        for sec_name, text in parsed_doc.sections.items():
            st.markdown(f"**{sec_name}**")
            st.caption(text)

# --- TAB 3: DCF SCHEDULE & SENSITIVITY ---
with tab3:
    st.subheader("5-Year Free Cash Flow to Firm (FCFF) Projection Schedule")
    sched_df = pd.DataFrame({
        "Metric ($ Millions)": [
            "Projected Revenue",
            "Operating Income (EBIT)",
            "NOPAT (EBIT x (1-t))",
            "Free Cash Flow (FCFF)",
            f"PV of FCFF (WACC = {base_dcf.wacc:.2%})",
        ],
        "Year 1": [
            f"${base_dcf.projected_revenues[0]:,.0f}",
            f"${base_dcf.projected_ebit[0]:,.0f}",
            f"${base_dcf.projected_ebit[0]*(1-custom_inputs.tax_rate):,.0f}",
            f"${base_dcf.projected_fcf[0]:,.0f}",
            f"${base_dcf.discounted_fcf[0]:,.0f}",
        ],
        "Year 2": [
            f"${base_dcf.projected_revenues[1]:,.0f}",
            f"${base_dcf.projected_ebit[1]:,.0f}",
            f"${base_dcf.projected_ebit[1]*(1-custom_inputs.tax_rate):,.0f}",
            f"${base_dcf.projected_fcf[1]:,.0f}",
            f"${base_dcf.discounted_fcf[1]:,.0f}",
        ],
        "Year 3": [
            f"${base_dcf.projected_revenues[2]:,.0f}",
            f"${base_dcf.projected_ebit[2]:,.0f}",
            f"${base_dcf.projected_ebit[2]*(1-custom_inputs.tax_rate):,.0f}",
            f"${base_dcf.projected_fcf[2]:,.0f}",
            f"${base_dcf.discounted_fcf[2]:,.0f}",
        ],
        "Year 4": [
            f"${base_dcf.projected_revenues[3]:,.0f}",
            f"${base_dcf.projected_ebit[3]:,.0f}",
            f"${base_dcf.projected_ebit[3]*(1-custom_inputs.tax_rate):,.0f}",
            f"${base_dcf.projected_fcf[3]:,.0f}",
            f"${base_dcf.discounted_fcf[3]:,.0f}",
        ],
        "Year 5": [
            f"${base_dcf.projected_revenues[4]:,.0f}",
            f"${base_dcf.projected_ebit[4]:,.0f}",
            f"${base_dcf.projected_ebit[4]*(1-custom_inputs.tax_rate):,.0f}",
            f"${base_dcf.projected_fcf[4]:,.0f}",
            f"${base_dcf.discounted_fcf[4]:,.0f}",
        ],
    })
    st.dataframe(sched_df, use_container_width=True)

    st.subheader("2D Valuation Sensitivity Matrix (WACC vs Terminal Growth Rate)")
    if base_dcf.sensitivity_table is not None:
        st.dataframe(base_dcf.sensitivity_table.style.highlight_max(axis=None, color="#d4edda"), use_container_width=True)

# --- TAB 4: RESEARCH MEMO ---
with tab4:
    st.subheader("Executive Equity Research Memo")
    memo_text = EquityResearchReportGenerator.generate_markdown_memo(summary)
    st.markdown(memo_text)
    st.download_button(
        label="📥 Download Research Memo (.md)",
        data=memo_text,
        file_name=f"{ticker_choice}_Equity_Research_Report.md",
        mime="text/markdown",
    )
