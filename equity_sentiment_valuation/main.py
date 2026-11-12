"""
Automated Equity Valuation & Financial Sentiment Pipeline CLI Runner.
Parses SEC 10-K filings, scores sentiment with FinBERT, and generates
sentiment-adjusted multi-stage DCF valuation models.
"""
import argparse
import os
import sys
import pandas as pd

# Add module to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.document_parser import FinancialDocumentParser
from src.sentiment_analyzer import FinBERTSentimentAnalyzer
from src.dcf_model import DCFInputs, DCFValuationEngine
from src.sentiment_dcf_integrator import SentimentDCFIntegrator
from src.report_generator import EquityResearchReportGenerator


# Preset fundamental defaults for top equities
PRESET_EQUITY_DATA = {
    "AAPL": DCFInputs(
        ticker="AAPL", current_share_price=225.0, shares_outstanding=15300.0,
        revenue_base=385000.0, revenue_growth_rates=[0.08, 0.07, 0.06, 0.05, 0.04],
        operating_margin=0.31, tax_rate=0.21, d_and_a_rate=0.035, capex_rate=0.028,
        risk_free_rate=0.045, market_risk_premium=0.055, beta=1.10, cost_of_debt=0.045,
        total_debt=105000.0, cash_and_equivalents=65000.0, terminal_growth_rate=0.025,
    ),
    "NVDA": DCFInputs(
        ticker="NVDA", current_share_price=135.0, shares_outstanding=24500.0,
        revenue_base=115000.0, revenue_growth_rates=[0.35, 0.25, 0.18, 0.12, 0.08],
        operating_margin=0.62, tax_rate=0.18, d_and_a_rate=0.025, capex_rate=0.040,
        risk_free_rate=0.045, market_risk_premium=0.055, beta=1.65, cost_of_debt=0.048,
        total_debt=12000.0, cash_and_equivalents=35000.0, terminal_growth_rate=0.030,
    ),
    "BAC": DCFInputs(
        ticker="BAC", current_share_price=39.5, shares_outstanding=7800.0,
        revenue_base=102000.0, revenue_growth_rates=[0.05, 0.045, 0.04, 0.035, 0.03],
        operating_margin=0.36, tax_rate=0.21, d_and_a_rate=0.020, capex_rate=0.020,
        risk_free_rate=0.045, market_risk_premium=0.055, beta=1.25, cost_of_debt=0.050,
        total_debt=285000.0, cash_and_equivalents=315000.0, terminal_growth_rate=0.022,
    ),
    "MSFT": DCFInputs(
        ticker="MSFT", current_share_price=415.0, shares_outstanding=7430.0,
        revenue_base=245000.0, revenue_growth_rates=[0.14, 0.12, 0.10, 0.08, 0.06],
        operating_margin=0.44, tax_rate=0.20, d_and_a_rate=0.050, capex_rate=0.065,
        risk_free_rate=0.045, market_risk_premium=0.055, beta=1.15, cost_of_debt=0.042,
        total_debt=75000.0, cash_and_equivalents=80000.0, terminal_growth_rate=0.025,
    ),
}


def run_valuation_pipeline(
    ticker: str,
    filing_path: str = None,
    custom_inputs: DCFInputs = None,
    export_dir: str = "output",
):
    print("=" * 70)
    print(" AUTOMATED EQUITY VALUATION & FINANCIAL SENTIMENT PIPELINE ")
    print(" Bank of America Equity Research & Quantitative Valuation ")
    print("=" * 70)
    ticker = ticker.upper()
    print(f"[*] Target Equity Ticker: {ticker}")

    # 1. Load Financial Inputs
    if custom_inputs is not None:
        inputs = custom_inputs
    elif ticker in PRESET_EQUITY_DATA:
        inputs = PRESET_EQUITY_DATA[ticker]
    else:
        print(f"[!] No custom inputs supplied; initializing default baseline DCF for {ticker}.")
        inputs = DCFInputs(ticker=ticker)

    # 2. Ingest & Parse SEC Filing / Transcripts
    print("\n[1/4] Ingesting Financial Documents & 10-K Disclosures...")
    parser = FinancialDocumentParser()
    raw_text = None
    if filing_path and os.path.exists(filing_path):
        with open(filing_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        raw_text = parser.load_sample_filing(ticker)

    if not raw_text:
        raw_text = (
            f"{ticker} reported solid top-line revenue growth surpassing consensus. "
            "Operating profitability and gross margins expanded, offset by macroeconomic volatility."
        )

    parsed_doc = parser.parse_sec_10k(raw_text, ticker=ticker)
    print(f"      Parsed {len(parsed_doc.sections)} key sections from document ({parsed_doc.metadata.get('word_count', 'N/A')} words).")

    # 3. FinBERT Sentiment Scoring
    print("\n[2/4] Executing FinBERT Financial Sentiment Classification...")
    sentiment_engine = FinBERTSentimentAnalyzer(use_transformers=True)
    section_scores = sentiment_engine.analyze_document_sections(parsed_doc.sections)
    aggregate_score = sentiment_engine.compute_aggregate_score(list(section_scores.values()))

    print(f"      Overall Polarity: {aggregate_score.label.upper()} (Confidence: {aggregate_score.confidence:.1%})")
    print(f"      Compound Score S: {aggregate_score.compound_score:+.3f} [Pos: {aggregate_score.positive_prob:.1%}, Neu: {aggregate_score.neutral_prob:.1%}, Neg: {aggregate_score.negative_prob:.1%}]")
    for sec, sc in section_scores.items():
        print(f"      - {sec:<24}: {sc.label:<8} (Compound: {sc.compound_score:+.2f})")

    # 4. DCF Valuation & Sentiment Coupling
    print("\n[3/4] Running Multi-Stage DCF Valuation & Dynamic Growth Adjustments...")
    integrator = SentimentDCFIntegrator()
    summary = integrator.evaluate_all_scenarios(inputs, aggregate_score)

    # 5. Output Research Memo & Artifacts
    print("\n[4/4] Generating Executive Equity Research Memo...")
    memo = EquityResearchReportGenerator.generate_markdown_memo(summary)
    print("\n" + memo)

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        json_path = os.path.join(export_dir, f"{ticker}_valuation_summary.json")
        md_path = os.path.join(export_dir, f"{ticker}_equity_research_report.md")
        EquityResearchReportGenerator.export_summary_json(summary, json_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(memo)
        print(f"\n[OK] Research memo saved to: {md_path}")
        print(f"[OK] Valuation JSON exported to: {json_path}")

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE. Launch Streamlit UI with: streamlit run app.py")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Automated Equity Valuation & Financial Sentiment Pipeline")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (e.g. AAPL, NVDA, BAC, MSFT)")
    parser.add_argument("--filing", default=None, help="Path to raw 10-K or transcript text file")
    parser.add_argument("--price", type=float, default=None, help="Current share price in USD")
    parser.add_argument("--export-dir", default="output", help="Output directory for reports")

    args = parser.parse_args()
    custom_inp = None
    if args.price is not None:
        custom_inp = PRESET_EQUITY_DATA.get(args.ticker.upper(), DCFInputs(ticker=args.ticker.upper()))
        custom_inp.current_share_price = args.price

    run_valuation_pipeline(
        ticker=args.ticker,
        filing_path=args.filing,
        custom_inputs=custom_inp,
        export_dir=args.export_dir,
    )


if __name__ == "__main__":
    main()
