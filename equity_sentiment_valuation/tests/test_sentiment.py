"""
Unit tests for FinancialDocumentParser and FinBERTSentimentAnalyzer.
Verifies section extraction, lexicon tokenization, and polarity scoring.
"""
import pytest
from src.document_parser import FinancialDocumentParser
from src.sentiment_analyzer import FinBERTSentimentAnalyzer, SentimentScore


def test_sec_10k_parsing():
    raw_doc = """
    ITEM 1. Business
    Apple Inc. designs and manufactures premium consumer electronics.
    
    ITEM 1A. Risk Factors
    The Company faces severe supply chain disruptions and regulatory scrutiny.
    
    ITEM 7. Management's Discussion and Analysis
    Revenue expanded significantly with record profit margins and dividend growth.
    """
    parser = FinancialDocumentParser()
    parsed = parser.parse_sec_10k(raw_doc, ticker="AAPL")
    
    assert parsed.ticker == "AAPL"
    assert "Item 1: Business" in parsed.sections
    assert "Item 1A: Risk Factors" in parsed.sections
    assert "Item 7: MD&A" in parsed.sections
    assert "electronics" in parsed.sections["Item 1: Business"]


def test_sentiment_analyzer_positive():
    analyzer = FinBERTSentimentAnalyzer(use_transformers=False)
    pos_text = "Net revenue accelerated dramatically with record profit margins, robust cash flows, and massive dividend growth."
    score = analyzer.analyze_chunk(pos_text)
    
    assert score.label == "Positive"
    assert score.compound_score > 0
    assert score.positive_prob > score.negative_prob


def test_sentiment_analyzer_negative():
    analyzer = FinBERTSentimentAnalyzer(use_transformers=False)
    neg_text = "The business suffered severe impairment losses, litigation penalties, regulatory scrutiny, and revenue deterioration."
    score = analyzer.analyze_chunk(neg_text)
    
    assert score.label == "Negative"
    assert score.compound_score < 0
    assert score.negative_prob > score.positive_prob


def test_sentiment_analyzer_neutral_empty():
    analyzer = FinBERTSentimentAnalyzer(use_transformers=False)
    score = analyzer.analyze_chunk("")
    
    assert score.label == "Neutral"
    assert score.compound_score == 0.0


def test_aggregate_score_calculation():
    analyzer = FinBERTSentimentAnalyzer(use_transformers=False)
    scores = [
        SentimentScore("Positive", 0.7, 0.1, 0.2, compound_score=0.6, confidence=0.7),
        SentimentScore("Positive", 0.8, 0.05, 0.15, compound_score=0.75, confidence=0.8),
        SentimentScore("Neutral", 0.2, 0.2, 0.6, compound_score=0.0, confidence=0.6),
    ]
    agg = analyzer.compute_aggregate_score(scores)
    
    assert agg.label == "Positive"
    assert agg.compound_score > 0.3
    assert -1.0 <= agg.compound_score <= 1.0
