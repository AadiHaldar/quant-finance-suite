"""
FinBERT Financial Sentiment Analysis & NLP Engine.
Classifies financial texts into Positive, Negative, and Neutral probabilities,
with Loughran-McDonald financial dictionary fallback for ultra-fast execution.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

# Loughran-McDonald Domain-Specific Financial Lexicon
FINANCIAL_LEXICON = {
    "positive": {
        "surpassed", "outperformed", "accelerated", "profitability", "expansion",
        "dividend", "robust", "strong", "momentum", "record", "exceeded",
        "growth", "tailwind", "gain", "efficiency", "upside", "lucrative",
        "synergies", "innovative", "scalable", "bullish", "accretive", "margin",
    },
    "negative": {
        "declined", "headwind", "impairment", "litigation", "scrutiny", "downturn",
        "shortfall", "loss", "volatility", "risk", "adversely", "inflationary",
        "compression", "restructuring", "recession", "deterioration", "default",
        "subdued", "layoffs", "penalties", "investigation", "tariff", "sluggish",
    },
    "intensifiers": {
        "significantly", "substantially", "massively", "dramatically", "exceptionally",
    },
}


@dataclass
class SentimentScore:
    """Quantitative representation of financial sentiment."""
    label: str  # 'Positive', 'Negative', 'Neutral'
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    compound_score: float  # S in [-1.0, 1.0]
    confidence: float


class FinBERTSentimentAnalyzer:
    """
    Analyzes sentiment in SEC 10-Ks, 10-Qs, and earnings calls using FinBERT
    with Loughran-McDonald rule-based fallback.
    """

    def __init__(self, use_transformers: bool = True, model_name: str = "ProsusAI/finbert"):
        self.use_transformers = use_transformers
        self.model_name = model_name
        self.pipeline = None
        self._initialized = False

        if self.use_transformers:
            self._try_load_finbert()

    def _try_load_finbert(self):
        """Attempts to load FinBERT pipeline via HuggingFace transformers."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import torch
            # Initialize with small truncation
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,
                device=-1,  # CPU for portability
            )
            self._initialized = True
        except Exception as e:
            # Fall back to financial dictionary engine
            self.pipeline = None
            self._initialized = False

    def analyze_chunk(self, text: str) -> SentimentScore:
        """
        Analyzes a single text paragraph or sentence.
        """
        if not text or len(text.strip()) == 0:
            return SentimentScore("Neutral", 0.0, 0.0, 1.0, 0.0, 1.0)

        # 1. FinBERT Neural Pathway
        if self._initialized and self.pipeline is not None:
            try:
                # Truncate text to 512 tokens
                results = self.pipeline(text[:1500])
                if results and isinstance(results, list) and isinstance(results[0], list):
                    prob_dict = {item["label"].lower(): float(item["score"]) for item in results[0]}
                    pos = prob_dict.get("positive", 0.0)
                    neg = prob_dict.get("negative", 0.0)
                    neu = prob_dict.get("neutral", 0.0)

                    compound = pos - neg
                    if pos > neg and pos > neu:
                        label = "Positive"
                    elif neg > pos and neg > neu:
                        label = "Negative"
                    else:
                        label = "Neutral"

                    confidence = max(pos, neg, neu)
                    return SentimentScore(
                        label=label,
                        positive_prob=round(pos, 4),
                        negative_prob=round(neg, 4),
                        neutral_prob=round(neu, 4),
                        compound_score=round(compound, 4),
                        confidence=round(confidence, 4),
                    )
            except Exception:
                pass  # Fall through to lexicon

        # 2. Loughran-McDonald Lexical Pathway (Fallback)
        return self._lexicon_analyze(text)

    def _lexicon_analyze(self, text: str) -> SentimentScore:
        """
        Rule-based financial sentiment scorer tuned to financial jargon.
        """
        words = text.lower().replace(",", " ").replace(".", " ").replace(";", " ").split()
        total_words = len(words)
        if total_words == 0:
            return SentimentScore("Neutral", 0.0, 0.0, 1.0, 0.0, 1.0)

        pos_count = 0.0
        neg_count = 0.0
        multiplier = 1.0

        for w in words:
            if w in FINANCIAL_LEXICON["intensifiers"]:
                multiplier = 1.6
                continue
            if w in FINANCIAL_LEXICON["positive"]:
                pos_count += 1.0 * multiplier
                multiplier = 1.0
            elif w in FINANCIAL_LEXICON["negative"]:
                neg_count += 1.0 * multiplier
                multiplier = 1.0
            else:
                multiplier = 1.0

        score_diff = pos_count - neg_count
        norm_factor = max(pos_count + neg_count, 1.0)
        compound = score_diff / norm_factor

        # Convert to pseudo-probabilities with softmax-style smoothing
        z_pos = np.exp(pos_count / max(1, pos_count + neg_count))
        z_neg = np.exp(neg_count / max(1, pos_count + neg_count))
        z_neu = np.exp(1.0) if (pos_count == 0 and neg_count == 0) else np.exp(0.5)
        z_sum = z_pos + z_neg + z_neu

        pos_prob = float(z_pos / z_sum)
        neg_prob = float(z_neg / z_sum)
        neu_prob = float(z_neu / z_sum)

        if compound > 0.15:
            label = "Positive"
        elif compound < -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        confidence = max(pos_prob, neg_prob, neu_prob)

        return SentimentScore(
            label=label,
            positive_prob=round(pos_prob, 4),
            negative_prob=round(neg_prob, 4),
            neutral_prob=round(neu_prob, 4),
            compound_score=round(float(compound), 4),
            confidence=round(confidence, 4),
        )

    def analyze_document_sections(self, sections: Dict[str, str]) -> Dict[str, SentimentScore]:
        """
        Analyzes distinct sections of a financial document (e.g. MD&A vs Risk Factors).
        """
        section_scores = {}
        for sec_name, text in sections.items():
            score = self.analyze_chunk(text)
            section_scores[sec_name] = score
        return section_scores

    def compute_aggregate_score(self, scores: List[SentimentScore]) -> SentimentScore:
        """
        Calculates confidence-weighted aggregate sentiment score across multiple chunks.
        """
        if not scores:
            return SentimentScore("Neutral", 0.0, 0.0, 1.0, 0.0, 1.0)

        weights = np.array([max(s.confidence, 0.1) for s in scores])
        weights = weights / np.sum(weights)

        avg_pos = float(np.sum([s.positive_prob * w for s, w in zip(scores, weights)]))
        avg_neg = float(np.sum([s.negative_prob * w for s, w in zip(scores, weights)]))
        avg_neu = float(np.sum([s.neutral_prob * w for s, w in zip(scores, weights)]))
        avg_comp = float(np.sum([s.compound_score * w for s, w in zip(scores, weights)]))

        if avg_comp > 0.10:
            label = "Positive"
        elif avg_comp < -0.10:
            label = "Negative"
        else:
            label = "Neutral"

        confidence = max(avg_pos, avg_neg, avg_neu)

        return SentimentScore(
            label=label,
            positive_prob=round(avg_pos, 4),
            negative_prob=round(avg_neg, 4),
            neutral_prob=round(avg_neu, 4),
            compound_score=round(avg_comp, 4),
            confidence=round(confidence, 4),
        )
