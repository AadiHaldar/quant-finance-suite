"""
Financial Document & SEC 10-K Parser.
Extracts MD&A, Risk Factors, and Executive Transcripts with sentence chunking.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import re


@dataclass
class FinancialDocument:
    """Represents a structured parsed financial filing or transcript."""
    ticker: str
    doc_type: str  # '10-K', '10-Q', 'Earnings Transcript', 'News'
    period: str
    full_text: str
    sections: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)


class FinancialDocumentParser:
    """
    Parses SEC 10-K filings, 10-Q disclosures, and earnings call transcripts
    into isolated strategic sections (MD&A, Risk Factors, Guidance).
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "filings"
        )

    def parse_sec_10k(self, raw_text: str, ticker: str = "GENERIC", period: str = "FY2024") -> FinancialDocument:
        """
        Extracts key SEC 10-K items:
        - Item 1: Business
        - Item 1A: Risk Factors
        - Item 7: Management's Discussion & Analysis (MD&A)
        - Item 7A: Quantitative & Qualitative Disclosures About Market Risk
        """
        sections: Dict[str, str] = {}

        patterns = {
            "Item 1: Business": r"(?:ITEM\s+1[\.\s\:\-]+Business)(.*?)(?=ITEM\s+1A[\.\s\:\-]|ITEM\s+2[\.\s\:\-]|$)",
            "Item 1A: Risk Factors": r"(?:ITEM\s+1A[\.\s\:\-]+Risk\s+Factors)(.*?)(?=ITEM\s+1B[\.\s\:\-]|ITEM\s+2[\.\s\:\-]|$)",
            "Item 7: MD&A": r"(?:ITEM\s+7[\.\s\:\-]+Management(?:'s)?\s+Discussion\s+and\s+Analysis)(.*?)(?=ITEM\s+7A[\.\s\:\-]|ITEM\s+8[\.\s\:\-]|$)",
            "Item 7A: Market Risk": r"(?:ITEM\s+7A[\.\s\:\-]+Quantitative\s+and\s+Qualitative\s+Disclosures)(.*?)(?=ITEM\s+8[\.\s\:\-]|$)",
        }

        for sec_name, pattern in patterns.items():
            match = re.search(pattern, raw_text, re.IGNORECASE | re.DOTALL)
            if match:
                clean_sec = self._clean_text(match.group(1))
                if len(clean_sec) > 50:
                    sections[sec_name] = clean_sec

        # If strict regex didn't find demarcated sections, segment into paragraphs
        if not sections:
            paragraphs = [self._clean_text(p) for p in raw_text.split("\n\n") if len(p.strip()) > 80]
            if paragraphs:
                half = len(paragraphs) // 2
                sections["Management Overview"] = " ".join(paragraphs[:half])
                sections["Risk & Operations"] = " ".join(paragraphs[half:])
            else:
                sections["General Disclosure"] = self._clean_text(raw_text)

        return FinancialDocument(
            ticker=ticker.upper(),
            doc_type="10-K",
            period=period,
            full_text=self._clean_text(raw_text),
            sections=sections,
            metadata={"word_count": str(len(raw_text.split()))},
        )

    def parse_earnings_transcript(self, raw_text: str, ticker: str = "GENERIC", period: str = "Q3 2024") -> FinancialDocument:
        """
        Splits earnings call transcripts into Prepared Remarks and Analyst Q&A.
        """
        sections: Dict[str, str] = {}
        qa_pattern = r"(?:QUESTION[\s\-]+AND[\s\-]+ANSWER[\s\-]+SESSION|Q&A\s+SESSION|Questions\s+and\s+Answers)(.*)"
        match = re.search(qa_pattern, raw_text, re.IGNORECASE | re.DOTALL)

        if match:
            qa_text = match.group(1)
            prep_text = raw_text[: match.start()]
            sections["Executive Remarks"] = self._clean_text(prep_text)
            sections["Analyst Q&A"] = self._clean_text(qa_text)
        else:
            sections["Executive Remarks"] = self._clean_text(raw_text)

        return FinancialDocument(
            ticker=ticker.upper(),
            doc_type="Earnings Transcript",
            period=period,
            full_text=self._clean_text(raw_text),
            sections=sections,
            metadata={"word_count": str(len(raw_text.split()))},
        )

    def chunk_text(self, text: str, max_words: int = 250) -> List[str]:
        """
        Splits text into contextually sized chunks suited for transformer tokenization limits.
        """
        sentences = re.split(r"(?<=[.?!])\s+", text)
        chunks = []
        current_chunk: List[str] = []
        current_count = 0

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            words_in_s = len(s_clean.split())
            if current_count + words_in_s > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [s_clean]
                current_count = words_in_s
            else:
                current_chunk.append(s_clean)
                current_count += words_in_s

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _clean_text(self, text: str) -> str:
        """Removes excess whitespace, unprintable characters, and page breaks."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text)
        return text.strip()

    def load_sample_filing(self, ticker: str) -> Optional[str]:
        """Loads sample raw text filing from data/filings directory."""
        filename = f"{ticker.upper()}_sample.txt"
        file_path = os.path.join(self.data_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
