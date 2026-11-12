# Automated Equity Valuation & Financial Sentiment Pipeline

An institutional-grade fundamental valuation framework combining **FinBERT Natural Language Processing (NLP)** sentiment classification with **Multi-Stage Discounted Cash Flow (DCF)** modeling and 2D sensitivity analysis.

---

## Key Features

1. **FinBERT NLP Sentiment Classifier (`src/sentiment_analyzer.py`)**:
   - Classifies financial text from SEC 10-K disclosures, 10-Q reports, and earnings call transcripts into **Positive, Negative, and Neutral** probabilities with confidence scores.
   - Includes **Loughran-McDonald Financial Lexicon fallback** for high-speed, zero-latency inference.
   - Computes normalized compound sentiment scores: $S = P(\text{Positive}) - P(\text{Negative}) \in [-1.0, +1.0]$.

2. **SEC 10-K & Transcript Ingestor (`src/document_parser.py`)**:
   - Parses Item 1 (Business Overview), Item 1A (Risk Factors), and Item 7 (MD&A) into segmented textual blocks.
   - Splits earnings calls into Executive Prepared Remarks vs Analyst Q&A sessions.

3. **Multi-Stage DCF Valuation Engine (`src/dcf_model.py`)**:
   - **Cost of Capital (WACC)**:
     $$WACC = \left(\frac{E}{V} \times K_e\right) + \left(\frac{D}{V} \times K_d \times (1 - t)\right)$$
     where $K_e = R_f + \beta \times ERP$ (CAPM).
   - **Free Cash Flow to Firm (FCFF)**:
     $$FCFF_t = \text{EBIT}_t \times (1 - t) + \text{D\&A}_t - \text{CapEx}_t - \Delta \text{NWC}_t$$
   - **Terminal Value & Enterprise Value**:
     $$TV = \frac{FCFF_N \times (1 + g)}{WACC - g}, \quad EV = \sum_{t=1}^N \frac{FCFF_t}{(1 + WACC)^t} + \frac{TV}{(1 + WACC)^N}$$
   - **Implied Target Price & Margin of Safety**:
     $$P_{\text{intrinsic}} = \frac{EV + \text{Cash} - \text{Debt}}{\text{Shares Outstanding}}, \quad \text{MoS} = \frac{P_{\text{intrinsic}} - P_{\text{market}}}{P_{\text{market}}}$$
   - Generates **2D Sensitivity Matrices** ($WACC \text{ vs } g$).

4. **Dynamic Sentiment-DCF Integrator (`src/sentiment_dcf_integrator.py`)**:
   - Dynamically modulates growth rates, operating margins, and equity risk premiums based on FinBERT confidence and scores.
   - Evaluates 4 distinct scenarios: **Base Case, Bull Case, Bear Case, and Sentiment-Informed Case**.

5. **Interactive Streamlit Workbench (`app.py`)**:
   - Real-time parameter tweaking, live filing analysis, scenario comparison charts, and executive memo export.

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run CLI Valuation
```bash
# Analyze Apple with preset fundamental data & sample 10-K
python main.py --ticker AAPL

# Analyze NVIDIA or Bank of America
python main.py --ticker NVDA
python main.py --ticker BAC
```

### 3. Launch Interactive Streamlit UI
```bash
streamlit run app.py
```

### 4. Run Test Suite
```bash
python -m pytest tests/ -v
```
