# Spotify AI Customer Support Agent

An end-to-end, production-ready AI customer support agent for **Spotify** (`@SpotifyCares`), built on real Twitter customer-support data (`twcs.csv`).

The agent:
1. **Classifies** incoming customer tweets into a data-derived 9-intent taxonomy.
2. **Retrieves** historically resolved Spotify support cases using embedding vector search.
3. **Decides** whether to **auto-handle** or **escalate** to a human support agent based on LLM evidence-sufficiency evaluation.
4. **Drafts** grounded support responses adhering strictly to historical brand policies (no fake refunds, unannounced features, or invented policies).

---

## 🚀 Quick Start & Reproducing Headline Results (< 15 Minutes)

### 1. Setup Environment

```bash
# Clone repository and enter project folder
cd customer-support-agent

# Create virtual environment (Python 3.9+)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables & API Keys

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY="your_gemini_api_key_here"
```

*(Note: Pre-cached evaluation results are checked in, so `pipeline.py evaluate` runs offline without requiring API quota).*

### 3. Reproduce Headline Evaluation Results

To run the complete evaluation harness across intent classifiers, retrieval quality, system baselines, LLM-as-Judge reply quality, and human agreement:

```bash
python src/pipeline.py evaluate
```

### 4. Interactive Pipeline Commands

**Classify a customer message:**
```bash
python src/pipeline.py classify "My Spotify desktop app keeps crashing when I open it"
```

**Run full support routing & reply generation:**
```bash
python src/pipeline.py respond "I was charged twice for my Premium subscription"
```

**Run Unit Tests:**
```bash
python -m unittest discover tests
```

---

## 📊 Intent Taxonomy (9 Spotify Intents)

Derived directly from Spotify support interaction patterns:
- `technical_issue`: App/web player crashes, installation, or error codes.
- `playback_audio`: Audio stopping, buffering, volume, or equalizer issues.
- `account_access`: Login problems, password resets, or family plan address verification.
- `billing_payment`: Charges, payment methods, duplicate billing, or refunds.
- `subscription`: Upgrading, canceling, or student/family plan eligibility.
- `content_availability`: Songs, albums, or features unavailable in specific regions.
- `feature_request`: UI suggestions, feature requests, or option updates.
- `general_information`: Offline song limits, Wrapped, device support, or general FAQs.
- `non_support`: Pleasantries, thank yous, or chatter requiring no support resolution.

---

## 📦 Raw Data Reproducibility

The pre-processed Spotify support pairs (`data/spotify_support_pairs.csv`) and golden evaluation set (`data/golden_set.csv`) are checked into this repository.

If you wish to rebuild `data/spotify_support_pairs.csv` from scratch:
1. Download the Kaggle **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`).
2. Extract `twcs.csv` into `data/twcs.csv`.
3. Run the dataset extraction CLI:
   ```bash
   python scripts/build_dataset.py
   ```

---

## 📁 Repository Structure

```
customer-support-agent/
├── README.md                      # Project documentation & quickstart
├── REPORT.md                      # Technical report (Problem framing, baselines, failure modes)
├── DECISION_LOG.md                # 12 non-obvious engineering decisions & justifications
├── requirements.txt               # Required Python dependencies
├── .env                           # API keys (GEMINI_API_KEY)
├── data/
│   ├── spotify_support_pairs.csv  # 1952 Spotify customer/support pairs
│   ├── golden_set.csv             # 200 hand-labeled golden evaluation set
│   ├── intent_examples_expanded.csv # 125 standardized intent training examples
│   ├── retrieval_eval.csv         # Retrieval evaluation dataset
│   ├── reply_judge_results.csv    # Pre-cached LLM-as-Judge evaluation results
│   ├── human_judge_agreement.csv  # 35 hand-labeled judge-vs-human agreement set
│   └── headline_results.json      # Saved evaluation metrics summary
├── src/
│   ├── pipeline.py                # Single pipeline CLI entry point (classify, respond, evaluate)
│   ├── intents.py                 # Spotify 9-intent taxonomy definition & descriptions
│   ├── intent_classifier.py       # TF-IDF nearest-neighbor intent classifier
│   ├── embedding_classifier.py    # SentenceTransformers centroid classifier
│   ├── retriever.py               # Embedding-based support pair retriever
│   ├── baselines.py               # System-level Trivial and Simple baselines
│   ├── evidence_checker.py        # Gemini evidence-sufficiency check
│   ├── response_generator.py      # Gemini grounded reply generator
│   ├── reply_judge.py             # LLM-as-Judge reply quality evaluator
│   ├── eval_agreement.py          # Cohen's Kappa & judge-vs-human agreement harness
│   └── load_data.py               # Raw twcs.csv dataset parser library
├── scripts/
│   └── build_dataset.py           # CLI script to rebuild support pairs from twcs.csv
└── tests/                         # Unit tests (unittest runner)
    ├── test_intents.py
    ├── test_load_data.py
    ├── test_retriever.py
    └── test_decision.py
```
