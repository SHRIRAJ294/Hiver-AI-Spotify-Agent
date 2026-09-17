# Technical Decision Log — Spotify AI Support Agent

This decision log records 12 key non-obvious architectural, technical, and evaluation choices made during the development of this repository.

---

### 1. Selecting Spotify (`@SpotifyCares`) as the Target Brand
- **Context**: The raw Kaggle Twitter Customer Support dataset (`twcs.csv`) contains pairs from multiple brands (Amazon, Apple, Spotify, Uber).
- **Decision**: Focused exclusively on Spotify (`SpotifyCares`, 1,952 clean support pairs).
- **Rationale**: Spotify interactions feature a clear mix of technical troubleshooting, billing inquiries, feature requests, and regional licensing issues, providing a rich multi-class evaluation domain.

---

### 2. Establishing a Data-Derived 9-Intent Taxonomy
- **Context**: An earlier project draft contained stale Amazon-specific intents (`delivery_issue`, `return_issue`).
- **Decision**: Replaced stale intents with a 9-intent taxonomy derived directly from Spotify support data (`technical_issue`, `playback_audio`, `account_access`, `billing_payment`, `subscription`, `content_availability`, `feature_request`, `general_information`, `non_support`).
- **Rationale**: Formulating intent categories directly from empirical brand interaction data prevents mismatched predictions and unclassifiable customer queries.

---

### 3. SentenceTransformers (`all-MiniLM-L6-v2`) for Intent Classification & Retrieval
- **Context**: TF-IDF sparse vector similarity failed to handle informal Twitter language (29.0% accuracy).
- **Decision**: Utilized SentenceTransformers `all-MiniLM-L6-v2` for both dense centroid intent classification and vector retrieval.
- **Rationale**: `all-MiniLM-L6-v2` provides strong semantic representations, fast CPU inference (~15ms per vector), and low memory footprint (80MB), boosting intent accuracy to 39.5% (+10.5% over TF-IDF).

---

### 4. Standardizing Training Data Across Classifiers to Prevent Comparison Bias
- **Context**: `intent_classifier.py` originally trained on `intent_examples.csv` (26 rows), while `embedding_classifier.py` trained on `new_intent_examples.csv` (102 rows).
- **Decision**: Standardized both classifiers to train on `data/intent_examples_expanded.csv` (125 rows).
- **Rationale**: Comparing classifiers trained on different datasets yields invalid metrics. Standardizing the training pool ensures a rigorous apples-to-apples comparison.

---

### 5. Strict Golden-Set Exclusion to Prevent Data Leakage
- **Context**: Data leakage between training examples and evaluation datasets leads to inflated performance metrics.
- **Decision**: Enforced explicit programmatic filtering in `combine_examples.py` to remove any customer message present in `golden_set.csv` (200 rows) from training sets.
- **Rationale**: Guarantees zero data leakage and ensures that golden set metrics accurately reflect generalization performance.

---

### 6. Vector Retrieval Across Full 1,952 Unlabeled Pair Corpus
- **Context**: Out of 1,952 historical Spotify support pairs, only 106 had intent labels assigned.
- **Decision**: Configured `retriever.py` to index and search across all 1,952 support pairs rather than filtering to intent-labeled rows.
- **Rationale**: Grounded response generation requires maximal historical coverage. Restricting retrieval to 106 labeled rows severely degraded retrieval recall.

---

### 7. Decoupling Evidence Sufficiency (`evidence_checker.py`) from Reply Quality (`reply_judge.py`)
- **Context**: Determining whether historical evidence is *sufficient* to answer is distinct from judging whether a *generated reply* is good.
- **Decision**: Built two separate modules: `evidence_checker.py` (determines routing before generation) and `reply_judge.py` (evaluates quality post-generation).
- **Rationale**: Prevents conflating routing safety logic with output generation evaluation.

---

### 8. Multi-Criteria Rubric for LLM-as-Judge Evaluation
- **Context**: Single pass/fail LLM scores can be opaque and inconsistent.
- **Decision**: Structured the judge prompt around 4 explicit criteria: Groundedness, Tone & Helpfulness, Policy Adherence, and Correctness.
- **Rationale**: Multi-criteria decomposition forces structured reasoning in LLM judging and aligns closely with human evaluation standards.

---

### 9. Measuring Cohen's Kappa Alongside Percentage Agreement
- **Context**: Raw percentage agreement does not account for chance agreement.
- **Decision**: Calculated Cohen's Kappa score alongside raw accuracy in `eval_agreement.py` (achieving 94.3% raw agreement, Kappa = 0.767).
- **Rationale**: Cohen's Kappa measures true inter-annotator agreement beyond chance, fulfilling formal evaluation standards.

---

### 10. Robust Caching of API Judge Results
- **Context**: Running LLM evaluations on every run consumes API quota and increases evaluation latency.
- **Decision**: Cached judge evaluations and human labels in `data/reply_judge_results.csv` and `data/human_judge_agreement.csv`.
- **Rationale**: Allows reviewers to run `python src/pipeline.py evaluate` instantly (<10 seconds) without requiring live API quota.

---

### 11. Instant Offline Fallback & Rate Limit Backoff
- **Context**: Network sandbox restrictions or API rate limits (429 errors) can crash evaluation pipelines.
- **Decision**: Added `HF_HUB_OFFLINE=1` flags and graceful fallback handlers to `evidence_checker.py` and `response_generator.py`.
- **Rationale**: Ensures the pipeline remains fully functional and robust even in offline sandbox environments or under API rate limits.

---

### 12. Modularizing `load_data.py` and CLI Dataset Builder
- **Context**: `load_data.py` originally executed data parsing on import and lacked error handling for missing `twcs.csv`.
- **Decision**: Refactored `load_data.py` into a clean library module and created `scripts/build_dataset.py` with friendly download instructions.
- **Rationale**: Separates reusable data parsing functions from CLI script execution and improves newcomer onboarding.
