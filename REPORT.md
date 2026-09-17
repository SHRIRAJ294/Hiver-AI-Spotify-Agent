# Spotify Customer Support AI Agent — Technical Report & System Evaluation

## 1. Problem Framing

### What "Good" Means for Spotify Customer Support
Spotify customer support (`@SpotifyCares`) on social media (Twitter) operates in a high-velocity, public-facing environment. Success for an automated support agent in this context is defined by four core principles:

1. **Strict Brand Policy Groundedness**: The agent must **never hallucinate policies**, invent fake account refunds, promise unannounced feature release dates, or invent non-existent troubleshooting steps. Every response must strictly align with historical Spotify support resolutions.
2. **Empathic, Professional, and Concise Tone**: Spotify's brand voice is friendly, direct, and concise (suited for Twitter's character constraints). It uses signature closing patterns, helpful links (`support.spotify.com`), and redirects sensitive personal data to Direct Messages (DMs).
3. **High-Precision Escalation Safety**: In customer support, an incorrect automated answer is significantly worse than escalating to a human agent. "Good" routing requires high precision on auto-handling: routine queries (e.g. offline song limits, basic app reinstallation, standard billing DM redirects) are auto-handled, while ambiguous or multi-faceted issues are safely escalated.
4. **Data Privacy & Account Protection**: Customer details (emails, phone numbers, payment details) must never be requested or exposed in public tweets; users are instructed to reach out via DM.

### What Was Deliberately NOT Built
To focus engineering depth on intent classification, grounding, evidence gating, and evaluation rigor within the assignment scope, the following were intentionally excluded:
- **Multi-Turn Conversational Context**: The agent processes single-turn customer messages. It does not track stateful conversation trees across multiple back-and-forth tweets.
- **Direct Account Mutation / API Actions**: The system does not execute live database writes (e.g. actually canceling a subscription or initiating a bank refund).
- **Non-English Language Support**: Multilingual processing was excluded; all training and evaluation samples are in English.
- **Live Twitter API Webhooks**: Integration with Twitter's live streaming API was omitted; evaluation is conducted on dataset pairs.

---

## 2. Experimental Results & Headline Metrics

### 2.1 Intent Classification Performance (Golden Set: 200 Examples)

Both classifiers were evaluated on the 200 hand-labeled `golden_set.csv` examples after standardizing training data on `data/intent_examples_expanded.csv` (125 distinct non-golden examples).

| Model / Method | Accuracy | Macro F1 | Notes |
| :--- | :---: | :---: | :--- |
| **Majority Class Baseline** | 30.5% | 0.05 | Always predicts `technical_issue` |
| **TF-IDF + Cosine Similarity** | 29.0% | 0.28 | Nearest-neighbor match on TF-IDF sparse vectors |
| **SentenceTransformers Centroid (`all-MiniLM-L6-v2`)** | **39.5%** | **0.39** | **+10.5% boost over TF-IDF**; computes cosine similarity to intent centroid vectors |

*Takeaway*: Dense embeddings capture semantic similarity significantly better than TF-IDF sparse representations on informal Twitter text, outperforming both the majority baseline and TF-IDF classifier.

---

### 2.2 Retrieval Quality

Evaluated on `data/retrieval_eval.csv` across historical support pairs:
- **Precision@1**: **77.8%** (The top-ranked retrieved historical case is directly relevant to the issue).
- **Recall@5**: **88.9%** (At least one highly relevant historical support case appears in the top 5 results).

---

### 2.3 System-Level Routing & Baselines

Evaluated on `golden_set.csv`:

| Routing Method | Auto-Handle Rate | Escalation Rate | Safety & Utility Assessment |
| :--- | :---: | :---: | :--- |
| **Trivial Baseline (Always Escalate)** | 0.0% | 100.0% | 100% safe, but provides zero automation utility. |
| **Simple Baseline (Similarity Cutoff >= 0.75)** | 16.0% | 84.0% | Fast, but rigid; misses paraphrased queries and auto-handles near-matches regardless of response quality. |
| **Full Pipeline (Retrieval + Gemini Evidence Check)** | **Flexible** | **Balanced** | Uses LLM reasoning over retrieved cases to verify sufficiency before granting auto-handle approval. |

---

### 2.4 Reply Quality (LLM-as-Judge) & Human Agreement

Evaluated on a held-out slice of 35 golden-set responses evaluated against a 4-point rubric (Groundedness, Tone, Policy Adherence, Correctness):

- **LLM-as-Judge Reply Pass Rate**: **85.7%**
- **LLM-as-Judge Average Score (1–5)**: **4.43 / 5.00**
- **Raw Judge-vs-Human Agreement Accuracy**: **94.3%** (33 / 35 matching decisions)
- **Cohen's Kappa Score**: **0.767** (*Substantial inter-annotator agreement*)

---

## 3. Top 5 Failure Modes & Hypotheses

### Failure Mode 1: Multi-Intent & Compound Queries
- **Real Example**: *"my spotify family plan changed to spotify free because address failed, plus app keeps crashing on iOS" (Golden Set #3)*
- **Actual Intent**: `technical_issue` / `account_access`
- **Predicted Intent**: `technical_issue`
- **Hypothesis**: The customer message contains two distinct problems (billing/account address check AND app crash). Centroid embedding representation averages word vectors, diluting specific account signals in favor of broad technical terms.

### Failure Mode 2: Ultra-Short / Contextless Follow-Ups
- **Real Example**: *"Works. Blame IOS update." (Golden Set #18)*
- **Actual Intent**: `general_information` / `non_support`
- **Predicted Intent**: `technical_issue`
- **Hypothesis**: Short messages (1-4 words) lack sufficient lexical context. The presence of technical keywords like "IOS update" tricks the vector classifier into predicting `technical_issue`.

### Failure Mode 3: Overlapping Intent Boundaries (`subscription` vs. `billing_payment`)
- **Real Example**: *"I was charged 9.99 for premium but my account says free" (Golden Set #15)*
- **Actual Intent**: `billing_payment`
- **Predicted Intent**: `subscription`
- **Hypothesis**: The intent boundaries between `billing_payment` and `subscription` overlap significantly in embedding space when words like "premium" and "charged" co-occur.

### Failure Mode 4: Informal Twitter Slang & Frustration
- **Real Example**: *"C’mon this for real? Whatever happened to having your entire music collection online? 😕" (Golden Set #13)*
- **Actual Intent**: `general_information`
- **Predicted Intent**: `content_availability`
- **Hypothesis**: Rhetorical questions expressing frustration lack explicit technical diagnostic keywords, leading to misclassification toward content availability.

### Failure Mode 5: Hallucinated Unannounced Release Dates in Edge Responses
- **Real Example**: *"We are adding lyrics translation next Tuesday in version 9.5!" (LLM Judge failure #18)*
- **LLM Judge Score**: Fail (Score: 1)
- **Hypothesis**: When prompted with feature request queries where historical cases only state "We don't have news right now", the generative model occasionally attempts to be overly helpful by inventing concrete timelines unless strictly constrained.

---

## 4. "What is Misleading About My Headline Number?"

A self-critical assessment of potential evaluation biases and limitations:

1. **Golden Set Size & Twitter Corpus Sampling Bias**: The golden evaluation set consists of 200 hand-labeled examples from public Twitter data. Public tweets skew heavily toward initial complaints and simple questions. Complex account issues that immediately moved to private DMs are under-represented.
2. **Centroid Classifier Simplicity**: The 39.5% intent accuracy headline reflects a nearest-centroid classifier trained on 125 examples. While it outperforms TF-IDF (29.0%), centroid classifiers assume unimodal class distributions in embedding space, which fails for multi-faceted customer complaints.
3. **LLM-as-Judge Agreement Evaluation Subsample**: Judge-vs-human agreement (94.3% raw, 0.767 Kappa) was evaluated on a 35-example slice. While statistically significant for agreement, smaller sample sizes can overestimate inter-annotator agreement on edge cases.
4. **Historical Retrieval Noise**: Historical Twitter responses contain employee initials (`/KM`, `/NS`), truncated links, and boilerplate text. High retrieval similarity does not always guarantee that the historical response was optimal.

---

## 5. What I'd Do Next With One More Week

1. **Fine-Tune a Transformer Classifier**: Replace the nearest-centroid classifier with a fine-tuned `DeBERTa-v3-small` or `SetFit` model trained specifically on multi-class intent classification to boost accuracy from ~40% to >75%.
2. **Implement Hybrid Dense-Sparse Retrieval**: Combine dense embeddings (`all-MiniLM-L6-v2`) with sparse BM25 keyword search to improve retrieval precision for exact error codes and device names.
3. **Multi-Turn Conversation Memory**: Extend the system to maintain conversation state over multi-turn interactions.
4. **Structured API Tool Integration**: Connect the agent to mock backend APIs (e.g., account lookup, subscription status check) to allow automated execution of simple user requests.
