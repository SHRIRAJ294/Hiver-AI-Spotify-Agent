import os
import pandas as pd

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from retriever import retrieve_similar_cases


def evaluate_trivial_baseline(golden_df):
    """
    Trivial Baseline: Always escalate to human agent.
    Auto-handle rate: 0.0
    Human escalation rate: 1.0
    """
    total = len(golden_df)
    decisions = []
    for _ in range(total):
        decisions.append({
            "decision": "escalate",
            "reason": "Trivial baseline: Always escalate to human support."
        })

    auto_handled = sum(1 for d in decisions if d["decision"] == "auto_handle")
    escalated = sum(1 for d in decisions if d["decision"] == "escalate")

    return {
        "name": "Trivial Baseline (Always Escalate)",
        "auto_handle_rate": auto_handled / total if total > 0 else 0.0,
        "escalate_rate": escalated / total if total > 0 else 0.0,
        "auto_handled_count": auto_handled,
        "escalated_count": escalated,
        "total": total
    }


def evaluate_simple_baseline(golden_df, similarity_threshold=0.75):
    """
    Simple Baseline: Cosine similarity threshold rule on retrieved cases.
    If best retrieved case similarity >= threshold -> auto_handle with retrieved response;
    else -> escalate.
    """
    total = len(golden_df)
    results = []

    for _, row in golden_df.iterrows():
        msg = row["customer_message"]
        retrieved_cases, best_sim = retrieve_similar_cases(msg, top_k=5, exclude_message=msg)

        if best_sim >= similarity_threshold:
            decision = "auto_handle"
            reply = retrieved_cases.iloc[0]["support_response"]
            reason = f"Retrieval similarity ({best_sim:.2f}) >= threshold ({similarity_threshold})"
        else:
            decision = "escalate"
            reply = None
            reason = f"Retrieval similarity ({best_sim:.2f}) < threshold ({similarity_threshold})"

        results.append({
            "customer_message": msg,
            "decision": decision,
            "reason": reason,
            "reply": reply,
            "similarity": best_sim
        })

    auto_handled = sum(1 for r in results if r["decision"] == "auto_handle")
    escalated = sum(1 for r in results if r["decision"] == "escalate")

    return {
        "name": f"Simple Baseline (TF-IDF/Embedding Sim Cutoff >= {similarity_threshold})",
        "threshold": similarity_threshold,
        "auto_handle_rate": auto_handled / total if total > 0 else 0.0,
        "escalate_rate": escalated / total if total > 0 else 0.0,
        "auto_handled_count": auto_handled,
        "escalated_count": escalated,
        "total": total,
        "details": results
    }


if __name__ == "__main__":
    golden = pd.read_csv("data/golden_set.csv")

    trivial_res = evaluate_trivial_baseline(golden)
    print("\n=== TRIVIAL BASELINE ===")
    print(f"Auto-handle rate: {trivial_res['auto_handle_rate']:.2%}")
    print(f"Escalation rate: {trivial_res['escalate_rate']:.2%}")

    simple_res = evaluate_simple_baseline(golden, similarity_threshold=0.75)
    print("\n=== SIMPLE BASELINE (Similarity Cutoff >= 0.75) ===")
    print(f"Auto-handle rate: {simple_res['auto_handle_rate']:.2%}")
    print(f"Escalation rate: {simple_res['escalate_rate']:.2%}")
