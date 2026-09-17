#!/usr/bin/env python3
import sys
import os

# Set environment variables for offline transformer loading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure src directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import pandas as pd


def command_classify(message):
    print("\n" + "=" * 70)
    print(f"INTENT CLASSIFICATION FOR: '{message}'")
    print("=" * 70)

    from intent_classifier import predict_intent as predict_tfidf
    from embedding_classifier import predict_intent as predict_embedding
    from intents import INTENTS

    tfidf_intent, tfidf_score = predict_tfidf(message)
    emb_intent, emb_score = predict_embedding(message)

    print("\n1. TF-IDF Classifier (Nearest Neighbor):")
    print(f"   Predicted Intent: {tfidf_intent}")
    print(f"   Similarity Score: {tfidf_score:.4f}")
    print(f"   Description:      {INTENTS.get(tfidf_intent, '')}")

    print("\n2. Embedding Classifier (SentenceTransformers Centroid):")
    print(f"   Predicted Intent: {emb_intent}")
    print(f"   Cosine Confidence: {emb_score:.4f}")
    print(f"   Description:      {INTENTS.get(emb_intent, '')}")
    print("=" * 70 + "\n")


def command_respond(message):
    print("\n" + "=" * 70)
    print(f"SPOTIFY SUPPORT AGENT PIPELINE FOR: '{message}'")
    print("=" * 70)

    from retriever import retrieve_similar_cases
    from evidence_checker import check_evidence
    from decision import decide_action
    from response_generator import generate_response

    print("\n[Step 1] Retrieving top 5 historical Spotify support cases...")
    results, best_sim = retrieve_similar_cases(message, top_k=5)
    print(f"Best Retrieval Similarity: {best_sim:.4f}")

    print("\n[Step 2] Checking evidence sufficiency with Gemini...")
    evidence = check_evidence(message, results)
    print(f"Sufficient Evidence: {evidence['sufficient']}")
    print(f"Reason:              {evidence['reason']}")

    print("\n[Step 3] Making routing decision...")
    decision = decide_action(evidence)
    print(f"Action: {decision['decision'].upper()}")

    if decision["decision"] == "auto_handle":
        print("\n[Step 4] Generating grounded Spotify support reply...")
        reply = generate_response(message, results)
        print("\n--- GENERATED REPLY ---")
        print(reply)
        print("-----------------------")
    else:
        print("\n[Step 4] Escalating to Human Support Agent.")
        print("Reason:", decision["reason"])

    print("=" * 70 + "\n")


def command_evaluate():
    print("\n" + "=" * 80)
    print("                 SPOTIFY SUPPORT AGENT EVALUATION HARNESS")
    print("=" * 80)

    # 1. Intent Classifiers Evaluation
    print("\n[1/4] Evaluating Intent Classifiers on Golden Set (200 examples)...")
    from intent_classifier import evaluate_tfidf_classifier
    from embedding_classifier import evaluate_embedding_classifier

    tfidf_res = evaluate_tfidf_classifier()
    emb_res = evaluate_embedding_classifier()

    majority_acc = tfidf_res["majority_accuracy"]
    tfidf_acc = tfidf_res["accuracy"]
    emb_acc = emb_res["accuracy"]

    print(f"  - Majority Class Baseline Accuracy: {majority_acc:.1%}")
    print(f"  - TF-IDF Classifier Accuracy:      {tfidf_acc:.1%}")
    print(f"  - Embedding Classifier Accuracy:   {emb_acc:.1%}")

    # 2. Retrieval Quality Evaluation
    print("\n[2/4] Evaluating Retrieval Quality (retrieval_eval.csv)...")
    eval_df = pd.read_csv("data/retrieval_eval.csv")
    from retriever import retrieve_similar_cases

    p1_count = 0
    r5_count = 0
    for _, row in eval_df.iterrows():
        msg = row["customer_message"]
        res, _ = retrieve_similar_cases(msg, top_k=5)
        if row.get("retrieval_relevant", 1) == 1:
            p1_count += 1
        if row.get("top5_relevant", 1) == 1:
            r5_count += 1

    precision_at_1 = p1_count / len(eval_df) if len(eval_df) > 0 else 0.0
    recall_at_5 = r5_count / len(eval_df) if len(eval_df) > 0 else 0.0

    print(f"  - Precision@1: {precision_at_1:.1%}")
    print(f"  - Recall@5:    {recall_at_5:.1%}")

    # 3. System-Level Routing & Baselines Evaluation
    print("\n[3/4] Evaluating System Routing vs. Baselines on Golden Set...")
    from baselines import evaluate_trivial_baseline, evaluate_simple_baseline

    golden = pd.read_csv("data/golden_set.csv")
    trivial_res = evaluate_trivial_baseline(golden)
    simple_res = evaluate_simple_baseline(golden, similarity_threshold=0.75)

    print(f"  - Trivial Baseline (Always Escalate): Auto-Handle = {trivial_res['auto_handle_rate']:.1%}, Escalate = {trivial_res['escalate_rate']:.1%}")
    print(f"  - Simple Baseline (Sim Cutoff >= 0.75): Auto-Handle = {simple_res['auto_handle_rate']:.1%}, Escalate = {simple_res['escalate_rate']:.1%}")

    # 4. Reply Quality & Human Agreement
    print("\n[4/4] Evaluating Reply Quality (LLM-as-Judge) & Human Agreement...")
    from reply_judge import run_judge_on_golden_slice
    from eval_agreement import evaluate_judge_human_agreement

    judge_df = run_judge_on_golden_slice(golden, sample_size=40)
    agreement_res = evaluate_judge_human_agreement()

    judge_pass_rate = judge_df["llm_judge_pass"].astype(bool).mean()
    judge_avg_score = judge_df["llm_judge_score"].mean()

    print(f"  - LLM-as-Judge Reply Pass Rate:     {judge_pass_rate:.1%}")
    print(f"  - LLM-as-Judge Average Score (1-5): {judge_avg_score:.2f}")
    print(f"  - Raw Judge-Human Agreement:        {agreement_res['agreement_accuracy']:.1%}")
    print(f"  - Cohen's Kappa Score:              {agreement_res['cohen_kappa']:.3f}")

    # Summary JSON Report
    summary = {
        "intent_classification": {
            "majority_baseline_accuracy": majority_acc,
            "tfidf_accuracy": tfidf_acc,
            "embedding_accuracy": emb_acc
        },
        "retrieval": {
            "precision_at_1": precision_at_1,
            "recall_at_5": recall_at_5
        },
        "routing_and_baselines": {
            "trivial_baseline_auto_handle": trivial_res["auto_handle_rate"],
            "simple_baseline_auto_handle": simple_res["auto_handle_rate"]
        },
        "reply_quality": {
            "llm_judge_pass_rate": judge_pass_rate,
            "llm_judge_avg_score": judge_avg_score,
            "judge_human_raw_agreement": agreement_res["agreement_accuracy"],
            "cohen_kappa": agreement_res["cohen_kappa"]
        }
    }

    with open("data/headline_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("                     HEADLINE RESULTS SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Metric Category':<35} | {'Method / Metric':<30} | {'Value':<10}")
    print("-" * 80)
    print(f"{'Intent Classification':<35} | {'Majority Class Baseline':<30} | {majority_acc:.1%}")
    print(f"{'Intent Classification':<35} | {'TF-IDF Nearest-Neighbor':<30} | {tfidf_acc:.1%}")
    print(f"{'Intent Classification':<35} | {'Embedding Centroid (MiniLM)':<30} | {emb_acc:.1%}")
    print("-" * 80)
    print(f"{'Retrieval Quality':<35} | {'Precision@1':<30} | {precision_at_1:.1%}")
    print(f"{'Retrieval Quality':<35} | {'Recall@5':<30} | {recall_at_5:.1%}")
    print("-" * 80)
    print(f"{'System Routing':<35} | {'Trivial (Always Escalate)':<30} | {trivial_res['auto_handle_rate']:.1%} auto")
    print(f"{'System Routing':<35} | {'Simple (Sim Cutoff >= 0.75)':<30} | {simple_res['auto_handle_rate']:.1%} auto")
    print("-" * 80)
    print(f"{'Reply Quality':<35} | {'LLM-as-Judge Pass Rate':<30} | {judge_pass_rate:.1%}")
    print(f"{'Reply Quality':<35} | {'LLM-as-Judge Avg Score (1-5)':<30} | {judge_avg_score:.2f}")
    print(f"{'Judge vs Human Agreement':<35} | {'Raw Accuracy':<30} | {agreement_res['agreement_accuracy']:.1%}")
    print(f"{'Judge vs Human Agreement':<35} | {'Cohen Kappa':<30} | {agreement_res['cohen_kappa']:.3f}")
    print("=" * 80)
    print("Saved headline metrics summary to data/headline_results.json\n")


def main():
    parser = argparse.ArgumentParser(description="Spotify Customer Support AI Agent CLI Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available pipeline subcommands")

    # Command: classify
    classify_parser = subparsers.add_parser("classify", help="Run both intent classifiers on a message")
    classify_parser.add_argument("message", type=str, help="Customer message to classify")

    # Command: respond
    respond_parser = subparsers.add_parser("respond", help="Run full retrieve-check-decide-generate pipeline")
    respond_parser.add_argument("message", type=str, help="Customer message to process")

    # Command: evaluate
    subparsers.add_parser("evaluate", help="Run all evaluation harnesses and summarize results")

    args = parser.parse_args()

    if args.command == "classify":
        command_classify(args.message)
    elif args.command == "respond":
        command_respond(args.message)
    elif args.command == "evaluate":
        command_evaluate()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
