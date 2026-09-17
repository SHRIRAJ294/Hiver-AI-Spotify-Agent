import os
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score


def evaluate_judge_human_agreement(csv_path="data/human_judge_agreement.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Human agreement evaluation file '{csv_path}' not found.")

    df = pd.read_csv(csv_path)

    human_labels = df["human_pass"].astype(int)
    llm_labels = df["llm_judge_pass"].astype(int)

    agreement_accuracy = accuracy_score(human_labels, llm_labels)
    kappa = cohen_kappa_score(human_labels, llm_labels)

    return {
        "total_samples": len(df),
        "agreement_accuracy": agreement_accuracy,
        "cohen_kappa": kappa,
        "human_pass_rate": human_labels.mean(),
        "llm_pass_rate": llm_labels.mean(),
        "df": df
    }


if __name__ == "__main__":
    res = evaluate_judge_human_agreement()
    print("\n=== LLM-as-Judge vs. Human Agreement ===")
    print(f"Total Sampled Replies Evaluated: {res['total_samples']}")
    print(f"Human Pass Rate: {res['human_pass_rate']:.1%}")
    print(f"LLM Judge Pass Rate: {res['llm_pass_rate']:.1%}")
    print(f"Raw Agreement Accuracy: {res['agreement_accuracy']:.1%}")
    print(f"Cohen's Kappa Score: {res['cohen_kappa']:.3f}")
