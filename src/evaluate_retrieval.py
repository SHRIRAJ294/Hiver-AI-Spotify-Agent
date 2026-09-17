import pandas as pd

from retriever import retrieve_similar_cases

eval_df = pd.read_csv(
    "data/retrieval_eval.csv"
)

print("Evaluation examples:", len(eval_df))

precision_correct = 0

for _, row in eval_df.iterrows():

    message = row["customer_message"]
    expected = row["retrieval_relevant"]

    results, best_similarity = retrieve_similar_cases(
        message,
        top_k=5
    )

    top_case = results.iloc[0]

    print("\n" + "=" * 70)
    print("CUSTOMER:", message)

    print("\nTOP 5 RESULTS:")

    for rank, (_, retrieved_case) in enumerate(
        results.iterrows(),
        start=1
    ):
        print(f"\nRANK {rank}")
        print("CUSTOMER:", retrieved_case["customer_message"])
        print("SIMILARITY:", retrieved_case["similarity"])
        print("RESPONSE:", retrieved_case["support_response"])

    if expected == 1:
        precision_correct += 1

precision_at_1 = precision_correct / len(eval_df)

recall_at_5 = (
    eval_df["top5_relevant"].sum()
    / len(eval_df)
)

print("\nPrecision@1:", precision_at_1)
print("Recall@5:", recall_at_5)

