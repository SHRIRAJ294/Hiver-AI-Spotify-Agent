import pandas as pd

from retriever import retrieve_similar_cases


eval_messages = pd.read_csv(
    "data/retrieval_eval_messages.csv"
)

rows = []

for _, row in eval_messages.iterrows():

    message = row["customer_message"]

    results, _ = retrieve_similar_cases(
        message,
        top_k=5,
        exclude_message=message
    )

    result = {
        "customer_message": message
    }

    for rank, (_, case) in enumerate(
        results.iterrows(),
        start=1
    ):

        result[f"top{rank}_customer"] = case["customer_message"]
        result[f"top{rank}_response"] = case["support_response"]
        result[f"top{rank}_similarity"] = case["similarity"]

    rows.append(result)


output = pd.DataFrame(rows)

output.to_csv(
    "data/retrieval_labeling.csv",
    index=False
)

print("Prepared labeling file:", len(output))
