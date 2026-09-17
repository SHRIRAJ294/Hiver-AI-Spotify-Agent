import pandas as pd

df = pd.read_csv(
    "data/spotify_support_pairs_labeled.csv"
)

df = df.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)

sample = df.sample(
    40,
    random_state=42
)

sample[
    ["customer_message"]
].to_csv(
    "data/retrieval_eval_messages.csv",
    index=False
)

print("Evaluation messages:", len(sample))