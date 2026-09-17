import pandas as pd

support_df = pd.read_csv("data/spotify_support_pairs.csv")
labeled_df = pd.read_csv("data/new_intent_examples.csv")

print("Support pairs:", len(support_df))
print("Labeled examples:", len(labeled_df))
# Match manually labeled examples to historical support cases

intent_lookup = dict(
    zip(
        labeled_df["customer_message"],
        labeled_df["intent"]
    )
)

support_df["intent"] = support_df["customer_message"].map(
    intent_lookup
)

print("\nIntent labels assigned:")
print(support_df["intent"].notna().sum())

print("\nUnlabeled cases:")
print(support_df["intent"].isna().sum())

print("\nIntent distribution:")
print(support_df["intent"].value_counts())

support_df.to_csv(
    "data/spotify_support_pairs_labeled.csv",
    index=False
)

print("\nSaved labeled support dataset.")