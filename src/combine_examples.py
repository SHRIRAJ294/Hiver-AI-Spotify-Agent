import pandas as pd

# Load existing intent example files
first = pd.read_csv("data/intent_examples.csv")
second = pd.read_csv("data/new_intent_examples.csv")
golden = pd.read_csv("data/golden_set.csv")

# Combine training examples
combined = pd.concat([first, second], ignore_index=True)
combined = combined.drop_duplicates(subset=["customer_message"])

# Strict exclusion of any golden set messages to prevent data leakage
combined = combined[~combined["customer_message"].isin(golden["customer_message"])].reset_index(drop=True)

combined.to_csv("data/intent_examples_expanded.csv", index=False)

print("Total expanded training examples:", len(combined))
print("\nDistribution:")
print(combined["intent"].value_counts())