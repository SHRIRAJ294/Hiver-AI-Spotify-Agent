import pandas as pd
from intents import INTENTS

df = pd.read_csv("data/spotify_support_pairs.csv")
golden = pd.read_csv("data/golden_set.csv")

# Remove golden-set examples
df = df[~df["customer_message"].isin(golden["customer_message"])]

# Find examples likely related to weaker intents
keywords = [
    "premium", "subscription", "subscribe", "upgrade", "cancel", "charge",
    "charged", "payment", "refund", "card", "billing", "password", "login",
    "log in", "sign in", "account", "can't access", "play", "playing",
    "playback", "sound", "audio", "volume", "shuffle", "feature", "add",
    "remove", "bring back", "option"
]

mask = df["customer_message"].str.lower().str.contains("|".join(keywords), na=False)
target_df = df[mask]
print("Potential targeted examples:", len(target_df))

sample = target_df.sample(min(50, len(target_df)), random_state=123).reset_index(drop=True)
intents = list(INTENTS.keys())

labeled_rows = []

for i, row in sample.iterrows():

    print("\n" + "=" * 80)
    print("Example", i + 1, "of", len(sample))
    print("CUSTOMER:", row["customer_message"])
    print("SUPPORT:", row["support_response"])

    print("\nChoose intent:")

    for number, intent in enumerate(intents, start=1):
        print(f"{number}. {intent}")

    while True:
        choice = input("\nEnter number (1-9): ")

        if choice.isdigit() and 1 <= int(choice) <= 9:
            intent = intents[int(choice) - 1]
            break

        print("Please enter a number from 1 to 9.")

    labeled_rows.append({
        "customer_message": row["customer_message"],
        "intent": intent
    })

    print("Saved as:", intent)


labeled_df = pd.DataFrame(labeled_rows)

output_file = "data/new_intent_examples.csv"

# Append to existing labels if the file already exists
try:
    existing = pd.read_csv(output_file)
    labeled_df = pd.concat(
        [existing, labeled_df],
        ignore_index=True
    )
except FileNotFoundError:
    pass

# Remove duplicate customer messages
labeled_df = labeled_df.drop_duplicates(
    subset=["customer_message"]
)

labeled_df.to_csv(
    output_file,
    index=False
)

print("\nDone!")
print("Total saved examples:", len(labeled_df))