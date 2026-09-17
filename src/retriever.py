import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"

support_df = pd.read_csv(
    "data/spotify_support_pairs_labeled.csv"
)

support_df = support_df.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

support_embeddings = model.encode(
    support_df["customer_message"].tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

def retrieve_similar_cases(message, top_k=5, exclude_message=None):

    message_embedding = model.encode(
        [message],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    similarities = np.dot(
        support_embeddings,
        message_embedding[0]
    )

    if exclude_message is not None:

        excluded_indices = (
            support_df["customer_message"] == exclude_message
        )

        similarities[excluded_indices.values] = -1

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = support_df.iloc[top_indices].copy()

    results["similarity"] = similarities[top_indices]

    best_similarity = similarities[top_indices[0]]

    return results, best_similarity
if __name__ == "__main__":

    test_messages = [
        "My Spotify app keeps crashing",
        "I was charged twice for my subscription",
        "Why is this song not available in India?",
        "I want to cancel my Premium subscription",
        "Can you add a lyrics translation feature?",
        "What is Spotify Wrapped?",
        "I forgot my password",
        "The music keeps buffering",
        "Do you guys support smart TVs?"
    ]

    def has_strong_evidence(best_similarity, threshold=0.80):
        return best_similarity >= threshold

    for test_message in test_messages:

        results, best_similarity = retrieve_similar_cases(
            test_message,
            top_k=3
        )

        print("\n" + "=" * 70)
        print("CUSTOMER:", test_message)
        print("BEST SIMILARITY:", best_similarity)

        if has_strong_evidence(best_similarity):
            print("DECISION: Strong historical evidence")
        else:
            print("DECISION: Weak historical evidence")

        print("\nTOP 3 RETRIEVED CASES:")

        for _, row in results.iterrows():

            print("\nCUSTOMER:", row["customer_message"])
            print("INTENT:", row["intent"])
            print("SIMILARITY:", row["similarity"])
            print("RESPONSE:", row["support_response"])