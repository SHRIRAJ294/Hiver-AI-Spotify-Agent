import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

examples = pd.read_csv("data/intent_examples_expanded.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

example_embeddings = model.encode(
    examples["customer_message"].tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

intent_embeddings = {}
for intent in examples["intent"].unique():
    intent_vectors = example_embeddings[examples["intent"] == intent]
    intent_embeddings[intent] = intent_vectors.mean(axis=0)

intent_names = list(intent_embeddings.keys())
centroid_matrix = np.array([intent_embeddings[i] for i in intent_names])  # (K, D)


def predict_intent(message):
    message_embedding = model.encode(
        [message],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores = np.dot(centroid_matrix, message_embedding[0])  # shape (K,)
    best_idx = np.argmax(scores)
    return intent_names[best_idx], float(scores[best_idx])


def predict_intents_batch(messages):
    embeddings = model.encode(
        messages,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")  # shape (N, D)

    sims = np.dot(embeddings, centroid_matrix.T)  # shape (N, K)
    best_indices = np.argmax(sims, axis=1)

    results = []
    for idx, best_i in enumerate(best_indices):
        results.append((intent_names[best_i], float(sims[idx, best_i])))
    return results


def evaluate_embedding_classifier():
    golden = pd.read_csv("data/golden_set.csv")
    predictions = [intent for intent, _ in predict_intents_batch(golden["customer_message"].tolist())]
    golden["predicted_intent"] = predictions
    accuracy = accuracy_score(golden["intent"], golden["predicted_intent"])

    return {
        "accuracy": accuracy,
        "golden_df": golden
    }


if __name__ == "__main__":
    test_message = "My Spotify app keeps crashing"
    intent, score = predict_intent(test_message)
    print("\nTest message:", test_message)
    print("Predicted intent:", intent)
    print("Confidence:", score)

    res = evaluate_embedding_classifier()
    print("\nEmbedding Classifier Accuracy:", res["accuracy"])
    print("\nClassification Report:")
    print(classification_report(res["golden_df"]["intent"], res["golden_df"]["predicted_intent"], zero_division=0))