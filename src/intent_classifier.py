import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

examples = pd.read_csv("data/intent_examples_expanded.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(examples["customer_message"])


def predict_intent(message):
    message_vector = vectorizer.transform([message])
    similarities = cosine_similarity(message_vector, X)
    best_index = similarities.argmax()
    best_score = float(similarities[0, best_index])
    best_intent = examples.iloc[best_index]["intent"]
    return best_intent, best_score


def evaluate_tfidf_classifier():
    golden = pd.read_csv("data/golden_set.csv")
    predictions = []
    scores = []

    for message in golden["customer_message"]:
        intent, score = predict_intent(message)
        predictions.append(intent)
        scores.append(score)

    golden["predicted_intent"] = predictions
    accuracy = accuracy_score(golden["intent"], golden["predicted_intent"])

    majority_intent = golden["intent"].value_counts().idxmax()
    majority_predictions = [majority_intent] * len(golden)
    majority_accuracy = accuracy_score(golden["intent"], majority_predictions)

    return {
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "golden_df": golden
    }


if __name__ == "__main__":
    golden = pd.read_csv("data/golden_set.csv")
    res = evaluate_tfidf_classifier()
    print("TF-IDF Classifier Accuracy:", res["accuracy"])
    print("\nClassification Report:")
    print(classification_report(golden["intent"], res["golden_df"]["predicted_intent"], zero_division=0))

    labels = sorted(golden["intent"].unique())
    cm = confusion_matrix(golden["intent"], res["golden_df"]["predicted_intent"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion Matrix:")
    print(cm_df)

    print("\nMajority Class Baseline:")
    print("Always predicting:", golden["intent"].value_counts().idxmax())
    print("Accuracy:", res["majority_accuracy"])