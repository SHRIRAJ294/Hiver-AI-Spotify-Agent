import os
import json
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def check_evidence(customer_message, retrieved_cases):

    context = ""

    for _, row in retrieved_cases.iterrows():

        context += f"""
Customer issue:
{row["customer_message"]}

Historical support response:
{row["support_response"]}

Similarity:
{row["similarity"]}

---
"""

    prompt = f"""
You are evaluating whether a customer-support system has enough
historical evidence to safely answer a customer's message.

Customer message:
{customer_message}

Historical support cases:
{context}

Decide whether the historical cases provide enough information
to give a useful and grounded response.

Return ONLY valid JSON in this format:

{{
    "sufficient": true,
    "reason": "brief explanation"
}}

Rules:
- sufficient=true only when the historical cases directly address
  the customer's issue or provide a clearly applicable response.
- sufficient=false when the cases are only loosely related.
- Do not use general knowledge.
- Do not invent Spotify policies or information.
- Pay attention to whether the historical responses actually answer
  the customer's question.
"""

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            err_str = str(e)
            if "nodename nor servname" in err_str or "NameResolutionError" in err_str or "gaierror" in err_str:
                max_sim = retrieved_cases["similarity"].max() if "similarity" in retrieved_cases else 0.8
                return {
                    "sufficient": bool(max_sim >= 0.75),
                    "reason": f"Retrieval evidence check (Max similarity: {max_sim:.2f})"
                }
            print(f"Gemini evidence check request failed (attempt {attempt + 1}/5): {err_str[:150]}...")
            if attempt == 4:
                max_sim = retrieved_cases["similarity"].max() if "similarity" in retrieved_cases else 0.8
                return {
                    "sufficient": bool(max_sim >= 0.75),
                    "reason": f"Fallback decision (Max similarity: {max_sim:.2f})"
                }
            time.sleep(5 * (attempt + 1))


if __name__ == "__main__":

    from retriever import retrieve_similar_cases

    test_messages = [
        "My Spotify app keeps crashing",
        "Can you add a lyrics translation feature?",
        "What is Spotify Wrapped?"
    ]

    for message in test_messages:

        results, _ = retrieve_similar_cases(
            message,
            top_k=5
        )

        result = check_evidence(
            message,
            results
        )

        print("\n" + "=" * 60)
        print("CUSTOMER:", message)
        print("SUFFICIENT:", result["sufficient"])
        print("REASON:", result["reason"])