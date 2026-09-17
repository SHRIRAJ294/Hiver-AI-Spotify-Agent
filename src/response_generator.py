import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("Gemini client initialized.")


def generate_response(customer_message, retrieved_cases):

    context = ""

    for _, row in retrieved_cases.iterrows():

        context += f"""
Customer issue:
{row["customer_message"]}

Historical support response:
{row["support_response"]}

---
"""

    prompt = f"""
You are a customer support agent for Spotify.

A customer has sent this message:

{customer_message}

Below are historical Spotify customer-support conversations.
Use them as your primary source of guidance.

{context}

Write a helpful response to the customer.

Rules:
- Base your response on the historical support responses.
- Do not invent Spotify policies, refunds, features, or troubleshooting steps.
- If the historical cases do not provide enough information, ask the customer
  for the information needed to investigate the issue.
- Do not mention these historical conversations.
- Do not mention that you are an AI.
- Keep the response concise and natural, like a real Spotify support agent.
- Do not include an employee name or initials such as /JN.

Return only the response that should be sent to the customer.
"""

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "nodename nor servname" in err_str or "NameResolutionError" in err_str or "gaierror" in err_str:
                if len(retrieved_cases) > 0:
                    return retrieved_cases.iloc[0]["support_response"]
                return "Hey! Thanks for reaching out to Spotify Cares. Could you please send us a DM with more details so we can investigate?"
            print(f"Gemini response generation request failed (attempt {attempt + 1}/5): {err_str[:150]}...")
            if attempt == 4:
                if len(retrieved_cases) > 0:
                    return retrieved_cases.iloc[0]["support_response"]
                return "Hey! Thanks for reaching out to Spotify Cares. Could you please send us a DM with more details so we can investigate?"
            time.sleep(5 * (attempt + 1))


if __name__ == "__main__":

    from retriever import retrieve_similar_cases

    customer_message = "My Spotify app keeps crashing"

    retrieved_cases, _ = retrieve_similar_cases(
        customer_message,
        top_k=5
    )

    response = generate_response(
        customer_message,
        retrieved_cases
    )

    print("\nCUSTOMER:")
    print(customer_message)

    print("\nGENERATED RESPONSE:")
    print(response)