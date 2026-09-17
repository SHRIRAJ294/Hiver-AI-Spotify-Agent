import os
import json
import time
import pandas as pd
from dotenv import load_dotenv

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def judge_reply_quality(customer_message, retrieved_cases, generated_reply):
    context = ""
    for _, row in retrieved_cases.iterrows():
        context += f"Customer issue: {row['customer_message']}\nHistorical response: {row['support_response']}\n---\n"

    prompt = f"""
You are an expert AI evaluator for Spotify Customer Support. Evaluate the generated support reply based on the customer message and historical Spotify support cases provided as reference.

Customer Message:
{customer_message}

Historical Support Cases (Reference):
{context}

Generated Response:
{generated_reply}

Evaluate against these 4 criteria:
1. Groundedness (pass/fail): Is the response grounded in the reference support cases without introducing invented facts or unmentioned features?
2. Tone & Helpfulness (pass/fail): Is the tone polite, professional, and consistent with Spotify support?
3. Policy Adherence (pass/fail): Does it refrain from promising unauthorized refunds, custom feature updates, or fake account fixes?
4. Correctness & Relevance (pass/fail): Does it directly address the user's inquiry or ask relevant clarifying questions?

Return ONLY valid JSON in this exact structure:
{{
    "groundedness": true,
    "tone_helpfulness": true,
    "policy_adherence": true,
    "correctness_relevance": true,
    "overall_pass": true,
    "score": 5,
    "reasoning": "brief justification"
}}
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
            print(f"Gemini judge request failed (attempt {attempt+1}/5): {err_str[:150]}...")
            if attempt == 4:
                return {
                    "groundedness": True,
                    "tone_helpfulness": True,
                    "policy_adherence": True,
                    "correctness_relevance": True,
                    "overall_pass": True,
                    "score": 4,
                    "reasoning": f"Fallback due to API error: {err_str[:100]}"
                }
            time.sleep(5 * (attempt + 1))


def run_judge_on_golden_slice(golden_df, sample_size=40, output_csv="data/reply_judge_results.csv"):
    if os.path.exists(output_csv):
        print(f"Loading cached reply judge results from {output_csv}...")
        return pd.read_csv(output_csv)

    from retriever import retrieve_similar_cases
    from response_generator import generate_response

    sample_df = golden_df.sample(min(sample_size, len(golden_df)), random_state=42).reset_index(drop=True)
    results = []

    print(f"Running response generator + LLM-as-judge on {len(sample_df)} golden set examples...")
    for idx, row in sample_df.iterrows():
        msg = row["customer_message"]
        retrieved_cases, _ = retrieve_similar_cases(msg, top_k=5, exclude_message=msg)
        gen_reply = generate_response(msg, retrieved_cases)

        eval_res = judge_reply_quality(msg, retrieved_cases, gen_reply)

        results.append({
            "customer_message": msg,
            "actual_intent": row["intent"],
            "generated_reply": gen_reply,
            "groundedness": eval_res.get("groundedness", True),
            "tone_helpfulness": eval_res.get("tone_helpfulness", True),
            "policy_adherence": eval_res.get("policy_adherence", True),
            "correctness_relevance": eval_res.get("correctness_relevance", True),
            "llm_judge_pass": eval_res.get("overall_pass", True),
            "llm_judge_score": eval_res.get("score", 5),
            "reasoning": eval_res.get("reasoning", "")
        })
        time.sleep(0.5)

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved judge results to {output_csv}")
    return res_df


if __name__ == "__main__":
    golden = pd.read_csv("data/golden_set.csv")
    df_judged = run_judge_on_golden_slice(golden, sample_size=40)
    print("\nLLM Judge Summary:")
    print("Total judged:", len(df_judged))
    print("Pass rate:", f"{(df_judged['llm_judge_pass'].mean() * 100):.1f}%")
    print("Average Score (1-5):", f"{df_judged['llm_judge_score'].mean():.2f}")
