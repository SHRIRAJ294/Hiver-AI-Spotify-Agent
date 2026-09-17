import os
import re
import pandas as pd


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_raw_twcs(file_path="data/twcs.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file '{file_path}' not found.\n"
            "To build the raw dataset from scratch:\n"
            "1. Download the Kaggle dataset 'Customer Support on Twitter' (thoughtvector/customer-support-on-twitter)\n"
            "2. Extract twcs.csv and place it at 'data/twcs.csv'\n"
            "3. Run 'python scripts/build_dataset.py'"
        )
    return pd.read_csv(file_path)


def get_tweet_by_id(df, tweet_id):
    tweet = df[df["tweet_id"] == tweet_id]
    if len(tweet) == 0:
        return None
    return tweet


def get_parent_tweet(df, tweet_id):
    tweet = get_tweet_by_id(df, tweet_id)
    if tweet is None:
        return None
    parent_id = tweet["in_response_to_tweet_id"].iloc[0]
    if pd.isna(parent_id):
        return None
    return get_tweet_by_id(df, int(parent_id))


def get_customer_message(df, spotify_tweet_id):
    spotify_tweet = get_tweet_by_id(df, spotify_tweet_id)
    if spotify_tweet is None:
        return None
    parent_id = spotify_tweet["in_response_to_tweet_id"].iloc[0]
    if pd.isna(parent_id):
        return None
    customer_tweet = get_tweet_by_id(df, int(parent_id))
    if customer_tweet is None:
        return None
    if customer_tweet["inbound"].iloc[0] != True:
        return None
    return customer_tweet


def create_support_dataset(df, brand="SpotifyCares", limit=2000):
    brand_tweets = df[
        (df["author_id"] == brand) &
        (df["inbound"] == False)
    ]
    pairs = []
    for _, tweet in brand_tweets.head(limit).iterrows():
        tweet_id = tweet["tweet_id"]
        customer = get_customer_message(df, tweet_id)
        if customer is None:
            continue
        pairs.append({
            "customer_message": customer["text"].iloc[0],
            "support_response": tweet["text"],
            "customer_tweet_id": customer["tweet_id"].iloc[0],
            "support_tweet_id": tweet_id
        })
    result_df = pd.DataFrame(pairs)
    if not result_df.empty:
        result_df["customer_message"] = result_df["customer_message"].apply(clean_text)
        result_df["support_response"] = result_df["support_response"].apply(clean_text)
        result_df = result_df[result_df["customer_message"].str.len() > 0].reset_index(drop=True)
    return result_df