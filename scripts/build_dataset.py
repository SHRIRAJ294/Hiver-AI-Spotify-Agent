#!/usr/bin/env python3
import sys
import os

# Add src to python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from load_data import load_raw_twcs, create_support_dataset


def main():
    print("Loading raw Twitter customer support dataset (twcs.csv)...")
    try:
        df = load_raw_twcs("data/twcs.csv")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print("Extracting Spotify customer-support pairs...")
    support_df = create_support_dataset(df, brand="SpotifyCares", limit=2000)

    output_path = "data/spotify_support_pairs.csv"
    support_df.to_csv(output_path, index=False)
    print(f"Successfully extracted {len(support_df)} support pairs -> {output_path}")


if __name__ == "__main__":
    main()
