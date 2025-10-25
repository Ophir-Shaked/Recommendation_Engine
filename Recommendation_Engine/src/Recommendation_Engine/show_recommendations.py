# -*- coding: utf-8 -*-  # Declares UTF-8 encoding to safely handle non-ASCII characters (e.g., names).

"""
show_recommendations.py
-------------------------------------------------
Utility to display personalized recommendation results for a given recipient.
Example: Hila Ronen — Test split, seed=3
"""

import os  # For file path operations and existence checks.
import pandas as pd  # For tabular data manipulation and display.
from .synthetic_corpus import build_big_corpus  # Function to rebuild the synthetic article dataset.

def show_hila_ronen(seed: int = 3):
    """
    Display Hila Ronen’s top-5 personalized recommendations and the full article details.

    Parameters
    ----------
    seed : int, optional
        Random seed used to generate the synthetic corpus and to locate the correct output folder.
    """
    # Path to the pre-generated Top-5 recommendation file.
    top5_path = f"reco_engine_artifacts_v5_5/seed_{seed}/test_top5_Hila_Ronen.txt"

    # Verify the file exists before trying to read it.
    if not os.path.exists(top5_path):
        print(f"[ERROR] File not found: {top5_path}")
        return

    # Nicely formatted console header.
    print(f"\n=== What we suggested to Hila Ronen (Test / seed={seed}) ===\n")
    print(" Hila Ronen — Personalized Recommendations (TEST split)")
    print("=" * 75)

    # Read and print the Top-5 recommendation text file.
    with open(top5_path, encoding="utf-8") as f:
        print(f.read())

    # Build the same synthetic article corpus used for training/evaluation.
    corpus, NOW = build_big_corpus(seed=seed)

    # Convert the list of Article objects into a pandas DataFrame for easier lookup.
    articles_df = pd.DataFrame([a.__dict__ for a in corpus])

    # Hard-coded Top-5 article IDs (these can later be parsed dynamically from the text file).
    top_ids = ["20156", "19556", "20145", "3698", "16548"]

    # Filter the DataFrame to include only those Top-5 articles.
    suggested = articles_df.loc[
        articles_df["article_id"].isin(top_ids),
        ["article_id", "title", "text", "category", "source"]
    ]

    print(" Full details of the 5 recommended articles:\n")

    try:
        display(suggested.reset_index(drop=True))
    except Exception:
        # Fallback to plain-text table output if IPython isn’t available.
        print(suggested.reset_index(drop=True).to_string(index=False))

if __name__ == "__main__":
    # Allow running the module directly from the command line:
    #   python -m src.recommendation_engine.show_recommendations
    show_hila_ronen(seed=3)
