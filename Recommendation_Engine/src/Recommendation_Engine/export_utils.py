# -*- coding: utf-8 -*-  # Ensures UTF-8 encoding for safe Unicode text handling.

import os, csv  # os for paths/directories; csv for structured output export.
import pandas as pd  # Used for handling DataFrame outputs of ranked recommendations.
from typing import Optional  # Type hinting for optional parameters.
from .env_and_imports import RECIPIENT_NAME  # Default user name for export files.

def _write_top_k_for_recipient(rec, agg_df: pd.DataFrame, k: int, out_dir: str,
                               recipient: str = RECIPIENT_NAME, split_name: str = "test") -> str:
    """Saves a compact CSV and a TXT with titles for recipient consumption."""
    # Create output directory if it doesn’t exist.
    os.makedirs(out_dir, exist_ok=True)

    # Select the top-K ranked items.
    topk = agg_df.head(k).copy()

    # Build export filenames (TXT for human reading, CSV for structured logs).
    txt_path = os.path.join(out_dir, f"{split_name}_top{int(k)}_{recipient.replace(' ', '_')}.txt")
    csv_path = os.path.join(out_dir, f"{split_name}_top{int(k)}_{recipient.replace(' ', '_')}.csv")

    # -------------------- Write human-readable TXT --------------------
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Top {k} recommendations for {recipient} ({split_name} split)\n")
        f.write("=" * 64 + "\n\n")
        # Iterate over top-K rows and print each article with metadata.
        for r, row in enumerate(topk.itertuples(index=False), start=1):
            aid   = getattr(row, "article_id")  # Article ID field from DataFrame.
            score = float(getattr(row, "score"))  # Recommendation score.
            a     = rec.articles[rec.id2idx[str(aid)]]  # Retrieve full article object from recommender.
            f.write(f"{r}. [{a.category}] {a.title}  |  score={score:.6f}\n")  # Rank, category, title, and score.

    # -------------------- Write structured CSV --------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank","article_id","title","category","score"])  # Header row.
        # Write one row per article with consistent ordering.
        for r, row in enumerate(topk.itertuples(index=False), start=1):
            aid   = getattr(row, "article_id")
            score = float(getattr(row, "score"))
            a     = rec.articles[rec.id2idx[str(aid)]]
            w.writerow([r, a.article_id, a.title, a.category, f"{score:.6f}"])

    # Return TXT path for downstream logging or UI linking.
    return txt_path
