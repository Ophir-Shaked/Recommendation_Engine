# -*- coding: utf-8 -*-

"""
Main entry point for Honest Recommender v5.5-lite.
This script launches the pipeline and summarizes outputs.
"""
import os

# Default preset (equivalent to `%env RECO_PRESET=ACCURATE`)
os.environ.setdefault("RECO_PRESET", "ACCURATE")
from src.honestrec.runner import run_many
from src.honestrec.env_and_imports import (
    PRESET, GENSIM_AVAILABLE, SEEDS, HISTORY_PROFILES, RUN_ALL_HISTORIES, ART_DIR, EXPORT_TOP_N
)

def main():
    print(f"[ENV] RECO_PRESET={os.environ.get('RECO_PRESET', PRESET)}")
    print("[W2V] Using gensim Word2Vec" if GENSIM_AVAILABLE else "[W2V] Using LSA-term pseudo2vec (fallback)")
    run_many(SEEDS, HISTORY_PROFILES, RUN_ALL_HISTORIES)
    print("\nArtifacts saved under:", ART_DIR)
    print(f" - seed_*/: ROC/PR/Confusion + interactions_all.csv + top{EXPORT_TOP_N} files")
    print(" - multi_run_summary.csv")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"[SystemExit caught] code={e.code}")
    except Exception as e:
        import traceback as _tb
        print(f"[FATAL] {e}\n{_tb.format_exc()}")