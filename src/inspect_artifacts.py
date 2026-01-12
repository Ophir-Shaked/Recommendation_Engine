# Inspect saved artifacts 

import os                       # file and folder handling
import json                     # read json outputs
import pandas as pd             # read csv outputs as tables

from IPython.display import display  # pretty display inside notebook

ART_DIR = "./artifacts"         # folder where we saved all outputs


def fmt_bytes(n: int) -> str:
    """
    Convert raw size in bytes into a readable string.
    Example: 12345 -> 12.1KB
    """
    n = float(n)                # make sure we can divide safely
    for unit in ["B", "KB", "MB", "GB"]:   # iterate units
        if n < 1024:            # stop once size is small enough
            return f"{n:.1f}{unit}"
        n /= 1024               # convert to next unit
    return f"{n:.1f}TB"         # fallback for huge files


def show_file_info(path: str, max_head_rows: int = 5) -> None:
    """
    Inspect a single file:
    If CSV: show shape, columns, head
    If JSON: show type, keys, preview
    If PNG: show image inline
    """
    if not os.path.exists(path):                # check file exists
        print(f"[MISSING] {path}")              # if not, warn user
        return                                  # stop here

    size = os.path.getsize(path)                # file size in bytes
    print(f"\n[FOUND] {path} | size={fmt_bytes(size)}")  # print summary

    low = path.lower()                          # lowercase suffix check

    if low.endswith(".csv"):                    # handle CSV case
        df = pd.read_csv(path)                  # read csv
        print(f"[CSV] shape: {df.shape[0]} rows × {df.shape[1]} cols")  # print shape
        print("[CSV] columns:", list(df.columns))  # print columns
        print("[CSV] preview:")                 # announce preview
        display(df.head(max_head_rows))         # show first rows nicely

    elif low.endswith(".json"):                 # handle JSON case
        with open(path, "r", encoding="utf-8") as f:  # open json file
            obj = json.load(f)                  # parse json

        print("[JSON] object type:", type(obj)) # print json type

        if isinstance(obj, dict):               # if json is a dict
            print("[JSON] keys:", list(obj.keys()))  # show keys

        print("[JSON] preview:")                # announce preview
        s = json.dumps(obj, ensure_ascii=False, indent=2)  # pretty json string
        print(s[:2000] + ("\n... (truncated)" if len(s) > 2000 else ""))  # limit output

    elif low.endswith(".png"):                  # handle image case
        try:
            from PIL import Image               # image open library
        except Exception:
            import sys, subprocess              # used to install pillow if missing
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pillow"])  # install pillow
            from PIL import Image               # import after install

        img = Image.open(path)                  # open image file
        print(f"[PNG] image size: {img.size[0]}×{img.size[1]}")  # print dimensions
        display(img)                            # show inside notebook

    else:
        print("[INFO] No preview handler for this file type.")  # unsupported extension


print("[ARTIFACTS] Listing files in artifacts directory")

if not os.path.exists(ART_DIR):                 # check artifacts folder exists
    print("[MISSING] artifacts folder not found:", ART_DIR)  # folder missing
else:
    for fn in sorted(os.listdir(ART_DIR)):      # loop through filenames
        p = os.path.join(ART_DIR, fn)           # build full path
        print("file:", fn, "| size:", fmt_bytes(os.path.getsize(p)))  # print without dashes

targets = [
    "results_folds.csv",                        # fold level results
    "results_summary.csv",                      # mean std summary per rep model
    "winner.json",                              # winner metadata
    "winner_oof_roc.png",                       # ROC curve plot
    "winner_oof_pr.png",                        # PR curve plot
    "winner_oof_calibration.png",               # calibration plot
    "winner_oof_calibration_bins.csv",          # calibration bins table
    "winner_oof_false_positives.csv",           # false positive examples
    "winner_oof_false_negatives.csv",           # false negative examples
    "winner_oof_most_uncertain.csv",            # uncertain examples
    "recommendations_topk.csv",                 # top k recommended docs
    "recommendations_history_idx.json",         # indices used as reading history
    "history_read_only.csv",                    # readable history docs
]

print("\n[ARTIFACTS] Detailed preview for expected outputs")

for t in targets:                               # go over expected output files
    full_path = os.path.join(ART_DIR, t)         # full path for each artifact
    show_file_info(full_path)                    # inspect it
