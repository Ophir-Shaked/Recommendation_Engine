#  View Top-K recommendations

import pandas as pd  # dataframe reader

# Read the saved recommendations file
df = pd.read_csv("./artifacts/recommendations_topk.csv")

# Make sure long text is not truncated in display
pd.set_option("display.max_colwidth", 200)   # show longer text per row
pd.set_option("display.width", 1200)         # wider output

print("[TOPK] Loaded recommendations_topk.csv")
print("[TOPK] shape:", df.shape)

# Show first 20 recommended documents
print("\n[TOPK] First 20 recommendations:")
display(df.head(20))
