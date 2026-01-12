import pandas as pd  # dataframe reader

# Read the saved simulated user history file 
hist = pd.read_csv("./artifacts/history_read_only.csv")

# Improve notebook display for long text
pd.set_option("display.max_colwidth", 200)   # show longer text per row
pd.set_option("display.width", 1200)         # wider output

# Print quick summary
print("[HISTORY] Loaded history_read_only.csv")
print("[HISTORY] history size:", len(hist))
print("[HISTORY] columns:", list(hist.columns))

# Display the full history table 
display(hist)
