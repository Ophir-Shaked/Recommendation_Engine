import pandas as pd  # Import pandas (commonly used for viewing or analyzing recommendation results).

# Path to her personalized recommendation file (TEST split)
top5_path = "reco_engine_artifacts_v5_5/seed_3/test_top5_Hila_Ronen.txt"  # Updated folder name to match new project identity.

# Read and print the Top-5 recommendations
print(" Hila Ronen — Personalized Recommendations (TEST split, seed=3)")  # Descriptive heading for the printed output.
print("=" * 75)  # Separator line for clarity and readability in console output.

# Open the Top-5 recommendation file in UTF-8 encoding (to handle Hebrew or special characters safely).
with open(top5_path, encoding="utf-8") as f:
    print(f.read())  # Print the entire contents of the file to the console.