import pandas as pd

# Load the dataset
df = pd.read_csv('data/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.csv')

# --- Initial EDA ---
print("--- First 5 Rows ---")
print(df.head())
print("\\n--- DataFrame Info ---")
df.info()
print("\\n--- Descriptive Statistics ---")
print(df.describe())
