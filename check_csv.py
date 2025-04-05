import pandas as pd

df = pd.read_csv("gesture_data.csv")
print("Number of columns:", len(df.columns) - 1)  # Excluding label
