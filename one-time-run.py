import pandas as pd
import glob
import os

df = pd.read_csv(f"nfl-power-ratings/final_sim_results_with_variance_week_1_2026.csv")
df['Day of Week'] = pd.to_datetime(df['Date_x']).dt.day_name()

df.to_csv(f"nfl-power-ratings/final_sim_results_with_variance_week_1_2026.csv", index=False)
