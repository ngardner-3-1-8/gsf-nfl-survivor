import pandas as pd
import glob
import os

for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    season_dir = f"nfl-power-ratings/final_data/{year}_final_data"
    if not os.path.exists(season_dir):
        continue
    
    # Find all weekly files in order
    weekly_files = sorted(
        glob.glob(f"{season_dir}/Week_*_{year}_Final_Data.csv"),
        key=lambda p: int(os.path.basename(p).split("_")[1])
    )
    
    if not weekly_files:
        print(f"{year}: no weekly files found")
        continue
    
    cumulative = pd.DataFrame()
    for wf in weekly_files:
        week_num = int(os.path.basename(wf).split("_")[1])
        week_df = pd.read_csv(wf)
        cumulative = pd.concat([cumulative, week_df], ignore_index=True)
        
        # Save cumulative season file through this week
        out = f"{season_dir}/Season_{year}_Through_Week_{week_num}_Final_Data.csv"
        cumulative.to_csv(out, index=False)
        print(f"{year} Week {week_num}: {len(cumulative)} cumulative rows → {out}")
