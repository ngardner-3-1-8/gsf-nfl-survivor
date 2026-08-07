"""
split_circa_historical.py

The per-year files (Circa_historical_data_{year}.csv) were all accidentally
overwritten with 2025 data. This rebuilds each year's file from the master
Circa_historical_data.csv, which contains all years.

Usage:
    python split_circa_historical.py
"""

import os
import pandas as pd

MASTER = "contest-historical-data/Circa_historical_data.csv"
OUT_DIR = "contest-historical-data"


def main():
    if not os.path.exists(MASTER):
        print(f"❌ Master file not found: {MASTER}")
        return

    df = pd.read_csv(MASTER)
    if "Year" not in df.columns:
        print("❌ Master file has no 'Year' column")
        return

    years = sorted(int(y) for y in df["Year"].dropna().unique())
    print(f"Master file: {len(df)} rows across years {years}")

    for year in years:
        year_df = df[df["Year"] == year].copy()
        out = os.path.join(OUT_DIR, f"Circa_historical_data_{year}.csv")

        # Safety: back up an existing file once before overwriting
        if os.path.exists(out) and not os.path.exists(out + ".bak"):
            os.rename(out, out + ".bak")
            print(f"   (backed up existing → {os.path.basename(out)}.bak)")

        year_df.to_csv(out, index=False)
        weeks = sorted(int(w) for w in year_df["Week"].dropna().unique())
        print(f"   ✅ {os.path.basename(out)}: {len(year_df)} rows, "
              f"weeks {weeks[0]}–{weeks[-1]} ({len(weeks)} weeks)")

    print("\n✅ Split complete. Re-run the backfill after committing these.")


if __name__ == "__main__":
    main()
