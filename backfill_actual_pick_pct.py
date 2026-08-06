"""
backfill_actual_pick_pct.py

One-time backfill: adds the ACTUAL Circa pick % (and a recomputed ACTUAL EV)
to historical Final_Data files that were generated before weekly_1 started
writing those columns.

For each (year, week) it:
  1. Reads the weekly file  Week_{w}_{year}_Final_Data.csv
  2. Looks up actual pick% from  Circa_historical_data_{year}.csv
  3. Writes "Home/Away Actual Pick %" (matching weekly_1's logic exactly)
  4. Recomputes "Home/Away Actual EV" from those actual pick %s
  5. Rewrites the weekly file, then rebuilds the cumulative
     Season_{year}_Through_Week_{w}_Final_Data.csv by concatenating weeks 1..w

Model columns ("Home/Away Pick %", the model EV columns) are left untouched,
so the Optimizer's model-vs-actual toggle has genuinely different inputs.

Usage:
    python backfill_actual_pick_pct.py 2024              # all weeks found for 2024
    python backfill_actual_pick_pct.py 2024 10           # just week 10
    python backfill_actual_pick_pct.py 2022 2023 2024    # several years, all weeks
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

# Same probability column the EV math uses. Adjust if your files use a
# different consensus win-prob column name.
HOME_PROB_COL_CANDIDATES = ["Consensus Home Win Pct", "Home Win %", "Home Win Pct"]
AWAY_PROB_COL_CANDIDATES = ["Consensus Away Win Pct", "Away Win %", "Away Win Pct"]

# Full team name → abbreviation. Circa historical data is keyed by abbreviation;
# Final_Data files are keyed by full team name.
FULL_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS", "Washington Football Team": "WAS",
}
# Circa data sometimes uses these variants
ABBR_ALIASES = {"LA": "LAR", "WSH": "WAS", "JAC": "JAX", "GNB": "GB", "KAN": "KC",
                "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV"}


def _first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _abbr(name):
    a = FULL_TO_ABBR.get(name)
    if a:
        return a
    up = str(name).strip().upper()
    return ABBR_ALIASES.get(up, up)


def load_actual_pick_map(year, week):
    """team abbr → actual pick% for a given week, from Circa historical data."""
    hist_file = f"contest-historical-data/Circa_historical_data_{year}.csv"
    if not os.path.exists(hist_file):
        print(f"   ⚠️  {hist_file} not found — cannot backfill {year}")
        return None
    hist = pd.read_csv(hist_file)
    wk = hist[(hist["Year"] == year) & (hist["Week"].astype(int) == week)]
    if wk.empty:
        print(f"   ⚠️  No {year} week {week} rows in {hist_file}")
        return None
    m = {}
    for _, r in wk.iterrows():
        team = str(r.get("Team", "")).strip()
        pct = r.get("Pick %")
        if team and pd.notna(pct):
            # Pick % may be stored as a string like "12.3%" or a float
            try:
                val = float(str(pct).replace("%", "")) 
                if val > 1.5:   # looks like a percentage, not a fraction
                    val /= 100.0
            except ValueError:
                continue
            m[ABBR_ALIASES.get(team.upper(), team.upper())] = val
    return m if m else None


def recompute_actual_ev(df, home_prob_col, away_prob_col):
    """
    Actual EV per team = win_prob / expected_survivors, where
    expected_survivors = Σ(win_prob × ACTUAL pick%) over eligible teams.
    Computed per week. Mirrors weekly_1's EV logic exactly.
    """
    df = df.copy()
    df["Home Actual EV"] = 0.0
    df["Away Actual EV"] = 0.0

    week_col = "Week_x" if "Week_x" in df.columns else "Week"
    for wk, grp in df.groupby(week_col):
        hp = grp[home_prob_col].fillna(0).values
        ap = grp[away_prob_col].fillna(0).values
        hpk = grp["Home Actual Pick %"].fillna(0).values
        apk = grp["Away Actual Pick %"].fillna(0).values

        exp_surv = (np.sum(hp[hpk > 0] * hpk[hpk > 0]) +
                    np.sum(ap[apk > 0] * apk[apk > 0]))

        for idx, hpi, api, hpki, apki in zip(grp.index, hp, ap, hpk, apk):
            if exp_surv > 0:
                df.at[idx, "Home Actual EV"] = hpi / exp_surv if hpki > 0 else 0.0
                df.at[idx, "Away Actual EV"] = api / exp_surv if apki > 0 else 0.0
    return df


def backfill_week_file(year, week):
    wpath = (f"nfl-power-ratings/final_data/{year}_final_data/"
             f"Week_{week}_{year}_Final_Data.csv")
    if not os.path.exists(wpath):
        return False
    df = pd.read_csv(wpath)

    pick_map = load_actual_pick_map(year, week)
    if pick_map is None:
        return False

    if "Home Actual Pick %" not in df.columns:
        df["Home Actual Pick %"] = np.nan
    if "Away Actual Pick %" not in df.columns:
        df["Away Actual Pick %"] = np.nan

    def lookup(full):
        return pick_map.get(_abbr(full), np.nan)

    home_pcts = df["Home Team"].apply(lookup)
    away_pcts = df["Away Team"].apply(lookup)
    df.loc[home_pcts.notna(), "Home Actual Pick %"] = home_pcts[home_pcts.notna()]
    df.loc[away_pcts.notna(), "Away Actual Pick %"] = away_pcts[away_pcts.notna()]

    hcol = _first_present(df, HOME_PROB_COL_CANDIDATES)
    acol = _first_present(df, AWAY_PROB_COL_CANDIDATES)
    if hcol and acol:
        df = recompute_actual_ev(df, hcol, acol)
    else:
        print(f"   ⚠️  No win-prob column in Week {week} — skipping actual EV")

    df.to_csv(wpath, index=False)
    n = home_pcts.notna().sum() + away_pcts.notna().sum()
    print(f"   ✅ Week {week}: actual pick% for {n} teams "
          f"({'+ actual EV' if hcol and acol else 'pick% only'})")
    return True


def rebuild_cumulative(year, through_week):
    """Rebuild Season_{year}_Through_Week_{w} by concatenating weeks 1..w."""
    frames = []
    for w in range(1, through_week + 1):
        wp = (f"nfl-power-ratings/final_data/{year}_final_data/"
              f"Week_{w}_{year}_Final_Data.csv")
        if os.path.exists(wp):
            frames.append(pd.read_csv(wp))
    if not frames:
        return
    season = pd.concat(frames, ignore_index=True)
    out = (f"nfl-power-ratings/final_data/{year}_final_data/"
           f"Season_{year}_Through_Week_{through_week}_Final_Data.csv")
    season.to_csv(out, index=False)
    print(f"   ✅ Rebuilt {os.path.basename(out)} ({len(season)} rows)")


def backfill_year(year, only_week=None):
    print(f"\n=== Backfilling {year}{f' week {only_week}' if only_week else ''} ===")
    wk_files = glob.glob(
        f"nfl-power-ratings/final_data/{year}_final_data/Week_*_{year}_Final_Data.csv")
    weeks = []
    for f in wk_files:
        try:
            weeks.append(int(os.path.basename(f).split("Week_")[1].split(f"_{year}")[0]))
        except (IndexError, ValueError):
            pass
    weeks = sorted(set(weeks))
    if only_week:
        weeks = [w for w in weeks if w == only_week]
    if not weeks:
        print(f"   No weekly files found for {year}")
        return

    done = []
    for w in weeks:
        if backfill_week_file(year, w):
            done.append(w)

    # Rebuild every cumulative file up through the max backfilled week
    for w in done:
        rebuild_cumulative(year, w)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python backfill_actual_pick_pct.py YEAR [WEEK]")
        print("   or: python backfill_actual_pick_pct.py YEAR1 YEAR2 ...")
        sys.exit(1)

    # Single year + optional week, or multiple years
    if len(args) == 2 and all(len(a) <= 2 or a.startswith("20") for a in args) \
       and int(args[1]) < 25:
        backfill_year(int(args[0]), only_week=int(args[1]))
    else:
        for y in args:
            backfill_year(int(y))

    print("\n✅ Backfill complete. Commit the updated Final_Data files.")
