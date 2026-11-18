#!/usr/bin/env python3
"""
build_live_schedule.py — build ONE "Live Schedule" file per season spanning the
whole season, so the Schedule and Optimizer tabs can look at (and optimize over)
ANY week, not just the upcoming one.

  completed weeks  <- nfl-power-ratings/final_data/{year}_final_data/
                      Season_{year}_Through_Week_{W}_Final_Data.csv   (cumulative 1..W, ACTUALS)
  upcoming+future  <- nfl-power-ratings/final_sim_results_with_variance_week_{U}_{year}.csv  (PROJECTIONS)

Output: nfl-power-ratings/live_schedule_{year}.csv

The actuals file is a column SUPERSET of the sim file (it adds the "Actual *"
columns, e.g. "Home/Away Actual Pick %", "Actual Circa Home/Away Pick %",
"Actual Home/Away Team Win %"/"EV"). Concatenating aligns on column name, so
completed weeks carry actuals and future weeks have NaN in the Actual columns.

  * mid-season : actuals(1..W) + projection(W+1..end)
  * season over: just the actuals file (no projection past the last week)
  * preseason  : just the projection file (no actuals yet)
"""
import argparse, glob, os, re
import pandas as pd

KEYS = ["NFL Week", "Home Team", "Away Team"]


def _latest_final_data(data_dir, year):
    pat = os.path.join(data_dir, "nfl-power-ratings", "final_data",
                       f"{year}_final_data",
                       f"Season_{year}_Through_Week_*_Final_Data.csv")
    best, best_w = None, -1
    for f in glob.glob(pat):
        m = re.search(r"Through_Week_(\d+)_", os.path.basename(f))
        if m and int(m.group(1)) > best_w:
            best_w, best = int(m.group(1)), f
    return best, (best_w if best is not None else None)


def _latest_sim(data_dir, year):
    pat = os.path.join(data_dir, "nfl-power-ratings",
                       f"final_sim_results_with_variance_week_*_{year}.csv")
    best, best_w = None, -1
    for f in glob.glob(pat):
        m = re.search(rf"week_(\d+)_{year}\.csv$", os.path.basename(f))
        if m and int(m.group(1)) > best_w:
            best_w, best = int(m.group(1)), f
    return best, (best_w if best is not None else None)


def build(data_dir=".", year=2026):
    fd_path, last_completed = _latest_final_data(data_dir, year)
    sim_path, upcoming = _latest_sim(data_dir, year)
    out = os.path.join(data_dir, "nfl-power-ratings", f"live_schedule_{year}.csv")

    frames = []
    if fd_path is not None:
        fd = pd.read_csv(fd_path)
        frames.append(fd)
        print(f"  actuals   : {os.path.basename(fd_path)} (NFL weeks 1..{last_completed}, {len(fd)} rows)")
    if sim_path is not None:
        sim = pd.read_csv(sim_path)
        if fd_path is not None and last_completed is not None and "NFL Week" in sim.columns:
            sim = sim[pd.to_numeric(sim["NFL Week"], errors="coerce") > last_completed]
        frames.append(sim)
        print(f"  projection: {os.path.basename(sim_path)} ({len(sim)} future-week rows kept)")

    if not frames:
        raise SystemExit(f"No final_data or sim file found for {year}; nothing to build.")

    combined = pd.concat(frames, ignore_index=True, sort=False)  # actuals first -> win de-dup
    keys = [k for k in KEYS if k in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="first")
    sortcol = "NFL Week" if "NFL Week" in combined.columns else ("Week_x" if "Week_x" in combined.columns else None)
    if sortcol:
        combined = combined.sort_values(
            by=sortcol, key=lambda s: pd.to_numeric(s, errors="coerce"), kind="stable"
        )

    combined = combined.copy()  # de-fragment before writing
    combined.to_csv(out, index=False)
    print(f"✅ wrote {out}: {len(combined)} rows x {combined.shape[1]} cols")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=int(os.environ.get("LIVE_SCHEDULE_YEAR", 2026)))
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "."))
    a = ap.parse_args()
    build(a.data_dir, a.year)
