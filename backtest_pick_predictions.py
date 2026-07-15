"""
backtest_pick_predictions.py

Compares two pick-percentage prediction models against actual results,
replaying each historical week IN-TIME (no future information leaks):

  Model A (new):  bottom-up behavioral model from entry_analytics
  Model B (old):  your existing top-down predictions, read from the
                  archived sim file's Home/Away Pick % columns

Actual pick% is computed directly from the picks CSV (ground truth).

Usage:
    python backtest_pick_predictions.py 2025
    python backtest_pick_predictions.py 2025 --weeks 2 3 4 5

Outputs:
    entry-analytics/backtest_{year}_results.csv   (per team-week detail)
    console summary of MAE / RMSE / correlation per model
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

from entry_analytics import (
    build_week_team_data, build_future_value, build_feature_tensors,
    build_entry_profiles, predict_upcoming_picks, norm_abbr,
    detect_holiday_weeks, build_holiday_future_value,
    build_contestant_priors, contestant_name,
    ALL_ABBRS, TEAM_IDX, FULL_TO_ABBR,
)


def load_week_sim_file(year, week):
    path = f"nfl-power-ratings/final_sim_results_with_variance_week_{week}_{year}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def load_season_final_latest(year):
    files = glob.glob(f"nfl-power-ratings/final_data/{year}_final_data/"
                      f"Season_{year}_Through_Week_*_Final_Data.csv")
    if not files:
        return None
    def wk(p):
        try:
            return int(os.path.basename(p).split("_Week_")[1].split("_Final")[0])
        except (IndexError, ValueError):
            return 0
    return pd.read_csv(max(files, key=wk))


def load_season_final(year, through_week):
    path = (f"nfl-power-ratings/final_data/{year}_final_data/"
            f"Season_{year}_Through_Week_{through_week}_Final_Data.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def truncate_picks(picks_df, week):
    """Return a copy with all picks from `week` onward blanked out —
    simulates what was known before week W. Also returns the alive set:
    entries that actually made a (non-ELIMINATED) pick in week W."""
    trunc = picks_df.copy()
    week_cols = [c for c in trunc.columns if c.startswith("Week_")]

    alive, actual_picks = [], {}
    wcol = f"Week_{week}"
    for _, row in trunc.iterrows():
        val = str(row.get(wcol, "") or "").strip()
        if val and val.upper() != "ELIMINATED":
            alive.append(row["EntryName"])
            actual_picks[row["EntryName"]] = norm_abbr(val)

    for col in week_cols:
        if int(col.replace("Week_", "")) >= week:
            trunc[col] = ""
    # Recompute wins as of before week W
    trunc["Total_Wins"] = week - 1
    return trunc, alive, actual_picks


def run_backtest(year, test_weeks=None):
    picks_path = f"circa-pick-history/{year}_survivor_picks.csv"
    picks_df = pd.read_csv(picks_path)
    total_entries = len(picks_df)

    # Cross-year contestant priors from the prior season (if available)
    contestant_priors = None
    prior_picks = f"circa-pick-history/{year-1}_survivor_picks.csv"
    prior_final = load_season_final_latest(year - 1)
    if os.path.exists(prior_picks) and prior_final is not None:
        contestant_priors = build_contestant_priors(prior_picks, prior_final)
    week_cols = [c for c in picks_df.columns if c.startswith("Week_")]
    max_week = len(week_cols)

    if test_weeks is None:
        # Weeks 2..max — week 1 has no history to learn from
        test_weeks = list(range(2, max_week + 1))

    results = []

    for W in test_weeks:
        sim = load_week_sim_file(year, W)
        if sim is None:
            print(f"Week {W}: no sim file — skipping")
            continue

        # Combine: sim file (in-time predictions W..end) + final data (actuals < W)
        hist = load_season_final(year, W - 1) if W > 1 else None
        combined = pd.concat([hist, sim], ignore_index=True) if hist is not None else sim

        wtd = build_week_team_data(combined, W)
        if W not in wtd:
            print(f"Week {W}: no matchup data — skipping")
            continue

        fv = build_future_value(wtd, W, max_week)
        holidays = detect_holiday_weeks(combined)
        hfv = build_holiday_future_value(wtd, holidays, 1, max_week)
        feats = build_feature_tensors(wtd, fv, list(range(1, max_week + 1)),
                                      holiday_fv=hfv)

        # In-time knowledge: truncate picks, derive alive set + ground truth
        trunc, alive, actual_picks = truncate_picks(picks_df, W)
        if len(alive) < 50:
            print(f"Week {W}: only {len(alive)} alive — skipping")
            continue

        profiles = build_entry_profiles(trunc, wtd, list(range(1, W)),
                                        contestant_priors=contestant_priors)

        # Used masks from truncated history
        used_masks = {}
        for _, row in trunc[trunc["EntryName"].isin(alive)].iterrows():
            mask = np.zeros(32, dtype=bool)
            for col in week_cols:
                t = norm_abbr(row.get(col, ""))
                if t in TEAM_IDX:
                    mask[TEAM_IDX[t]] = True
            used_masks[row["EntryName"]] = mask

        # ── Model A: behavioral prediction (coordinated + staged) ──
        stage = float(np.log(max(len(alive), 1) / max(total_entries, 1)))
        contestant_of = {e: contestant_name(e) for e in alive}
        entry_dists, model_a = predict_upcoming_picks(
            alive, used_masks, profiles, feats, W,
            contestant_of=contestant_of, stage=stage)

        # ── Entry-level metrics (behavioral only — topdown has none) ──
        lls, hits = [], []
        for e, actual_team in actual_picks.items():
            d = entry_dists.get(e)
            if d is None or actual_team not in TEAM_IDX:
                continue
            ti = TEAM_IDX[actual_team]
            lls.append(-np.log(max(d[ti], 1e-9)))
            hits.append(1.0 if int(np.argmax(d)) == ti else 0.0)
        entry_ll = float(np.mean(lls)) if lls else None
        entry_top1 = float(np.mean(hits)) if hits else None

        # ── Ground truth from actual picks ──
        actual = np.zeros(32)
        for e, t in actual_picks.items():
            if t in TEAM_IDX:
                actual[TEAM_IDX[t]] += 1
        actual = actual / max(1, len(actual_picks))

        # ── Model B: your existing top-down prediction (from sim file wk W) ──
        model_b = np.zeros(32)
        week_col = "Week_x" if "Week_x" in sim.columns else "Week"
        for _, row in sim[sim[week_col] == W].iterrows():
            h = FULL_TO_ABBR.get(row.get("Home Team"), norm_abbr(row.get("Home Team")))
            a = FULL_TO_ABBR.get(row.get("Away Team"), norm_abbr(row.get("Away Team")))
            if h in TEAM_IDX:
                model_b[TEAM_IDX[h]] = float(row.get("Home Pick %", 0) or 0)
            if a in TEAM_IDX:
                model_b[TEAM_IDX[a]] = float(row.get("Away Pick %", 0) or 0)

        # Per-team rows for teams playing this week
        playing = feats[W]["plays"].astype(bool)
        for i in np.where(playing)[0]:
            results.append({
                "week": W, "team": ALL_ABBRS[i],
                "actual": round(actual[i], 5),
                "behavioral": round(model_a[i], 5),
                "topdown": round(model_b[i], 5),
                "behavioral_err": round(model_a[i] - actual[i], 5),
                "topdown_err": round(model_b[i] - actual[i], 5),
                "alive": len(alive),
                "entry_logloss": round(entry_ll, 4) if entry_ll is not None else None,
                "entry_top1": round(entry_top1, 4) if entry_top1 is not None else None,
            })

        wk_a_mae = np.abs(model_a[playing] - actual[playing]).mean()
        wk_b_mae = np.abs(model_b[playing] - actual[playing]).mean()
        winner = "behavioral" if wk_a_mae < wk_b_mae else "topdown"
        extra = ""
        if entry_ll is not None:
            extra = f"  entry-LL={entry_ll:.3f} top1={entry_top1*100:.0f}%"
        print(f"Week {W:>2}: MAE behavioral={wk_a_mae:.4f}  "
              f"topdown={wk_b_mae:.4f}  → {winner} "
              f"({len(alive)} alive){extra}")

    if not results:
        print("No weeks backtested.")
        return

    df = pd.DataFrame(results)
    os.makedirs("entry-analytics", exist_ok=True)
    out = f"entry-analytics/backtest_{year}_results.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*58}")
    print(f"OVERALL — {year}, {df['week'].nunique()} weeks, {len(df)} team-weeks")
    print(f"{'='*58}")
    for label, col in [("Behavioral (new)", "behavioral"), ("Top-down (old)", "topdown")]:
        mae = np.abs(df[col] - df["actual"]).mean()
        rmse = np.sqrt(((df[col] - df["actual"]) ** 2).mean())
        corr = df[[col, "actual"]].corr().iloc[0, 1]
        print(f"  {label:<20} MAE={mae:.4f}  RMSE={rmse:.4f}  corr={corr:.4f}")

    # Where does each model win?
    wk_summary = df.groupby("week").apply(
        lambda g: pd.Series({
            "behavioral_mae": np.abs(g["behavioral"] - g["actual"]).mean(),
            "topdown_mae": np.abs(g["topdown"] - g["actual"]).mean(),
        }), include_groups=False)
    b_wins = int((wk_summary["behavioral_mae"] < wk_summary["topdown_mae"]).sum())
    print(f"\n  Weekly wins: behavioral {b_wins} — "
          f"topdown {len(wk_summary) - b_wins}")
    ll_rows = df.dropna(subset=["entry_logloss"]).groupby("week").first()
    if len(ll_rows):
        print(f"  Entry-level (behavioral only — topdown has no entry model):")
        print(f"    avg log-loss={ll_rows['entry_logloss'].mean():.3f}  "
              f"avg top-1 hit={ll_rows['entry_top1'].mean()*100:.1f}%")
    print(f"  Detail saved → {out}")


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    weeks = None
    if "--weeks" in sys.argv:
        idx = sys.argv.index("--weeks")
        weeks = [int(x) for x in sys.argv[idx + 1:]]
    run_backtest(year, weeks)
