"""
blend_pick_estimates.py

Per-contest, all-weeks blend of the two independent pick-% predictions the
pipeline produces:

  * TOP-DOWN  -- daily_2's game/market model projects each team's pick % for
    every remaining week of the season, for each contest
    ('Home/Away Pick %' for Circa; 'Home/Away Big Splash Pick %' /
    'Home/Away World Championship Pick %' for the Splash contests).

  * BEHAVIORAL -- daily_4 projects, entry by entry, who each alive entry will
    pick every remaining week (the archetype + choice model), aggregated up to
    a team-level 'Estimated_Pick_Pct' per week, per contest, written to
    entry_archetype_pick_estimates[_<tag>]/{year}/week_{w}_team_pick_estimates.csv

This module blends them with the SAME fitted stacked-ridge weights the
existing upcoming-week Circa blend uses (entry-analytics/blend_model.json:
blended ~= intercept + c_b*behavioral + c_t*topdown + c_w*week + c_a*log_alive),
for EVERY remaining week and EVERY contest, and writes the blended values back
into the sim file's per-contest pick-% columns. That gives a blended pick-%
surface for the whole season so downstream EV (daily_3) and the optimizer can
plan survivor paths on projected EV, not just the upcoming week.

WHY IT'S CALLED AT THE END OF daily_2, READING daily_4's LATEST OUTPUT
=====================================================================
daily_4 needs daily_2's sim file as input, so within one daily cycle the order
is daily_2 -> daily_3 -> daily_4. This blend therefore consumes the MOST RECENT
daily_4 estimates (produced on the previous cycle) -- exactly the "next time
daily_2 runs, blend with daily_4" behaviour. On the very first run, before
daily_4 has ever produced a contest's files, that contest's weeks simply keep
their pure top-down projection (no-op), so nothing breaks.

PRESERVING CIRCA'S UPCOMING-WEEK BLEND
======================================
daily_2 already blends Circa's UPCOMING week with a live-crowd behavioral model
(entry_analytics) right before this step, and stashes the pure projection in
'Topdown Home/Away Pick %'. To avoid clobbering that live-informed number, this
module blends Circa only for weeks AFTER the upcoming one; the Splash contests
have no such prior blend, so they are blended for every remaining week
including the upcoming one.
"""

import os
import json
import math

import numpy as np
import pandas as pd

from contest_config import CONTESTS, tagged, flavor_cols
from team_codes import canonical_pick_code, FULLNAME_TO_ABBR

BLEND_MODEL_PATH = os.path.join("entry-analytics", "blend_model.json")
SIM_FILE_PATTERN = "nfl-power-ratings/final_sim_results_with_variance_week_{week}_{year}.csv"
DAILY4_FILE = "week_{week}_team_pick_estimates.csv"

# Contests to blend, in order. Circa keeps its upcoming-week entry_analytics
# blend (blend_from_upcoming=False -> only weeks strictly after upcoming);
# Splash contests are blended for every remaining week.
_BLEND_CONTESTS = [
    ("circa", False),
    ("big_splash", True),
    ("world_championship", True),
]


def _full_to_abbr(name):
    """Sim-file full team name ('Los Angeles Rams') -> canonical abbr ('LA')."""
    s = str(name).strip()
    return canonical_pick_code(FULLNAME_TO_ABBR.get(s, s))


def _contest_cols(contest_key):
    """(home_out, away_out, home_topdown, away_topdown) column names for a
    contest. Circa keeps its original 'Home/Away Pick %'; the Splash contests
    use their projected-column names."""
    prefix = CONTESTS[contest_key].get("proj_prefix", "")
    if prefix:
        home_out, away_out = f"Home {prefix} Pick %", f"Away {prefix} Pick %"
    else:
        home_out, away_out = "Home Pick %", "Away Pick %"
    return home_out, away_out, f"Topdown {home_out}", f"Topdown {away_out}"


def _daily4_dir(contest_key, year):
    return os.path.join(tagged("entry_archetype_pick_estimates", "", contest_key),
                        str(year))


def _load_blend_model():
    if not os.path.exists(BLEND_MODEL_PATH):
        return None
    with open(BLEND_MODEL_PATH) as f:
        m = json.load(f)
    # feature order is fixed: [behavioral, topdown, week, log_alive]
    return float(m["intercept"]), [float(c) for c in m["coef"]]


def _ridge_blend(behavioral, topdown, week, alive, model):
    """Apply the fitted stacked-ridge blend to aligned per-team arrays and
    renormalize to the week's original top-down mass (so blending never
    changes the total pick mass daily_3's coverage guard expects; multi-pick
    weeks whose top-down sums to ~2.0 stay at ~2.0)."""
    intercept, coef = model
    la = math.log(max(float(alive or 1), 1.0))
    pred = (intercept
            + coef[0] * behavioral
            + coef[1] * topdown
            + coef[2] * float(week)
            + coef[3] * la)
    pred = np.clip(pred, 0.0, None)
    target_mass = float(np.nansum(topdown))
    s = float(pred.sum())
    if s > 0 and target_mass > 0:
        pred = pred * (target_mass / s)
    else:
        # Degenerate (no behavioral signal placed): fall back to top-down.
        pred = np.nan_to_num(topdown, nan=0.0)
    return pred


def blend_all_contests(target_year, upcoming_week, sim_path=None, verbose=True):
    """Blend top-down (daily_2) with behavioral (daily_4) for every contest and
    every remaining week, writing the result into the sim file's per-contest
    pick-% columns. Returns a dict of {contest_key: n_weeks_blended}."""
    model = _load_blend_model()
    if model is None and verbose:
        # No model → we still record the Predicted / Top Down / Archetype flavor
        # columns (Predicted stays pure top-down), and only skip the ridge
        # re-blend. This keeps the accuracy-study columns populated even in a
        # fresh checkout or a contest with no fitted blend model.
        print(f"ℹ️  No {BLEND_MODEL_PATH}; recording flavor columns but skipping "
              f"the ridge re-blend (Predicted stays pure top-down).")

    sim_path = sim_path or SIM_FILE_PATTERN.format(week=upcoming_week, year=target_year)
    if not os.path.exists(sim_path):
        if verbose:
            print(f"⚠️  Sim file not found ({sim_path}); nothing to blend.")
        return {}

    sim = pd.read_csv(sim_path, low_memory=False)
    week_col = "Week_x" if "Week_x" in sim.columns else "Week"
    pool_col = "Total Remaining Entries at Start of Week"
    result = {}

    for contest_key, blend_from_upcoming in _BLEND_CONTESTS:
        home_out, away_out, home_td, away_td = _contest_cols(contest_key)
        if home_out not in sim.columns or away_out not in sim.columns:
            if verbose:
                print(f"ℹ️  {CONTESTS[contest_key]['label']}: no "
                      f"'{home_out}' column in sim file — skipping (contest not "
                      f"projected this run).")
            continue

        # Preserve the pure top-down projection once, before we overwrite the
        # output columns with blended values. (Circa's Topdown columns were
        # already created by the upcoming-week entry_analytics step; create
        # them here too if that step didn't run.)
        # Preserve the pure top-down projection once, before we overwrite the
        # output columns with blended values. (Circa's Topdown columns were
        # already created by the upcoming-week entry_analytics step; create
        # them here too if that step didn't run.)
        for out_col, td_col in ((home_out, home_td), (away_out, away_td)):
            if td_col not in sim.columns:
                sim[td_col] = sim[out_col]

        # Per-contest, per-flavor comparison columns for the pick-% accuracy
        # study. Predicted == the blended working value; Top Down == daily_2's
        # pure market projection; Archetype == daily_4's behavioral estimate.
        # Recorded for every remaining week, whether or not it gets re-blended.
        pred_home, pred_away = flavor_cols("Predicted", contest_key)
        tdf_home, tdf_away = flavor_cols("Top Down", contest_key)
        arch_home, arch_away = flavor_cols("Archetype", contest_key)
        for _c in (pred_home, pred_away, tdf_home, tdf_away, arch_home, arch_away):
            if _c not in sim.columns:
                sim[_c] = np.nan

        d4_dir = _daily4_dir(contest_key, target_year)
        weeks = sorted(int(w) for w in sim[week_col].dropna().unique()
                       if int(w) >= upcoming_week)
        n_blended = 0

        for week in weeks:
            wk_mask = sim[week_col] == week
            wk = sim.loc[wk_mask]
            if wk.empty:
                continue

            # Record Top Down (pure projection) and the current Predicted value
            # for every row this week. Predicted == the working column: Circa's
            # upcoming week already holds the live-crowd blend; every other week
            # holds pure top-down until it is re-blended just below.
            sim.loc[wk_mask, tdf_home] = pd.to_numeric(sim.loc[wk_mask, home_td], errors="coerce")
            sim.loc[wk_mask, tdf_away] = pd.to_numeric(sim.loc[wk_mask, away_td], errors="coerce")
            sim.loc[wk_mask, pred_home] = pd.to_numeric(sim.loc[wk_mask, home_out], errors="coerce")
            sim.loc[wk_mask, pred_away] = pd.to_numeric(sim.loc[wk_mask, away_out], errors="coerce")

            # Behavioral (archetype) estimate for this week, if daily_4 produced
            # one. Used to record the Archetype flavor and (where allowed) to
            # re-blend the working / Predicted columns.
            d4_path = os.path.join(d4_dir, DAILY4_FILE.format(week=week))
            if not os.path.exists(d4_path):
                continue  # no behavioral estimate yet -> leave pure top-down
            beh = pd.read_csv(d4_path)
            if "Team" not in beh.columns or "Estimated_Pick_Pct" not in beh.columns:
                continue
            beh_map = {canonical_pick_code(t): p for t, p in
                       zip(beh["Team"], beh["Estimated_Pick_Pct"])}

            # Flatten this week's games into one row per playing team, carrying
            # each team's own top-down value and its row index + side so we can
            # write the blended value straight back.
            recs = []
            for idx in wk.index:
                h_ab = _full_to_abbr(sim.at[idx, "Home Team"])
                a_ab = _full_to_abbr(sim.at[idx, "Away Team"])
                recs.append((idx, "home", h_ab, sim.at[idx, home_td]))
                recs.append((idx, "away", a_ab, sim.at[idx, away_td]))
            rec_df = pd.DataFrame(recs, columns=["idx", "side", "abbr", "topdown"])
            rec_df["topdown"] = pd.to_numeric(rec_df["topdown"], errors="coerce").fillna(0.0)
            rec_df["behavioral"] = rec_df["abbr"].map(beh_map)

            # Record the Archetype (behavioral) flavor wherever daily_4 placed a
            # team, regardless of whether this week is re-blended below.
            for _, r in rec_df.iterrows():
                if pd.notna(r["behavioral"]):
                    col = arch_home if r["side"] == "home" else arch_away
                    sim.at[r["idx"], col] = float(r["behavioral"])

            # No fitted blend model → flavors are recorded above; skip the ridge
            # re-blend (Predicted keeps the pure top-down value already written).
            if model is None:
                continue

            # If daily_4 has no behavioral estimate for ANY team this week, the
            # blend has nothing to add -> keep pure top-down for the week.
            if rec_df["behavioral"].notna().sum() == 0:
                continue

            # Circa keeps its live-crowd upcoming-week blend already applied, so
            # re-blend only weeks strictly after the upcoming one; Splash
            # contests re-blend every remaining week. (Flavors above are still
            # recorded for the skipped upcoming week.)
            if not blend_from_upcoming and week == upcoming_week:
                continue

            rec_df["behavioral"] = rec_df["behavioral"].fillna(0.0)
            alive = wk[pool_col].iloc[0] if pool_col in wk.columns else None
            blended = _ridge_blend(rec_df["behavioral"].to_numpy(dtype=float),
                                   rec_df["topdown"].to_numpy(dtype=float),
                                   week, alive, model)
            rec_df["blended"] = blended

            for _, r in rec_df.iterrows():
                out_c = home_out if r["side"] == "home" else away_out
                pred_c = pred_home if r["side"] == "home" else pred_away
                sim.at[r["idx"], out_c] = r["blended"]
                sim.at[r["idx"], pred_c] = r["blended"]
            n_blended += 1

        result[contest_key] = n_blended
        if verbose:
            span = "all remaining weeks" if blend_from_upcoming else "weeks after the upcoming one"
            print(f"🔀 {CONTESTS[contest_key]['label']}: blended top-down × behavioral "
                  f"for {n_blended} week(s) ({span}); pure projection kept in "
                  f"'{home_td}' / '{away_td}'; flavors recorded in "
                  f"'{pred_home}' / '{tdf_home}' / '{arch_home}'.")

    sim.to_csv(sim_path, index=False)
    if verbose:
        print(f"💾 Blended pick % written back to {sim_path}")
    return result


if __name__ == "__main__":
    # Manual/standalone run: derive the week context the same way the daily
    # scripts do, then blend the current sim file in place.
    from season_dates import resolve_week_context
    from datetime import datetime
    ctx = resolve_week_context(datetime.now().strftime("%m/%d/%Y"))
    blend_all_contests(target_year=ctx.target_year, upcoming_week=ctx.upcoming_week)
