"""
Additions to entry_analytics.py for PRESEASON mode.

Paste these functions into entry_analytics.py, and replace run_entry_analytics
with the version at the bottom (only the top section changed — it now accepts
synthetic-field parameters).

Usage in preseason (before entries are official):

    from entry_analytics import run_entry_analytics
    import pandas as pd

    # Harvest realistic behavior profiles from last year's field
    prior_picks = "circa-pick-history/2025_survivor_picks.csv"
    prior_final = pd.read_csv(
        "nfl-power-ratings/final_data/2025_final_data/"
        "Season_2025_Through_Week_20_Final_Data.csv")

    rankings, predicted_picks = run_entry_analytics(
        picks_csv_path=None,                 # ← no real entries yet
        sim_df=df,                           # 2026 sim file
        upcoming_week=1,
        target_year=2026,
        synthetic_n_entries=24000,           # ← your estimate
        prior_picks_csv=prior_picks,         # ← behavior source
        prior_season_df=prior_final,
    )
"""

import numpy as np
import pandas as pd


def harvest_profile_pool(prior_picks_csv, prior_season_df, n_weeks=20):
    """
    Builds the empirical distribution of behavioral profiles from a prior
    year's field. Returns a list of profile dicts to bootstrap-sample from.
    """
    prior_picks = pd.read_csv(prior_picks_csv)
    prior_wtd = build_week_team_data(prior_season_df, upcoming_week=n_weeks + 1)
    completed = [w for w in range(1, n_weeks + 1) if w in prior_wtd]
    profiles = build_entry_profiles(prior_picks, prior_wtd, completed)
    # Keep only profiles with real history (they inform the distribution)
    pool = [p for p in profiles.values() if p["n_picks"] >= 3]
    print(f"   Harvested {len(pool)} behavioral profiles from prior year")
    return pool


def generate_synthetic_field(n_entries, profile_pool=None, rng=None):
    """
    Creates a synthetic field of n_entries with empty pick history.
    Profiles are bootstrap-sampled from profile_pool (realistic behavior mix)
    or default to league-average with noise if no pool provided.

    Returns (alive_entries, used_masks, profiles) matching the shapes
    run_entry_analytics expects.
    """
    rng = rng or np.random.default_rng(42)
    alive_entries, used_masks, profiles = [], {}, {}

    for i in range(n_entries):
        name = f"SYNTH-{i+1:05d}"
        alive_entries.append(name)
        used_masks[name] = np.zeros(32, dtype=bool)

        if profile_pool:
            base = profile_pool[rng.integers(len(profile_pool))]
            # jitter so identical source profiles still diverge slightly
            profiles[name] = {
                "home_rate": float(np.clip(base["home_rate"] + rng.normal(0, 0.05), 0, 1)),
                "fav_rate":  float(np.clip(base["fav_rate"]  + rng.normal(0, 0.05), 0, 1)),
                "chalk":     float(np.clip(base["chalk"]     + rng.normal(0, 0.05), 0, 1)),
                "ev_align":  float(np.clip(base["ev_align"]  + rng.normal(0, 0.05), 0, 1)),
                "win_pref":  float(np.clip(base["win_pref"]  + rng.normal(0, 0.03), 0.4, 0.9)),
                # Full-history weight so profile_to_weights doesn't shrink
                # these sampled behaviors back toward the league prior
                "n_picks":   10,
            }
        else:
            profiles[name] = {
                "home_rate": float(np.clip(rng.normal(0.55, 0.12), 0, 1)),
                "fav_rate":  float(np.clip(rng.normal(0.90, 0.08), 0.5, 1)),
                "chalk":     float(np.clip(rng.normal(0.50, 0.20), 0, 1)),
                "ev_align":  float(np.clip(rng.normal(0.50, 0.18), 0, 1)),
                "win_pref":  float(np.clip(rng.normal(0.65, 0.06), 0.45, 0.85)),
                "n_picks":   10,
            }

    return alive_entries, used_masks, profiles


# ── REPLACEMENT run_entry_analytics (top section changed only) ──────────────
def run_entry_analytics(picks_csv_path, sim_df, upcoming_week, target_year,
                        output_dir="entry-analytics", seed=42,
                        synthetic_n_entries=None,
                        prior_picks_csv=None, prior_season_df=None):
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    week_team_data = build_week_team_data(sim_df, upcoming_week)
    max_week = max(week_team_data.keys()) if week_team_data else 20
    completed_weeks = list(range(1, upcoming_week))
    remaining_weeks = list(range(upcoming_week, max_week + 1))

    future_value = build_future_value(week_team_data, upcoming_week, max_week)
    feats = build_feature_tensors(week_team_data, future_value,
                                  completed_weeks + remaining_weeks)

    if synthetic_n_entries:
        # ── PRESEASON MODE — synthetic field ──
        print(f"\n🎯 Entry analytics — {target_year} PRESEASON "
              f"({synthetic_n_entries} synthetic entries)")
        pool = None
        if prior_picks_csv and prior_season_df is not None:
            pool = harvest_profile_pool(prior_picks_csv, prior_season_df)
        alive_entries, used_masks, profiles = generate_synthetic_field(
            synthetic_n_entries, pool, rng)
        total_entries = synthetic_n_entries
        contestant_of = {e: e for e in alive_entries}
        wins_of = {e: 0 for e in alive_entries}
        is_synthetic = True
    else:
        # ── REGULAR MODE — real entries ──
        print(f"\n🎯 Entry analytics — {target_year} week {upcoming_week}")
        picks_df = pd.read_csv(picks_csv_path)
        picks_df["Total_Wins"] = pd.to_numeric(
            picks_df["Total_Wins"], errors="coerce").fillna(0).astype(int)
        week_cols = [c for c in picks_df.columns if c.startswith("Week_")]

        alive_df = picks_df[picks_df["Total_Wins"] >= upcoming_week - 1]
        alive_entries, used_masks = [], {}
        for _, row in alive_df.iterrows():
            entry = row["EntryName"]
            mask = np.zeros(32, dtype=bool)
            for col in week_cols:
                t = norm_abbr(row.get(col, ""))
                if t in TEAM_IDX:
                    mask[TEAM_IDX[t]] = True
            alive_entries.append(entry)
            used_masks[entry] = mask
        print(f"   {len(alive_entries)} alive entries of {len(picks_df)} total")

        profiles = build_entry_profiles(picks_df, week_team_data, completed_weeks)
        total_entries = len(picks_df)
        contestant_of = dict(zip(picks_df["EntryName"],
                                 picks_df["EntryName"].astype(str).str.replace(
                                     r"-\d+$", "", regex=True)))
        wins_of = dict(zip(alive_df["EntryName"], alive_df["Total_Wins"]))
        is_synthetic = False

    # ── Everything below is unchanged from the original ──
    entry_dists, field_pick_pct = predict_upcoming_picks(
        alive_entries, used_masks, profiles, feats, upcoming_week)

    total_pot = total_entries * ENTRY_FEE * POT_MULT
    survival, fair_value, avg_survivors = run_season_simulation(
        alive_entries, used_masks, profiles, week_team_data,
        feats, remaining_weeks, total_pot, rng)
    print(f"   Avg expected end-of-season survivors: {avg_survivors:.2f}")
    print(f"   Total pot: ${total_pot:,.0f}")

    rows = []
    for ei, entry in enumerate(alive_entries):
        win_path, ev_path = optimal_paths(used_masks[entry], feats, remaining_weeks)
        dist = entry_dists.get(entry)
        top_picks = ""
        if dist is not None:
            order = np.argsort(-dist)[:3]
            top_picks = "; ".join(
                f"{ALL_ABBRS[i]} {dist[i]*100:.0f}%" for i in order if dist[i] > 0.01)
        rows.append({
            "entry": entry,
            "contestant": contestant_of.get(entry, entry),
            "wins": int(wins_of.get(entry, 0)),
            "teams_remaining": int((~used_masks[entry]).sum()),
            "optimal_win_path_prob": round(float(win_path), 6),
            "optimal_ev_path_score": round(float(ev_path), 3),
            "survival_prob": round(float(survival[ei]), 6),
            "fair_value": round(float(fair_value[ei]), 2),
            "predicted_next_picks": top_picks,
        })

    rankings = pd.DataFrame(rows).sort_values(
        "fair_value", ascending=False).reset_index(drop=True)
    rankings.insert(0, "rank", rankings.index + 1)

    suffix = "preseason" if is_synthetic else f"week_{upcoming_week}"
    rank_file = os.path.join(output_dir, f"{target_year}_{suffix}_entry_rankings.csv")
    rankings.to_csv(rank_file, index=False)
    print(f"   ✅ Entry rankings → {rank_file}")

    pick_rows = [{"team": ALL_ABBRS[i],
                  "predicted_pick_pct": round(float(field_pick_pct[i]), 6)}
                 for i in range(32) if field_pick_pct[i] > 0]
    pick_df = pd.DataFrame(pick_rows).sort_values(
        "predicted_pick_pct", ascending=False)
    pick_file = os.path.join(output_dir, f"{target_year}_{suffix}_predicted_pick_pct.csv")
    pick_df.to_csv(pick_file, index=False)
    print(f"   ✅ Predicted field pick% → {pick_file}")

    return rankings, pick_df
