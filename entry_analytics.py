"""
entry_analytics.py — called from daily_2

Three capabilities:
  1. Entry ranking by remaining-team strength (matchup-aware win% path + EV path)
  2. Per-entry probabilistic pick prediction for the upcoming week
     (fractional picks, e.g. entry X → 0.9 Chiefs, 0.1 elsewhere)
  3. Season-long survival simulation → survival probability and fair entry value

Model design notes:
  - Behavioral profiles are learned from each entry's pick history:
    home/away tendency, favorite vs contrarian, chalk-following, EV alignment.
  - Pick prediction = softmax over a utility function whose weights are
    personalized from the entry's profile.
  - KEY SIMULATION INSIGHT: an entry's pick sequence does NOT depend on game
    outcomes (picks are made before games; an eliminated entry's counterfactual
    picks don't matter). So we sample K pick-paths per entry independently of
    S game-outcome simulations, then cross them. This turns an intractable
    per-entry-per-sim sequential problem into two cheap independent samples.
  - Fair value handles correlation properly: chalk entries die together, so
    sims where a contrarian survives tend to have few survivors → bigger pot
    share. FairValue_i = mean_s [ P(i survives | outcomes_s) * POT / E[survivors_s] ]

Outputs (per run):
  entry-analytics/{year}_week_{W}_entry_rankings.csv
  entry-analytics/{year}_week_{W}_predicted_pick_pct.csv
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ── Tunable parameters ──────────────────────────────────────────────────────
N_OUTCOME_SIMS = 500      # game-outcome simulations
N_PICK_PATHS   = 25       # sampled pick-paths per entry
SOFTMAX_TEMP   = 1.0      # lower = entries more deterministic
ENTRY_FEE      = 1000.0   # Circa entry fee
POT_MULT       = 1.0      # pot = entries * fee * POT_MULT (adjust for rake if any)

ABBR_TO_FULL = {
    "ARI": "Arizona Cardinals",    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",     "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",   "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",       "DEN": "Denver Broncos",
    "DET": "Detroit Lions",        "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",       "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC":  "Kansas City Chiefs",
    "LA":  "Los Angeles Rams",     "LAC": "Los Angeles Chargers",
    "LV":  "Las Vegas Raiders",    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",    "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",   "NYG": "New York Giants",
    "NYJ": "New York Jets",        "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",  "SEA": "Seattle Seahawks",
    "SF":  "San Francisco 49ers",  "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",     "WAS": "Washington Commanders",
}
FULL_TO_ABBR = {v: k for k, v in ABBR_TO_FULL.items()}
VARIANT_MAP = {"JAC": "JAX", "LAR": "LA", "GNB": "GB", "KAN": "KC",
               "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV", "WSH": "WAS"}
ALL_ABBRS = sorted(ABBR_TO_FULL.keys())
TEAM_IDX = {t: i for i, t in enumerate(ALL_ABBRS)}   # team → 0..31


def norm_abbr(t):
    t = str(t).strip().upper()
    return VARIANT_MAP.get(t, t)


# ═══════════════════════════════════════════════════════════════════════════
# 1. GAME DATA — matchup grid for all weeks (past actuals + future predictions)
# ═══════════════════════════════════════════════════════════════════════════
def build_week_team_data(sim_df, upcoming_week):
    """
    Returns dict: week -> {abbr -> {win_pct, ev, pick_pct, is_home, opp}}
    Uses Circa week numbering via 'Circa Week' where numeric, else Week_x.
    """
    week_col = "Week_x" if "Week_x" in sim_df.columns else "Week"
    data = defaultdict(dict)

    for _, row in sim_df.iterrows():
        try:
            w = int(row[week_col])
        except (ValueError, TypeError):
            continue
        home_full, away_full = row.get("Home Team"), row.get("Away Team")
        home = FULL_TO_ABBR.get(home_full, norm_abbr(home_full))
        away = FULL_TO_ABBR.get(away_full, norm_abbr(away_full))
        if home not in TEAM_IDX or away not in TEAM_IDX:
            continue

        h_win = float(row.get("Consensus Home Win Pct", 0.5) or 0.5)
        a_win = float(row.get("Consensus Away Win Pct", 0.5) or 0.5)
        h_ev  = float(row.get("consensus_Home_EV", 0) or 0)
        a_ev  = float(row.get("consensus_Away_EV", 0) or 0)
        h_pk  = float(row.get("Home Pick %", 0) or 0)
        a_pk  = float(row.get("Away Pick %", 0) or 0)

        data[w][home] = {"win_pct": h_win, "ev": h_ev, "pick_pct": h_pk,
                         "is_home": 1.0, "opp": away}
        data[w][away] = {"win_pct": a_win, "ev": a_ev, "pick_pct": a_pk,
                         "is_home": 0.0, "opp": home}
    return dict(data)


def build_future_value(week_team_data, upcoming_week, max_week):
    """FV(team, week) = sum over future weeks of max(0, win% - 0.65).
    High FV = team worth saving. Used as a penalty for sophisticated entries."""
    fv = defaultdict(lambda: defaultdict(float))
    for w in range(upcoming_week, max_week + 1):
        for t in ALL_ABBRS:
            total = 0.0
            for w2 in range(w + 1, max_week + 1):
                d = week_team_data.get(w2, {}).get(t)
                if d:
                    total += max(0.0, d["win_pct"] - 0.65)
            fv[w][t] = total
    return fv


# ═══════════════════════════════════════════════════════════════════════════
# 2. BEHAVIORAL PROFILES — learned from each entry's completed picks
# ═══════════════════════════════════════════════════════════════════════════
def build_entry_profiles(picks_df, week_team_data, completed_weeks):
    """
    Per entry: home_rate, fav_rate, chalk_score, ev_align, avg_win_pref, n_picks.
    chalk_score: mean of (picked team's field pick% / max pick% that week);
      1.0 = pure chalk, near 0 = contrarian.
    ev_align: mean EV percentile of picks; 1.0 = always the EV-optimal pick.
    """
    profiles = {}
    week_cols = [c for c in picks_df.columns if c.startswith("Week_")]

    # Precompute per-week maxima / rankings for normalization
    week_max_pick = {}
    week_ev_sorted = {}
    for w in completed_weeks:
        teams = week_team_data.get(w, {})
        if teams:
            week_max_pick[w] = max((d["pick_pct"] for d in teams.values()), default=1) or 1
            week_ev_sorted[w] = sorted(d["ev"] for d in teams.values())

    for _, row in picks_df.iterrows():
        home_hits, fav_hits, chalk_vals, ev_pctls, win_prefs, n = 0, 0, [], [], [], 0
        for col in week_cols:
            w = int(col.replace("Week_", ""))
            if w not in completed_weeks:
                continue
            pick = norm_abbr(row.get(col, ""))
            if pick in ("", "ELIMINATED", "NAN") or pick not in TEAM_IDX:
                continue
            d = week_team_data.get(w, {}).get(pick)
            if not d:
                continue
            n += 1
            home_hits += d["is_home"]
            fav_hits += 1 if d["win_pct"] > 0.5 else 0
            chalk_vals.append(d["pick_pct"] / week_max_pick.get(w, 1))
            win_prefs.append(d["win_pct"])
            evs = week_ev_sorted.get(w, [])
            if evs:
                ev_pctls.append(np.searchsorted(evs, d["ev"]) / max(1, len(evs) - 1))

        if n == 0:  # no history — league-average profile
            profiles[row["EntryName"]] = dict(
                home_rate=0.55, fav_rate=0.9, chalk=0.5, ev_align=0.5,
                win_pref=0.65, n_picks=0)
        else:
            profiles[row["EntryName"]] = dict(
                home_rate=home_hits / n,
                fav_rate=fav_hits / n,
                chalk=float(np.mean(chalk_vals)) if chalk_vals else 0.5,
                ev_align=float(np.mean(ev_pctls)) if ev_pctls else 0.5,
                win_pref=float(np.mean(win_prefs)) if win_prefs else 0.65,
                n_picks=n)
    return profiles


def profile_to_weights(p):
    """Map a behavioral profile to utility weights. All tunable."""
    # Shrink toward league average when history is thin
    shrink = min(1.0, p["n_picks"] / 8.0)
    def blend(v, prior):
        return prior + (v - prior) * shrink

    chalk    = blend(p["chalk"], 0.5)
    home     = blend(p["home_rate"], 0.55)
    ev_align = blend(p["ev_align"], 0.5)
    win_pref = blend(p["win_pref"], 0.65)

    return {
        "w_win":  1.0 + 4.0 * (win_pref - 0.5),      # safety seekers weight win%
        "w_pop":  6.0 * (chalk - 0.5),               # contrarians get NEGATIVE
        "w_home": 3.0 * (home - 0.55),               # home/away lean
        "w_ev":   3.0 * (ev_align - 0.5),            # EV optimizers
        "w_fv":   1.5 * max(0.0, ev_align - 0.5),    # sophisticated → save teams
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. UTILITY MATRICES — precomputed per (entry, week, team), fully vectorized
# ═══════════════════════════════════════════════════════════════════════════
def build_feature_tensors(week_team_data, future_value, weeks):
    """
    Returns per-week z-scored feature arrays over the 32-team index:
      feats[w] = dict of np arrays shape (32,): win, ev, pop, home, fv, plays
    'plays' is a 0/1 mask of teams with a game that week.
    """
    feats = {}
    for w in weeks:
        teams = week_team_data.get(w, {})
        win = np.zeros(32); ev = np.zeros(32); pop = np.zeros(32)
        home = np.zeros(32); fv = np.zeros(32); plays = np.zeros(32)
        for t, d in teams.items():
            i = TEAM_IDX[t]
            plays[i] = 1
            win[i], ev[i], pop[i], home[i] = d["win_pct"], d["ev"], d["pick_pct"], d["is_home"]
            fv[i] = future_value[w][t]

        def z(x):
            m = plays.astype(bool)
            if m.sum() < 2:
                return x
            mu, sd = x[m].mean(), x[m].std()
            out = np.zeros_like(x)
            out[m] = (x[m] - mu) / (sd if sd > 1e-9 else 1)
            return out

        feats[w] = {"win": z(win), "ev": z(ev), "pop": z(pop),
                    "home": home, "fv": z(fv), "plays": plays,
                    "raw_win": win}
    return feats


def entry_utilities(weights, feats_w):
    """Utility vector (32,) for one entry in one week."""
    return (weights["w_win"]  * feats_w["win"]
          + weights["w_ev"]   * feats_w["ev"]
          + weights["w_pop"]  * feats_w["pop"]
          + weights["w_home"] * (feats_w["home"] - 0.5)
          - weights["w_fv"]   * feats_w["fv"])


def pick_distribution(utility, available_mask, plays_mask, temp=SOFTMAX_TEMP):
    """Softmax over teams that are available AND playing. Returns (32,) probs."""
    mask = available_mask & plays_mask.astype(bool)
    if not mask.any():
        return None
    u = utility.copy()
    u[~mask] = -1e9
    u = (u - u[mask].max()) / temp
    ex = np.exp(u)
    ex[~mask] = 0
    s = ex.sum()
    return ex / s if s > 0 else None


# ═══════════════════════════════════════════════════════════════════════════
# 4. UPCOMING WEEK PREDICTION — the fractional pick model
# ═══════════════════════════════════════════════════════════════════════════
def predict_upcoming_picks(alive_entries, used_masks, profiles, feats, upcoming_week):
    """
    Returns:
      entry_dists: {entry -> (32,) probability vector}
      field_pick_pct: (32,) aggregated predicted pick% for the week
    """
    entry_dists = {}
    agg = np.zeros(32)
    fw = feats[upcoming_week]
    for entry in alive_entries:
        weights = profile_to_weights(profiles[entry])
        u = entry_utilities(weights, fw)
        dist = pick_distribution(u, ~used_masks[entry], fw["plays"])
        if dist is None:
            continue
        entry_dists[entry] = dist
        agg += dist
    n = max(1, len(entry_dists))
    return entry_dists, agg / n


# ═══════════════════════════════════════════════════════════════════════════
# 5. SEASON SIMULATION — pick-paths × outcome-sims
# ═══════════════════════════════════════════════════════════════════════════
def sample_game_outcomes(feats, weeks, n_sims, rng):
    """wins[s, w_idx, team] = 1 if team won its game in sim s."""
    wins = np.zeros((n_sims, len(weeks), 32), dtype=bool)
    for wi, w in enumerate(weeks):
        p = feats[w]["raw_win"]           # (32,) win prob (0 for bye teams)
        plays = feats[w]["plays"].astype(bool)
        r = rng.random((n_sims, 32))
        wins[:, wi, :] = (r < p) & plays
        # Enforce zero-sum per game: if team "won" in a sim, its opp must lose.
        # raw_win probs already sum to 1 per game, so independent draws per team
        # slightly misspecify; correct by resolving each matchup with one draw:
    # Cleaner: resolve per matchup below (overwrites the above per pair)
    return wins


def sample_game_outcomes_paired(week_team_data, weeks, n_sims, rng):
    """Correct paired version: one random draw per game per sim."""
    wins = np.zeros((n_sims, len(weeks), 32), dtype=bool)
    for wi, w in enumerate(weeks):
        seen = set()
        for t, d in week_team_data.get(w, {}).items():
            opp = d["opp"]
            key = tuple(sorted((t, opp)))
            if key in seen:
                continue
            seen.add(key)
            p_t = d["win_pct"]
            draw = rng.random(n_sims)
            ti, oi = TEAM_IDX[t], TEAM_IDX[opp]
            wins[:, wi, ti] = draw < p_t
            wins[:, wi, oi] = ~wins[:, wi, ti]
    return wins


def sample_pick_paths(entry, used_mask, weights, feats, weeks, n_paths, rng):
    """
    Samples n_paths pick sequences for one entry over remaining weeks.
    Returns int array (n_paths, n_weeks) of team indices, -1 = no valid pick.
    """
    paths = np.full((n_paths, len(weeks)), -1, dtype=np.int8)
    for k in range(n_paths):
        avail = ~used_mask.copy()
        for wi, w in enumerate(weeks):
            fw = feats[w]
            u = entry_utilities(weights, fw)
            dist = pick_distribution(u, avail, fw["plays"])
            if dist is None:
                break
            pick = rng.choice(32, p=dist)
            paths[k, wi] = pick
            avail[pick] = False
    return paths


def run_season_simulation(alive_entries, used_masks, profiles, week_team_data,
                          feats, remaining_weeks, total_pot, rng):
    """
    Returns per-entry: survival_prob (to end of season), fair_value.
    Handles survivor-count correlation via shared outcome sims.
    """
    n_entries = len(alive_entries)
    weeks = remaining_weeks
    W = len(weeks)

    print(f"   Simulating {n_entries} entries × {N_PICK_PATHS} paths × "
          f"{N_OUTCOME_SIMS} outcome sims over {W} weeks...")

    outcome_wins = sample_game_outcomes_paired(week_team_data, weeks,
                                               N_OUTCOME_SIMS, rng)  # (S, W, 32)

    # P(entry survives | sim s) for every entry — accumulate expected survivors
    survive_prob_per_sim = np.zeros((n_entries, N_OUTCOME_SIMS))

    for ei, entry in enumerate(alive_entries):
        weights = profile_to_weights(profiles[entry])
        paths = sample_pick_paths(entry, used_masks[entry], weights,
                                  feats, weeks, N_PICK_PATHS, rng)  # (P, W)

        # survived[p, s] = all picks in path p won under sim s
        survived = np.ones((N_PICK_PATHS, N_OUTCOME_SIMS), dtype=bool)
        for wi in range(W):
            picks = paths[:, wi]                       # (P,)
            valid = picks >= 0
            # win lookup: outcome_wins[s, wi, pick]
            wk_wins = outcome_wins[:, wi, :]           # (S, 32)
            pick_won = np.ones((N_PICK_PATHS, N_OUTCOME_SIMS), dtype=bool)
            if valid.any():
                pick_won[valid] = wk_wins[:, picks[valid]].T   # (n_valid, S)
            # A path with no pick (ran out of teams) is treated as eliminated
            pick_won[~valid] = False
            survived &= pick_won

        survive_prob_per_sim[ei] = survived.mean(axis=0)   # avg over paths

        if (ei + 1) % 2000 == 0:
            print(f"      ... {ei + 1}/{n_entries} entries")

    # Expected survivors per sim (correlation preserved through shared outcomes)
    exp_survivors = survive_prob_per_sim.sum(axis=0)        # (S,)

    # Fair value: your share of pot when you survive, given how many others do.
    # Floor survivors at 1 (if you survived, at least you did).
    # Note: Circa's "most wins split if all eliminated" rule is approximated —
    # sims where exp_survivors < 1 give the full pot to whoever survives.
    denom = np.maximum(exp_survivors, 1.0)
    fair_values = (survive_prob_per_sim * (total_pot / denom)).mean(axis=1)
    survival_probs = survive_prob_per_sim.mean(axis=1)

    return survival_probs, fair_values, exp_survivors.mean()


# ═══════════════════════════════════════════════════════════════════════════
# 6. MATCHUP-AWARE ENTRY RANKING (deterministic optimal paths)
# ═══════════════════════════════════════════════════════════════════════════
def optimal_paths(used_mask, feats, weeks):
    """
    Greedy optimal remaining path through unused teams.
    Returns (win_path_prob, ev_path_sum).
      win_path_prob = PRODUCT of win% along greedy-best-win% path
        (probability of surviving all remaining weeks with those picks —
         matchup-aware since win% is per-game)
      ev_path_sum = sum of EV along greedy-best-EV path
    """
    # Win path
    avail = ~used_mask.copy()
    win_prob = 1.0
    for w in weeks:
        fw = feats[w]
        cand = np.where(avail & fw["plays"].astype(bool))[0]
        if len(cand) == 0:
            win_prob = 0.0
            break
        best = cand[np.argmax(fw["raw_win"][cand])]
        win_prob *= fw["raw_win"][best]
        avail[best] = False

    # EV path (re-walk with EV objective)
    avail = ~used_mask.copy()
    ev_sum = 0.0
    ev_raw = {w: feats[w]["ev"] for w in weeks}  # z-scored; fine for ranking
    for w in weeks:
        fw = feats[w]
        cand = np.where(avail & fw["plays"].astype(bool))[0]
        if len(cand) == 0:
            break
        best = cand[np.argmax(ev_raw[w][cand])]
        ev_sum += ev_raw[w][best]
        avail[best] = False

    return win_prob, ev_sum


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN ENTRY POINT — call this from daily_2
# ═══════════════════════════════════════════════════════════════════════════
def run_entry_analytics(picks_csv_path, sim_df, upcoming_week, target_year,
                        output_dir="entry-analytics", seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    picks_df = pd.read_csv(picks_csv_path)
    picks_df["Total_Wins"] = pd.to_numeric(
        picks_df["Total_Wins"], errors="coerce").fillna(0).astype(int)

    week_cols = [c for c in picks_df.columns if c.startswith("Week_")]
    max_week = len(week_cols)
    completed_weeks = list(range(1, upcoming_week))
    remaining_weeks = list(range(upcoming_week, max_week + 1))

    print(f"\n🎯 Entry analytics — {target_year} week {upcoming_week}")
    week_team_data = build_week_team_data(sim_df, upcoming_week)
    future_value = build_future_value(week_team_data, upcoming_week, max_week)
    feats = build_feature_tensors(week_team_data, future_value,
                                  completed_weeks + remaining_weeks)

    # Alive entries + used-team bitmasks
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

    # Behavioral profiles
    profiles = build_entry_profiles(picks_df, week_team_data, completed_weeks)

    # Upcoming week fractional pick predictions
    entry_dists, field_pick_pct = predict_upcoming_picks(
        alive_entries, used_masks, profiles, feats, upcoming_week)

    # Season simulation → survival + fair value
    total_pot = len(picks_df) * ENTRY_FEE * POT_MULT
    survival, fair_value, avg_survivors = run_season_simulation(
        alive_entries, used_masks, profiles, week_team_data,
        feats, remaining_weeks, total_pot, rng)
    print(f"   Avg expected end-of-season survivors: {avg_survivors:.2f}")
    print(f"   Total pot: ${total_pot:,.0f}")

    # Deterministic optimal-path rankings
    rows = []
    contestant_of = dict(zip(picks_df["EntryName"],
                             picks_df["EntryName"].astype(str).str.replace(
                                 r"-\d+$", "", regex=True)))
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
            "wins": int(alive_df[alive_df["EntryName"] == entry]["Total_Wins"].iloc[0]),
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

    rank_file = os.path.join(
        output_dir, f"{target_year}_week_{upcoming_week}_entry_rankings.csv")
    rankings.to_csv(rank_file, index=False)
    print(f"   ✅ Entry rankings → {rank_file}")

    # Aggregated predicted field pick% (feed this back into your EV pipeline!)
    pick_rows = [{"team": ALL_ABBRS[i],
                  "predicted_pick_pct": round(float(field_pick_pct[i]), 6)}
                 for i in range(32) if field_pick_pct[i] > 0]
    pick_df = pd.DataFrame(pick_rows).sort_values(
        "predicted_pick_pct", ascending=False)
    pick_file = os.path.join(
        output_dir, f"{target_year}_week_{upcoming_week}_predicted_pick_pct.csv")
    pick_df.to_csv(pick_file, index=False)
    print(f"   ✅ Predicted field pick% → {pick_file}")

    print(f"\n   📊 Top 5 entries by fair value:")
    for _, r in rankings.head(5).iterrows():
        print(f"      #{r['rank']} {r['entry']:<24} "
              f"${r['fair_value']:>10,.2f}  "
              f"(survive: {r['survival_prob']*100:.2f}%, next: {r['predicted_next_picks']})")

    return rankings, pick_df
