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
SURVIVAL_DECAY = 0.75     # assumed weekly field survival for stage projection
DIVERSIFY_COEFF = 2.0     # multi-entry anti-correlation strength (0 = off)
SELF_CONSISTENCY_ITERS = 1  # popularity feedback iterations (1 = off; ablation: 3 hurts)
SELF_CONSISTENCY_DAMPING = 0.5  # partial pop update per iteration (kills overshoot)
FE_SCALE = 1.0            # team fixed-effect strength (0 = off)
USE_CONTESTANT_PRIORS = True  # cross-year identity priors (False = off)
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

        # Thursday detection (picks lock at kickoff; Thu teams behave differently)
        is_thu = 0.0
        _d = row.get("Date_x") or row.get("Date")
        if _d is not None and not pd.isna(_d):
            try:
                is_thu = 1.0 if pd.to_datetime(str(_d)).weekday() == 3 else 0.0
            except Exception:
                is_thu = 0.0

        h_win = float(row.get("Consensus Home Win Pct", 0.5) or 0.5)
        a_win = float(row.get("Consensus Away Win Pct", 0.5) or 0.5)
        h_ev  = float(row.get("consensus_Home_EV", 0) or 0)
        a_ev  = float(row.get("consensus_Away_EV", 0) or 0)
        h_pk  = float(row.get("Home Pick %", 0) or 0)
        a_pk  = float(row.get("Away Pick %", 0) or 0)

        data[w][home] = {"win_pct": h_win, "ev": h_ev, "pick_pct": h_pk,
                         "is_home": 1.0, "opp": away, "is_thu": is_thu}
        data[w][away] = {"win_pct": a_win, "ev": a_ev, "pick_pct": a_pk,
                         "is_home": 0.0, "opp": home, "is_thu": is_thu}
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


def detect_holiday_weeks(sim_df):
    """
    Returns {week_num: "Thanksgiving"|"Christmas"} by scanning the
    'Circa Week' column. Handles 2020 (no Christmas week) automatically.
    """
    holidays = {}
    if "Circa Week" not in sim_df.columns:
        return holidays
    week_col = "Week_x" if "Week_x" in sim_df.columns else "Week"
    for _, row in sim_df[[week_col, "Circa Week"]].drop_duplicates().iterrows():
        label = str(row.get("Circa Week", "")).lower()
        try:
            w = int(row[week_col])
        except (ValueError, TypeError):
            continue
        if "thanks" in label:
            holidays[w] = "Thanksgiving"
        elif "christ" in label:
            holidays[w] = "Christmas"
    return holidays


def build_holiday_future_value(week_team_data, holiday_weeks, upcoming_week, max_week):
    """
    HFV(team, week) = value of this team on FUTURE holiday weeks specifically.
    Entries hoard teams that are strong on short holiday slates, so a high
    HFV suppresses a team's pick probability in the weeks BEFORE the holiday.
    On the holiday week itself a team's HFV drops to 0 (the holiday is now),
    so the hoarded teams flood out — the observed week-13 dynamic emerges
    naturally from the feature.
    """
    from collections import defaultdict as _dd
    hfv = _dd(lambda: _dd(float))
    hol_weeks = sorted(w for w in holiday_weeks if w <= max_week)
    for w in range(upcoming_week, max_week + 1):
        future_hols = [h for h in hol_weeks if h > w]
        for t in ALL_ABBRS:
            total = 0.0
            for h in future_hols:
                d = week_team_data.get(h, {}).get(t)
                if d:
                    # Scarcity premium: short slates make strong holiday
                    # teams more valuable than the same win% midseason
                    total += max(0.0, d["win_pct"] - 0.60) * 2.0
            hfv[w][t] = total
    return hfv


# ═══════════════════════════════════════════════════════════════════════════
# 2. BEHAVIORAL PROFILES — learned from each entry's completed picks
# ═══════════════════════════════════════════════════════════════════════════
import re as _re

def contestant_name(entry):
    return _re.sub(r"-\d+$", "", str(entry).strip())


def build_contestant_priors(prior_picks_csv, prior_season_df):
    """
    Cross-year identity: profiles from a PRIOR year keyed by contestant name,
    used as informed priors for thin-history entries in the current year.
    Multiple entries per contestant are averaged, weighted by picks observed.
    """
    prior_picks = pd.read_csv(prior_picks_csv)
    week_cols = [c for c in prior_picks.columns if c.startswith("Week_")]
    wtd = build_week_team_data(prior_season_df, len(week_cols) + 1)
    completed = sorted(wtd.keys())
    entry_profiles = build_entry_profiles(prior_picks, wtd, completed)

    agg = {}
    for entry, p in entry_profiles.items():
        if p["n_picks"] < 3:
            continue
        c = contestant_name(entry)
        if c not in agg:
            agg[c] = {"w": 0.0, **{k: 0.0 for k in
                      ("home_rate", "fav_rate", "chalk", "ev_align", "win_pref")}}
        w = p["n_picks"]
        for k in ("home_rate", "fav_rate", "chalk", "ev_align", "win_pref"):
            agg[c][k] += p[k] * w
        agg[c]["w"] += w

    priors = {}
    for c, a in agg.items():
        if a["w"] > 0:
            priors[c] = {k: a[k] / a["w"] for k in
                         ("home_rate", "fav_rate", "chalk", "ev_align", "win_pref")}
    print(f"   Built cross-year priors for {len(priors)} contestants")
    return priors


def build_entry_profiles(picks_df, week_team_data, completed_weeks,
                         contestant_priors=None):
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

    # ── Cross-year identity: blend thin profiles toward the contestant's
    #    prior-year behavior instead of the league average ──
    if contestant_priors and USE_CONTESTANT_PRIORS:
        matched = 0
        for entry, p in profiles.items():
            if p["n_picks"] >= 8:
                continue
            prior = contestant_priors.get(contestant_name(entry))
            if not prior:
                continue
            w = p["n_picks"] / 8.0
            for k in ("home_rate", "fav_rate", "chalk", "ev_align", "win_pref"):
                p[k] = w * p[k] + (1 - w) * prior[k]
            # Prior counts as ~5 picks of evidence → less shrink downstream
            p["n_picks"] = min(8, p["n_picks"] + 5)
            matched += 1
        print(f"   Applied cross-year priors to {matched} thin-history entries")
    return profiles


_CALIBRATED = None
_TEAM_FE = np.zeros(32)
_GLOBAL_W = {"w_win_stage": 0.0, "w_pop_stage": 0.0, "w_thu": 0.0}
_CAL_PATH = os.path.join("entry-analytics", "calibrated_weights.json")
if os.path.exists(_CAL_PATH):
    try:
        import json as _json
        with open(_CAL_PATH) as _f:
            _CALIBRATED = _json.load(_f)
        if "team_fe" in _CALIBRATED:
            _TEAM_FE = np.array(_CALIBRATED["team_fe"], dtype=float)
        _b = _CALIBRATED.get("beta", [])
        if len(_b) >= 9:      # v3 format: [6 base] + [win_stage, pop_stage, thu]
            _GLOBAL_W = {"w_win_stage": _b[6], "w_pop_stage": _b[7], "w_thu": _b[8]}
        print(f"   Loaded calibrated pick-model weights "
              f"(fit on {_CALIBRATED.get('n_decisions', '?')} decisions)")
    except Exception:
        _CALIBRATED = None


def profile_to_weights(p):
    """Map a behavioral profile to utility weights.
    Uses calibrated conditional-logit weights when available,
    falling back to hand-tuned heuristics."""
    shrink = min(1.0, p["n_picks"] / 8.0)
    def blend(v, prior):
        return prior + (v - prior) * shrink

    devs = {
        "win":  blend(p["win_pref"], 0.65) - 0.65,
        "ev":   blend(p["ev_align"], 0.5) - 0.5,
        "pop":  blend(p["chalk"], 0.5) - 0.5,
        "home": blend(p["home_rate"], 0.55) - 0.55,
        "fv":   blend(p["ev_align"], 0.5) - 0.5,
        "hfv":  blend(p["ev_align"], 0.5) - 0.5,
    }

    if _CALIBRATED:
        beta = list(_CALIBRATED["beta"])
        gamma = list(_CALIBRATED["gamma"])
        while len(beta) < 6:
            beta.append(0.8)
        while len(gamma) < 6:
            gamma.append(2.0)
        order = ["win", "ev", "pop", "home", "fv", "hfv"]
        w = {f"w_{k}": beta[i] + gamma[i] * devs[k]
             for i, k in enumerate(order)}
        w.update(_GLOBAL_W)
        return w

    w = {
        "w_win":  1.0 + 4.0 * devs["win"],
        "w_pop":  6.0 * devs["pop"],
        "w_home": 3.0 * devs["home"],
        "w_ev":   3.0 * devs["ev"],
        "w_fv":   1.5 * max(0.0, devs["fv"]),
        "w_hfv":  2.0 * max(0.0, devs["hfv"]),
    }
    w.update(_GLOBAL_W)
    return w


# ═══════════════════════════════════════════════════════════════════════════
# 3. UTILITY MATRICES — precomputed per (entry, week, team), fully vectorized
# ═══════════════════════════════════════════════════════════════════════════
def build_feature_tensors(week_team_data, future_value, weeks, holiday_fv=None):
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
        hfv = np.zeros(32); thu = np.zeros(32)
        for t, d in teams.items():
            i = TEAM_IDX[t]
            plays[i] = 1
            win[i], ev[i], pop[i], home[i] = d["win_pct"], d["ev"], d["pick_pct"], d["is_home"]
            fv[i] = future_value[w][t]
            thu[i] = d.get("is_thu", 0.0)
            if holiday_fv is not None:
                hfv[i] = holiday_fv[w][t]

        def z(x):
            m = plays.astype(bool)
            if m.sum() < 2:
                return x
            mu, sd = x[m].mean(), x[m].std()
            out = np.zeros_like(x)
            out[m] = (x[m] - mu) / (sd if sd > 1e-9 else 1)
            return out

        feats[w] = {"win": z(win), "ev": z(ev), "pop": z(pop),
                    "home": home, "fv": z(fv), "hfv": z(hfv), "thu": thu,
                    "plays": plays, "raw_win": win, "raw_pop": pop}
    return feats


def entry_utilities(weights, feats_w, stage=0.0):
    """Utility vector (32,) for one entry in one week.
    stage = log(alive/initial): survival-stage interactions shift the
    behavioral mix as the field shrinks. Team fixed effects (brand bias)
    are applied globally from the calibration."""
    u = (weights["w_win"]  * feats_w["win"]
       + weights["w_ev"]   * feats_w["ev"]
       + weights["w_pop"]  * feats_w["pop"]
       + weights["w_home"] * (feats_w["home"] - 0.5)
       - weights["w_fv"]   * feats_w["fv"]
       - weights.get("w_hfv", 0.0) * feats_w.get("hfv", 0.0))
    if stage != 0.0:
        u = u + stage * (weights.get("w_win_stage", 0.0) * feats_w["win"]
                       + weights.get("w_pop_stage", 0.0) * feats_w["pop"])
    thu = feats_w.get("thu")
    if thu is not None:
        u = u + weights.get("w_thu", 0.0) * thu
    return u + FE_SCALE * _TEAM_FE


def _zscore(x, plays_mask):
    m = plays_mask.astype(bool)
    if m.sum() < 2:
        return x
    mu, sd = x[m].mean(), x[m].std()
    out = np.zeros_like(x, dtype=float)
    out[m] = (x[m] - mu) / (sd if sd > 1e-9 else 1)
    return out


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


def apply_blend(field_pick_pct, feats, upcoming_week,
                alive_count=None, total_entries=None):
    """
    Blends behavioral and top-down predictions.
    Prefers the stacked ridge model (blend_model.json) when fitted;
    falls back to week-banded alpha (blend_alpha.json); else no-op.
    """
    import json as _json
    fw = feats[upcoming_week]
    topdown = fw.get("raw_pop")
    if topdown is None or topdown.sum() == 0:
        return field_pick_pct
    playing = fw["plays"].astype(bool)

    ridge_path = os.path.join("entry-analytics", "blend_model.json")
    if os.path.exists(ridge_path):
        with open(ridge_path) as f:
            m = _json.load(f)
        la = np.log(max(alive_count or 1, 1))
        X = np.stack([field_pick_pct, topdown,
                      np.full(32, upcoming_week, dtype=float),
                      np.full(32, la)], axis=1)
        pred = m["intercept"] + X @ np.array(m["coef"])
        pred = np.clip(pred, 0, None)
        pred[~playing] = 0
        s = pred.sum()
        return pred / s if s > 0 else field_pick_pct

    alpha_path = os.path.join("entry-analytics", "blend_alpha.json")
    if os.path.exists(alpha_path):
        with open(alpha_path) as f:
            bands = _json.load(f)["bands"]
        alpha = 0.5
        for band, a in bands.items():
            lo, hi = (int(x) for x in band.split("-"))
            if lo <= upcoming_week <= hi:
                alpha = a
                break
        blended = alpha * field_pick_pct + (1 - alpha) * topdown
        s = blended.sum()
        return blended / s if s > 0 else field_pick_pct

    return field_pick_pct


# ═══════════════════════════════════════════════════════════════════════════
# 4. UPCOMING WEEK PREDICTION — the fractional pick model
# ═══════════════════════════════════════════════════════════════════════════
def predict_upcoming_picks(alive_entries, used_masks, profiles, feats,
                           upcoming_week, contestant_of=None, stage=0.0,
                           n_iter=None, diversify_coeff=None):
    """
    Per-entry fractional pick prediction with two upgrades:

    SELF-CONSISTENCY: the 'pop' (herding) feature initially uses the top-down
    prediction; after each pass we replace it with our own aggregate and
    re-predict, converging to the herding fixed point — chalk entries respond
    to what the field will ACTUALLY do.

    MULTI-ENTRY COORDINATION: contestants with several entries diversify
    deliberately. Entries are processed grouped by contestant, with a utility
    penalty proportional to probability already allocated by that contestant's
    other entries — entry 2 avoids entry 1's team.
    """
    if n_iter is None:
        n_iter = SELF_CONSISTENCY_ITERS
    if diversify_coeff is None:
        diversify_coeff = DIVERSIFY_COEFF
    fw = dict(feats[upcoming_week])   # shallow copy — we mutate pop only
    plays = fw["plays"]

    # Group entries by contestant for coordination
    if contestant_of is None:
        contestant_of = {e: e for e in alive_entries}
    by_contestant = {}
    for e in alive_entries:
        by_contestant.setdefault(contestant_of.get(e, e), []).append(e)

    entry_dists, agg = {}, np.zeros(32)
    for it in range(max(1, n_iter)):
        entry_dists, agg = {}, np.zeros(32)
        for contestant, entries in by_contestant.items():
            alloc = np.zeros(32)   # this contestant's cumulative allocation
            for entry in entries:
                weights = profile_to_weights(profiles[entry])
                u = entry_utilities(weights, fw, stage=stage)
                if len(entries) > 1:
                    u = u - diversify_coeff * alloc
                dist = pick_distribution(u, ~used_masks[entry], plays)
                if dist is None:
                    continue
                entry_dists[entry] = dist
                agg += dist
                alloc += dist
        n = max(1, len(entry_dists))
        agg = agg / n
        # Feed our aggregate back in as the popularity feature — DAMPED
        # (full replacement overshoots: herding feedback piles mass onto the
        # leader; a partial step converges to the fixed point without runaway)
        if it < n_iter - 1:
            d = SELF_CONSISTENCY_DAMPING
            new_pop = d * agg + (1 - d) * fw["raw_pop"]
            fw["pop"] = _zscore(new_pop, plays)
            fw["raw_pop"] = new_pop

    return entry_dists, agg


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


def sample_pick_paths(entry, used_mask, weights, feats, weeks, n_paths, rng,
                      stage_by_week=None):
    """
    Samples n_paths pick sequences for one entry over remaining weeks.
    Returns int array (n_paths, n_weeks) of team indices, -1 = no valid pick.
    """
    paths = np.full((n_paths, len(weeks)), -1, dtype=np.int8)
    for k in range(n_paths):
        avail = ~used_mask.copy()
        for wi, w in enumerate(weeks):
            fw = feats[w]
            # A week with NO games at all is a data gap (or a scheduling quirk),
            # not an elimination. Mark it -2 so the survival loop carries the
            # entry forward instead of killing it. -1 stays "no legal pick".
            if fw["plays"].sum() == 0:
                paths[k, wi] = -2
                continue
            st = stage_by_week.get(w, 0.0) if stage_by_week else 0.0
            u = entry_utilities(weights, fw, stage=st)
            dist = pick_distribution(u, avail, fw["plays"])
            if dist is None:
                break
            pick = rng.choice(32, p=dist)
            paths[k, wi] = pick
            avail[pick] = False
    return paths


def run_season_simulation(alive_entries, used_masks, profiles, week_team_data,
                          feats, remaining_weeks, total_pot, rng,
                          stage_by_week=None):
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
                                  feats, weeks, N_PICK_PATHS, rng,
                                  stage_by_week=stage_by_week)  # (P, W)

        # survived[p, s] = all picks in path p won under sim s
        survived = np.ones((N_PICK_PATHS, N_OUTCOME_SIMS), dtype=bool)
        for wi in range(W):
            picks = paths[:, wi]                       # (P,)
            valid = picks >= 0
            skipped = picks == -2                      # no games that week
            # win lookup: outcome_wins[s, wi, pick]
            wk_wins = outcome_wins[:, wi, :]           # (S, 32)
            pick_won = np.ones((N_PICK_PATHS, N_OUTCOME_SIMS), dtype=bool)
            if valid.any():
                pick_won[valid] = wk_wins[:, picks[valid]].T   # (n_valid, S)
            # Ran out of legal picks (-1) = eliminated. No games (-2) = carry on.
            pick_won[~valid & ~skipped] = False
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
        if fw["plays"].sum() == 0:
            continue          # no games scheduled/known this week — skip it
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
        if fw["plays"].sum() == 0:
            continue
        cand = np.where(avail & fw["plays"].astype(bool))[0]
        if len(cand) == 0:
            break
        best = cand[np.argmax(ev_raw[w][cand])]
        ev_sum += ev_raw[w][best]
        avail[best] = False

    return win_prob, ev_sum


# ═══════════════════════════════════════════════════════════════════════════
# 7. PRESEASON SUPPORT — synthetic field generation
# ═══════════════════════════════════════════════════════════════════════════
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
    if not os.path.exists(picks_csv_path):
        print(f"⚠️  Picks file not found: {picks_csv_path} — "
              f"returning empty analytics.")
        return None, None   # match whatever the caller unpacks
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    week_team_data = build_week_team_data(sim_df, upcoming_week)
    max_week = max(week_team_data.keys()) if week_team_data else 20
    completed_weeks = list(range(1, upcoming_week))
    remaining_weeks = list(range(upcoming_week, max_week + 1))

    # Surface data gaps — a missing week silently zeroed survival before
    _gap_weeks = [w for w in range(upcoming_week, max_week + 1)
                  if w not in week_team_data]
    if _gap_weeks:
        print(f"   ⚠️  No game data for week(s) {_gap_weeks} — they will be "
              f"skipped in the simulation (check the season final-data file)")

    future_value = build_future_value(week_team_data, upcoming_week, max_week)
    holiday_weeks = detect_holiday_weeks(sim_df)
    if holiday_weeks:
        print(f"   Holiday weeks detected: {holiday_weeks}")
    holiday_fv = build_holiday_future_value(
        week_team_data, holiday_weeks, 1, max_week)
    feats = build_feature_tensors(week_team_data, future_value,
                                  completed_weeks + remaining_weeks,
                                  holiday_fv=holiday_fv)

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

        # Only weeks BEFORE the upcoming week are known at this point in time.
        # Iterating every week column would leak future picks — an entry that
        # finished the season would show teams from weeks 13-20 as already used
        # while you're viewing week 13.
        past_week_cols = [c for c in week_cols
                          if int(c.replace("Week_", "")) < upcoming_week]

        alive_entries, used_masks = [], {}
        for _, row in alive_df.iterrows():
            entry = row["EntryName"]
            mask = np.zeros(32, dtype=bool)
            for col in past_week_cols:
                t = norm_abbr(row.get(col, ""))
                if t in TEAM_IDX:
                    mask[TEAM_IDX[t]] = True
            alive_entries.append(entry)
            used_masks[entry] = mask
        print(f"   {len(alive_entries)} alive entries of {len(picks_df)} total")

        contestant_priors = None
        if prior_picks_csv and prior_season_df is not None:
            contestant_priors = build_contestant_priors(
                prior_picks_csv, prior_season_df)
        profiles = build_entry_profiles(picks_df, week_team_data,
                                        completed_weeks,
                                        contestant_priors=contestant_priors)
        total_entries = len(picks_df)
        contestant_of = dict(zip(picks_df["EntryName"],
                                 picks_df["EntryName"].astype(str).str.replace(
                                     r"-\d+$", "", regex=True)))
        # Wins AS OF this week — an entry that finished 20-0 should display 12
        # wins when you're viewing week 13, not its final total.
        wins_of = {e: min(int(w), upcoming_week - 1)
                   for e, w in zip(alive_df["EntryName"], alive_df["Total_Wins"])}
        is_synthetic = False

    # ── Everything below is unchanged from the original ──
    stage0 = float(np.log(max(len(alive_entries), 1) / max(total_entries, 1)))
    stage_by_week = {w: stage0 + np.log(SURVIVAL_DECAY) * (w - upcoming_week)
                     for w in remaining_weeks}

    entry_dists, field_pick_pct = predict_upcoming_picks(
        alive_entries, used_masks, profiles, feats, upcoming_week,
        contestant_of=contestant_of, stage=stage0)
    field_pick_pct = apply_blend(field_pick_pct, feats, upcoming_week,
                                 alive_count=len(alive_entries),
                                 total_entries=total_entries)

    total_pot = total_entries * ENTRY_FEE * POT_MULT
    survival, fair_value, avg_survivors = run_season_simulation(
        alive_entries, used_masks, profiles, week_team_data,
        feats, remaining_weeks, total_pot, rng,
        stage_by_week=stage_by_week)
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
