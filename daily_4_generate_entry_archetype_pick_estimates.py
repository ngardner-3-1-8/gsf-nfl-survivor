"""
generate_entry_archetype_pick_estimates.py

Uses the trained entry-choice model (entry_pick_choice_model_training.py)
to estimate, for every currently-alive entry in the current season, a
probability split over its available teams -- for the UPCOMING week (the
next one not yet played) AND every remaining week of the season, e.g.
"30% SF, 20% LAC, 15% PHI...".

THE UPCOMING WEEK VS. EVERY WEEK AFTER IT
============================================
The upcoming week uses the entry's real, true available pool (whatever
it has genuinely not used yet, per its actual pick history) and the
model's full feature set, including the real Public_Pick_Pct for that
week (Circa's live crowd consensus). Every week after that is a
projection: we don't yet know what the entry will actually pick between
now and then, and Public_Pick_Pct for a week that hasn't opened for
picking yet simply doesn't exist (confirmed empty in every sim file
beyond its first week) -- LightGBM handles that missing feature natively,
same as it already does for Predicted_Pick_Pct throughout training.

FRACTIONAL FORWARD SIMULATION (weeks beyond the upcoming one)
================================================================
Rather than a hard "assume the entry picks its #1 team, mark it used,
repeat" walk (which commits to one guess per week and compounds error),
or ignoring the shrinking pool entirely, each entry carries a continuous
per-team `remaining_weight` in [0, 1] forward from week to week:

  - Teams the entry has ACTUALLY already picked (real history through
    the last completed week) start at weight 0 -- hard-excluded, no
    uncertainty there.
  - Every other team starts at weight 1.0 (fully available).
  - Each week, a team's weight gets folded into the model's raw score
    as an additive log term before softmax: adjusted_utility = raw_logit
    + log(remaining_weight). This is the standard way to fold a
    continuous availability prior into a softmax choice model --
    exp(logit + log(w)) = exp(logit) * w, so a team's whole contribution
    to that week's probability (numerator AND the normalizing
    denominator) scales exactly by its remaining weight. A weight of
    exactly 0 (a really-already-used team) contributes exactly 0
    probability, so the "hard truth" case for the upcoming week falls
    out of this same formula with no special-casing needed.
  - After softmax gives this week's Estimated_Pick_Pct per team, decay:
        remaining_weight[team] *= (1 - Estimated_Pick_Pct[team])
    A team predicted at 30% this week loses 30% of its remaining
    eligibility for future weeks.
  - Entry survival compounds forward too, using each team's sportsbook
    win probability:
        alive_prob *= sum(Estimated_Pick_Pct[team] * team_win_prob[team])
    An entry that's fractionally more likely to have already busted out
    is recorded as such (Entry_Alive_Prob_Entering_Week in the output)
    but is NOT excluded from later weeks' files -- that's for whoever
    reads the output to weight by, not a reason to drop rows.

An entry ACTUALLY eliminated already (its most recent real pick is
recorded as a loss in the picks file) is dropped entirely -- no
projection is made for it.

INPUTS
======
- circa-pick-history/{year}_survivor_picks.csv (real pick history, and
  to determine who is currently alive)
- nfl-power-ratings/final_data/{year}_final_data/Week_{N}_{year}_Final_Data.csv
  (to score each entry's own REAL past picks, for archetype state --
  same as build_entry_pick_training_data.py)
- nfl-power-ratings/final_sim_results_with_variance_week_{upcoming}_{year}.csv
  (ONE file that covers the upcoming week through the end of the season
  -- confirmed from a real sample: a "week_10_2020" file contains rows
  for weeks 10 through 18 all at once, refreshed weekly as actual results
  roll in and the model's own "upcoming week" moves forward)
- models/entry_pick_choice_model.pkl + entry_pick_choice_model_features.json
  (from entry_pick_choice_model_training.py)

OUTPUT
======
entry_archetype_pick_estimates/{year}/week_{week}_entry_archetype_pick_estimates.csv
-- one file per remaining week (upcoming week through the season's last
week, both taken from the sim file itself, not hardcoded), each with one
row per (alive entry, candidate team) pair.

entry_archetype_pick_estimates/{year}/week_{week}_team_pick_estimates.csv
-- a companion per-TEAM summary for that same week: one row per candidate
team with its opponent, sportsbook win %, the pool-wide Estimated_Pick_Pct
(the per-entry Estimated_Pick_Pct values above, aggregated up to "what
fraction of the still-alive pool is expected to pick this team"), and an
Estimated_EV (see compute_estimated_ev below).

YEAR SELECTION
==============
Operates on the current/most recent season with a picks file present --
same "no hardcoded year" approach as the rest of this pipeline, so this
needs no edits at a season transition.
"""

import datetime
import glob
import json
import os
import re

import joblib
import numpy as np
import pandas as pd

from team_codes import canonical_pick_code
from contest_config import get_contest, tagged

_CFG = get_contest()
_IS_CIRCA = (_CFG['out_tag'] == 'circa')

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
# Contest-driven (SURVIVOR_CONTEST env var; default 'circa'). Splash
# contests start in 2026, so EARLIEST_SEASON comes from the config and a run
# for an earlier season finds no picks file and estimates nothing.
EARLIEST_SEASON = _CFG['start_season']

PICKS_PATTERN = _CFG['picks_pattern']
MULTI_PICK_WEEKS = _CFG['multi_pick_weeks']
FINAL_DATA_PATTERN = (
    "nfl-power-ratings/final_data/{year}_final_data/"
    "Week_{week}_{year}_Final_Data.csv"
)
SIM_FILE_PATTERN = "nfl-power-ratings/final_sim_results_with_variance_week_{week}_{year}.csv"

MODEL_PATH = tagged("models/entry_pick_choice_model", ".pkl")
FEATURE_META_PATH = tagged("models/entry_pick_choice_model_features", ".json")

# Splash contests write to their own tagged output dir so they never collide
# with Circa's estimates.
OUT_DIR_PATTERN = (tagged("entry_archetype_pick_estimates", "") + "/{year}")
OUT_FILE_PATTERN = "week_{week}_entry_archetype_pick_estimates.csv"
TEAM_OUT_FILE_PATTERN = "week_{week}_team_pick_estimates.csv"

TEAM_FULLNAME_TO_ABBR = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAC',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
}

# Same candidate feature map as build_entry_pick_training_data.py -- kept
# duplicated here so this script stays standalone, matching the rest of
# this pipeline's convention.
CANDIDATE_FEATURE_MAP = {
    'Win_Pct':               '{prefix} Team Fair Odds',
    'Sportsbook_EV':         'sportsbook_{prefix}_EV',
    'Future_Value':          '{prefix} Team Star Rating',
    'Public_Pick_Pct':       '{prefix} Team Public Pick %',
    'Predicted_Pick_Pct':    '{prefix} Predicted_Pick_Pct',
    'Expected_Availability': '{prefix} Team Expected Availability',
    'Pre_Thanksgiving':      '{prefix} Team Pre Thanksgiving',
    'Pre_Christmas':         '{prefix} Team Pre Christmas',
}
CANDIDATE_GAME_LEVEL_COLS = ['Divisional Matchup Boolean', 'Circa Week',
                              'Total Remaining Entries at Start of Week']

# The literal sportsbook win probability (distinct from the blended
# "Team Fair Odds" used as the Win_Pct MODEL feature) -- used only for
# the fractional survival simulation, per the user's explicit ask to use
# "the sportsbook win %" for that.
WIN_PROB_COL_TMPL = '{prefix} Team Sportsbook Fair Odds'

ARCHETYPE_SCORE_COLS = ['Planner', 'Contrarian', 'EV Hunter', 'Sprinter', 'Hoarder', 'Tourist']
DIM_COLS = ['win_pct_pctile', 'ev_pctile', 'fv_pctile', 'unpopularity_pctile']


# --------------------------------------------------------------------
# 0. Which season to run -- no hardcoded year, matches the rest of the
#    pipeline. Picks the highest year that actually has a picks file.
# --------------------------------------------------------------------
def current_season_year():
    candidates = []
    for year in range(EARLIEST_SEASON, datetime.date.today().year + 2):
        if os.path.exists(PICKS_PATTERN.format(year=year)):
            candidates.append(year)
    if not candidates:
        raise FileNotFoundError(
            f"No picks file found for any year {EARLIEST_SEASON}-"
            f"{datetime.date.today().year + 1} matching {PICKS_PATTERN}."
        )
    return max(candidates)


# --------------------------------------------------------------------
# 1. Picks file helpers -- same logic as build_entry_pick_training_data.py
# --------------------------------------------------------------------
def load_picks_long(picks_path):
    picks_wide = pd.read_csv(picks_path)
    week_cols = sorted(
        [c for c in picks_wide.columns if re.fullmatch(r'Week_\d+', c)],
        key=lambda c: int(c.split('_')[1]),
    )
    long_rows = []
    for wc in week_cols:
        week_num = int(wc.split('_')[1])
        sub = picks_wide[['EntryName', wc]].rename(columns={wc: 'Team'})
        sub = sub.dropna(subset=['Team'])
        # Canonicalize pick codes (e.g. 'LAR' -> 'LA') so an entry's used
        # teams and past picks line up with the model's team categories and
        # the FULLNAME_TO_ABBR-derived pool; otherwise 'LAR' picks neither
        # score nor get marked used.
        sub['Team'] = sub['Team'].astype(str).str.strip().map(canonical_pick_code)
        sub = sub[(sub['Team'] != '') & (sub['Team'] != 'ELIMINATED')]
        sub = sub.assign(Week=week_num)
        long_rows.append(sub)
    picks_long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
        columns=['EntryName', 'Team', 'Week'])
    return picks_long.sort_values(['EntryName', 'Week']).reset_index(drop=True), week_cols, picks_wide


def _resolve_col(df, candidates, prefix, label, path):
    for tmpl in candidates:
        col = tmpl.format(prefix=prefix)
        if col in df.columns:
            return df[col]
    tried = [t.format(prefix=prefix) for t in candidates]
    raise KeyError(f"{label} column not found for '{prefix}' in {path}. Tried: {tried}.")


def contest_actual_pick_pct(picks_long):
    """{week: {TEAM: observed_pick_pct}} for THIS contest, from its own
    canonicalized picks -- the crowd a Splash entry's past picks are scored
    against for the Contrarian dimension (Circa keeps final_data's pick %)."""
    out = {}
    for week, g in picks_long.groupby('Week'):
        counts = g['Team'].value_counts()
        total = int(counts.sum())
        if total > 0:
            out[int(week)] = (counts / total).to_dict()
    return out


def load_week_actual_table(path, actual_pick_override=None):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)

    def _side(prefix):
        return pd.DataFrame({
            'Team_Full': df[f'{prefix} Team'],
            'Win %': _resolve_col(
                df, ['{prefix} Team Sportsbook Fair Odds', 'Actual {prefix} Team Win %'],
                prefix, 'Win %', path),
            'EV': _resolve_col(
                df, ['sportsbook_{prefix}_EV', 'Actual {prefix} Team EV'],
                prefix, 'EV', path),
            'Future Value': df[f'{prefix} Team Star Rating'],
            'Actual Pick %': _resolve_col(
                df, ['{prefix} Pick %', '{prefix} Actual Pick %', 'Actual {prefix} Team Pick %'],
                prefix, 'Pick %', path),
        })

    long_df = pd.concat([_side('Home'), _side('Away')], ignore_index=True)
    long_df['Team'] = long_df['Team_Full'].map(TEAM_FULLNAME_TO_ABBR)
    long_df = long_df.dropna(subset=['Team']).drop(columns=['Team_Full'])

    # Contest-specific crowd for the Contrarian/unpopularity dimension: Circa
    # uses final_data's pick %; a Splash contest overrides with its own.
    if actual_pick_override is not None:
        long_df['Actual Pick %'] = long_df['Team'].map(actual_pick_override).fillna(0.0)

    long_df = long_df.dropna(subset=['Win %', 'EV', 'Future Value', 'Actual Pick %'])
    return long_df.reset_index(drop=True)


def load_week_outcomes(path):
    """Win/loss per team for an already-played week, from 'Actual {side}
    Team Win %' -- confirmed to be exactly 1.0 for a win and 0.0 for a
    loss. Per the user: treat anything other than 1.0 as a loss (a tie
    is a loss in Circa Survivor, and this also degrades safely if some
    other non-1.0/0.0 value ever shows up). Used only to catch entries
    eliminated on the MOST RECENT completed week -- see the comment at
    its call site for why that can't be read off the picks file's
    'ELIMINATED' markers alone."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)

    def _side(prefix):
        col = f'Actual {prefix} Team Win %'
        if col not in df.columns:
            return None
        return pd.DataFrame({
            'Team_Full': df[f'{prefix} Team'],
            'Won': df[col] == 1.0,
        })

    sides = [s for s in (_side('Home'), _side('Away')) if s is not None]
    if not sides:
        return None
    long_df = pd.concat(sides, ignore_index=True)
    long_df['Team'] = long_df['Team_Full'].map(TEAM_FULLNAME_TO_ABBR)
    long_df = long_df.dropna(subset=['Team'])
    return long_df.set_index('Team')['Won'].to_dict()


# --------------------------------------------------------------------
# 2. Per-pick percentile scores -- identical methodology to weekly_4 /
#    build_entry_pick_training_data.py, used only to score entries' OWN
#    REAL past picks (already-realized weeks).
# --------------------------------------------------------------------
def compute_pick_scores(picks_long, week_actual_tables):
    records = []
    used_before_map = {}
    seen = {}
    for entry, team in zip(picks_long['EntryName'], picks_long['Team']):
        used_before_map.setdefault(entry, []).append(frozenset(seen.get(entry, set())))
        seen.setdefault(entry, set()).add(team)

    idx_counters = {}
    for entry, picked_team, week in zip(picks_long['EntryName'], picks_long['Team'], picks_long['Week']):
        i = idx_counters.get(entry, 0)
        used_before = used_before_map[entry][i]
        idx_counters[entry] = i + 1

        week_tbl = week_actual_tables.get(week)
        if week_tbl is None:
            continue
        pool = week_tbl[~week_tbl['Team'].isin(used_before)]
        if picked_team not in pool['Team'].values or len(pool) < 2:
            continue

        def pctile(col, higher_is_better):
            ranks = pool[col].rank(pct=True, method='average')
            val = ranks[pool['Team'] == picked_team].iloc[0]
            return val if higher_is_better else 1 - val

        win_p = pctile('Win %', True)
        ev_p = pctile('EV', True)
        fv_p = pctile('Future Value', True)
        unpop_p = pctile('Actual Pick %', False)

        top_fv_team = pool.loc[pool['Future Value'].idxmax(), 'Team']
        hoarder = 0.0 if top_fv_team == picked_team else (1 - fv_p) * win_p
        dims = np.array([ev_p, win_p, fv_p])
        planner = max(0.0, 1 - dims.std() / 0.5)

        records.append({
            'EntryName': entry, 'Week': week,
            'win_pct_pctile': win_p, 'ev_pctile': ev_p, 'fv_pctile': fv_p,
            'unpopularity_pctile': unpop_p,
            'Sprinter_pick': win_p, 'EV_Hunter_pick': ev_p,
            'Contrarian_pick': unpop_p, 'Hoarder_pick': hoarder,
            'Planner_pick': planner,
        })
    return pd.DataFrame.from_records(records)


def _dominant_dim(row):
    vals = {d: row[d] for d in DIM_COLS}
    return max(vals, key=vals.get)


# --------------------------------------------------------------------
# 3. Entry state "as of right now" -- inclusive of the entry's last REAL
#    pick, for predicting the NOT-yet-played upcoming pick. An expanding
#    mean evaluated at the last row is the same as a plain mean over all
#    rows, so this is the training script's compute_asof_entry_states
#    one step further forward, computed directly.
# --------------------------------------------------------------------
def compute_current_entry_state(pick_scores):
    cols = ['Picks_Used_So_Far', 'Primary_Archetype_AsOf'] + ARCHETYPE_SCORE_COLS
    if pick_scores.empty:
        return pd.DataFrame(columns=cols).rename_axis('EntryName')

    ps = pick_scores.copy()
    ps['Dominant_Dim'] = ps.apply(_dominant_dim, axis=1)
    ps['Dist_From_Mid'] = (ps[DIM_COLS] - 0.5).abs().mean(axis=1)
    ps['Middling_This_Pick'] = 1 - ps['Dist_From_Mid'] / 0.5

    pick_cols = ['Planner_pick', 'Contrarian_pick', 'EV_Hunter_pick', 'Sprinter_pick', 'Hoarder_pick']
    out_cols = ['Planner', 'Contrarian', 'EV Hunter', 'Sprinter', 'Hoarder']
    state = ps.groupby('EntryName')[pick_cols].mean()
    state.columns = out_cols

    middling_mean = ps.groupby('EntryName')['Middling_This_Pick'].mean()

    ps_sorted = ps.sort_values(['EntryName', 'Week']).copy()
    prev_dom = ps_sorted.groupby('EntryName')['Dominant_Dim'].shift(1)
    switched = (ps_sorted['Dominant_Dim'] != prev_dom).astype(float)
    switched[prev_dom.isna()] = np.nan
    ps_sorted['_switch'] = switched
    inconsistency_mean = ps_sorted.groupby('EntryName')['_switch'].mean()

    picks_used = ps.groupby('EntryName').size()

    tourist = pd.Series(index=state.index, dtype=float)
    for name in state.index:
        n = picks_used.get(name, 0)
        mid = middling_mean.get(name, np.nan)
        inc = inconsistency_mean.get(name, np.nan)
        tourist[name] = (0.5 * mid + 0.5 * inc) if n >= 2 else mid
    state['Tourist'] = tourist

    for c in ARCHETYPE_SCORE_COLS:
        state[c] = state[c] * 100

    state['Picks_Used_So_Far'] = picks_used
    state['Primary_Archetype_AsOf'] = state[ARCHETYPE_SCORE_COLS].idxmax(axis=1)
    no_state = state[ARCHETYPE_SCORE_COLS].isna().all(axis=1)
    state.loc[no_state, 'Primary_Archetype_AsOf'] = 'No_Prior_Picks'

    return state.rename_axis('EntryName')


# --------------------------------------------------------------------
# 4. Candidate table for one week, from the already-loaded full-season
#    sim file (build_entry_pick_training_data.py's load_week_candidate_
#    table, but reading an in-memory slice instead of a per-week file).
# --------------------------------------------------------------------
OPPOSITE_PREFIX = {'Home': 'Away', 'Away': 'Home'}


def build_candidate_table_from_sim(sim_df, week):
    wk_df = sim_df[sim_df['Week'] == week]
    if wk_df.empty:
        return pd.DataFrame()

    frames = []
    for prefix in ('Home', 'Away'):
        opp_prefix = OPPOSITE_PREFIX[prefix]
        cols = {
            'Team_Full': wk_df[f'{prefix} Team'],
            # Each team's opponent that week -- needed for the per-team
            # summary output (week_{week}_team_pick_estimates.csv), not
            # used as a model feature.
            'Opponent_Full': wk_df[f'{opp_prefix} Team'],
        }
        for out_name, tmpl in CANDIDATE_FEATURE_MAP.items():
            col = tmpl.format(prefix=prefix)
            cols[out_name] = wk_df[col] if col in wk_df.columns else np.nan
        for game_col in CANDIDATE_GAME_LEVEL_COLS:
            cols[game_col] = wk_df[game_col] if game_col in wk_df.columns else np.nan
        win_prob_col = WIN_PROB_COL_TMPL.format(prefix=prefix)
        cols['Win_Prob_For_Survival'] = wk_df[win_prob_col] if win_prob_col in wk_df.columns else np.nan
        frames.append(pd.DataFrame(cols))

    long_df = pd.concat(frames, ignore_index=True)
    long_df['Team'] = long_df['Team_Full'].map(TEAM_FULLNAME_TO_ABBR)
    long_df['Opponent'] = long_df['Opponent_Full'].map(TEAM_FULLNAME_TO_ABBR)
    unmapped = long_df.loc[long_df['Team'].isna(), 'Team_Full'].unique()
    if len(unmapped):
        print(f"   ⚠️  Week {week}: team name(s) didn't map to an abbreviation "
              f"and will be dropped: {list(unmapped)}.")
    long_df = long_df.dropna(subset=['Team']).drop(columns=['Team_Full', 'Opponent_Full'])

    for feat_col in ['Win_Pct', 'Sportsbook_EV', 'Future_Value']:
        if feat_col in long_df.columns:
            long_df[feat_col + '_Pctile'] = long_df[feat_col].rank(pct=True, method='average')

    return long_df.reset_index(drop=True)


# --------------------------------------------------------------------
# 4b. Estimated EV for the per-team summary output. The pipeline already
#    has a "sportsbook_{prefix}_EV" concept elsewhere, computed from the
#    real/observed Public Pick %, which only exists for a week that has
#    already opened for picking. For a future week that hasn't opened
#    yet, the only "pick %" available is this script's own simulated,
#    pool-wide Estimated_Pick_Pct, so EV here is computed from that
#    instead: win probability divided by how much of the still-alive
#    pool is expected to also pick that team. This rewards exactly what
#    makes a pick valuable in a knockout/survivor pool -- safe (high win
#    probability) AND differentiated (few others also surviving on it) --
#    the same intuition as the existing sportsbook EV figure, just fed
#    with an estimated pick share instead of an observed one.
# --------------------------------------------------------------------
def compute_estimated_ev(win_prob, pick_pct, min_pick_pct=1e-4):
    safe_pick_pct = np.clip(pick_pct, min_pick_pct, 1.0)
    return win_prob / safe_pick_pct


# --------------------------------------------------------------------
# 5. Robust categorical matching -- LightGBM encodes Team / Primary_
#    Archetype_AsOf / Circa Week as fixed integer codes learned at
#    training time (model.booster_.pandas_categorical carries the exact
#    category lists). A value from a fresh source file (e.g. sim file
#    "Circa Week" values are plain "10", but training data's were
#    recovered from a mixed-type Parquet column that may have gone
#    through as "10.0") needs to land on the SAME category string the
#    model learned, or the prediction is silently wrong, not an error.
#    This tries a few plausible reformattings and reports anything that
#    still doesn't match instead of guessing wrong silently.
# --------------------------------------------------------------------
def match_category(value, known_categories, label):
    known_set = set(known_categories)
    candidates = [str(value)]
    try:
        f = float(value)
        candidates.append(str(f))
        if f.is_integer():
            candidates.append(str(int(f)))
    except (TypeError, ValueError):
        pass
    for c in candidates:
        if c in known_set:
            return c
    print(f"   ⚠️  {label} value {value!r} didn't match any category the model "
          f"was trained on (tried {candidates}); treating as unknown/missing "
          f"for rows with this value.")
    return None


def map_to_known_categories(series, known_categories, label):
    uniques = series.dropna().unique()
    lookup = {v: match_category(v, known_categories, label) for v in uniques}
    return series.map(lookup)


# --------------------------------------------------------------------
# 6. Main per-season run
# --------------------------------------------------------------------
def run_pick_estimates(year, model, all_features, cat_lookup):
    team_cats, archetype_cats, circaweek_cats = cat_lookup

    picks_path = PICKS_PATTERN.format(year=year)
    picks_long, week_cols, picks_wide = load_picks_long(picks_path)
    if picks_long.empty:
        print(f"⚠️  {year}: no real picks recorded yet, nothing to estimate.")
        return

    last_completed_week = int(picks_long['Week'].max())
    upcoming_week = last_completed_week + 1
    print(f"{year}: last completed week {last_completed_week} -> "
          f"estimating from week {upcoming_week} through season end.")

    # Marker-based elimination: catches anyone eliminated more than one
    # week ago -- the picks file already shows 'ELIMINATED' for them.
    # Restricted to weeks that have actually happened -- a picks file
    # that pre-creates every week's column for the whole season could
    # default a not-yet-reached week to 'ELIMINATED' as a "no valid pick
    # yet" placeholder, which would otherwise make every entry look
    # eliminated the moment any future column exists, unfilled.
    completed_week_cols = [c for c in week_cols if int(c.split('_')[1]) <= last_completed_week]
    elim_mask = (picks_wide[completed_week_cols].astype(str) == 'ELIMINATED').any(axis=1)
    alive_names = set(picks_wide.loc[~elim_mask, 'EntryName'])

    # Cross-check the LAST completed week's real pick against the actual
    # game outcome. This catches entries eliminated on the MOST RECENT
    # week, which the picks file can't show yet -- an 'ELIMINATED'
    # marker only gets written into the FOLLOWING week's column once the
    # picks file is next refreshed, which doesn't exist yet for the week
    # that just finished.
    last_week_path = FINAL_DATA_PATTERN.format(year=year, week=last_completed_week)
    outcomes = load_week_outcomes(last_week_path)
    if outcomes is None:
        print(f"   ⚠️  Couldn't load win/loss outcomes for week {last_completed_week} "
              f"({last_week_path}) -- can't catch freshly-eliminated entries this "
              f"run; relying on 'ELIMINATED' markers only.")
    else:
        last_week_picks = picks_long[picks_long['Week'] == last_completed_week] \
            .set_index('EntryName')['Team']
        freshly_out = set()
        for name, team in last_week_picks.items():
            if name not in alive_names:
                continue
            won = outcomes.get(team)
            if won is False:
                freshly_out.add(name)
            elif won is None:
                print(f"   ⚠️  No outcome found for team {team!r} (entry "
                      f"{name!r}'s week {last_completed_week} pick) -- "
                      f"leaving its alive status as-is.")
        if freshly_out:
            print(f"   {len(freshly_out)} entr{'y is' if len(freshly_out) == 1 else 'ies are'} "
                  f"freshly eliminated (week {last_completed_week} pick lost, not yet "
                  f"marked 'ELIMINATED' in the picks file) -- excluding from projections.")
            alive_names -= freshly_out
    picks_long_alive = picks_long[picks_long['EntryName'].isin(alive_names)]
    if picks_long_alive.empty:
        print(f"⚠️  {year}: no currently-alive entries with real pick history found.")
        return
    print(f"   {picks_long_alive['EntryName'].nunique():,} alive entries with pick history.")

    used_teams = picks_long_alive.groupby('EntryName')['Team'].apply(set).to_dict()

    # Score past picks' unpopularity against this contest's own full field
    # (all entries that week, not just the currently-alive subset) -- Circa
    # keeps final_data's pick % (override stays None).
    contest_pick_pct = None if _IS_CIRCA else contest_actual_pick_pct(picks_long)

    actual_tables = {}
    for week in sorted(picks_long_alive['Week'].unique()):
        path = FINAL_DATA_PATTERN.format(year=year, week=week)
        override = None if contest_pick_pct is None else contest_pick_pct.get(week, {})
        try:
            actual_tables[week] = load_week_actual_table(path, actual_pick_override=override)
        except Exception as e:
            print(f"   ⚠️  {year} Week {week}: couldn't load actual table ({path}): {e}")
            actual_tables[week] = None

    pick_scores = compute_pick_scores(picks_long_alive, actual_tables)
    current_state = compute_current_entry_state(pick_scores)
    current_state = current_state.reindex(sorted(used_teams.keys()))
    current_state['Picks_Used_So_Far'] = current_state['Picks_Used_So_Far'].fillna(0)
    current_state['Primary_Archetype_AsOf'] = current_state['Primary_Archetype_AsOf'].fillna('No_Prior_Picks')

    sim_path = SIM_FILE_PATTERN.format(week=upcoming_week, year=year)
    if not os.path.exists(sim_path):
        print(f"⚠️  {year}: sim file not found ({sim_path}) -- can't build "
              f"estimates for week {upcoming_week} onward yet.")
        return
    sim_df = pd.read_csv(sim_path, low_memory=False)
    remaining_weeks = sorted(sim_df['Week'].unique())
    print(f"   Sim file covers weeks {remaining_weeks}.")

    entries_df = current_state.reset_index()
    entry_names = entries_df['EntryName'].tolist()

    teams_df = pd.DataFrame({'Team': team_cats})
    remaining_weight_df = entries_df[['EntryName']].merge(teams_df, how='cross')
    remaining_weight_df['Weight'] = 1.0
    used_long_rows = [(name, t) for name, teams in used_teams.items() for t in teams]
    if used_long_rows:
        used_long = pd.DataFrame(used_long_rows, columns=['EntryName', 'Team'])
        used_long['_used'] = True
        remaining_weight_df = remaining_weight_df.merge(used_long, on=['EntryName', 'Team'], how='left')
        remaining_weight_df.loc[remaining_weight_df['_used'] == True, 'Weight'] = 0.0
        remaining_weight_df = remaining_weight_df.drop(columns='_used')

    alive_prob_df = entries_df[['EntryName']].copy()
    alive_prob_df['AliveProb'] = 1.0

    for week in remaining_weeks:
        is_upcoming = bool(week == upcoming_week)
        cand = build_candidate_table_from_sim(sim_df, week)
        if cand.empty:
            print(f"   ⚠️  Week {week}: no candidate teams resolved from sim file, skipping.")
            continue

        pool = entries_df.merge(cand, how='cross')
        pool = pool.merge(remaining_weight_df, on=['EntryName', 'Team'], how='left')
        pool['Weight'] = pool['Weight'].fillna(0.0)
        pool = pool[pool['Weight'] > 1e-9].copy()
        if pool.empty:
            print(f"   Week {week}: every alive entry has exhausted every team playing -- skipping.")
            continue

        pool['Teams_Remaining_In_Pool'] = pool.groupby('EntryName')['Team'].transform('count')
        pool['Week'] = week

        pool['Team'] = map_to_known_categories(pool['Team'], team_cats, 'Team')
        pool['Primary_Archetype_AsOf'] = map_to_known_categories(
            pool['Primary_Archetype_AsOf'], archetype_cats, 'Primary_Archetype_AsOf')
        pool['Circa Week'] = map_to_known_categories(pool['Circa Week'], circaweek_cats, 'Circa Week')

        X = pool.copy()
        X['Team'] = pd.Categorical(X['Team'], categories=team_cats)
        X['Primary_Archetype_AsOf'] = pd.Categorical(X['Primary_Archetype_AsOf'], categories=archetype_cats)
        X['Circa Week'] = pd.Categorical(X['Circa Week'], categories=circaweek_cats)

        missing_cols = [c for c in all_features if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Week {week}: pool is missing model feature(s): {missing_cols}")

        raw_logits = model.predict(X[all_features], raw_score=True)
        pool['adj_utility'] = raw_logits + np.log(np.clip(pool['Weight'].values, 1e-12, 1.0))

        pool['u_shift'] = pool.groupby('EntryName')['adj_utility'].transform('max')
        pool['exp_u'] = np.exp(pool['adj_utility'] - pool['u_shift'])
        pool['sum_exp'] = pool.groupby('EntryName')['exp_u'].transform('sum')
        pool['Estimated_Pick_Pct'] = pool['exp_u'] / pool['sum_exp']

        pool = pool.merge(alive_prob_df, on='EntryName', how='left')

        out_df = pool[['EntryName', 'Primary_Archetype_AsOf', 'Picks_Used_So_Far',
                        'Team', 'Estimated_Pick_Pct', 'AliveProb']].copy()
        out_df = out_df.rename(columns={'AliveProb': 'Entry_Alive_Prob_Entering_Week'})
        out_df.insert(0, 'Week', int(week))
        out_df.insert(0, 'Year', year)
        out_df['Is_Upcoming_Week'] = is_upcoming
        out_df = out_df.sort_values(['EntryName', 'Estimated_Pick_Pct'], ascending=[True, False])

        out_dir = OUT_DIR_PATTERN.format(year=year)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, OUT_FILE_PATTERN.format(week=int(week)))
        out_df.to_csv(out_path, index=False)
        print(f"   ✅ Week {week}: wrote {len(out_df):,} rows "
              f"({out_df['EntryName'].nunique():,} entries) to {out_path}")

        # ---- Per-team pool-wide summary for this week -----------------
        # Aggregates the per-entry Estimated_Pick_Pct values up to "what
        # fraction of the STILL-ALIVE pool is expected to pick this team",
        # weighted by each entry's own Entry_Alive_Prob_Entering_Week
        # (pool['AliveProb'] here) so an entry that's fractionally more
        # likely to already be busted out (per the survival simulation)
        # doesn't count as a full vote for whatever it would have picked.
        pool['_alive_weighted_pick'] = pool['Estimated_Pick_Pct'] * pool['AliveProb']
        team_pick_sum = pool.groupby('Team')['_alive_weighted_pick'].sum().rename('Alive_Weighted_Pick_Sum')
        # AliveProb is one value per entry (repeated across its candidate-
        # team rows), so de-dup by EntryName before summing it as the
        # normalizing denominator.
        total_alive_weight = pool.drop_duplicates('EntryName')['AliveProb'].sum()

        team_summary = cand[['Team', 'Opponent', 'Win_Prob_For_Survival']].drop_duplicates('Team').copy()
        team_summary = team_summary.merge(team_pick_sum, on='Team', how='left')
        team_summary['Alive_Weighted_Pick_Sum'] = team_summary['Alive_Weighted_Pick_Sum'].fillna(0.0)
        if total_alive_weight > 0:
            team_summary['Estimated_Pick_Pct'] = team_summary['Alive_Weighted_Pick_Sum'] / total_alive_weight
        else:
            team_summary['Estimated_Pick_Pct'] = np.nan
        team_summary['Estimated_EV'] = compute_estimated_ev(
            team_summary['Win_Prob_For_Survival'], team_summary['Estimated_Pick_Pct'])

        team_summary = team_summary.rename(columns={'Win_Prob_For_Survival': 'Sportsbook_Win_Pct'})
        team_summary = team_summary[['Team', 'Opponent', 'Sportsbook_Win_Pct',
                                      'Estimated_Pick_Pct', 'Estimated_EV']]
        team_summary.insert(0, 'Week', int(week))
        team_summary.insert(0, 'Year', year)
        team_summary['Is_Upcoming_Week'] = is_upcoming
        team_summary = team_summary.sort_values('Estimated_Pick_Pct', ascending=False)

        team_out_path = os.path.join(out_dir, TEAM_OUT_FILE_PATTERN.format(week=int(week)))
        team_summary.to_csv(team_out_path, index=False)
        print(f"   ✅ Week {week}: wrote {len(team_summary):,} team rows to {team_out_path}")

        decay = pool[['EntryName', 'Team', 'Estimated_Pick_Pct']].copy()
        decay['DecayFactor'] = 1 - decay['Estimated_Pick_Pct']
        remaining_weight_df = remaining_weight_df.merge(
            decay[['EntryName', 'Team', 'DecayFactor']], on=['EntryName', 'Team'], how='left')
        remaining_weight_df['DecayFactor'] = remaining_weight_df['DecayFactor'].fillna(1.0)
        remaining_weight_df['Weight'] = remaining_weight_df['Weight'] * remaining_weight_df['DecayFactor']
        remaining_weight_df = remaining_weight_df.drop(columns='DecayFactor')

        pool['_survive_contrib'] = pool['Estimated_Pick_Pct'] * pool['Win_Prob_For_Survival'].fillna(0.0)
        week_survive = pool.groupby('EntryName')['_survive_contrib'].sum().rename('WeekSurvive')
        alive_prob_df = alive_prob_df.merge(week_survive, on='EntryName', how='left')
        alive_prob_df['WeekSurvive'] = alive_prob_df['WeekSurvive'].fillna(1.0)
        alive_prob_df['AliveProb'] = alive_prob_df['AliveProb'] * alive_prob_df['WeekSurvive']
        alive_prob_df = alive_prob_df.drop(columns='WeekSurvive')


def main():
    year = current_season_year()
    print(f"Running entry archetype pick estimates for the {year} season.")

    model = joblib.load(MODEL_PATH)
    with open(FEATURE_META_PATH) as f:
        meta = json.load(f)
    all_features = meta['features']

    cat_lookup = model.booster_.pandas_categorical
    if not cat_lookup or len(cat_lookup) != 3:
        raise ValueError(
            "Expected the model to carry exactly 3 categorical feature "
            f"mappings (Team, Primary_Archetype_AsOf, Circa Week), got: {cat_lookup}"
        )

    run_pick_estimates(year, model, all_features, cat_lookup)


if __name__ == '__main__':
    main()
