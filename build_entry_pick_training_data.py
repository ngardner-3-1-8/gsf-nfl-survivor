"""
build_entry_pick_training_data.py

Assembles a historical training set for predicting each SURVIVOR ENTRY's
next pick as a probability distribution over its available teams (e.g.
"30% SF, 20% LAC, 15% PHI...").

WHY THIS SHAPE (choice-set expansion)
======================================
Each entry picks exactly one team from whatever it still has left that
week -- a "discrete choice" problem over a variable, entry-specific set
of options, not a fixed 32-way classification. The standard way to train
that: expand every real pick into one row per team the entry COULD have
picked that week, label the team that WAS picked 1 and every alternative
0, train a binary scorer on those rows, then at inference time normalize
that scorer's outputs (softmax) within one entry's choice set to get a
probability split. This script produces that expanded, labeled table.
Fitting the actual model is a separate, later script.

TWO DIFFERENT LEAKAGE RULES -- THIS IS THE PART THAT MATTERS
===============================================================
1. ENTRY-STATE features (Planner/Contrarian/EV Hunter/Sprinter/Hoarder/
   Tourist, Picks_Used) going INTO week N must be computed using ONLY
   that entry's picks from weeks STRICTLY BEFORE N. These are "as-of"
   values, recomputed per week with an expanding-then-shifted mean --
   NOT the full-season archetype weekly_4_identify_entry_archetypes.py
   produces, which looks at the whole season including picks that
   happen AFTER any given week.

2. CANDIDATE TEAM features for the week actually being chosen (that
   team's Win%, EV, Public Pick%, etc.) must be values that existed
   BEFORE that week's picks closed -- never anything derived from what
   actually happened in that week's games. Season_{year}_Through_Week_
   {N}_Final_Data.csv is a superset file: it keeps the ORIGINAL pre-game
   columns (Team Fair Odds, sportsbook_*_EV, Public Pick %,
   Predicted_Pick_Pct, Splash Pick %, Pre-Thanksgiving/Christmas flags)
   AND appends "Actual ..." / "... Points" / "... Win %" columns after
   the games are played. This script reads ONLY the pre-game columns
   for the week being predicted -- the same columns a genuine pre-game
   file like final_sim_results_with_variance_week_{N}_{year}.csv would
   have. Training on the "Actual" columns here would mean training on
   information that doesn't exist yet at real prediction time -- the
   same class of train/inference mismatch this pipeline has hit before.

   Contrast this with entry-state features: those look at PAST weeks,
   which are fully realized by the time week N starts, so it's correct
   for them to use the same "Actual Pick %"-based percentile scoring
   weekly_4 already uses. Only the CURRENT week's candidate features
   are restricted to pre-game columns.

WEEK-DEPENDENT ARCHETYPE BEHAVIOR
===================================
An EV Hunter's choice in Week 1 (16+ teams, no constraints) and Week 16
(maybe 3 teams left, holiday teams already burned or saved on purpose)
aren't the same decision even though the label is the same. Rather than
trust the model to infer this from Week alone, this script also emits
Teams_Remaining_In_Pool and holiday-distance features so the model can
learn archetype x season-stage interactions explicitly.

INPUTS (per year in YEARS_TO_PROCESS)
======================================
- circa-pick-history/{year}_survivor_picks.csv
- nfl-power-ratings/final_data/{year}_final_data/
      Season_{year}_Through_Week_{N}_Final_Data.csv
  for every week N that has at least one actual pick that year.

OUTPUT
======
training_data/entry_pick_choice_training_data.csv
One row per (Year, Week, EntryName, candidate Team). `Picked` is the
label (1 for exactly one candidate row per Year/Week/EntryName group,
0 for the rest).
"""

import os
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
YEARS_TO_PROCESS = [2020, 2021, 2022, 2023, 2024, 2025]  # completed seasons

PICKS_PATTERN = "circa-pick-history/{year}_survivor_picks.csv"
FINAL_DATA_PATTERN = (
    "nfl-power-ratings/final_data/{year}_final_data/"
    "Week_{week}_{year}_Final_Data.csv"
)
OUT_PATH = "training_data/entry_pick_choice_training_data.csv"

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

DIM_COLS = ['win_pct_pctile', 'ev_pctile', 'fv_pctile', 'unpopularity_pctile']
ARCHETYPE_SCORE_COLS = ['Planner', 'Contrarian', 'EV Hunter', 'Sprinter', 'Hoarder', 'Tourist']

# Pre-game-only candidate features. Every one of these must exist BEFORE
# a week's picks close -- do not add "Actual ..." / "... Points" / "...
# Win %" columns here, see module docstring.
#
# NOTE on Win_Pct: intentionally "{prefix} Team Fair Odds", NOT "{prefix}
# Team Sportsbook Fair Odds". weekly_1_collect_last_weeks_pick_data.py
# (lines ~2795-2796) OVERWRITES "... Sportsbook Fair Odds" with the real
# CLOSING line pulled from nflreadpy after that week's games are played,
# as part of building this exact file -- so by the time it's saved, that
# column is a post-hoc value for the week being predicted, even though
# it's the right (and requested) column to use for scoring an entry's
# ALREADY-REALIZED past weeks (see load_week_actual_table below, which
# does use "... Sportsbook Fair Odds" on purpose). "{prefix} Team Fair
# Odds" is a separate, earlier-pipeline blended field that isn't part of
# that overwrite. If that turns out not to be true, swap this one line.
#
# Predicted_Pick_Pct is included but has come back 100% NaN in every
# file checked so far (2025 and 2026 alike) -- kept here in case it gets
# populated later, but don't expect signal from it yet.
CANDIDATE_FEATURE_MAP = {
    'Win_Pct':            '{prefix} Team Fair Odds',
    'Sportsbook_EV':      'sportsbook_{prefix}_EV',
    'Future_Value':       '{prefix} Team Star Rating',
    'Public_Pick_Pct':    '{prefix} Team Public Pick %',
    'Predicted_Pick_Pct': '{prefix} Predicted_Pick_Pct',
    'Expected_Availability': '{prefix} Team Expected Availability',
    'Pre_Thanksgiving':   '{prefix} Team Pre Thanksgiving',
    'Pre_Christmas':      '{prefix} Team Pre Christmas',
}
CANDIDATE_GAME_LEVEL_COLS = ['Divisional Matchup Boolean', 'Circa Week',
                              'Total Remaining Entries at Start of Week']


# --------------------------------------------------------------------
# 1. Picks file -> long format + each entry's available pool per pick
#    (mirrors weekly_4_identify_entry_archetypes.py's load_picks_long /
#    attach_available_pool -- kept duplicated here so this script stays
#    standalone like the rest of the pipeline; if you change the NaN/
#    ELIMINATED handling in weekly_4, mirror it here too.)
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
        sub['Team'] = sub['Team'].astype(str).str.strip()
        sub = sub[(sub['Team'] != '') & (sub['Team'] != 'ELIMINATED')]
        sub = sub.assign(Week=week_num)
        long_rows.append(sub)

    picks_long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
        columns=['EntryName', 'Team', 'Week'])
    picks_long = picks_long.sort_values(['EntryName', 'Week']).reset_index(drop=True)
    return picks_long


def attach_available_pool(picks_long):
    used_before = []
    seen = {}
    for entry, team in zip(picks_long['EntryName'], picks_long['Team']):
        prior = seen.get(entry, set())
        used_before.append(frozenset(prior))
        seen.setdefault(entry, set()).add(team)
    out = picks_long.copy()
    out['Used_Before_This_Week'] = used_before
    return out


# --------------------------------------------------------------------
# 2. Weekly team tables
#    (a) "actual" table -- Actual Pick %, used ONLY to score entries'
#        OWN past picks (weeks already realized by week N).
#    (b) "candidate" table -- pre-game-only columns, used for the
#        options available IN week N, the week being predicted.
# --------------------------------------------------------------------
def load_week_actual_table(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)

    def _side(prefix):
        return pd.DataFrame({
            'Team_Full': df[f'{prefix} Team'],
            # Sportsbook Fair Odds on purpose here (unlike the candidate
            # table below) -- this is scoring an ALREADY-PLAYED week, so
            # the real closing line is the right number to rank against,
            # per the user-specified "official" columns.
            'Win %': df[f'{prefix} Team Sportsbook Fair Odds'],
            'EV': df[f'sportsbook_{prefix}_EV'],
            'Future Value': df[f'{prefix} Team Star Rating'],
            'Actual Pick %': df[f'{prefix} Actual Pick %'],
        })

    long_df = pd.concat([_side('Home'), _side('Away')], ignore_index=True)
    long_df['Team'] = long_df['Team_Full'].map(TEAM_FULLNAME_TO_ABBR)
    long_df = long_df.dropna(subset=['Team']).drop(columns=['Team_Full'])
    long_df = long_df.dropna(subset=['Win %', 'EV', 'Future Value', 'Actual Pick %'])
    return long_df.reset_index(drop=True)


def load_week_candidate_table(path):
    """Pre-game-only features for every team playing that week -- this
    is what a genuine pre-game file (final_sim_results_with_variance_
    week_{N}_{year}.csv) would also have."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)

    frames = []
    for prefix in ('Home', 'Away'):
        cols = {'Team_Full': df[f'{prefix} Team']}
        for out_name, tmpl in CANDIDATE_FEATURE_MAP.items():
            col = tmpl.format(prefix=prefix)
            cols[out_name] = df[col] if col in df.columns else np.nan
        for game_col in CANDIDATE_GAME_LEVEL_COLS:
            cols[game_col] = df[game_col] if game_col in df.columns else np.nan
        frames.append(pd.DataFrame(cols))

    long_df = pd.concat(frames, ignore_index=True)
    long_df['Team'] = long_df['Team_Full'].map(TEAM_FULLNAME_TO_ABBR)

    unmapped = long_df.loc[long_df['Team'].isna(), 'Team_Full'].unique()
    if len(unmapped):
        print(f"⚠️  Team name(s) didn't map to an abbreviation and will be "
              f"dropped: {list(unmapped)}. Add to TEAM_FULLNAME_TO_ABBR if real.")

    long_df = long_df.dropna(subset=['Team']).drop(columns=['Team_Full'])
    return long_df.reset_index(drop=True)


# --------------------------------------------------------------------
# 3. Per-pick percentile scores (for AS-OF archetype state only)
#    -- same methodology as weekly_4's compute_pick_scores.
# --------------------------------------------------------------------
def compute_pick_scores(picks_long, week_actual_tables):
    records = []
    for entry, picked_team, week, used_before in zip(
        picks_long['EntryName'], picks_long['Team'],
        picks_long['Week'], picks_long['Used_Before_This_Week'],
    ):
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
# 4. AS-OF (expanding, shifted) entry-state features -- the leakage-
#    safe version of weekly_4's full-season aggregate_entry_archetypes.
#    For an entry's row at Week=N, every value here reflects picks
#    STRICTLY BEFORE week N (never including week N's own pick).
# --------------------------------------------------------------------
def compute_asof_entry_states(pick_scores):
    if pick_scores.empty:
        return pick_scores.assign(**{c: [] for c in ARCHETYPE_SCORE_COLS + [
            'Picks_Used_So_Far', 'Primary_Archetype_AsOf']})

    ps = pick_scores.sort_values(['EntryName', 'Week']).reset_index(drop=True)
    ps['Dominant_Dim'] = ps.apply(_dominant_dim, axis=1)
    ps['Dist_From_Mid'] = (ps[DIM_COLS] - 0.5).abs().mean(axis=1)
    ps['Middling_This_Pick'] = 1 - ps['Dist_From_Mid'] / 0.5

    grp = ps.groupby('EntryName')

    # expanding().mean() includes the CURRENT row; shift(1) drops it so
    # week N's state reflects only weeks < N.
    pick_cols = ['Planner_pick', 'Contrarian_pick', 'EV_Hunter_pick',
                 'Sprinter_pick', 'Hoarder_pick']
    for col, out_col in zip(pick_cols, ['Planner', 'Contrarian', 'EV Hunter',
                                         'Sprinter', 'Hoarder']):
        ps[out_col] = grp[col].transform(lambda s: s.expanding().mean().shift(1))

    ps['_middling_asof'] = grp['Middling_This_Pick'].transform(
        lambda s: s.expanding().mean().shift(1))

    prev_dom = grp['Dominant_Dim'].shift(1)
    switched = (ps['Dominant_Dim'] != prev_dom).astype(float)
    switched[prev_dom.isna()] = np.nan  # no prior pick -> no comparison yet
    ps['_switch_this_pick'] = switched
    # inconsistency-as-of: mean switch rate over all PRIOR transitions
    ps['_inconsistency_asof'] = grp['_switch_this_pick'].transform(
        lambda s: s.expanding().mean().shift(1))

    ps['Picks_Used_So_Far'] = grp.cumcount()  # already 0-based -> = picks before this one

    ps['Tourist'] = np.where(
        ps['Picks_Used_So_Far'] >= 2,
        0.5 * ps['_middling_asof'] + 0.5 * ps['_inconsistency_asof'],
        ps['_middling_asof'],
    )

    for c in ARCHETYPE_SCORE_COLS:
        ps[c] = (ps[c] * 100)

    score_cols_present = [c for c in ARCHETYPE_SCORE_COLS]
    # idxmax raises on an all-NaN row (true for every entry's very first
    # pick, before it has any prior-week state) -- guard those separately.
    ps['Primary_Archetype_AsOf'] = 'No_Prior_Picks'
    has_state = ps[score_cols_present].notna().any(axis=1)
    if has_state.any():
        ps.loc[has_state, 'Primary_Archetype_AsOf'] = (
            ps.loc[has_state, score_cols_present].idxmax(axis=1)
        )
    # first pick of the season has no prior picks at all -> no state yet
    ps.loc[ps['Picks_Used_So_Far'] == 0, score_cols_present] = np.nan
    ps.loc[ps['Picks_Used_So_Far'] == 0, 'Primary_Archetype_AsOf'] = 'No_Prior_Picks'

    return ps[['EntryName', 'Week', 'Picks_Used_So_Far', 'Primary_Archetype_AsOf']
              + ARCHETYPE_SCORE_COLS]


# --------------------------------------------------------------------
# 5. Choice-set expansion: one row per (entry, week, candidate team)
# --------------------------------------------------------------------
def build_choice_rows(year, picks_long, asof_states, candidate_tables):
    asof_lookup = asof_states.set_index(['EntryName', 'Week'])
    rows = []

    for entry, picked_team, week, used_before in zip(
        picks_long['EntryName'], picks_long['Team'],
        picks_long['Week'], picks_long['Used_Before_This_Week'],
    ):
        cand_tbl = candidate_tables.get(week)
        if cand_tbl is None:
            continue

        pool = cand_tbl[~cand_tbl['Team'].isin(used_before)].copy()
        if picked_team not in pool['Team'].values or len(pool) < 2:
            continue

        try:
            state = asof_lookup.loc[(entry, week)]
        except KeyError:
            continue  # shouldn't happen -- asof_states built from same picks_long

        n_pool = len(pool)
        for feat_col in ['Win_Pct', 'Sportsbook_EV', 'Future_Value']:
            if feat_col in pool.columns:
                pool[feat_col + '_Pctile'] = pool[feat_col].rank(pct=True, method='average')

        for _, cand in pool.iterrows():
            row = {
                'Year': year, 'Week': week,
                'EntryName': entry, 'Team': cand['Team'],
                'Picked': int(cand['Team'] == picked_team),
                'Teams_Remaining_In_Pool': n_pool,
            }
            for c in ARCHETYPE_SCORE_COLS + ['Picks_Used_So_Far', 'Primary_Archetype_AsOf']:
                row[c] = state[c]
            for feat_col in list(CANDIDATE_FEATURE_MAP.keys()) + CANDIDATE_GAME_LEVEL_COLS:
                if feat_col in cand.index:
                    row[feat_col] = cand[feat_col]
            for feat_col in ['Win_Pct', 'Sportsbook_EV', 'Future_Value']:
                pct_col = feat_col + '_Pctile'
                if pct_col in cand.index:
                    row[pct_col] = cand[pct_col]
            rows.append(row)

    return pd.DataFrame.from_records(rows)


# --------------------------------------------------------------------
# 6. Orchestration
# --------------------------------------------------------------------
def process_year(year):
    picks_path = PICKS_PATTERN.format(year=year)
    if not os.path.exists(picks_path):
        print(f"⚠️  {year}: picks file not found at {picks_path}, skipping.")
        return None

    picks_long = load_picks_long(picks_path)
    if picks_long.empty:
        print(f"⚠️  {year}: no real picks found, skipping.")
        return None
    picks_long = attach_available_pool(picks_long)

    weeks = sorted(picks_long['Week'].unique())
    actual_tables, candidate_tables = {}, {}
    for week in weeks:
        path = FINAL_DATA_PATTERN.format(year=year, week=week)
        actual_tables[week] = load_week_actual_table(path)
        candidate_tables[week] = load_week_candidate_table(path)
        if actual_tables[week] is None:
            print(f"⚠️  {year} Week {week}: final_data file not found "
                  f"({path}); this week will be skipped.")

    pick_scores = compute_pick_scores(picks_long, actual_tables)
    asof_states = compute_asof_entry_states(pick_scores)

    choice_rows = build_choice_rows(year, picks_long, asof_states, candidate_tables)
    print(f"✅ {year}: {len(choice_rows)} choice rows from "
          f"{choice_rows['EntryName'].nunique() if not choice_rows.empty else 0} entries "
          f"across {len(weeks)} weeks.")
    return choice_rows


def main():
    all_years = []
    for year in YEARS_TO_PROCESS:
        result = process_year(year)
        if result is not None and not result.empty:
            all_years.append(result)

    if not all_years:
        raise ValueError("No training rows produced for any year -- check "
                          "PICKS_PATTERN / FINAL_DATA_PATTERN paths above.")

    full = pd.concat(all_years, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    full.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Saved {len(full)} training rows ({full['Picked'].sum()} positive) "
          f"to {OUT_PATH}")
    print(f"   Years: {sorted(full['Year'].unique())}")
    print(f"   Entries: {full['EntryName'].nunique()}")
    print(f"   Positive rate: {full['Picked'].mean():.4f} "
          f"(sanity check -- should roughly equal 1 / avg pool size)")


if __name__ == '__main__':
    main()
