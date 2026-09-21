"""
identify_entry_archetypes.py

Classifies each Circa survivor ENTRY (not player -- the same person's two
entries can get two different labels) into a blend of six behavioral
archetypes, based on the picks that entry has actually made so far this
season:

    Planner       - balanced consideration of EV, Future Value, and Win %
    Contrarian    - seeks out low-pick% (unpopular) options to fade the field
    EV Hunter     - chases the single best sportsbook EV available each week
    Sprinter      - chases the single highest win% (safest) option each week
    Hoarder       - saves high-Future-Value teams, takes "good enough" now
    Tourist       - no consistent pattern across the four signals below

METHODOLOGY
===========
For every pick an entry has actually made, the picked team is ranked
against every team THAT SAME ENTRY still had available that week -- i.e.
all teams minus whatever it had already used in earlier weeks, not the
whole public pool. Two entries in week 6 can have completely different
remaining options depending on their own pick history, so "available"
has to be computed per entry, not per week.

That produces four percentiles per pick (0 = worst option this entry had
available, 1 = best):

    win_pct_pctile       - percentile of the picked team's Win % (Fair Odds)
    ev_pctile            - percentile of the picked team's Sportsbook EV
    fv_pctile            - percentile of the picked team's Future Value (Star Rating)
    unpopularity_pctile  - percentile of (1 - Actual Pick %) -- how quiet the pick was

Those four percentiles become five per-pick archetype scores (0-1):

    Sprinter_score   = win_pct_pctile
    EV_Hunter_score  = ev_pctile
    Contrarian_score = unpopularity_pctile
    Hoarder_score    = (1 - fv_pctile) * win_pct_pctile, zeroed out on any
                        pick where the picked team WAS the highest-Future-Value
                        option available (nothing was actually saved)
    Planner_score    = 1 - (spread across [ev_pctile, win_pct_pctile, fv_pctile])
                        -- rewards picks that don't lean hard on just one axis

Per-entry archetype scores are the MEAN of these across every pick made so
far, scaled to 0-100. Tourist_score is entry-level: how close, on average,
each pick's four percentiles sit to 0.5 (a genuinely aimless pick tends to
look unremarkable on every axis rather than extreme on any one), blended
--once an entry has 2+ picks-- with how often the entry's single strongest
dimension actually changes week to week (a real strategy keeps favoring
the same dimension; wandering doesn't).

Primary_Archetype is the argmax of the six scores. Archetype_Mix lists
every archetype within 15 points of the top score, so a genuinely blended
entry shows up as a blend instead of being forced into one box.

WEEK-1 / SMALL-SAMPLE CAVEAT
=============================
With one week of evidence, every score here is a single data point
wearing a profile's clothing. Picks_Used and Confidence are included
specifically so nothing downstream treats an early-season label as
settled. This script does not change week to week -- rerun it as more
Week_N pick columns and more Season_{year}_Through_Week_{N}_Final_Data.csv
files become available, and profiles will fill in and stabilize on their
own as more picks accumulate.

INPUTS
======
- circa-pick-history/{year}_survivor_picks.csv
- nfl-power-ratings/final_data/{year}_final_data/
      Season_{year}_Through_Week_{N}_Final_Data.csv
  for every week N that has at least one actual pick in the picks file.

OUTPUT
======
circa-pick-history/{year}_survivor_picks_with_archetypes_week_{last_week}.csv
(last_week = the most recent week that has any actual picks)
"""

import numpy as np
import pandas as pd

from team_codes import canonical_pick_code
from contest_config import get_contest

_CFG = get_contest()

# --------------------------------------------------------------------
# Config -- adjust YEAR (or wire it up to your existing target_year
# logic elsewhere in the pipeline) each season.
# --------------------------------------------------------------------
YEAR = 2026

# Contest-driven (SURVIVOR_CONTEST env var; default 'circa').
PICKS_PATH = _CFG['picks_pattern'].format(year=YEAR)

FINAL_DATA_PATTERN = (
    f"nfl-power-ratings/final_data/{YEAR}_final_data/"
    f"Season_{YEAR}_Through_Week_{{week}}_Final_Data.csv"
)

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


# --------------------------------------------------------------------
# 1. Weekly team-level stats table (one row per team per week)
# --------------------------------------------------------------------
def load_week_team_table(week):
    """Melts one week's final_data file (one row per GAME) into one row
    per TEAM with the four stats archetype scoring needs, joined onto the
    team abbreviations the picks file uses. Returns None if that week's
    file doesn't exist yet (future/not-yet-run weeks)."""
    path = FINAL_DATA_PATTERN.format(week=week)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, low_memory=False)

    def _side(prefix):
        return pd.DataFrame({
            'Team_Full': df[f'{prefix} Team'],
            'Win %': df[f'{prefix} Team Fair Odds'],
            'EV': df[f'sportsbook_{prefix}_EV'],
            'Future Value': df[f'{prefix} Team Star Rating'],
            'Actual Pick %': df[f'{prefix} Actual Pick %'],
        })

    long_df = pd.concat([_side('Home'), _side('Away')], ignore_index=True)
    long_df['Team'] = long_df['Team_Full'].map(TEAM_FULLNAME_TO_ABBR)

    unmapped = long_df.loc[long_df['Team'].isna(), 'Team_Full'].unique()
    if len(unmapped):
        print(f"⚠️ Week {week}: team name(s) didn't map to an abbreviation and "
              f"will be dropped from scoring: {list(unmapped)}. Add them to "
              f"TEAM_FULLNAME_TO_ABBR if these are real teams (e.g. a "
              f"relocated/renamed franchise).")

    long_df = long_df.dropna(subset=['Team']).drop(columns=['Team_Full'])
    # A team missing any of the four metrics that week can't be fairly
    # ranked against its peers -- drop it from the pool rather than let a
    # NaN silently poison every percentile computed against this week.
    long_df = long_df.dropna(subset=['Win %', 'EV', 'Future Value', 'Actual Pick %'])
    long_df['Week'] = week
    return long_df.reset_index(drop=True)


# --------------------------------------------------------------------
# 2. Picks file -> long format + each entry's available pool per pick
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
        # dropna() BEFORE stringifying -- on newer pandas (PyArrow-backed
        # string dtype), astype(str) on a real NaN can silently keep it as
        # a null rather than the literal text "nan", which would slip
        # straight past a `!= 'nan'` string filter. dropna() is robust to
        # that regardless of which string backend pandas is using.
        sub = sub.dropna(subset=['Team'])
        sub['Team'] = sub['Team'].astype(str).str.strip()
        # Split multi-pick "TEAM1;TEAM2" cells (no-op for single-pick), then
        # canonicalize (e.g. 'LAR' -> 'LA') BEFORE the pool-membership check
        # downstream or those picks silently drop. Also filter 'ELIMINATED'
        # to match weekly_5 / daily_4 (this loader previously kept it).
        sub = sub.assign(Team=sub['Team'].str.split(';')).explode('Team')
        sub['Team'] = sub['Team'].str.strip().map(canonical_pick_code)
        sub = sub[(sub['Team'] != '') & (sub['Team'] != 'ELIMINATED')]
        sub = sub.assign(Week=week_num)
        long_rows.append(sub)

    picks_long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
        columns=['EntryName', 'Team', 'Week'])
    picks_long = picks_long.sort_values(['EntryName', 'Week']).reset_index(drop=True)
    return picks_wide, picks_long, week_cols


def attach_available_pool(picks_long):
    """For every pick, the set of teams that entry had ALREADY used going
    into that week (everything it picked in a strictly earlier week).
    Computed per week (not per row) so a multi-pick week's two picks share
    the same prior-weeks pool; identical to the old row-by-row version for
    single-pick data."""
    out = picks_long.sort_values(['EntryName', 'Week']).reset_index(drop=True)
    used_before = [frozenset()] * len(out)
    for _entry, g in out.groupby('EntryName', sort=False):
        prior: set = set()
        for wk in sorted(g['Week'].unique()):
            wk_idx = g.index[g['Week'] == wk]
            fs = frozenset(prior)
            for i in wk_idx:
                used_before[i] = fs
            prior |= set(out.loc[wk_idx, 'Team'])
    out['Used_Before_This_Week'] = used_before
    return out

# --------------------------------------------------------------------
# 3. Per-pick percentiles -> per-pick archetype scores
# --------------------------------------------------------------------
def compute_pick_scores(picks_long, week_team_tables):
    records = []
    for entry, picked_team, week, used_before in zip(
        picks_long['EntryName'], picks_long['Team'],
        picks_long['Week'], picks_long['Used_Before_This_Week'],
    ):
        week_tbl = week_team_tables.get(week)
        if week_tbl is None:
            continue  # no final_data file for this week yet

        pool = week_tbl[~week_tbl['Team'].isin(used_before)]
        if picked_team not in pool['Team'].values or len(pool) < 2:
            # Picked team is missing from this week's scoreable pool
            # (data gap, or already excluded) or there's nothing to rank
            # against -- skip rather than guess.
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
        # max possible population std for 3 values confined to [0, 1] is 0.5
        # (two at one extreme, one at the other) -- normalize against that.
        planner = max(0.0, 1 - dims.std() / 0.5)

        records.append({
            'EntryName': entry, 'Week': week, 'Team': picked_team,
            'win_pct_pctile': win_p, 'ev_pctile': ev_p, 'fv_pctile': fv_p,
            'unpopularity_pctile': unpop_p,
            'Sprinter_pick': win_p, 'EV_Hunter_pick': ev_p,
            'Contrarian_pick': unpop_p, 'Hoarder_pick': hoarder,
            'Planner_pick': planner,
        })

    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------
# 4. Aggregate to one row per entry
# --------------------------------------------------------------------
def _dominant_dim(row):
    vals = {d: row[d] for d in DIM_COLS}
    return max(vals, key=vals.get)


def _middling_score(sub):
    # 0 = every pick sat right at the 50th percentile on every axis
    # (maximally unremarkable); 1 = picks were consistently extreme.
    dist_from_mid = (sub[DIM_COLS] - 0.5).abs().mean(axis=1)
    return float((1 - dist_from_mid / 0.5).mean())


def _inconsistency_score(sub):
    if len(sub) < 2:
        return np.nan
    sub = sub.sort_values('Week')
    dominant = sub.apply(_dominant_dim, axis=1)
    switches = (dominant.to_numpy()[1:] != dominant.to_numpy()[:-1]).sum()
    return switches / (len(sub) - 1)


def aggregate_entry_archetypes(pick_scores):
    grouped = pick_scores.groupby('EntryName')

    agg = grouped[['Planner_pick', 'Contrarian_pick', 'EV_Hunter_pick',
                    'Sprinter_pick', 'Hoarder_pick']].mean()
    agg.columns = ['Planner', 'Contrarian', 'EV Hunter', 'Sprinter', 'Hoarder']

    middling = grouped.apply(_middling_score, include_groups=False)
    inconsistency = grouped.apply(_inconsistency_score, include_groups=False)
    n_picks = grouped.size().rename('Picks_Used')

    tourist_df = pd.DataFrame({
        'middling': middling, 'inconsistency': inconsistency, 'Picks_Used': n_picks,
    })
    # With only 1 scoreable pick, Tourist is purely the "how unremarkable
    # was it" signal. With 2+, blend in the week-to-week inconsistency
    # signal too, now that there's something to be consistent (or not) about.
    tourist_df['Tourist'] = np.where(
        tourist_df['Picks_Used'] >= 2,
        0.5 * tourist_df['middling'] + 0.5 * tourist_df['inconsistency'],
        tourist_df['middling'],
    )

    agg = agg.join(tourist_df[['Tourist', 'Picks_Used']])

    for c in ARCHETYPE_SCORE_COLS:
        agg[c] = (agg[c] * 100).round(1)

    agg['Primary_Archetype'] = agg[ARCHETYPE_SCORE_COLS].idxmax(axis=1)

    def _mix_label(row):
        top = row[ARCHETYPE_SCORE_COLS].max()
        close = [c for c in ARCHETYPE_SCORE_COLS if row[c] >= top - 15]
        close_sorted = sorted(close, key=lambda c: -row[c])
        return ' / '.join(f"{c} ({row[c]:.0f})" for c in close_sorted)

    agg['Archetype_Mix'] = agg.apply(_mix_label, axis=1)

    def _confidence(n):
        if n <= 1:
            return 'Low'
        if n <= 3:
            return 'Medium'
        return 'High'

    agg['Confidence'] = agg['Picks_Used'].apply(_confidence)

    return agg.reset_index()


# --------------------------------------------------------------------
# 5. Orchestration
# --------------------------------------------------------------------
def main():
    picks_wide, picks_long, week_cols = load_picks_long(PICKS_PATH)
    weeks_with_picks = sorted(picks_long['Week'].unique())
    if not weeks_with_picks:
        raise ValueError(f"No picks found in {PICKS_PATH}.")
    last_week = int(max(weeks_with_picks))

    picks_long = attach_available_pool(picks_long)

    week_team_tables = {}
    for week in weeks_with_picks:
        tbl = load_week_team_table(week)
        if tbl is None:
            print(f"⚠️ No final_data file found for Week {week} "
                  f"({FINAL_DATA_PATTERN.format(week=week)}); picks made that "
                  f"week will be skipped until that file exists.")
        week_team_tables[week] = tbl

    pick_scores = compute_pick_scores(picks_long, week_team_tables)
    if pick_scores.empty:
        raise ValueError(
            "Could not score any picks. Most likely causes: the "
            "final_data file(s) for the relevant week(s) are missing, or "
            "team names in those files aren't mapping to the abbreviations "
            "used in the picks file -- check the '⚠️ team name(s) didn't "
            "map' warnings above."
        )

    entry_archetypes = aggregate_entry_archetypes(pick_scores)

    output = picks_wide.merge(entry_archetypes, on='EntryName', how='left')
    output['Primary_Archetype'] = output['Primary_Archetype'].fillna('Insufficient Data')
    output['Confidence'] = output['Confidence'].fillna('None')
    output['Picks_Used'] = output['Picks_Used'].fillna(0).astype(int)
    output['Archetype_Mix'] = output['Archetype_Mix'].fillna('')
    for c in ARCHETYPE_SCORE_COLS:
        output[c] = output[c].fillna(0.0)

    # Sits next to this contest's picks file (Circa keeps its original name).
    out_path = PICKS_PATH.replace('.csv', f'_with_archetypes_week_{last_week}.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output.to_csv(out_path, index=False)

    print(f"\n✅ Saved {len(output)} entries to {out_path}")
    print(f"   Scored {len(pick_scores)} individual picks across "
          f"{pick_scores['EntryName'].nunique()} entries.")
    print("\nPrimary archetype distribution:")
    print(output['Primary_Archetype'].value_counts().to_string())
    print("\nConfidence distribution:")
    print(output['Confidence'].value_counts().to_string())


if __name__ == '__main__':
    main()
