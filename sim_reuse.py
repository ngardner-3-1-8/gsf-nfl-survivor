"""
sim_reuse.py

Historical-replay reuse of daily_2's Monte Carlo results.

The per-game simulation (simulate_matchup x SIMULATIONS) and the season
survivor simulation are daily_2's slowest steps. During a historical replay we
don't need to re-run them: that week's committed sim file
(nfl-power-ratings/final_sim_results_with_variance_week_{w}_{year}.csv) already
holds full-fidelity results from the original run. These helpers let daily_2
reuse them with a tiny, flat edit -- keeping the messy logic out of daily_2's
big function so a copy/paste can't mangle its indentation.

Everything here is gated on run_config.is_replay(): live runs get None / the
full trial count and re-simulate exactly as before.
"""

import os

import pandas as pd

from run_config import is_replay


def _force_resim():
    """True when FORCE_RESIM is set in the environment. Overrides the reuse:
    even during a replay, the per-game AND season Monte Carlo sims are re-run
    at full fidelity (cache ignored). Use it to rebuild a week's sims from
    scratch without leaving replay mode (which would change unrelated
    point-in-time behavior). You don't need to delete the cached sim file --
    FORCE_RESIM ignores it."""
    return bool(os.environ.get('FORCE_RESIM'))

SIM_FILE_PATTERN = ("nfl-power-ratings/final_sim_results_with_variance_week_"
                    "{week}_{year}.csv")

# Per-game columns to lift from the cache (everything simulate_matchup produces).
_REUSE_COLS = [
    'Wind', 'Temperature', 'Precipitation', 'Sim_Weather_Source', 'Dome',
    'Away_Starting_QB', 'Home_Starting_QB',
    'Sim_Spread_Mean', 'Sim_Spread_Median', 'Sim_Spread_Std',
    'Sim_Spread_Variance', 'Sim_Spread_Variance_Label',
    'Sim_Spread_25th', 'Sim_Spread_75th',
    'Sim_Total_Mean', 'Sim_Total_Median', 'Sim_Total_Std',
    'Sim_Total_10th_Floor', 'Sim_Total_90th_Ceiling',
    'Sim_Home_Win_Pct', 'Sim_Away_Win_Pct',
    'Sim_Prob_Land_3', 'Sim_Prob_Land_7',
    'Sim_Home_Cover_Prob', 'Sim_Away_Cover_Prob',
    'Sim_Prob_Over', 'Sim_Prob_Under',
]


def load_cached_sim_lookup(upcoming_week, target_year, verbose=True):
    """(reuse_on, lookup) for the per-game sims.

    reuse_on is True only during a replay when a cached sim file with Sim_*
    columns exists. lookup maps (int Week, Home Team, Away Team) -> {col: value}
    for the reusable columns, matched on the raw matchup so holiday-week
    renumbering can't misalign it."""
    if not is_replay():
        return False, {}
    if _force_resim():
        if verbose:
            print("🔁 FORCE_RESIM set; re-running per-game sims (cache ignored).")
        return False, {}
    path = SIM_FILE_PATTERN.format(week=upcoming_week, year=target_year)
    if not os.path.exists(path):
        if verbose:
            print(f"ℹ️  No cached sim file {path}; running full per-game sims.")
        return False, {}
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        if verbose:
            print(f"⚠️  Could not read sim cache {path} ({e}); running full sims.")
        return False, {}
    if not {'Week', 'Home Team', 'Away Team', 'Sim_Home_Win_Pct'}.issubset(df.columns):
        if verbose:
            print(f"⚠️  {path} has no Sim_* columns; running full per-game sims.")
        return False, {}
    have = [c for c in _REUSE_COLS if c in df.columns]
    lookup = {}
    for _, r in df.iterrows():
        try:
            k = (int(r['Week']), str(r['Home Team']).strip(), str(r['Away Team']).strip())
        except (ValueError, TypeError):
            continue
        lookup[k] = {c: r[c] for c in have}
    if verbose:
        print(f"♻️  Reusing cached per-game sims from {path} "
              f"({len(lookup)} matchups); re-simulating only matchups not cached.")
    return True, lookup


def cached_sim_res(index, row, lookup):
    """A simulation_results dict for this matchup built from the cache, or None
    if it isn't cached. `index` becomes Matchup_ID (daily_2's merge key)."""
    try:
        k = (int(row['Week']), str(row['Home Team']).strip(), str(row['Away Team']).strip())
    except (ValueError, TypeError, KeyError):
        return None
    cv = lookup.get(k)
    if cv is None:
        return None
    res = {'Matchup_ID': index}
    res.update(cv)
    res.setdefault('Week', row.get('Week'))
    # Carry Date + Matchup like the full-sim path so the schedule<->sim merge
    # still creates the 'Date_x' column the downstream projection relies on.
    try:
        res['Date'] = pd.to_datetime(row['Date'])
    except Exception:
        res['Date'] = row.get('Date')
    res['Matchup'] = f"{row.get('Away Team')} @ {row.get('Home Team')}"
    return res


def load_cached_season_summary(upcoming_week, target_year, verbose=True):
    """The cached season-survivor summary (Week / Avg Survivors / Avg
    Eliminations) during a replay, or None when it can't be reused (live run,
    no cache, or the columns are absent) so daily_2 falls back to running it."""
    if not is_replay():
        return None
    if _force_resim():
        # Re-run the season MC too (season_trials() returns the full count
        # under FORCE_RESIM), so don't hand back a cached summary.
        return None
    path = SIM_FILE_PATTERN.format(week=upcoming_week, year=target_year)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    cols = {'Week', 'Avg Survivors', 'Avg Eliminations'}
    if not cols.issubset(df.columns):
        return None
    out = (df[['Week', 'Avg Survivors', 'Avg Eliminations']]
           .dropna(subset=['Week']).drop_duplicates('Week').copy())
    if verbose:
        print(f"♻️  Reusing cached season survivor summary from {path}; "
              f"skipping the season Monte Carlo.")
    return out


def season_trials():
    """Season-MC trial count: the full 1000 live or under FORCE_RESIM, and 1
    during a replay (cheap fallback when there's no cached summary to reuse)."""
    if _force_resim():
        return 1000
    return 1 if is_replay() else 1000
