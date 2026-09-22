"""
build_contest_historical_data.py

Builds the per-contest historical training files that daily_2's per-contest
pick-percentage projection trains on:

    contest-historical-data/BigSplash_historical_data.csv
    contest-historical-data/WorldChampionship_historical_data.csv

WHY THIS IS SAFE / CORRECT
==========================
Every feature column in Circa_historical_data.csv is *contest-agnostic* --
they describe the game/slate, not Circa's crowd (Win %, Future Value,
holiday flags, weekly rank/relative/lookahead features, Public Pick %
[SurvivorGrid], ...). The ONLY genuinely contest-specific column is the
target, 'Pick %' (the fraction of that contest's entries who picked each
team that week). So a contest's historical training file is exactly the
Circa feature rows with 'Pick %' swapped for THAT contest's observed
pick %, restricted to the (year, week) slates the contest actually ran.

That keeps every contest a first-class, isolated model -- each trains on
its own crowd's behaviour -- while reusing the one authoritative source of
game features, so the contests can never silently disagree about, say, a
team's Win % or holiday status.

OBSERVED PICK %
===============
Read from the canonical wide picks files produced by
weekly_3b_normalize_splash_pick_history.py:

    big-splash-pick-history/{year}_big_splash_picks.csv
    splash-world-championship-pick-history/{year}_world_championship_picks.csv

For each Week_N column, observed Pick %(team) = (entries who picked team) /
(entries who made a pick that week). Multi-pick weeks (an entry picks two
teams, stored "TEAM1;TEAM2") naturally sum to ~2.0, matching how the
downstream target-sum logic treats those weeks; single-pick weeks sum to
~1.0, exactly like Circa's 'Pick %'.

COLD START
==========
Splash contests are 2026+ and, as of this writing, only 2026 Week 1 has
real pick data. This builder therefore emits a *thin* file (one slate).
daily_2's projection owns the cold-start fallback (use Public Pick % as the
Week-1 proxy when there is not yet enough real history to train on); this
builder just makes the real rows available and never fabricates a slate a
contest didn't run.

Usage:
    python build_contest_historical_data.py
"""

import os
import re
import glob

import pandas as pd

from team_codes import canonical_pick_code, VALID_ABBRS

MASTER = "contest-historical-data/Circa_historical_data.csv"
OUT_DIR = "contest-historical-data"

# contest key -> (canonical picks glob, output basename)
CONTESTS = {
    'big_splash': (
        'big-splash-pick-history/*_big_splash_picks.csv',
        'BigSplash_historical_data',
    ),
    'world_championship': (
        'splash-world-championship-pick-history/*_world_championship_picks.csv',
        'WorldChampionship_historical_data',
    ),
}


def _year_from_filename(path):
    m = re.match(r'(\d{4})_', os.path.basename(path))
    return int(m.group(1)) if m else None


def observed_pick_pct(picks_path):
    """{(year, week): {'pct': {TEAM: pct}, 'count': {TEAM: n}}} for one
    canonical wide picks file. Denominator per week = entries who made a pick
    that week, so single-pick weeks sum to ~1.0 and multi-pick weeks to ~2.0.
    Counts are the raw number of entries picking each team (the contest's
    'Calculated Current Week Picks')."""
    year = _year_from_filename(picks_path)
    if year is None:
        print(f"⚠️  Couldn't parse a year from {os.path.basename(picks_path)}; skipping.")
        return {}
    df = pd.read_csv(picks_path)
    week_cols = [c for c in df.columns if re.fullmatch(r'Week_\d+', str(c))]
    out = {}
    for col in week_cols:
        week = int(col.split('_')[1])
        picks = df[col].astype(str).str.strip()
        # entries that actually made a pick this week (the denominator)
        made = picks[(picks != '') & (picks.str.lower() != 'nan')
                     & (picks.str.upper() != 'ELIMINATED')]
        n_entries = len(made)
        if n_entries == 0:
            continue
        # split multi-pick "TEAM1;TEAM2", canonicalize, keep valid teams only
        exploded = made.str.split(';').explode().str.strip()
        exploded = exploded[exploded != ''].map(canonical_pick_code)
        exploded = exploded[exploded.isin(VALID_ABBRS)]
        counts = exploded.value_counts()
        out[(year, week)] = {
            'pct': (counts / n_entries).to_dict(),
            'count': counts.astype(int).to_dict(),
        }
    return out


def build_contest(contest_key, master_df):
    picks_glob, out_base = CONTESTS[contest_key]
    files = sorted(glob.glob(picks_glob))
    if not files:
        print(f"ℹ️  {contest_key}: no canonical picks files match {picks_glob} "
              f"(run weekly_3b first). Nothing to build -- expected for a "
              f"season before the contest existed.")
        return None

    # (year, week) -> {team: pct}
    observed = {}
    for path in files:
        observed.update(observed_pick_pct(path))
    if not observed:
        print(f"⚠️  {contest_key}: picks files present but no usable weeks parsed.")
        return None

    slates = sorted(observed)
    print(f"📊 {contest_key}: observed pick% for {len(slates)} slate(s): {slates}")

    # Restrict the Circa feature rows to the slates this contest actually ran,
    # then overwrite 'Pick %' with this contest's observed pick %.
    master = master_df.copy()
    master['_canon'] = master['Team'].astype(str).map(canonical_pick_code)

    kept = []
    for (year, week), obs in observed.items():
        team_pct = obs['pct']
        team_cnt = obs['count']
        slate = master[(master['Year'] == year) & (master['Week'] == week)].copy()
        if slate.empty:
            print(f"   ⚠️  no Circa feature rows for {year} Week {week}; "
                  f"the projection has no features to train on for that slate.")
            continue
        # Target: this contest's observed pick %.
        slate['Pick %'] = slate['_canon'].map(team_pct).fillna(0.0)
        # Contest-specific count columns (were Circa's in the source rows).
        # Recompute them for THIS contest so nobody later trains on Circa's
        # numbers by mistake. 'Prior Week Picks by Alive Entries' is 0 in the
        # first week of a contest (no prior usage) and stays consistent as
        # more weeks accrue.
        if 'Calculated Current Week Picks' in slate.columns:
            slate['Calculated Current Week Picks'] = \
                slate['_canon'].map(team_cnt).fillna(0).astype(int)
        if 'Calculated Prior Week Picks by Alive Entries' in slate.columns and week == 1:
            slate['Calculated Prior Week Picks by Alive Entries'] = 0.0
        kept.append(slate)

    if not kept:
        print(f"⚠️  {contest_key}: no overlapping feature slates; nothing written.")
        return None

    out_df = pd.concat(kept, ignore_index=True)
    out_df = out_df.drop(columns=['_canon'])

    out_path = os.path.join(OUT_DIR, f"{out_base}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"✅ {contest_key}: wrote {len(out_df):,} rows x {out_df.shape[1]} cols "
          f"-> {out_path}")

    # Per-year splits, mirroring Circa_historical_data_{year}.csv, so any
    # tooling that reads the per-year files finds the contest's too.
    for year in sorted(out_df['Year'].dropna().unique()):
        yr_df = out_df[out_df['Year'] == year]
        yr_path = os.path.join(OUT_DIR, f"{out_base}_{int(year)}.csv")
        yr_df.to_csv(yr_path, index=False)
        print(f"   ↳ {os.path.basename(yr_path)}: {len(yr_df):,} rows")
    return out_path


def main():
    if not os.path.exists(MASTER):
        print(f"❌ Master feature file not found: {MASTER}")
        return
    master_df = pd.read_csv(MASTER)
    for contest_key in CONTESTS:
        build_contest(contest_key, master_df)


if __name__ == '__main__':
    main()
