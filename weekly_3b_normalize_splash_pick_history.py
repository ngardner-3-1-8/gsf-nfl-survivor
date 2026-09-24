"""
weekly_3b_normalize_splash_pick_history.py

Converts the raw Splash contest exports into the SAME canonical wide
"picks history" format the rest of the pipeline already understands
(EntryName, Week_1, Week_2, ...), so the Splash contests can flow through
the exact same archetype / choice-model machinery as Circa -- one method,
no Circa proxy.

RAW INPUT (one file per contest per week, as exported from Splash)
=================================================================
big-splash-pick-history/Splash_Picks_-_The_Big_Splash_-_Week_{N}.csv
splash-world-championship-pick-history/Splash_Picks_-_The_Survivor_World_Championship_-_Week_{N}.csv

Each raw file has: User Name, Entry Number, Status, and one pick column
named "Week {N}" holding a team NICKNAME ("Jaguars", "Chargers", ...).
A user can hold many entries, so the unique entry key is (User Name,
Entry Number) -> EntryName = "{User Name}-{Entry Number}" (same "name-N"
convention the Circa picks files already use).

MULTI-PICK WEEKS (Picks Required = 2)
=====================================
On a Splash multi-pick week an entry selects two teams. If/when a raw
export represents that as more than one pick column for the same week
(e.g. "Week 12" and "Week 12 (2)"), every such column is collected and the
week's picks are stored as a "TEAM1;TEAM2" string in the canonical file.
Downstream loaders split on ";" so a normal week (one team, no ";") and a
multi-pick week (two teams) are handled uniformly. As of this writing only
single-pick Week 1 data exists, so this path is exercised by the column
detection but not yet by real two-pick data.

OUTPUT (canonical, one file per contest per season)
===================================================
big-splash-pick-history/{year}_big_splash_picks.csv
splash-world-championship-pick-history/{year}_world_championship_picks.csv
Columns: EntryName, Status, Week_1, Week_2, ...  (team ABBREVIATIONS,
canonicalized via team_codes; missing/eliminated weeks left blank.)

SEASON
======
Splash contests are 2026+. The raw files carry no year, so YEAR is set
here (defaults to 2026); update it each season. Running this for a season
with no raw export files simply writes nothing and says so -- it never
fabricates Splash history for a year the contest didn't exist.
"""

import os
import re
import glob

import pandas as pd

from team_codes import NICKNAME_TO_ABBR, canonical_pick_code

from run_config import as_of_year
# Season to normalize: the central as-of year (PIPELINE_AS_OF_DATE env when the
# orchestrator drives us, else run_config.HISTORICAL_DATES[0], else today).
YEAR = as_of_year()

# contest key -> (raw export directory, raw filename glob, canonical output path)
CONTESTS = {
    'big_splash': (
        'big-splash-pick-history',
        'Splash_Picks_-_The_Big_Splash_-_Week_*.csv',
        'big-splash-pick-history/{year}_big_splash_picks.csv',
    ),
    'world_championship': (
        'splash-world-championship-pick-history',
        'Splash_Picks_-_The_Survivor_World_Championship_-_Week_*.csv',
        'splash-world-championship-pick-history/{year}_world_championship_picks.csv',
    ),
}


def _nickname_to_abbr(value):
    """Map a Splash team nickname ("Jaguars") to a canonical abbreviation.
    Falls back to canonical_pick_code() in case a file ever carries an
    abbreviation instead of a nickname. Blanks/sentinels pass through."""
    if pd.isna(value):
        return ''
    s = str(value).strip()
    if s == '' or s.upper() == 'ELIMINATED':
        return s
    if s in NICKNAME_TO_ABBR:
        return NICKNAME_TO_ABBR[s]
    # Not a known nickname -- maybe it's already an abbreviation/full code.
    return canonical_pick_code(s)


def _week_from_filename(path):
    m = re.search(r'Week_(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


def _pick_columns_for_week(df, week):
    """All columns in this raw file that hold a pick for `week`. Normally
    just "Week {week}"; a multi-pick week may add a second column whose
    header still starts with "Week {week}" (e.g. "Week 12 (2)")."""
    exact = f'Week {week}'
    cols = [c for c in df.columns if c == exact]
    cols += [c for c in df.columns
             if c != exact and re.match(rf'^Week\s*{week}\b', str(c))]
    return cols


def normalize_contest(contest_key, year=YEAR):
    raw_dir, raw_glob, out_tmpl = CONTESTS[contest_key]
    files = sorted(glob.glob(os.path.join(raw_dir, raw_glob)),
                   key=lambda p: (_week_from_filename(p) or 0))
    if not files:
        print(f"ℹ️  {contest_key}: no raw Splash export files in {raw_dir} "
              f"(expected {raw_glob}) -- nothing to normalize. This is "
              f"expected for any season before the contest existed.")
        return None

    entry_index = None          # ordered EntryName list (first-seen order)
    status_by_entry = {}
    week_maps = {}              # week_num -> {EntryName: "ABBR" or "ABBR1;ABBR2"}

    for path in files:
        week = _week_from_filename(path)
        if week is None:
            print(f"⚠️  Couldn't parse a week number from {path}; skipping.")
            continue
        df = pd.read_csv(path)
        if 'User Name' not in df.columns or 'Entry Number' not in df.columns:
            print(f"⚠️  {path}: missing 'User Name'/'Entry Number'; skipping.")
            continue

        entry_names = (df['User Name'].astype(str).str.strip() + '-'
                       + df['Entry Number'].astype(str).str.strip())

        pick_cols = _pick_columns_for_week(df, week)
        if not pick_cols:
            print(f"⚠️  {path}: no 'Week {week}' pick column found "
                  f"(columns: {list(df.columns)}); skipping.")
            continue

        abbr_cols = [df[c].map(_nickname_to_abbr) for c in pick_cols]
        combined = abbr_cols[0].astype(str)
        for extra in abbr_cols[1:]:
            extra = extra.astype(str)
            both = [';'.join([a for a in (x, y) if a and a != 'ELIMINATED'])
                    for x, y in zip(combined, extra)]
            combined = pd.Series(both, index=combined.index)

        wk_map = {}
        for name, pick in zip(entry_names, combined):
            wk_map[name] = pick
        week_maps[week] = wk_map

        if 'Status' in df.columns:
            for name, st in zip(entry_names, df['Status'].astype(str)):
                status_by_entry[name] = st  # latest week's status wins

        if entry_index is None:
            entry_index = list(dict.fromkeys(entry_names))
        else:
            for name in entry_names:
                if name not in entry_index:
                    entry_index.append(name)

        n_multi = int((combined.str.contains(';')).sum())
        print(f"   {os.path.basename(path)}: week {week}, {len(entry_names):,} entries"
              + (f", {n_multi:,} multi-pick" if n_multi else ""))

    if not week_maps:
        print(f"⚠️  {contest_key}: no usable weeks parsed; nothing written.")
        return None

    out = pd.DataFrame({'EntryName': entry_index})
    out['Status'] = out['EntryName'].map(status_by_entry).fillna('')
    for week in sorted(week_maps):
        out[f'Week_{week}'] = out['EntryName'].map(week_maps[week]).fillna('')

    out_path = out_tmpl.format(year=year)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    wk_cols = [c for c in out.columns if re.fullmatch(r'Week_\d+', c)]
    print(f"✅ {contest_key}: wrote {len(out):,} entries x {len(wk_cols)} week(s) "
          f"to {out_path}")
    return out_path


def main():
    # Splash contests are 2026+. In a historical replay of an earlier season
    # the contests didn't exist, so normalize nothing rather than mislabel the
    # real (2026) raw exports -- which aren't year-stamped -- as that year.
    from contest_config import CONTESTS as _CC
    splash_start = min(_CC[c]['start_season']
                       for c in ('big_splash', 'world_championship'))
    if YEAR < splash_start:
        print(f"ℹ️  Splash contests start {splash_start}; nothing to normalize "
              f"for {YEAR} (historical replay before the contests existed).")
        return
    for contest_key in CONTESTS:
        normalize_contest(contest_key, YEAR)


if __name__ == '__main__':
    main()
