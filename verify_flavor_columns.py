"""
verify_flavor_columns.py

Post-migration sanity check for the per-contest pick-% "flavor" columns
(Predicted / Actual / Archetype / Top Down) x (Circa / Big Splash / World
Championship) x (Home / Away) introduced by the column-rename change.

Loads a Season Final_Data file (the latest cumulative file for a year, unless a
path is given) and, for every contest that existed in that season, checks that
each flavor's Home/Away columns are present and sane:

  * PRESENT   -- the column exists in the file
  * numeric   -- values parse as numbers
  * in range  -- non-null values sit in [0, ~2] (multi-pick Splash weeks can
                 sum to ~2 per team-week is not expected, but a single team's
                 pick % should never exceed ~1; 2.0 is a generous guard)
  * non-empty -- at least one non-null value (a WARN, not a failure: Archetype
                 is legitimately empty for early replay years with no trained
                 choice model, and Splash before 2026 doesn't exist at all)

Exit code is non-zero if any REQUIRED column is missing, so it can gate a
workflow. Empty/optional columns only warn.

Usage:
    python verify_flavor_columns.py 2026
    python verify_flavor_columns.py 2026 --week 5
    python verify_flavor_columns.py --file path/to/Season_2026_Through_Week_5_Final_Data.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd

from contest_config import CONTESTS, PICK_FLAVORS, flavor_cols

# Flavors that must exist for a contest that ran this season. Archetype is
# allowed to be entirely empty (no trained model yet in early years), so it is
# checked for presence but an empty Archetype column is only a warning.
REQUIRED_FLAVORS = ("Predicted", "Actual", "Top Down")
MAX_REASONABLE_PCT = 2.0  # generous upper guard for a single team's pick %


def latest_season_file(year):
    pattern = (f"nfl-power-ratings/final_data/{year}_final_data/"
               f"Season_{year}_Through_Week_*_Final_Data.csv")
    files = glob.glob(pattern)
    if not files:
        return None

    def wk(p):
        try:
            return int(os.path.basename(p).split("_Through_Week_")[1].split("_Final")[0])
        except (IndexError, ValueError):
            return 0

    return max(files, key=wk)


def check_column(df, col):
    """Return (present, n_nonnull, out_of_range_count)."""
    if col not in df.columns:
        return False, 0, 0
    s = pd.to_numeric(df[col], errors="coerce")
    nonnull = int(s.notna().sum())
    oor = int(((s < -0.01) | (s > MAX_REASONABLE_PCT)).sum())
    return True, nonnull, oor


def verify(df, year):
    print(f"\nVerifying {len(df):,} rows"
          + (f" for {year}" if year is not None else "") + " ...\n")
    failures = []   # missing required columns
    warnings = []   # present but empty, or out-of-range

    for key, cfg in CONTESTS.items():
        active = year is None or int(year) >= int(cfg["start_season"])
        tag = cfg["col_tag"]
        status = "active" if active else f"n/a before {cfg['start_season']}"
        print(f"── {cfg['label']} ({tag}) — {status}")

        for flavor in PICK_FLAVORS:
            home, away = flavor_cols(flavor, key)
            for col in (home, away):
                present, nonnull, oor = check_column(df, col)
                required = active and flavor in REQUIRED_FLAVORS

                if not present:
                    if required:
                        failures.append(col)
                        print(f"     ❌ MISSING (required): {col}")
                    else:
                        # Not required (Archetype, or an inactive Splash contest)
                        note = "optional" if active else "expected absent"
                        print(f"     ·  absent ({note}): {col}")
                    continue

                flags = []
                if nonnull == 0:
                    flags.append("EMPTY")
                    warnings.append(f"{col} (empty)")
                if oor > 0:
                    flags.append(f"{oor} OUT-OF-RANGE")
                    warnings.append(f"{col} ({oor} out of range)")
                mark = "⚠️ " if flags else "✅"
                detail = f"{nonnull:,} non-null" + (f"; {', '.join(flags)}" if flags else "")
                print(f"     {mark} {col}: {detail}")
        print()

    print("=" * 60)
    if failures:
        print(f"❌ FAIL — {len(failures)} required column(s) missing:")
        for c in failures:
            print(f"     - {c}")
    else:
        print("✅ PASS — all required flavor columns present.")
    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s) (not fatal):")
        for w in warnings:
            print(f"     - {w}")
    print("=" * 60)
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser(description="Verify per-contest pick-% flavor columns.")
    ap.add_argument("year", nargs="?", type=int, help="Season year (finds the latest Season file).")
    ap.add_argument("--file", help="Explicit CSV path to check instead of the latest Season file.")
    args = ap.parse_args()

    if args.file:
        path = args.file
        year = args.year
    else:
        if args.year is None:
            ap.error("Provide a YEAR or --file.")
        path = latest_season_file(args.year)
        year = args.year
        if path is None:
            print(f"❌ No Season Final_Data file found for {args.year}.")
            sys.exit(1)

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        sys.exit(1)

    print(f"Reading {path}")
    df = pd.read_csv(path, low_memory=False)
    sys.exit(verify(df, year))


if __name__ == "__main__":
    main()
