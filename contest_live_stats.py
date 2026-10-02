"""
contest_live_stats.py

Live contest entry counts, read straight from the committed picks files, so the
UI no longer needs those numbers hand-maintained. The STATIC fields (entry fee,
total prize, double-pick weeks) stay in each contest's config; only the counts
that change weekly are derived here.

  total_entries     = number of rows in the picks file (the field size)
  surviving_entries = entries still alive as of the upcoming week:
                        • Splash picks carry a 'Status' column (Alive/Eliminated)
                          -> count the ones that aren't 'Eliminated'
                        • Circa picks carry 'Total_Wins'
                          -> count Total_Wins >= upcoming_week - 1
"""

import os

import pandas as pd

from contest_config import CONTESTS


def live_entry_counts(contest_key, year, upcoming_week=None):
    """(total_entries, surviving_entries) for a contest from its picks file, or
    (None, None) when the file is missing/unreadable (caller falls back to the
    manual config). Handles both the Splash 'Status' and Circa 'Total_Wins'
    survival signals."""
    cfg = CONTESTS.get(contest_key)
    if not cfg:
        return (None, None)
    pattern = cfg.get("picks_pattern")
    if not pattern:
        return (None, None)
    path = pattern.format(year=year)
    if not os.path.exists(path):
        return (None, None)
    try:
        df = pd.read_csv(path)
    except Exception:
        return (None, None)

    total = int(len(df))
    surviving = None
    if "Status" in df.columns:
        status = df["Status"].astype(str).str.strip().str.lower()
        surviving = int((status != "eliminated").sum())
    elif "Total_Wins" in df.columns and upcoming_week is not None:
        wins = pd.to_numeric(df["Total_Wins"], errors="coerce").fillna(0)
        surviving = int((wins >= (upcoming_week - 1)).sum())
    return (total, surviving)
