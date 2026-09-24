"""
run_config.py

THE ONE PLACE to control historical replay for the whole pipeline.

Set HISTORICAL_DATES to one or more "MM/DD/YYYY" strings to run every script
"as if" today were each of those dates (each date is the day a weekly cycle
kicks off -- the same convention daily_2's old date list used). Leave it empty
to run for the real today.

    HISTORICAL_DATES = [
        "09/16/2026",   # leading up to Week 2
        "09/23/2026",   # leading up to Week 3
    ]

HOW IT DRIVES THE PIPELINE
==========================
run_pipeline.py loops these dates (outer loop) and, for each one, runs every
stage in order (inner loop) before moving to the next date -- so each
historical week is fully rebuilt start-to-finish, rather than running one
script across all weeks at a time.

It passes the current date to each stage via the PIPELINE_AS_OF_DATE
environment variable. Every script derives its season/week from that date:
  * date-driven scripts (daily_1/2/3, weekly_1) loop get_run_dates();
  * the rest read as_of_datetime() / as_of_year() instead of the real clock;
  * season_dates.resolve_week_context() defaults to as_of_date_str().

So a single run_config edit reaches the whole system -- no per-file edits.
"""

import os
from datetime import datetime

# ── EDIT THIS ────────────────────────────────────────────────────────────────
# Empty = run for the real today. Add "MM/DD/YYYY" strings to replay history.
HISTORICAL_DATES = [
    # "09/16/2026",
    # "09/23/2026",
]
# ─────────────────────────────────────────────────────────────────────────────

_DATE_FMT = "%m/%d/%Y"
_ENV_KEY = "PIPELINE_AS_OF_DATE"


def _today_str():
    return datetime.now().strftime(_DATE_FMT)


def as_of_date_str():
    """The single date this process should treat as 'today' (MM/DD/YYYY).

    The orchestrator sets PIPELINE_AS_OF_DATE per date; when it isn't set we
    fall back to the first HISTORICAL_DATES entry (so a lone script run still
    honors a pinned replay date) and finally to the real today."""
    env = os.environ.get(_ENV_KEY)
    if env:
        return env.strip()
    if HISTORICAL_DATES:
        return str(HISTORICAL_DATES[0]).strip()
    return _today_str()


def as_of_datetime():
    """as_of_date_str() as a datetime (midnight)."""
    return datetime.strptime(as_of_date_str(), _DATE_FMT)


def as_of_year():
    """Calendar year of the as-of date -- for scripts that key off a year."""
    return as_of_datetime().year


def get_run_dates():
    """The list of dates a DATE-DRIVEN script should iterate in its __main__.

    - Orchestrated (PIPELINE_AS_OF_DATE set): just that one date.
    - Standalone with HISTORICAL_DATES set: all of them (quick single-file
      backfill without the orchestrator).
    - Otherwise: the real today.
    """
    env = os.environ.get(_ENV_KEY)
    if env:
        return [env.strip()]
    if HISTORICAL_DATES:
        return [str(d).strip() for d in HISTORICAL_DATES]
    return [_today_str()]
