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
#        "09/09/2020", #Leading up to Week 1 - THIS WEEK WILL NEVER WORK BECAUISE THERE IS NO PREVIOUS DATA
#        "09/16/2020", #Leading up to Week 2
#        "09/23/2020", #Leading up to Week 3
#        "09/30/2020", #Leading up to Week 4
#        "10/07/2020", #Leading up to Week 5
#        "10/14/2020", #Leading up to Week 6
#        "10/21/2020", #Leading up to Week 7
#        "10/28/2020", #Leading up to Week 8
#        "11/04/2020", #Leading up to Week 9
#        "11/11/2020", #Leading up to Week 10
#        "11/18/2020", #Leading up to Week 11
#        "11/25/2020", #Leading up to Week 12
#        "11/28/2020", #Leading up to Week 13
#        "12/02/2020", #Leading up to Week 14
#        "12/09/2020", #Leading up to Week 15
#        "12/16/2020", #Leading up to Week 16
#        "12/23/2020", #Leading up to Week 17
#        "12/30/2020", #Leading up to Week 18

#        "09/08/2021", #Leading up to Week 1
#        "09/15/2021", #Leading up to Week 2
#        "09/22/2021", #Leading up to Week 3
#        "09/29/2021", #Leading up to Week 4
#        "10/06/2021", #Leading up to Week 5
#        "10/13/2021", #Leading up to Week 6
#        "10/20/2021", #Leading up to Week 7
#        "10/27/2021", #Leading up to Week 8
#        "11/03/2021", #Leading up to Week 9
#        "11/10/2021", #Leading up to Week 10
#        "11/17/2021", #Leading up to Week 11
#        "11/24/2021", #Leading up to Week 12
#        "11/27/2021", #Leading up to Week 13
#        "12/01/2021", #Leading up to Week 14
#        "12/08/2021", #Leading up to Week 15
#        "12/15/2021", #Leading up to Week 16
#        "12/22/2021", #Leading up to Week 17
#        "12/26/2021", #Leading up to Week 18
#        "12/29/2021", #Leading up to Week 19
#        "01/05/2022", #Leading up to Week 20

#        "09/07/2022", #Leading up to Week 1
#        "09/14/2022", #Leading up to Week 2
#        "09/21/2022", #Leading up to Week 3
#        "09/28/2022", #Leading up to Week 4
#        "10/05/2022", #Leading up to Week 5
#        "10/12/2022", #Leading up to Week 6
#        "10/19/2022", #Leading up to Week 7
#        "10/26/2022", #Leading up to Week 8
#        "11/02/2022", #Leading up to Week 9
#        "11/09/2022", #Leading up to Week 10
#        "11/16/2022", #Leading up to Week 11
#        "11/23/2022", #Leading up to Week 12
#        "11/26/2022", #Leading up to Week 13
#        "11/30/2022", #Leading up to Week 14
#        "12/07/2022", #Leading up to Week 15
#        "12/14/2022", #Leading up to Week 16
#        "12/21/2022", #Leading up to Week 17
#        "12/25/2022", #Leading up to Week 18
#        "12/28/2022", #Leading up to Week 19
#        "01/04/2023", #Leading up to Week 20

#        "09/06/2023", #Leading up to Week 1
#        "09/13/2023", #Leading up to Week 2
#        "09/20/2023", #Leading up to Week 3
#        "09/27/2023", #Leading up to Week 4
#        "10/04/2023", #Leading up to Week 5
#        "10/11/2023", #Leading up to Week 6
#        "10/18/2023", #Leading up to Week 7
#        "10/25/2023", #Leading up to Week 8
#        "11/01/2023", #Leading up to Week 9
#        "11/08/2023", #Leading up to Week 10
#        "11/15/2023", #Leading up to Week 11
#        "11/22/2023", #Leading up to Week 12
#        "11/25/2023", #Leading up to Week 13
#        "11/29/2023", #Leading up to Week 14
#        "12/06/2023", #Leading up to Week 15
#        "12/13/2023", #Leading up to Week 16
#        "12/20/2023", #Leading up to Week 17
#        "12/25/2023",  #Leading up to Week 18
#        "12/27/2023", #Leading up to Week 19
#        "01/03/2024", #Leading up to Week 20

#        "09/04/2024", #Leading up to Week 1
#        "09/11/2024", #Leading up to Week 2
#        "09/18/2024", #Leading up to Week 3
#        "09/25/2024", #Leading up to Week 4
#        "10/02/2024", #Leading up to Week 5
#        "10/09/2024", #Leading up to Week 6
#        "10/16/2024", #Leading up to Week 7
#        "10/23/2024", #Leading up to Week 8
#        "10/30/2024", #Leading up to Week 9
#        "11/06/2024", #Leading up to Week 10
#        "11/13/2024", #Leading up to Week 11
#        "11/20/2024", #Leading up to Week 12
#        "11/27/2024", #Leading up to Week 13
#        "11/30/2024", #Leading up to Week 14
#        "12/04/2024", #Leading up to Week 15
#        "12/11/2024", #Leading up to Week 16
#        "12/18/2024", #Leading up to Week 17
#        "12/24/2024", #Leading up to Week 18
#        "12/27/2024", #Leading up to Week 19
#        "01/01/2025", #Leading up to Week 20
        
#        "09/03/2025", #Leading up to Week 1
#        "09/10/2025", #Leading up to Week 2
#        "09/17/2025", #Leading up to Week 3
#        "09/24/2025", #Leading up to Week 4
#        "10/01/2025", #Leading up to Week 5
#        "10/08/2025", #Leading up to Week 6
#        "10/15/2025", #Leading up to Week 7
#        "10/22/2025", #Leading up to Week 8
#        "10/29/2025", #Leading up to Week 9
#        "11/05/2025", #Leading up to Week 10
#        "11/12/2025", #Leading up to Week 11
#        "11/19/2025", #Leading up to Week 12
#        "11/26/2025", #Leading up to Week 13
#        "11/29/2025", #Leading up to Week 14
#        "12/03/2025", #Leading up to Week 15
#        "12/10/2025", #Leading up to Week 16
#        "12/17/2025", #Leading up to Week 17
#        "12/24/2025", #Leading up to Week 18
#        "12/26/2025", #Leading up to Week 19
#        "12/31/2025", #Leading up to Week 20

#        "09/08/2026", #Leading up to Week 1
#		 "09/15/2026", #Leading up to Week 2
#        "09/22/2026", #Leading up to Week 3
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


def is_replay():
    """True when the pipeline is replaying a historical as-of date rather than
    running live.

    A replay is active when the orchestrator pins a date (PIPELINE_AS_OF_DATE)
    or when HISTORICAL_DATES is non-empty. Stages use this to avoid writing
    shared, all-years artifacts (e.g. the master Circa_historical_data.csv)
    with truncated point-in-time data during a backfill.
    """
    return bool(os.environ.get(_ENV_KEY)) or bool(HISTORICAL_DATES)


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
