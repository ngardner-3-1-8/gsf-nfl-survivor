"""
season_dates.py

Single source of truth for "what season and week is it, given a date?" —
the logic daily_2_consolidate_predict_weekly_data.py uses, extracted so
every script derives the target year and week the same way instead of each
carrying its own (drifting) copy. Historically daily_3 and weekly_1 each
re-implemented this with hardcoded per-year `if/elif` Christmas branches and
their own fallbacks; this module replaces those with one config-driven pass.

WHAT IT COMPUTES (mirrors daily_2 exactly)
==========================================
- target_year: the SEASON a given calendar date belongs to. Jan–May
  (month < 6) is still finishing the previous season, so it maps to
  current_cal_year - 1; June onward maps to current_cal_year.
- The season's first game date, Thanksgiving / Black Friday / Christmas /
  Boxing Day, and the thanksgiving_week / christmas_week indices.
- starting_week: the raw NFL week now in progress / upcoming = last fully
  completed NFL week + 1 (a week counts as completed once its LAST game is
  strictly before `today`), capped at 18.
- upcoming_week: starting_week shifted forward for Circa's separate holiday
  contest weeks — +1 once the date is past Black Friday, +1 more once it is
  on/after that season's Christmas cutoff (see CIRCA_HOLIDAY_CONFIG).
- last_completed_week: the raw NFL week whose last game is already done
  (starting_week - 1 in the normal case, 0 before the season starts).
- thanksgiving_shift_applied / christmas_shift_applied: the two booleans
  behind the upcoming_week shift, exposed so a caller that maintains its own
  bookkeeping (e.g. weekly_1) can apply the SAME shifts to its own variables
  instead of re-deriving the cutoffs.

CIRCA HOLIDAY WEEKS
===================
Circa carves Thanksgiving and Christmas into their own contest weeks, which
pushes every later game up a week. Thanksgiving is uniform every season
(handled in code). Christmas placement depends on the actual schedule, so it
is listed per season in CIRCA_HOLIDAY_CONFIG — the SAME table daily_2's
schedule build uses (daily_2 imports it from here). To add a season, read
that year's schedule and enter the Christmas slate date(s) and the first date
that should roll into the following week.
"""

import calendar
from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

SCHEDULE_PATTERN = "nfl-schedules/schedule_{year}.csv"

# christmas_shift_from : games on/after this date get +1 to the Circa week
#                        (None = no separate Circa Christmas week that season)
# christmas_label_dates: dates shown as 'Christmas' in the 'Circa Week' column
CIRCA_HOLIDAY_CONFIG = {
    2020: {'christmas_shift_from': None,
           'christmas_label_dates': []},
    2021: {'christmas_shift_from': datetime(2021, 12, 26),
           'christmas_label_dates': [datetime(2021, 12, 25), datetime(2021, 12, 23)]},
    2022: {'christmas_shift_from': datetime(2022, 12, 25),
           'christmas_label_dates': [datetime(2022, 12, 25), datetime(2022, 12, 26)]},
    2023: {'christmas_shift_from': datetime(2023, 12, 25),
           'christmas_label_dates': [datetime(2023, 12, 25)]},
    2024: {'christmas_shift_from': datetime(2024, 12, 27),
           'christmas_label_dates': [datetime(2024, 12, 25), datetime(2024, 12, 26)]},
    2025: {'christmas_shift_from': datetime(2025, 12, 26),
           'christmas_label_dates': [datetime(2025, 12, 25)]},
    2026: {'christmas_shift_from': datetime(2026, 12, 26),
           'christmas_label_dates': [datetime(2026, 12, 25), datetime(2026, 12, 24)]},
}


def default_circa_holiday(target_year):
    """Fallback for any season not listed in CIRCA_HOLIDAY_CONFIG: mirror the
    common pattern (Christmas week starts on Boxing Day; label Christmas Day +
    Boxing Day)."""
    christmas_day = datetime(target_year, 12, 25)
    boxing_day = datetime(target_year, 12, 26)
    return {'christmas_shift_from': boxing_day,
            'christmas_label_dates': [christmas_day, boxing_day]}


def get_thanksgiving(year):
    """4th Thursday in November."""
    c = calendar.monthcalendar(year, 11)
    thursdays = [row[calendar.THURSDAY] for row in c if row[calendar.THURSDAY] != 0]
    return datetime(year, 11, thursdays[3])


def resolve_target_year(today):
    """Season a calendar date belongs to (Jan–May -> previous season)."""
    return today.year - 1 if today.month < 6 else today.year


def resolve_week_context(date_str=None, schedule_df=None):
    """Return a SimpleNamespace describing the season/week for `date_str`,
    computed exactly the way daily_2 does it.

    `date_str` defaults to the pipeline's as-of date (run_config.as_of_date_str
    -- the PIPELINE_AS_OF_DATE env var the historical orchestrator sets, else
    the real today), so a script that just calls resolve_week_context() with no
    argument automatically respects a historical replay.

    Pass `schedule_df` to reuse an already-loaded schedule (must have 'Date'
    and 'Week' columns); otherwise the season's schedule CSV is loaded from
    SCHEDULE_PATTERN. Access fields by attribute, e.g. ctx.target_year,
    ctx.upcoming_week, ctx.starting_week, ctx.last_completed_week.
    """
    if date_str is None:
        from run_config import as_of_date_str
        date_str = as_of_date_str()
    today = pd.to_datetime(date_str)
    current_cal_year = today.year
    target_year = resolve_target_year(today)

    if schedule_df is None:
        schedule_df = pd.read_csv(SCHEDULE_PATTERN.format(year=target_year))
    else:
        schedule_df = schedule_df.copy()
    schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])
    first_game_date = schedule_df['Date'].min()

    thanksgiving_date = get_thanksgiving(target_year)
    black_friday = thanksgiving_date + timedelta(days=1)
    black_wednesday = thanksgiving_date - timedelta(days=1)
    christmas_day = datetime(target_year, 12, 25)
    boxing_day = datetime(target_year, 12, 26)

    # +1 because the first game date is week 1, not week 0; +2 for Christmas
    # additionally accounts for the separate Thanksgiving contest week.#################Might change this back to thanksgiving_date instead of black_wednesday
    thanksgiving_week = int((black_wednesday - first_game_date).days / 7) + 1
    christmas_week = int((christmas_day - first_game_date).days / 7) + 2

    thanksgiving_shift_applied = False
    christmas_shift_applied = False

    if today <= first_game_date:
        # Before the season starts: week 1, and (as daily_2 does) load the
        # PRIOR season's history.
        starting_week = 1
        upcoming_week = 1
        last_completed_week = 0
        target_year_load = target_year - 1
    else:
        target_year_load = target_year
        week_end_dates = schedule_df.groupby('Week')['Date'].max()
        completed_weeks = week_end_dates[week_end_dates < today]
        if not completed_weeks.empty:
            last_completed_week = int(completed_weeks.index.max())
            starting_week = last_completed_week + 1
            upcoming_week = starting_week
            # Circa holiday shifts (same boundaries as the schedule build).
            if today > black_friday:
                thanksgiving_shift_applied = True
                upcoming_week += 1
            xmas_from = CIRCA_HOLIDAY_CONFIG.get(
                target_year, default_circa_holiday(target_year)).get('christmas_shift_from')
            if xmas_from is not None and today >= pd.to_datetime(xmas_from):
                christmas_shift_applied = True
                upcoming_week += 1
            if starting_week > 18:
                starting_week = 18
        else:
            starting_week = 1
            upcoming_week = 1
            last_completed_week = 0

    return SimpleNamespace(
        date_str=date_str,
        today=today,
        current_cal_year=current_cal_year,
        target_year=target_year,
        target_year_load=target_year_load,
        schedule_df=schedule_df,
        first_game_date=first_game_date,
        thanksgiving_date=thanksgiving_date,
        black_friday=black_friday,
        christmas_day=christmas_day,
        boxing_day=boxing_day,
        thanksgiving_week=thanksgiving_week,
        christmas_week=christmas_week,
        starting_week=starting_week,
        upcoming_week=upcoming_week,
        last_completed_week=last_completed_week,
        thanksgiving_shift_applied=thanksgiving_shift_applied,
        christmas_shift_applied=christmas_shift_applied,
    )
