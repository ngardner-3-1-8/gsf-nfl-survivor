"""
splash_config.py

Manual weekly configuration for the Splash Sports survivor contests.
You edit this file by hand each week — there's no automation for Splash data.

Each contest has:
  - display_name : shown in the UI sub-tab selector
  - entries      : current number of entries alive (for pool-size / EV context)
  - survivors    : entries that have survived so far (same as entries if you
                   track only the living pool)
  - double_pick_weeks : NFL week numbers requiring TWO picks (both must win)
  - weekly_pick_data  : {week: {TEAM_ABBR: actual_pick_fraction}} — optional,
                   fill in as the contest progresses; used to override the
                   model's estimated pick% with what's actually happening.
  - weekly_availability : {week: {TEAM_ABBR: available_fraction}} — optional,
                   from Splash's per-team availability report. When present for
                   a week, it's used DIRECTLY instead of the model's estimated
                   availability. Fraction 0.0–1.0 = share of surviving entries
                   that still have that team unused.

Pick % values are FRACTIONS (0.0–1.0), same scale as the model.
Weeks are RAW NFL WEEKS (no Thanksgiving/Christmas insertion).

To add a new week's data: add entries to `weekly_pick_data`, update `entries`
and `survivors`, and add the week number to `double_pick_weeks` if it's a
double-pick week.
"""

SPLASH_CONTESTS = {
    "big_splash": {
        "display_name": "The Big Splash",
        "total_entries": 33000,          # ← update each week
        "surviving_entries": 33000,        # ← update each week
        "entry_fee": 100,
        "total_prize": 3000000,
        "double_pick_weeks": [3, 6, 9, 12, 13, 14, 15, 16],   # ← e.g. [10, 14, 17] as they're announced
        "weekly_pick_data": {
            #1: {"ARI": 1.0, "ATL": 1.0, "BAL": 1.0, "BUF": 1.0, "CAR": 1.0, "CHI": 1.0, "CIN": 1.0, "CLE": 1.0, "DAL": 1.0, "DEN": 1.0, "DET": 1.0, "GB": 1.0, "HOU": 1.0, "IND": 1.0, "JAX": 1.0, "KC": 1.0, 
            #    "LA": 1.0, "LAC": 1.0, "LV": 1.0, "MIA": 1.0, "MIN": 1.0, "NE": 1.0, "NO": 1.0, "NYG": 1.0, "NYJ": 1.0, "PHI": 1.0, "PIT": 1.0, "SEA": 1.0, "SF": 1.0, "TB": 1.0, "TEN": 1.0, "WAS": 1.0,
            #    },
            #2: {"ARI": 1.0, "ATL": 1.0, "BAL": 1.0, "BUF": 1.0, "CAR": 1.0, "CHI": 1.0, "CIN": 1.0, "CLE": 1.0, "DAL": 1.0, "DEN": 1.0, "DET": 1.0, "GB": 1.0, "HOU": 1.0, "IND": 1.0, "JAX": 1.0, "KC": 1.0, 
            #    "LA": 1.0, "LAC": 1.0, "LV": 1.0, "MIA": 1.0, "MIN": 1.0, "NE": 1.0, "NO": 1.0, "NYG": 1.0, "NYJ": 1.0, "PHI": 1.0, "PIT": 1.0, "SEA": 1.0, "SF": 1.0, "TB": 1.0, "TEN": 1.0, "WAS": 1.0,
            #    },
        },
        "weekly_availability": {
            1: {"ARI": 1.0, "ATL": 1.0, "BAL": 1.0, "BUF": 1.0, "CAR": 1.0, "CHI": 1.0, "CIN": 1.0, "CLE": 1.0, "DAL": 1.0, "DEN": 1.0, "DET": 1.0, "GB": 1.0, "HOU": 1.0, "IND": 1.0, "JAX": 1.0, "KC": 1.0, 
                "LA": 1.0, "LAC": 1.0, "LV": 1.0, "MIA": 1.0, "MIN": 1.0, "NE": 1.0, "NO": 1.0, "NYG": 1.0, "NYJ": 1.0, "PHI": 1.0, "PIT": 1.0, "SEA": 1.0, "SF": 1.0, "TB": 1.0, "TEN": 1.0, "WAS": 1.0,
                },
            #2: {"ARI": 1.0, "ATL": 1.0, "BAL": 1.0, "BUF": 1.0, "CAR": 1.0, "CHI": 1.0, "CIN": 1.0, "CLE": 1.0, "DAL": 1.0, "DEN": 1.0, "DET": 1.0, "GB": 1.0, "HOU": 1.0, "IND": 1.0, "JAX": 1.0, "KC": 1.0, 
            #    "LA": 1.0, "LAC": 1.0, "LV": 1.0, "MIA": 1.0, "MIN": 1.0, "NE": 1.0, "NO": 1.0, "NYG": 1.0, "NYJ": 1.0, "PHI": 1.0, "PIT": 1.0, "SEA": 1.0, "SF": 1.0, "TB": 1.0, "TEN": 1.0, "WAS": 1.0,
            #    },
        },
    },
    "survivor_world_championship": {
        "display_name": "The Survivor World Championship",
        "total_entries": 21000,           # ← update each week
        "surviving_entries": 21000,         # ← update each week
        "entry_fee": 1000,
        "total_prize": 21000000,
        "double_pick_weeks": [9, 12, 13, 14, 15, 16],   # ← e.g. [6, 12] as they're announced
        "weekly_pick_data": {
            #1: {"ARI": 1.0, "ATL": 1.0, "BAL": 1.0, "BUF": 1.0, "CAR": 1.0, "CHI": 1.0, "CIN": 1.0, "CLE": 1.0, "DAL": 1.0, "DEN": 1.0, "DET": 1.0, "GB": 1.0, "HOU": 1.0, "IND": 1.0, "JAX": 1.0, "KC": 1.0, 
            #    "LA": 1.0, "LAC": 1.0, "LV": 1.0, "MIA": 1.0, "MIN": 1.0, "NE": 1.0, "NO": 1.0, "NYG": 1.0, "NYJ": 1.0, "PHI": 1.0, "PIT": 1.0, "SEA": 1.0, "SF": 1.0, "TB": 1.0, "TEN": 1.0, "WAS": 1.0,
            #    },
            #2: {"ARI": 1.0, "ATL": 1.0, "BAL": 1.0, "BUF": 1.0, "CAR": 1.0, "CHI": 1.0, "CIN": 1.0, "CLE": 1.0, "DAL": 1.0, "DEN": 1.0, "DET": 1.0, "GB": 1.0, "HOU": 1.0, "IND": 1.0, "JAX": 1.0, "KC": 1.0, 
            #    "LA": 1.0, "LAC": 1.0, "LV": 1.0, "MIA": 1.0, "MIN": 1.0, "NE": 1.0, "NO": 1.0, "NYG": 1.0, "NYJ": 1.0, "PHI": 1.0, "PIT": 1.0, "SEA": 1.0, "SF": 1.0, "TB": 1.0, "TEN": 1.0, "WAS": 1.0,
            #    },
        },
        "weekly_availability": {
            1: {"ARI": 1.0, "ATL": 1.0, "BAL": 1.0, "BUF": 1.0, "CAR": 1.0, "CHI": 1.0, "CIN": 1.0, "CLE": 1.0, "DAL": 1.0, "DEN": 1.0, "DET": 1.0, "GB": 1.0, "HOU": 1.0, "IND": 1.0, "JAX": 1.0, "KC": 1.0, 
                "LA": 1.0, "LAC": 1.0, "LV": 1.0, "MIA": 1.0, "MIN": 1.0, "NE": 1.0, "NO": 1.0, "NYG": 1.0, "NYJ": 1.0, "PHI": 1.0, "PIT": 1.0, "SEA": 1.0, "SF": 1.0, "TB": 1.0, "TEN": 1.0, "WAS": 1.0,
                },
            #2: {"ARI": 1.0, "ATL": 1.0, "BAL": 1.0, "BUF": 1.0, "CAR": 1.0, "CHI": 1.0, "CIN": 1.0, "CLE": 1.0, "DAL": 1.0, "DEN": 1.0, "DET": 1.0, "GB": 1.0, "HOU": 1.0, "IND": 1.0, "JAX": 1.0, "KC": 1.0, 
            #    "LA": 1.0, "LAC": 1.0, "LV": 1.0, "MIA": 1.0, "MIN": 1.0, "NE": 1.0, "NO": 1.0, "NYG": 1.0, "NYJ": 1.0, "PHI": 1.0, "PIT": 1.0, "SEA": 1.0, "SF": 1.0, "TB": 1.0, "TEN": 1.0, "WAS": 1.0,
            #    },
        },
    },
}


def get_contest(contest_key):
    """Return the config dict for a contest key, or None if unknown."""
    return SPLASH_CONTESTS.get(contest_key)


def list_contests():
    """[(key, display_name)] for populating the UI selector."""
    return [(k, v["display_name"]) for k, v in SPLASH_CONTESTS.items()]


def get_double_pick_weeks(contest_key):
    c = SPLASH_CONTESTS.get(contest_key)
    return list(c["double_pick_weeks"]) if c else []


def get_weekly_availability(contest_key):
    """
    {week(int): {TEAM: available_fraction}} from Splash's availability report,
    for weeks the user has entered. Empty when none entered — the simulation
    then falls back to its own estimated availability for those weeks.
    """
    c = SPLASH_CONTESTS.get(contest_key)
    if not c:
        return {}
    return {int(w): dict(teams)
            for w, teams in c.get("weekly_availability", {}).items()}


def get_weekly_pick_overrides(contest_key):
    """
    Flatten weekly_pick_data into the custom_pick_percentages shape the
    optimizer already understands: {"week_{n}": {TEAM: pct}}.
    """
    c = SPLASH_CONTESTS.get(contest_key)
    if not c:
        return {}
    out = {}
    for week, teams in c.get("weekly_pick_data", {}).items():
        out[f"week_{week}"] = dict(teams)
    return out
