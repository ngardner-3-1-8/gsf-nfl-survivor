"""
circa_config.py

Static Circa Survivor contest config (manual, set once per season). The live
entry counts (total / surviving) come from the picks file via
contest_live_stats; only the static fields below are hand-set each year.
"""

CIRCA_CONTEST = {
    "key": "circa",
    "display_name": "Circa Survivor",
    "entry_fee": 1000,            # $ per entry
    "total_prize": 25017000,      # $ total prize pool
    "double_pick_weeks": [],      # Circa has no double-pick weeks
}


def get_circa_contest():
    """A fresh copy of the static Circa contest config."""
    return dict(CIRCA_CONTEST)
