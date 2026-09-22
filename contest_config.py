"""
contest_config.py

One place that says which survivor contest a pipeline run targets and
where that contest's data lives, so weekly_4 / weekly_5 / weekly_6 /
daily_4 are a single method that serves Circa and both Splash contests
with no per-script edits -- pick the contest with the SURVIVOR_CONTEST
environment variable (default 'circa').

    SURVIVOR_CONTEST=big_splash python weekly_5_build_entry_pick_training_data.py
    SURVIVOR_CONTEST=world_championship python daily_4_generate_entry_archetype_pick_estimates.py

SEASON GATING (answers "what happens if I run this for 2023?")
==============================================================
Each contest declares the first season it existed (`start_season`). The
Splash contests are 2026+, so a run for an earlier season finds no picks
file and is skipped with a clear message -- the pipeline never fabricates
Splash history for a year the contest didn't exist. Circa runs unchanged
for every season it has data.

MULTI-PICK WEEKS
================
`multi_pick_weeks` lists the NFL weeks where an entry selects TWO teams
(the Splash "multi-pick" weeks). Loaders split a "TEAM1;TEAM2" cell into
both picks; the builder labels both as chosen. Empty for Circa.
"""

import os

# Selected contest for this run (default keeps every existing Circa run
# behaving exactly as before).
CONTEST = os.environ.get('SURVIVOR_CONTEST', 'circa').strip().lower()

CONTESTS = {
    'circa': {
        'label': 'Circa Survivor',
        'picks_pattern': 'circa-pick-history/{year}_survivor_picks.csv',
        # Per-contest historical training file for daily_2's pick-% projection
        # (game features + that contest's observed 'Pick %'). Built by
        # build_contest_historical_data.py; Circa's is maintained directly.
        'historical_csv': 'contest-historical-data/Circa_historical_data.csv',
        # Basename daily_2 writes its projected pick-% schedule to, and the
        # short prefix used for this contest's projected columns when they are
        # merged into the shared schedule ('' keeps Circa's original
        # 'Home Pick %' / 'Away Pick %' names untouched).
        'predicted_out': 'Circa_Predicted_pick_percent.csv',
        'proj_prefix': '',
        'start_season': 2020,
        'multi_pick_weeks': set(),
        'out_tag': 'circa',
    },
    'big_splash': {
        'label': 'Splash Big Splash',
        'picks_pattern': 'big-splash-pick-history/{year}_big_splash_picks.csv',
        'historical_csv': 'contest-historical-data/BigSplash_historical_data.csv',
        'predicted_out': 'BigSplash_Predicted_pick_percent.csv',
        'proj_prefix': 'Big Splash',
        'start_season': 2026,
        'multi_pick_weeks': {3, 6, 9, 12, 13, 14, 15, 16},
        'out_tag': 'big_splash',
    },
    'world_championship': {
        'label': 'Splash Survivor World Championship',
        'picks_pattern': 'splash-world-championship-pick-history/{year}_world_championship_picks.csv',
        'historical_csv': 'contest-historical-data/WorldChampionship_historical_data.csv',
        'predicted_out': 'WorldChampionship_Predicted_pick_percent.csv',
        'proj_prefix': 'World Championship',
        'start_season': 2026,
        'multi_pick_weeks': {9, 12, 13, 14, 15, 16},
        'out_tag': 'world_championship',
    },
}


def get_contest(name=None):
    """Return the config dict for `name` (or the SURVIVOR_CONTEST default)."""
    key = (name or CONTEST or 'circa').strip().lower()
    if key not in CONTESTS:
        raise ValueError(
            f"Unknown contest {key!r}. Set SURVIVOR_CONTEST to one of: "
            f"{list(CONTESTS)}."
        )
    return CONTESTS[key]


# Historically the pipeline used one un-tagged filename per artifact (Circa
# was the only contest). To stay backward compatible, the Circa artifacts
# keep their original names and only the Splash contests get a tag suffix.
def tagged(path_no_ext, ext, contest=None):
    """Build an artifact path, appending the contest tag for non-Circa
    contests so contests never overwrite each other's files while Circa's
    existing filenames are preserved unchanged."""
    cfg = get_contest(contest)
    if cfg['out_tag'] == 'circa':
        return f"{path_no_ext}{ext}"
    return f"{path_no_ext}_{cfg['out_tag']}{ext}"
