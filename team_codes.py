"""
team_codes.py

Single source of truth for NFL team-code handling across the survivor
pipeline (weekly_4 / weekly_5 / weekly_6 / daily_4, and available to
daily_2). Historically each script carried its own TEAM_FULLNAME_TO_ABBR
copy, and the pick-history files use inconsistent abbreviations across
seasons and sources. Centralizing the maps here removes that drift and,
more importantly, canonicalizes pick codes so a stale/legacy code doesn't
silently fall out of the pipeline.

THE BUG THIS FIXES
==================
The Circa picks files encode the Rams as 'LAR' in 2020-2024 and 'LA' in
2025. Every archetype/choice script builds each week's available-team
pool by mapping full team names from the Final_Data files through
FULLNAME_TO_ABBR, which yields 'LA'. So a pick recorded as 'LAR' failed
the "is the picked team in this entry's available pool?" check in
compute_pick_scores() / build_choice_rows() and was silently skipped --
4,426 Rams picks across 2020-2024 never contributed to archetype scoring
or choice-model training. Routing every pick code through
canonical_pick_code() at load time collapses these aliases onto the same
canonical abbreviation the pool uses, so the picks line up.

CANONICAL ABBREVIATIONS
=======================
The canonical code for each franchise is whatever FULLNAME_TO_ABBR
produces (that is the code the Final_Data-derived pool uses, so it is the
join key everything else must agree with). Notably: Rams -> 'LA',
Jacksonville -> 'JAC', Washington -> 'WAS'. PICK_CODE_ALIASES maps every
legacy/alternate spelling onto that canonical code.
"""

# Full team name (as it appears in the Final_Data files) -> canonical abbr.
FULLNAME_TO_ABBR = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAC',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
}

# Team nickname (as it appears in the Splash entry CSVs, e.g. "Jaguars")
# -> canonical abbr. Needed to fold the Splash contests into this pipeline.
NICKNAME_TO_ABBR = {
    'Cardinals': 'ARI', 'Falcons': 'ATL', 'Ravens': 'BAL', 'Bills': 'BUF',
    'Panthers': 'CAR', 'Bears': 'CHI', 'Bengals': 'CIN', 'Browns': 'CLE',
    'Cowboys': 'DAL', 'Broncos': 'DEN', 'Lions': 'DET', 'Packers': 'GB',
    'Texans': 'HOU', 'Colts': 'IND', 'Jaguars': 'JAC', 'Chiefs': 'KC',
    'Raiders': 'LV', 'Chargers': 'LAC', 'Rams': 'LA', 'Dolphins': 'MIA',
    'Vikings': 'MIN', 'Patriots': 'NE', 'Saints': 'NO', 'Giants': 'NYG',
    'Jets': 'NYJ', 'Eagles': 'PHI', 'Steelers': 'PIT', '49ers': 'SF',
    'Seahawks': 'SEA', 'Buccaneers': 'TB', 'Titans': 'TEN', 'Commanders': 'WAS',
}

# Legacy / alternate abbreviations seen in picks files and other feeds,
# mapped onto the canonical code above. Left side is what might appear in
# the raw data; right side is canonical.
PICK_CODE_ALIASES = {
    'LAR': 'LA', 'STL': 'LA',                 # Rams (LA relocation, old St. Louis)
    'JAX': 'JAC',                             # Jacksonville
    'WSH': 'WAS', 'WFT': 'WAS',               # Washington
    'OAK': 'LV',                              # Raiders (old Oakland)
    'SD': 'LAC',                              # Chargers (old San Diego)
    'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU',  # feed-specific spellings
}

# Every code the rest of the pipeline treats as a real, canonical team.
VALID_ABBRS = set(FULLNAME_TO_ABBR.values())


def canonical_pick_code(code):
    """Normalize a raw pick code onto its canonical abbreviation.

    Uppercases/strips, then applies PICK_CODE_ALIASES (e.g. 'LAR' -> 'LA').
    Non-team sentinels like '' and 'ELIMINATED' pass through unchanged so
    callers can still filter them; an unrecognized code is returned as-is
    (upper/stripped) rather than dropped, so a genuinely new code surfaces
    downstream instead of vanishing silently.
    """
    if code is None:
        return code
    c = str(code).strip()
    if c == '' or c.upper() == 'ELIMINATED':
        return c
    c = c.upper()
    return PICK_CODE_ALIASES.get(c, c)
