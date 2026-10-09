# TYPICAL STARTERS MAP (Primary 2025 Starters)
TYPICAL_STARTERS = {
    'ARI': 'J.Brissett',
    'ATL': 'T.Tagovailoa',
    'BAL': 'L.Jackson',
    'BUF': 'J.Allen',
    'CAR': 'B.Young',
    'CHI': 'C.Williams',
    'CIN': 'J.Burrow',
    'CLE': 'D.Watson',
    'DAL': 'D.Prescott',
    'DEN': 'B.Nix',
    'DET': 'J.Goff',
    'GB': 'J.Love',
    'HOU': 'C.Stroud',
    'IND': 'D.Jones',
    'JAX': 'T.Lawrence',
    'KC': 'P.Mahomes',
    'LA': 'M.Stafford',
    'LAC': 'J.Herbert',
    'LV': 'K.Cousins',
    'MIA': 'M.Willis',
    'MIN': 'K.Murray',
    'NE': 'D.Maye',
    'NO': 'T.Shough',
    'NYG': 'J.Dart',
    'NYJ': 'G.Smith',
    'PHI': 'J.Hurts',
    'PIT': 'A.Rodgers',
    'SEA': 'S.Darnold',
    'SF': 'B.Purdy',
    'TB': 'B.Mayfield',
    'TEN': 'C.Ward',
    'WAS': 'J.Daniels'
}

# MANUAL OVERRIDE: [Backup Name, Wk1, Wk2, Wk3, Wk4...]
# True = Backup is starting, False = Typical Starter is playing
MANUAL_CURRENT_STARTERS = {
    'ARI': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'ATL': ['C.Rush', 'C.Rush', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix', 'M.Penix'],
    'BAL': [None, None, None, None, 'T.Huntley', None, None, None, None, None, None, None, None, None, None, None, None, None],
    'BUF': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'CAR': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'CHI': [None, None, 'T.Bagent', 'T.Bagent', 'T.Bagent', None, None, None, None, None, None, None, None, None, None, None, None, None],
    'CIN': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'CLE': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'DAL': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'DEN': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'DET': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'GB': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'HOU': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'IND': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'JAX': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'KC': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'LV': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'LAC': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'LA': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'MIA': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'MIN': [None, 'C.Wentz', None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'NE': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'NO': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'NYG': [None, None, 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston', 'J.Winston'],
    'NYJ': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'PHI': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'PIT': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'SEA': [None, 'D.Lock', None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'SF': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'TB': [None, None, None, 'J.Daniels', 'J.Daniels', None, None, None, None, None, None, None, None, None, None, None, None, None],
    'TEN': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'WAS': [None, None, 'M.Mariota', 'M.Mariota', None, None, None, None, None, None, None, None, None, None, None, None, None, None]
}

# FORCE REPLACEMENT-LEVEL RATING
# -----------------------------------------------------------------------------
# A set of (TEAM, 'F.Last') pairs pinned to replacement level in daily_1,
# overriding the normal rating lookup entirely.
#
# Use this ONLY for the one case the play-by-play data cannot resolve on its
# own: a spot-starter / backup whose short 'F.Last' name collides with an
# established NFL player AND who has no NFL snaps of his own yet. Until he takes
# a snap he is invisible to the data, so the rating resolver would otherwise
# hand him the established player's rating (e.g. a pre-debut Tampa Bay
# 'J.Daniels' borrowing Jayden Daniels' rating).
#
# Once the player has his own snaps the resolver separates them by team on its
# own, so REMOVE the entry at that point if you want his real (shrinkage-
# regressed) rating instead of a hard replacement floor.
#
# Example — Jalon Daniels on TB before his Week 4 2026 debut:
#     FORCE_REPLACEMENT = {('TB', 'J.Daniels')}
FORCE_REPLACEMENT = set()
