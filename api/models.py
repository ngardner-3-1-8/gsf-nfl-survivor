from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

# ─────────────────────────────────────────────
# Scheduling / situational constraint toggles
# ─────────────────────────────────────────────
class SchedulingConstraints(BaseModel):
    avoid_away_short_rest: bool = False
    avoid_away_divisional: bool = False
    avoid_3_in_10: bool = False
    avoid_4_in_17: bool = False
    avoid_cumulative_rest: bool = False
    avoid_thursday_all: bool = False       # all teams in TNF
    avoid_thursday_away: bool = False      # only away teams in TNF
    avoid_back_to_back_away: bool = False
    avoid_international: bool = False
    avoid_weekly_rest_disadvantage: bool = False
    avoid_travel_disadvantage: bool = False

    # These two have sub-parameters
    avoid_close_divisional: bool = False
    min_div_spread: float = 3.0            # only active if avoid_close_divisional=True

    avoid_away_close: bool = False
    min_away_spread: float = 3.0           # only active if avoid_away_close=True

    # Bayesian / ranking consistency constraint
    # ── Massey-Peabody Bayesian constraints ──
    mp_bayesian_all_metrics: bool = False          # Same Winner Across All Metrics
    mp_bayesian_preseason_and_current: bool = False # Same Current and Preseason Adjusted Winner
    mp_bayesian_current_and_adjusted: bool = False  # Same Current and Adjusted Current Winner
    
    # ── Generic Sports Fan Bayesian constraints ──
    gsf_bayesian_adjusted: bool = False            # Same Adjusted Winner
    gsf_bayesian_preseason_and_current: bool = False # Current and Preseason Adjusted Winner
    gsf_bayesian_current_and_adjusted: bool = False  # Same Current and Adjusted Current Winner
    
    # ── Sportsbook / Sim / Consensus cross-model constraints ──
    sportsbook_bayesian_preseason_and_current: bool = False
    sim_bayesian_preseason_and_current: bool = False
    consensus_bayesian_preseason_and_current: bool = False
    
    # Require ALL selected Bayesian constraints to be satisfied (True)
    # vs. ANY one of them (False)
    bayesian_require_all: bool = False


# ─────────────────────────────────────────────
# Main optimizer request
# ─────────────────────────────────────────────
class OptimizeRequest(BaseModel):

    # --- Objective ---
    objective: Literal[
        "consensus", "sportsbook", "mp", "gsf", "sim", "win_pct"
    ] = "consensus"

    # How many top solutions to return per method (EV-based + ranking-based)
    number_solutions: Literal[1, 5, 10, 25, 50, 100] = 10

    # --- Week range ---
    start_week: int = Field(default=1, ge=1, le=20)
    end_week: int = Field(default=20, ge=1, le=20)

    # --- Contest pool size ---
    # -1 = auto-calculate from historical pick data
    current_week_entries: int = Field(default=-1, ge=-1)

    # --- Season-long prohibitions ---
    # Teams the user has already picked OR never wants picked
    prohibited_teams: List[str] = []

    # --- Required picks: lock a team to a specific week ---
    # {"Kansas City Chiefs": 3, "Buffalo Bills": 7}
    required_picks: Dict[str, int] = {}

    # --- Prohibited weekly picks: ban a team from specific weeks ---
    # {"Cleveland Browns": [1, 4, 9]}
    prohibited_weekly_picks: Dict[str, List[int]] = {}

    # --- Must be favored ---
    must_be_favored: bool = False
    favored_qualifier: Literal[
        "sportsbook",
        "mp",
        "gsf",
        "sim",
        "consensus",
        "all"       # must be favored by every model
    ] = "sportsbook"

    # --- Team availability this week ---
    # Fraction 0.0–1.0 per team. -1.0 = auto-estimate
    # {"Kansas City Chiefs": 0.42, "Buffalo Bills": -1.0}
    team_availabilities: Dict[str, float] = {}
    use_live_availability: bool = True

    # --- Custom team rankings ---
    # Override the default power ratings. "Default" = use system value
    # {"Kansas City Chiefs": 7.5, "New York Jets": -4.0}
    custom_rankings: Dict[str, float] = {}

    # --- Custom pick percentages ---
    # Per team per week. -1.0 = auto-estimate
    # {"week_3": {"Kansas City Chiefs": 0.32, "Buffalo Bills": -1.0}}
    custom_pick_percentages: Dict[str, Dict[str, float]] = {}

    # --- Scheduling / situational constraints ---
    scheduling: SchedulingConstraints = SchedulingConstraints()


# ─────────────────────────────────────────────
# Optimizer response
# ─────────────────────────────────────────────
class PickResult(BaseModel):
    week: int
    team: str
    ev: float
    win_pct: float
    pick_pct: float
    home_or_away: str
    opponent: str
    spread: Optional[float] = None
    temperature: Optional[float] = None
    precipitation: Optional[float] = None
    wind: Optional[float] = None
    is_thanksgiving: bool = False
    is_christmas: bool = False

class OptimizeResponse(BaseModel):
    # EV-optimized solutions
    ev_solutions: List[List[PickResult]] = []

    # Win-pct-optimized solutions
    ranking_solutions: List[List[PickResult]] = []

    # Aggregate stats for each solution
    total_ev: float = 0.0
    total_win_pct: float = 0.0

    feasible: bool = True
    message: str = ""
