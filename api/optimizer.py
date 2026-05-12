# api/optimizer.py
#
# Circa Survivor OR-Tools optimizer.
# Receives a merged sim DataFrame and an OptimizeRequest,
# runs the SCIP solver N times (forbidding each previous solution),
# and returns two lists of solutions: one maximizing EV, one maximizing win %.

import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from typing import List, Tuple

from api.models import OptimizeRequest, OptimizeResponse, PickResult

def safe_float(val, default=None):
    """Convert to float, returning default if None, NaN or inf."""
    try:
        if val is None:
            return default
        f = float(val)
        if f != f:  # NaN check
            return default
        if f == float('inf') or f == float('-inf'):
            return default
        return f
    except (TypeError, ValueError):
        return default

# ─────────────────────────────────────────────────────────────
# Column mapping — current CSV → internal solver names
# ─────────────────────────────────────────────────────────────

AWAY_COL_MAP = {
    "Away Team":                                        "Team",
    "Home Team":                                        "Opponent",
    "Week":                                             "Week_Num",
    "Actual Stadium":                                   "Actual Stadium",
    "International Game":                               "International Game",
    "Divisional Matchup?":                             "Divisional Matchup?",
    "Away Team Short Rest":                            "Away Team Short Rest",
    "Away Team 3 games in 10 days":                    "3 Games in 10 Days",
    "Away Team 4 games in 17 days":                    "4 Games in 17 Days",
    "Back to Back Away Games":                         "Back to Back Away Games",
    "Away Travel Advantage":                           "Travel Advantage",
    "Away Weekly Timezone Difference":                 "Weekly Timezone Difference",
    "Away Team Weekly Rest":                           "Weekly Rest",
    "Weekly Away Rest Advantage":                      "Weekly Rest Advantage",
    "Away Cumulative Rest Advantage":                  "Season-Long Rest Advantage",
    "Away Team Current Week Cumulative Rest Advantage":"Season-Long Rest Advantage Including This Week",
    "Away Team Massey-Peabody Current Rank":           "MP Current Rank",
    "Away Team Generic Sports Fan Current Rank":       "GSF Current Rank",
    "Away Team Adjusted Massey-Peabody Current Rank":  "Adjusted Current Rank",
    "Away Team Adjusted Generic Sports Fan Current Rank": "Opp Adjusted Current Rank_team",
    "Home Team Adjusted Massey-Peabody Current Rank":  "Opp Adjusted Current Rank",
    "Adjusted MP + GSF Average Current Difference":    "Adjusted Current Difference",
    "Adjusted MP + GSF Average Current Winner":        "Adjusted Current Winner",
    "Away Team Sportsbook Fair Odds":                  "Fair Odds Based on Sportsbook Odds",
    "Away Team Massey-Peabody Fair Odds":              "Fair Odds Based on MP",
    "Away Team Generic Sports Fan Fair Odds":          "Fair Odds Based on GSF",
    "Consensus Away Win Pct":                          "Fair Odds Consensus",
    "Away Team Sportsbook Spread":                     "Spread Based on Sportsbook Odds",
    "Massey-Peabody Away Team Spread":                 "Spread Based on MP",
    "Favorite":                                        "Favorite",
    "Away Team Massey-Peabody Preseason Rank":         "Preseason Rank",
    "Away Team Adjusted MP + GSF Average Preseason Rank": "Adjusted Preseason Rank",
    "Home Team Adjusted MP + GSF Average Preseason Rank": "Opp Adjusted Preseason Rank",
    "Away Team Thanksgiving Favorite":                 "Thanksgiving Favorite",
    "Away Team Christmas Favorite":                    "Christmas Favorite",
    "Away Team Thanksgiving Underdog":                 "Thanksgiving Underdog",
    "Away Team Christmas Underdog":                    "Christmas Underdog",
    "Away Team Expected Availability":                 "Expected Availability",
    "Away Pick %":                                     "Expected Pick Percent",
    "Away Expected Survival Rate":                     "Expected Survival Rate",
    "Away Expected Elimination Percent":               "Expected Contest Elimination Percent",
    "Expected Away Team Picks":                        "Expected Picks",
    "Expected Away Team Survivors":                    "Expected Survivors",
    "Expected Away Team Eliminations":                 "Expected Eliminations",
    "Total Remaining Entries at Start of Week":        "Total Remaining Entries at Start of Week",
    "Away Team Previous Opponent":                     "Previous Opponent",
    "Away Team Previous Location":                     "Previous Game Location",
    "Away Team Next Opponent":                         "Next Opponent",
    "Away Team Next Location":                         "Next Game Location",
    "Date_x":                                          "Date",
    "Time":                                            "Game_Time",
    "Dome":                                            "Dome",
    "Away_Starting_QB":                                "Starting_QB",
    "Thursday Night Game":                             "Thursday Night Game",
    "Away Team Weekly Rest":                           "Days_of_Rest",
    "Weekly Away Rest Advantage":                      "Rest_Advantage",
    "Away Cumulative Rest Advantage":                  "Cumulative_Rest",
    "Circa Week":                                      "Circa_Week",
    "Temperature":                                     "Temperature",
    "Precipitation":                                   "Precipitation",
    "Wind":                                            "Wind",
}

HOME_COL_MAP = {
    "Home Team":                                        "Team",
    "Away Team":                                        "Opponent",
    "Week":                                             "Week_Num",
    "Actual Stadium":                                   "Actual Stadium",
    "International Game":                               "International Game",
    "Divisional Matchup?":                             "Divisional Matchup?",
    "Away Team Short Rest":                            "Away Team Short Rest",
    "Home Team 3 games in 10 days":                    "3 Games in 10 Days",
    "Home Team 4 games in 17 days":                    "4 Games in 17 Days",
    "Back to Back Away Games":                         "Back to Back Away Games",
    "Home Travel Advantage":                           "Travel Advantage",
    "Home Weekly Timezone Difference":                 "Weekly Timezone Difference",
    "Home Team Weekly Rest":                           "Weekly Rest",
    "Weekly Home Rest Advantage":                      "Weekly Rest Advantage",
    "Home Cumulative Rest Advantage":                  "Season-Long Rest Advantage",
    "Home Team Current Week Cumulative Rest Advantage":"Season-Long Rest Advantage Including This Week",
    "Home Team Massey-Peabody Current Rank":           "MP Current Rank",
    "Home Team Generic Sports Fan Current Rank":       "GSF Current Rank",
    "Home Team Adjusted Massey-Peabody Current Rank":  "Adjusted Current Rank",
    "Away Team Adjusted Massey-Peabody Current Rank":  "Opp Adjusted Current Rank",
    "Adjusted MP + GSF Average Current Difference":    "Adjusted Current Difference",
    "Adjusted MP + GSF Average Current Winner":        "Adjusted Current Winner",
    "Home Team Sportsbook Fair Odds":                  "Fair Odds Based on Sportsbook Odds",
    "Home Team Massey-Peabody Fair Odds":              "Fair Odds Based on MP",
    "Home Team Generic Sports Fan Fair Odds":          "Fair Odds Based on GSF",
    "Consensus Home Win Pct":                          "Fair Odds Consensus",
    "Home Team Sportsbook Spread":                     "Spread Based on Sportsbook Odds",
    "Massey-Peabody Home Team Spread":                 "Spread Based on MP",
    "Favorite":                                        "Favorite",
    "Home Team Massey-Peabody Preseason Rank":         "Preseason Rank",
    "Home Team Adjusted MP + GSF Average Preseason Rank": "Adjusted Preseason Rank",
    "Away Team Adjusted MP + GSF Average Preseason Rank": "Opp Adjusted Preseason Rank",
    "Home Team Thanksgiving Favorite":                 "Thanksgiving Favorite",
    "Home Team Christmas Favorite":                    "Christmas Favorite",
    "Home Team Thanksgiving Underdog":                 "Thanksgiving Underdog",
    "Home Team Christmas Underdog":                    "Christmas Underdog",
    "Home Team Expected Availability":                 "Expected Availability",
    "Home Pick %":                                     "Expected Pick Percent",
    "Home Expected Survival Rate":                     "Expected Survival Rate",
    "Home Expected Elimination Percent":               "Expected Contest Elimination Percent",
    "Expected Home Team Picks":                        "Expected Picks",
    "Expected Home Team Survivors":                    "Expected Survivors",
    "Expected Home Team Eliminations":                 "Expected Eliminations",
    "Total Remaining Entries at Start of Week":        "Total Remaining Entries at Start of Week",
    "Home Team Previous Opponent":                     "Previous Opponent",
    "Home Team Previous Location":                     "Previous Game Location",
    "Home Team Next Opponent":                         "Next Opponent",
    "Home Team Next Location":                         "Next Game Location",
    "Date_x":                                          "Date",
    "Time":                                            "Game_Time",
    "Dome":                                            "Dome",
    "Home_Starting_QB":                                "Starting_QB",
    "Thursday Night Game":                             "Thursday Night Game",
    "Home Team Weekly Rest":                           "Days_of_Rest",
    "Weekly Home Rest Advantage":                      "Rest_Advantage",
    "Home Cumulative Rest Advantage":                  "Cumulative_Rest",
    "Circa Week":                                      "Circa_Week",
    "Temperature":                                     "Temperature",
    "Precipitation":                                   "Precipitation",
    "Wind":                                            "Wind",
}

OBJECTIVE_EV_COLS = {
    "consensus":  ("consensus_Away_EV",  "consensus_Home_EV"),
    "sportsbook": ("sportsbook_Away_EV", "sportsbook_Home_EV"),
    "mp":         ("mp_Away_EV",         "mp_Home_EV"),
    "gsf":        ("gsf_Away_EV",        "gsf_Home_EV"),
    "sim":        ("sim_Away_EV",        "sim_Home_EV"),
    "win_pct":    ("Consensus Away Win Pct", "Consensus Home Win Pct"),
}

OBJECTIVE_WIN_PCT_COLS = {
    "consensus":  ("Consensus Away Win Pct",             "Consensus Home Win Pct"),
    "sportsbook": ("Away Team Sportsbook Fair Odds",     "Home Team Sportsbook Fair Odds"),
    "mp":         ("Away Team Massey-Peabody Fair Odds", "Home Team Massey-Peabody Fair Odds"),
    "gsf":        ("Away Team Generic Sports Fan Fair Odds", "Home Team Generic Sports Fan Fair Odds"),
    "sim":        ("Sim_Away_Win_Pct",                   "Sim_Home_Win_Pct"),
    "win_pct":    ("Consensus Away Win Pct",             "Consensus Home Win Pct"),
}


# ─────────────────────────────────────────────────────────────
# Step 1 — Prepare the team-centric DataFrame
# ─────────────────────────────────────────────────────────────

def prepare_df(sim_df: pd.DataFrame, request: OptimizeRequest) -> pd.DataFrame:
    """
    Converts the game-centric sim CSV into a team-centric DataFrame
    (one row per team per game), then applies week range and
    prohibited team filters.

    IMPORTANT: always returns a DataFrame with a clean 0-based integer index
    so that positional indexing (df.iloc[i]) and dict keys (picks[i]) align.
    """
    df = sim_df.copy().reset_index(drop=True)

    away_ev_col, home_ev_col = OBJECTIVE_EV_COLS.get(
        request.objective, ("consensus_Away_EV", "consensus_Home_EV")
    )
    away_win_col, home_win_col = OBJECTIVE_WIN_PCT_COLS.get(
        request.objective, ("Consensus Away Win Pct", "Consensus Home Win Pct")
    )

    # ── Away team rows ──
    away_cols = {k: v for k, v in AWAY_COL_MAP.items() if k in df.columns}
    away_df = df[list(away_cols.keys())].rename(columns=away_cols).copy()
    away_df["Team Is Away"] = True
    away_df["EV"] = df[away_ev_col].values if away_ev_col in df.columns else 0.0
    away_df["Win Pct"] = df[away_win_col].values if away_win_col in df.columns else 0.0
    away_df["Sportsbook Spread"] = df["Away Team Sportsbook Spread"].values if "Away Team Sportsbook Spread" in df.columns else 0.0
    away_df["Game_Index"] = df.index
    # Away display columns — bypass duplicate key limitation in AWAY_COL_MAP
    away_df["Days_of_Rest"] = df["Away Team Weekly Rest"].values if "Away Team Weekly Rest" in df.columns else None
    away_df["Rest_Advantage"] = df["Weekly Away Rest Advantage"].values if "Weekly Away Rest Advantage" in df.columns else None
    away_df["Cumulative_Rest"] = df["Away Cumulative Rest Advantage"].values if "Away Cumulative Rest Advantage" in df.columns else None

    # ── Home team rows ──
    home_cols = {k: v for k, v in HOME_COL_MAP.items() if k in df.columns}
    home_df = df[list(home_cols.keys())].rename(columns=home_cols).copy()
    home_df["Team Is Away"] = False
    home_df["EV"] = df[home_ev_col].values if home_ev_col in df.columns else 0.0
    home_df["Win Pct"] = df[home_win_col].values if home_win_col in df.columns else 0.0
    home_df["Sportsbook Spread"] = df["Home Team Sportsbook Spread"].values if "Home Team Sportsbook Spread" in df.columns else 0.0
    home_df["Game_Index"] = df.index
    home_df["Away Team Short Rest"] = "No"
    home_df["Back to Back Away Games"] = False
    # Home display columns
    home_df["Days_of_Rest"] = df["Home Team Weekly Rest"].values if "Home Team Weekly Rest" in df.columns else None
    home_df["Rest_Advantage"] = df["Weekly Home Rest Advantage"].values if "Weekly Home Rest Advantage" in df.columns else None
    home_df["Cumulative_Rest"] = df["Home Cumulative Rest Advantage"].values if "Home Cumulative Rest Advantage" in df.columns else None

    # Concatenate and immediately reset to a clean 0-based index
    combined = pd.concat([away_df, home_df], ignore_index=True)

    # Filter week range — reset index after every filter
    combined = combined[
        (combined["Week_Num"] >= request.start_week) &
        (combined["Week_Num"] <= request.end_week)
    ].reset_index(drop=True)

    # Filter prohibited teams — reset index after every filter
    if request.prohibited_teams:
        combined = combined[
            ~combined["Team"].isin(request.prohibited_teams)
        ].reset_index(drop=True)

    # Custom pick percentage overrides
    for week_key, team_overrides in request.custom_pick_percentages.items():
        try:
            week_num = int(week_key.replace("week_", ""))
        except ValueError:
            continue
        for team, pct in team_overrides.items():
            if pct >= 0:
                mask = (combined["Week_Num"] == week_num) & (combined["Team"] == team)
                combined.loc[mask, "Expected Pick Percent"] = pct

    # Custom ranking overrides
    for team, ranking in request.custom_rankings.items():
        mask = combined["Team"] == team
        combined.loc[mask, "Adjusted Current Rank"] = ranking

    # Final safety reset — guarantees iloc[i] == picks key i
    return combined.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Step 2 — Apply constraints to the solver
# ─────────────────────────────────────────────────────────────
def apply_constraints(
    solver: pywraplp.Solver,
    picks: dict,
    df: pd.DataFrame,
    request: OptimizeRequest,
) -> None:
    """
    Adds all scheduling and situational constraints to the solver.
    df MUST have a clean 0-based index (guaranteed by prepare_df).
    """
    s = request.scheduling

    # ── Required picks FIRST — force specific team-week combinations ──
    # Must be done before the per-row loop so required_positions is defined
    required_positions = set()
    for team, req_week in request.required_picks.items():
        if req_week > 0:
            positions = [
                i for i in range(len(df))
                if df.iloc[i]["Team"] == team and df.iloc[i]["Week_Num"] == req_week
            ]
            if positions:
                solver.Add(picks[positions[0]] == 1)
                required_positions.add(positions[0])
                # Force all other teams in this week to 0
                for j in range(len(df)):
                    if j != positions[0] and df.iloc[j]["Week_Num"] == req_week:
                        solver.Add(picks[j] == 0)

    # ── Built once outside the row loop ──
    BAYESIAN_CHECKS = [
        (s.mp_bayesian_all_metrics,
            "Massey-Peabody Bayesian Same Winner Across All Metrics"),
        (s.mp_bayesian_preseason_and_current,
            "Massey-Peabody Bayesian Same Current and Preseason Adjusted Winner"),
        (s.mp_bayesian_current_and_adjusted,
            "Massey-Peabody Bayesian Same Current and Adjusted Current Winner"),
        (s.gsf_bayesian_adjusted,
            "Generic Sports Fan Bayesian Same Adjusted Winner Across All Metrics"),
        (s.gsf_bayesian_preseason_and_current,
            "Generic Sports Fan Bayesian Current and Preseason Adjusted Winner"),
        (s.gsf_bayesian_current_and_adjusted,
            "Generic Sports Fan Bayesian Same Current and Adjusted Current Winner"),
        (s.sportsbook_bayesian_preseason_and_current,
            "Sportsbook Bayesian Same Current and Preseason Adjusted Winner"),
        (s.sim_bayesian_preseason_and_current,
            "Sim Bayesian Same Current and Preseason Adjusted Winner"),
        (s.consensus_bayesian_preseason_and_current,
            "Consensus Bayesian Same Current and Preseason Adjusted Winner"),
    ]
    active_bayesian_checks = [(enabled, col) for enabled, col in BAYESIAN_CHECKS if enabled]

    CLOSE_MATCHUP_COLS = {
        "sportsbook": "Fair Odds Based on Sportsbook Odds",
        "mp":         "Fair Odds Based on MP",
        "gsf":        "Fair Odds Based on GSF",
        "sim":        "Win Pct",
        "consensus":  "Fair Odds Consensus",
        "win_pct":    "Fair Odds Consensus",
    }

    FAIR_ODDS_COLS = {
        "sportsbook": "Fair Odds Based on Sportsbook Odds",
        "mp":         "Fair Odds Based on MP",
        "gsf":        "Fair Odds Based on GSF",
        "sim":        "Win Pct",
        "consensus":  "Fair Odds Consensus",
        "win_pct":    "Fair Odds Consensus",
    }

    def is_favored_by(model: str, team: str, row, is_away: bool) -> bool:
        col = FAIR_ODDS_COLS.get(model)
        if not col:
            return True
        if col not in df.columns:
            if model == "consensus":
                col = "Win Pct"
            else:
                return True
        if col not in df.columns:
            return True
        val = row.get(col, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return True
        return float(val) > 0.5

    fq = request.favored_qualifier

    # ── Per-row constraints — skip required picks ──
    for i in range(len(df)):
        # Required picks bypass all constraints
        if i in required_positions:
            continue

        row = df.iloc[i]
        is_away = bool(row.get("Team Is Away", False))
        team = str(row.get("Team", ""))
        week_num = int(row.get("Week_Num", 0))

        # Must be favored
        if request.must_be_favored:
            if fq == "all":
                all_models = ["sportsbook", "mp", "gsf", "sim", "consensus"]
                if not all(is_favored_by(m, team, row, is_away) for m in all_models):
                    solver.Add(picks[i] == 0)
            else:
                if not is_favored_by(fq, team, row, is_away):
                    solver.Add(picks[i] == 0)

        # Away teams in close matchups
        if s.avoid_away_close and is_away:
            col = CLOSE_MATCHUP_COLS.get(request.objective, "Fair Odds Consensus")
            if col in df.columns:
                val = row.get(col, None)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    if float(val) < 0.65:
                        solver.Add(picks[i] == 0)

        # Close divisional matchups
        if s.avoid_close_divisional:
            is_div = row.get("Divisional Matchup?", 0)
            if is_div == 1 or is_div == "Divisional" or is_div is True:
                col = CLOSE_MATCHUP_COLS.get(request.objective, "Fair Odds Consensus")
                if col in df.columns:
                    val = row.get(col, None)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        if float(val) < 0.65:
                            solver.Add(picks[i] == 0)

        # Away divisional matchups
        if s.avoid_away_divisional and is_away:
            is_div = row.get("Divisional Matchup?", 0)
            if is_div == 1 or is_div == "Divisional" or is_div is True:
                solver.Add(picks[i] == 0)

        # Away short rest
        if s.avoid_away_short_rest:
            val = str(row.get("Away Team Short Rest", "No")).strip().lower()
            if val in ("yes", "true", "1"):
                solver.Add(picks[i] == 0)

        # 3 in 10
        if s.avoid_3_in_10:
            val = str(row.get("3 Games in 10 Days", "No")).strip().lower()
            if val in ("yes", "true", "1"):
                solver.Add(picks[i] == 0)

        # 4 in 17
        if s.avoid_4_in_17:
            val = str(row.get("4 Games in 17 Days", "No")).strip().lower()
            if val in ("yes", "true", "1"):
                solver.Add(picks[i] == 0)

        # International
        if s.avoid_international:
            if bool(row.get("International Game", False)):
                solver.Add(picks[i] == 0)

        # TNF all teams
        if s.avoid_thursday_all:
            val = str(row.get("Thursday Night Game", "False")).strip().lower()
            if val in ("yes", "true", "1"):
                solver.Add(picks[i] == 0)

        # TNF away only
        if s.avoid_thursday_away and is_away:
            val = str(row.get("Thursday Night Game", "False")).strip().lower()
            if val in ("yes", "true", "1"):
                solver.Add(picks[i] == 0)

        # Back to back away
        if s.avoid_back_to_back_away and is_away:
            val = str(row.get("Back to Back Away Games", "False")).strip().lower()
            if val in ("yes", "true", "1"):
                solver.Add(picks[i] == 0)

        # Weekly rest disadvantage
        if s.avoid_weekly_rest_disadvantage:
            rest_adv = row.get("Rest_Advantage", None)
            if rest_adv is not None and not (isinstance(rest_adv, float) and np.isnan(rest_adv)):
                if float(rest_adv) < 0:
                    solver.Add(picks[i] == 0)

        # Cumulative rest disadvantage
        if s.avoid_cumulative_rest:
            cum_adv = row.get("Cumulative_Rest", None)
            if cum_adv is not None and not (isinstance(cum_adv, float) and np.isnan(cum_adv)):
                if float(cum_adv) < 0:
                    solver.Add(picks[i] == 0)

        # Travel disadvantage
        if s.avoid_travel_disadvantage and is_away:
            travel = row.get("Travel Advantage", None)
            if travel is not None and not (isinstance(travel, float) and np.isnan(travel)):
                if float(travel) < 0:
                    solver.Add(picks[i] == 0)

        # Bayesian constraints
        if active_bayesian_checks:
            bay_results = []
            for _, col in active_bayesian_checks:
                if col not in df.columns:
                    continue  # column missing — skip, don't penalise
                val = row.get(col, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue  # empty value — skip, don't penalise
                bay_results.append(
                    str(val).strip().lower() in ("true", "yes", "1", "same")
                )

            if bay_results:  # only constrain if we got at least one real value
                if s.bayesian_require_all:
                    if not all(bay_results):
                        solver.Add(picks[i] == 0)
                else:
                    if not any(bay_results):
                        solver.Add(picks[i] == 0)

        # Prohibited weekly picks
        if team in request.prohibited_weekly_picks:
            if week_num in request.prohibited_weekly_picks[team]:
                solver.Add(picks[i] == 0)

    # ── One pick per week — skip required weeks (already handled above) ──
    required_weeks = {df.iloc[i]["Week_Num"] for i in required_positions}
    for week in df["Week_Num"].unique():
        if week in required_weeks:
            continue  # required picks already enforce one pick for this week
        weekly_picks = [
            picks[i] for i in range(len(df))
            if df.iloc[i]["Week_Num"] == week
        ]
        if weekly_picks:
            solver.Add(solver.Sum(weekly_picks) == 1)

    # ── Each team picked at most once ──
    for team in df["Team"].unique():
        team_picks = [picks[i] for i in range(len(df)) if df.iloc[i]["Team"] == team]
        if team_picks:
            solver.Add(solver.Sum(team_picks) <= 1)


# ─────────────────────────────────────────────────────────────
# Step 3 — Run the solver N times
# ─────────────────────────────────────────────────────────────

def run_solver(
    df: pd.DataFrame,
    request: OptimizeRequest,
    maximize_ev: bool,
) -> Tuple[List[List[PickResult]], bool, str]:
    """
    Runs the SCIP solver up to request.number_solutions times.
    Each iteration adds a constraint forbidding the previous solution.
    df must have a clean 0-based index.
    """
    # Guarantee clean index for this solver run
    df = df.reset_index(drop=True)

    solutions: List[List[PickResult]] = []
    forbidden_solutions: List[List[int]] = []
    objective_label = "EV" if maximize_ev else "Win %"

    for iteration in range(request.number_solutions):
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            return [], False, "SCIP solver unavailable"

        picks = {i: solver.IntVar(0, 1, f"pick_{i}") for i in range(len(df))}

        apply_constraints(solver, picks, df, request)

        # Forbid all previous solutions
        for prev_indices in forbidden_solutions:
            prev_vars = [picks[i] for i in prev_indices if i in picks]
            if prev_vars:
                solver.Add(solver.Sum([1 - v for v in prev_vars]) >= 1)

        # Set objective
        if maximize_ev:
            solver.Maximize(solver.Sum([
                picks[i] * float(df.iloc[i].get("EV", 0) or 0)
                for i in range(len(df))
            ]))
        else:
            solver.Maximize(solver.Sum([
                picks[i] * float(df.iloc[i].get("Win Pct", 0) or 0)
                for i in range(len(df))
            ]))

        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            if iteration == 0:
                return [], False, (
                    f"No feasible solution found for {objective_label} objective. "
                    f"Try relaxing some constraints."
                )
            else:
                break

        chosen_indices = [i for i in range(len(df)) if picks[i].solution_value() > 0.5]
        forbidden_solutions.append(chosen_indices)

        pick_results = []
        for i in sorted(chosen_indices, key=lambda x: df.iloc[x]["Week_Num"]):
            row = df.iloc[i]

            # Update holiday detection to handle both string and integer values
            thanksgiving_val = str(row.get("Thanksgiving Favorite", "")).strip()
            christmas_val = str(row.get("Christmas Favorite", "")).strip()
            underdog_thanksgiving = str(row.get("Thanksgiving Underdog", "")).strip()
            underdog_christmas = str(row.get("Christmas Underdog", "")).strip()
            
            is_thanksgiving = thanksgiving_val in ("1", "Thanksgiving", "True", "true") or \
                              underdog_thanksgiving in ("1", "Thanksgiving", "True", "true")
            is_christmas = christmas_val in ("1", "Christmas", "True", "true") or \
                           underdog_christmas in ("1", "Christmas", "True", "true")
            dome_val = row.get("Dome", None)
            is_dome = bool(dome_val) if dome_val is not None and pd.notna(dome_val) else None
            # Derive day of week from Date column
            day_label = None
            try:
                from datetime import datetime as dt
                date_val = row.get("Date")
                if date_val and pd.notna(date_val):
                    date_obj = pd.to_datetime(date_val)
                    day_name = date_obj.strftime("%A")  # Monday, Tuesday, etc.
                    game_time = str(row.get("Game_Time", "") or "")
                    is_tnf = str(row.get("Thursday Night Game", "False")).strip() == "True"
                    # Check for Monday Night Football (after 7pm ET on Monday)
                    is_mnf = day_name == "Monday" and game_time >= "19:00"
                    # Check for Sunday Night Football (after 7pm ET on Sunday)
                    is_snf = day_name == "Sunday" and game_time >= "19:00"
            
                    if is_tnf:
                        day_label = "Thu 🌙"
                    elif is_mnf:
                        day_label = "Mon 🌙"
                    elif is_snf:
                        day_label = "Sun 🌙"
                    elif day_name == "Sunday":
                        day_label = "Sun"
                    elif day_name == "Saturday":
                        day_label = "Sat"
                    elif day_name == "Friday":
                        day_label = "Fri"
                    else:
                        day_label = day_name[:3]
            except Exception:
                pass
            
            pick_results.append(PickResult(
                week=int(row["Week_Num"]),
                circa_week=str(row["Circa_Week"]) if pd.notna(row.get("Circa_Week")) else None,
                team=str(row["Team"]),
                ev=safe_float(row.get("EV"), 0.0),
                win_pct=safe_float(row.get("Win Pct"), 0.0),
                pick_pct=safe_float(row.get("Expected Pick Percent"), 0.0),
                home_or_away="Away" if bool(row.get("Team Is Away", False)) else "Home",
                opponent=str(row.get("Opponent", "")),
                spread=safe_float(row.get("Sportsbook Spread")),
                temperature=safe_float(row.get("Temperature")),
                precipitation=safe_float(row.get("Precipitation")),
                wind=safe_float(row.get("Wind")),
                dome=is_dome,
                starting_qb=str(row["Starting_QB"]) if pd.notna(row.get("Starting_QB")) else None,
                is_thanksgiving=is_thanksgiving,
                is_christmas=is_christmas,
                day_of_week=day_label,
                days_of_rest=int(row["Days_of_Rest"]) if pd.notna(row.get("Days_of_Rest")) else None,
                rest_advantage=safe_float(row.get("Rest_Advantage")),
                cumulative_rest=safe_float(row.get("Cumulative_Rest")),
                stadium=str(row["Actual Stadium"]) if pd.notna(row.get("Actual Stadium")) else None,
                is_international=bool(row.get("International Game", False)),
            ))

        solutions.append(pick_results)

    n = len(solutions)
    return solutions, True, f"Found {n} {objective_label} solution{'s' if n != 1 else ''}."


# ─────────────────────────────────────────────────────────────
# Step 4 — Main entry point called by main.py
# ─────────────────────────────────────────────────────────────

def run_optimizer(sim_df: pd.DataFrame, request: OptimizeRequest) -> OptimizeResponse:
    """
    Full optimizer pipeline:
      1. Prepare team-centric DataFrame
      2. Run EV solver N times
      3. Run win% solver N times
      4. Return combined OptimizeResponse
    """
    try:
        df = prepare_df(sim_df, request)
    except Exception as e:
        return OptimizeResponse(
            feasible=False,
            message=f"Data preparation failed: {str(e)}"
        )

    if df.empty:
        return OptimizeResponse(
            feasible=False,
            message="No games found for the selected week range and constraints."
        )

    ev_solutions, ev_feasible, ev_message = run_solver(df, request, maximize_ev=True)
    rank_solutions, rank_feasible, rank_message = run_solver(df, request, maximize_ev=False)

    feasible = ev_feasible or rank_feasible
    message_parts = []
    if ev_message:
        message_parts.append(f"EV: {ev_message}")
    if rank_message:
        message_parts.append(f"Win%: {rank_message}")

    total_ev = sum(p.ev for p in ev_solutions[0]) if ev_solutions else 0.0
    total_win_pct = sum(p.win_pct for p in ev_solutions[0]) if ev_solutions else 0.0

    return OptimizeResponse(
        ev_solutions=ev_solutions,
        ranking_solutions=rank_solutions,
        total_ev=round(total_ev, 4),
        total_win_pct=round(total_win_pct, 4),
        feasible=feasible,
        message=" | ".join(message_parts),
    )
