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

# ─────────────────────────────────────────────────────────────
# Column mapping — current CSV → internal solver names
# ─────────────────────────────────────────────────────────────
# These are the columns we actually need from the sim CSV.
# Keys = current CSV column names, Values = internal names used in this file.

AWAY_COL_MAP = {
    "Away Team":                                        "Team",
    "Home Team":                                        "Opponent",
    "Week":                                             "Week_Num",       # Circa contest week
    "Date_x":                                          "Date",
    "Time":                                            "Time",
    "Location":                                        "Location",        # not in CSV — use Actual Stadium
    "Actual Stadium":                                  "Location",
    "Thursday Night Game":                             "Thursday Night Game",
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
    "Away Team Fair Odds":                             "Fair Odds Consensus",
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
    # EV columns — model-specific, added dynamically
}

HOME_COL_MAP = {
    "Home Team":                                        "Team",
    "Away Team":                                        "Opponent",
    "Week":                                             "Week_Num",
    "Date_x":                                          "Date",
    "Time":                                            "Time",
    "Actual Stadium":                                  "Location",
    "Thursday Night Game":                             "Thursday Night Game",
    "Divisional Matchup?":                             "Divisional Matchup?",
    "Away Team Short Rest":                            "Away Team Short Rest",   # same col — short rest flag is always about away team
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
    "Home Team Fair Odds":                             "Fair Odds Consensus",
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
}

# Maps request.objective → EV column suffix in the CSV
OBJECTIVE_EV_COLS = {
    "consensus":  ("consensus_Away_EV",  "consensus_Home_EV"),
    "sportsbook": ("sportsbook_Away_EV", "sportsbook_Home_EV"),
    "mp":         ("mp_Away_EV",         "mp_Home_EV"),
    "gsf":        ("gsf_Away_EV",        "gsf_Home_EV"),
    "sim":        ("sim_Away_EV",        "sim_Home_EV"),
    "win_pct":    ("Consensus Away Win Pct", "Consensus Home Win Pct"),
}

# Maps request.objective → win probability column for PickResult output
OBJECTIVE_WIN_PCT_COLS = {
    "consensus":  ("Consensus Away Win Pct",    "Consensus Home Win Pct"),
    "sportsbook": ("Away Team Sportsbook Fair Odds", "Home Team Sportsbook Fair Odds"),
    "mp":         ("Away Team Massey-Peabody Fair Odds", "Home Team Massey-Peabody Fair Odds"),
    "gsf":        ("Away Team Generic Sports Fan Fair Odds", "Home Team Generic Sports Fan Fair Odds"),
    "sim":        ("Sim_Away_Win_Pct",          "Sim_Home_Win_Pct"),
    "win_pct":    ("Consensus Away Win Pct",    "Consensus Home Win Pct"),
}


# ─────────────────────────────────────────────────────────────
# Step 1 — Prepare the team-centric DataFrame
# ─────────────────────────────────────────────────────────────

def prepare_df(sim_df: pd.DataFrame, request: OptimizeRequest) -> pd.DataFrame:
    """
    Converts the game-centric sim CSV into a team-centric DataFrame
    (one row per team per game), then applies week range and 
    prohibited team filters.
    """
    df = sim_df.copy()

    # Resolve the EV column for this objective
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
    away_df["EV"] = df[away_ev_col] if away_ev_col in df.columns else 0.0
    away_df["Win Pct"] = df[away_win_col] if away_win_col in df.columns else 0.0
    away_df["Sportsbook Spread"] = df["Away Team Sportsbook Spread"] if "Away Team Sportsbook Spread" in df.columns else 0.0
    away_df["Game_Index"] = df.index

    # ── Home team rows ──
    home_cols = {k: v for k, v in HOME_COL_MAP.items() if k in df.columns}
    home_df = df[list(home_cols.keys())].rename(columns=home_cols).copy()
    home_df["Team Is Away"] = False
    home_df["EV"] = df[home_ev_col] if home_ev_col in df.columns else 0.0
    home_df["Win Pct"] = df[home_win_col] if home_win_col in df.columns else 0.0
    home_df["Sportsbook Spread"] = df["Home Team Sportsbook Spread"] if "Home Team Sportsbook Spread" in df.columns else 0.0
    home_df["Game_Index"] = df.index
    # Home teams are never on short rest (the flag is always about the away team)
    home_df["Away Team Short Rest"] = "No"
    home_df["Back to Back Away Games"] = False

    combined = pd.concat([away_df, home_df], ignore_index=True)

    # ── Filter week range ──
    combined = combined[
        (combined["Week_Num"] >= request.start_week) &
        (combined["Week_Num"] <= request.end_week)
    ].reset_index(drop=True)

    # ── Filter prohibited teams ──
    if request.prohibited_teams:
        combined = combined[
            ~combined["Team"].isin(request.prohibited_teams)
        ].reset_index(drop=True)

    # ── Apply custom EV overrides ──
    for week_key, team_overrides in request.custom_pick_percentages.items():
        try:
            week_num = int(week_key.replace("week_", ""))
        except ValueError:
            continue
        for team, pct in team_overrides.items():
            if pct >= 0:
                mask = (combined["Week_Num"] == week_num) & (combined["Team"] == team)
                combined.loc[mask, "Expected Pick Percent"] = pct

    # ── Apply custom rankings ──
    for team, ranking in request.custom_rankings.items():
        mask = combined["Team"] == team
        combined.loc[mask, "Adjusted Current Rank"] = ranking

    return combined


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
    Each constraint sets picks[i] == 0 for rows that violate it.
    """
    s = request.scheduling
    fq = request.favored_qualifier  # "sportsbook", "internal", or "both"

    # ── Bayesian constraints ──
    # Maps each toggle → the CSV column it checks
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
            "Sportsbook Same Current and Preseason Adjusted Winner"),
        (s.sim_bayesian_preseason_and_current,
            "Sim Same Current and Preseason Adjusted Winner"),
        (s.consensus_bayesian_preseason_and_current,
            "Consensus Same Current and Preseason Adjusted Winner")
    ]
            def is_favored_by(model: str) -> bool:
                if model == "sportsbook":
                    return team == str(row.get("Favorite", ""))
                elif model == "mp":
                    col = "Massey-Peabody Current Winner"
                    if col not in df.columns:
                        return True
                    return team == str(row.get(col, ""))
                elif model == "gsf":
                    col = "Generic Sports Fan Current Winner"
                    if col not in df.columns:
                        return True
                    return team == str(row.get(col, ""))
                elif model == "sim":
                    col = "Sim Favorite"   # col 292 — now exists directly
                    if col not in df.columns:
                        # fall back to win pct comparison
                        pct_col = "Sim_Away_Win_Pct" if is_away else "Sim_Home_Win_Pct"
                        return float(row.get(pct_col, 0.5) or 0.5) > 0.5
                    return team == str(row.get(col, ""))
                elif model == "consensus":
                    col = "Consensus Favorite"   # col 293 — now exists directly
                    if col not in df.columns:
                        pct_col = "Consensus Away Win Pct" if is_away else "Consensus Home Win Pct"
                        return float(row.get(pct_col, 0.5) or 0.5) > 0.5
                    return team == str(row.get(col, ""))
                return True
    for i in range(len(df)):
        row = df.iloc[i]
        is_away = bool(row.get("Team Is Away", False))
        spread_sb = float(row.get("Sportsbook Spread", 0) or 0)
        spread_int = float(row.get("Adjusted Current Difference", 0) or 0)
        adj_winner = str(row.get("Adjusted Current Winner", ""))
        favorite = str(row.get("Favorite", ""))
        team = str(row.get("Team", ""))
        week_num = int(row.get("Week_Num", 0))

        # ── Must be favored ──
        # ── Must be favored ──
        if request.must_be_favored:
            fq = request.favored_qualifier
        
            # Maps each qualifier → the CSV column that identifies the favorite
            # For spread-based models, the team is favored if their spread is negative
            # For fair odds models, the team is favored if their odds > 0.5
            FAVORED_COL_MAP = {
                "sportsbook": ("Favorite",              None),
                "mp":         ("Massey-Peabody Current Winner",         None),
                "gsf":        ("Generic Sports Fan Current Winner",     None),
                "sim":        ("Sim_Home_Win_Pct",       "Sim_Away_Win_Pct"),   # compare by value
                "consensus":  ("Consensus Home Win Pct", "Consensus Away Win Pct"),
            }
        
            if fq == "all":
                # Team must be favored by every single model
                all_models = ["sportsbook", "mp", "gsf", "sim", "consensus"]
                if not all(is_favored_by(m) for m in all_models):
                    solver.Add(picks[i] == 0)
            else:
                # Team must be favored by the selected model
                if not is_favored_by(fq):
                    solver.Add(picks[i] == 0)

        # ── Away teams in close matchups ──
        if s.avoid_away_close and is_away:
            if fq == "internal":
                if spread_int <= s.min_away_spread:
                    solver.Add(picks[i] == 0)
            else:
                # Sportsbook spread for away team is negative when favored
                # e.g. -3.5 means favored by 3.5. Avoid if not favored by enough
                if spread_sb > -s.min_away_spread:
                    solver.Add(picks[i] == 0)

        # ── Close divisional matchups ──
        if s.avoid_close_divisional:
            is_div = row.get("Divisional Matchup?", 0)
            if is_div == 1 or is_div == "Divisional" or is_div is True:
                if fq == "internal":
                    if spread_int <= s.min_div_spread:
                        solver.Add(picks[i] == 0)
                else:
                    if spread_sb > -s.min_div_spread:
                        solver.Add(picks[i] == 0)

        # ── Away divisional matchups ──
        if s.avoid_away_divisional and is_away:
            is_div = row.get("Divisional Matchup?", 0)
            if is_div == 1 or is_div == "Divisional" or is_div is True:
                solver.Add(picks[i] == 0)

        # ── Away team short rest ──
        if s.avoid_away_short_rest:
            if str(row.get("Away Team Short Rest", "No")).strip() == "Yes":
                solver.Add(picks[i] == 0)

        # ── 3 games in 10 days ──
        if s.avoid_3_in_10:
            if str(row.get("3 Games in 10 Days", "No")).strip() == "Yes":
                solver.Add(picks[i] == 0)

        # ── 4 games in 17 days ──
        if s.avoid_4_in_17:
            if str(row.get("4 Games in 17 Days", "No")).strip() == "Yes":
                solver.Add(picks[i] == 0)

        # ── International games ──
        if s.avoid_international:
            location = str(row.get("Location", "")).lower()
            if "london" in location or "munich" in location or "madrid" in location:
                solver.Add(picks[i] == 0)

        # ── Thursday Night Football — all teams ──
        if s.avoid_thursday_all:
            if str(row.get("Thursday Night Game", "False")).strip() == "True":
                solver.Add(picks[i] == 0)

        # ── Thursday Night Football — away teams only ──
        if s.avoid_thursday_away and is_away:
            if str(row.get("Thursday Night Game", "False")).strip() == "True":
                solver.Add(picks[i] == 0)

        # ── Back to back away games ──
        if s.avoid_back_to_back_away and is_away:
            if str(row.get("Back to Back Away Games", "False")).strip() == "True":
                solver.Add(picks[i] == 0)

        # ── Weekly rest disadvantage ──
        if s.avoid_weekly_rest_disadvantage:
            rest_adv = float(row.get("Weekly Rest Advantage", 0) or 0)
            if rest_adv < 0:
                solver.Add(picks[i] == 0)

        # ── Cumulative rest disadvantage ──
        if s.avoid_cumulative_rest:
            rest_adv = float(row.get("Season-Long Rest Advantage Including This Week", 0) or 0)
            if is_away and rest_adv < -10:
                solver.Add(picks[i] == 0)
            elif not is_away and rest_adv < -5:
                solver.Add(picks[i] == 0)

        # ── Travel disadvantage ──
        if s.avoid_travel_disadvantage and is_away:
            travel = float(row.get("Travel Advantage", 0) or 0)
            if travel < -850:
                solver.Add(picks[i] == 0)
        
        # Only evaluate constraints the user has actually toggled on
        active_checks = [(enabled, col) for enabled, col in BAYESIAN_CHECKS if enabled]
        
        if active_checks:
            results = []
            for enabled, col in active_checks:
                # Gracefully handle missing or empty columns
                if col not in df.columns:
                    # Column doesn't exist — treat as not satisfied
                    results.append(False)
                    continue
                val = row.get(col, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    # Empty value — treat as not satisfied
                    results.append(False)
                    continue
                # Accept True, 1, "True", "Yes", "1" as passing
                results.append(str(val).strip() in ("True", "Yes", "1", "true", "yes"))
        
            if s.bayesian_require_all:
                # ALL active constraints must pass
                if not all(results):
                    solver.Add(picks[i] == 0)
            else:
                # AT LEAST ONE active constraint must pass
                if not any(results):
                    solver.Add(picks[i] == 0)

        # ── Prohibited weekly picks ──
        if team in request.prohibited_weekly_picks:
            if week_num in request.prohibited_weekly_picks[team]:
                solver.Add(picks[i] == 0)

    # ── Required picks (force specific team-week combinations) ──
    for team, req_week in request.required_picks.items():
        if req_week > 0:
            required_indices = df[
                (df["Team"] == team) & (df["Week_Num"] == req_week)
            ].index.tolist()
            if required_indices:
                solver.Add(picks[required_indices[0]] == 1)

    # ── One pick per week ──
    for week in df["Week_Num"].unique():
        weekly_picks = [picks[i] for i in range(len(df)) if df.iloc[i]["Week_Num"] == week]
        if weekly_picks:
            solver.Add(solver.Sum(weekly_picks) == 1)

    # ── Each team can only be picked once across the whole season ──
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

    maximize_ev=True  → maximize EV (ev_solutions)
    maximize_ev=False → maximize win probability (ranking_solutions)

    Returns (solutions, feasible, message)
    """
    solutions: List[List[PickResult]] = []
    forbidden_solutions: List[List[int]] = []  # list of game indices per solution
    objective_label = "EV" if maximize_ev else "Win %"

    for iteration in range(request.number_solutions):
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            return [], False, "SCIP solver unavailable"

        # Create binary decision variables
        picks = {i: solver.IntVar(0, 1, f"pick_{i}") for i in range(len(df))}

        # Apply all constraints
        apply_constraints(solver, picks, df, request)

        # Forbid all previous solutions
        for prev_indices in forbidden_solutions:
            prev_vars = [picks[i] for i in prev_indices if i in picks]
            if prev_vars:
                # At least one pick from the previous solution must NOT be chosen
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
                # No solution at all — constraints are too restrictive
                return [], False, (
                    f"No feasible solution found for {objective_label} objective. "
                    f"Try relaxing some constraints."
                )
            else:
                # Ran out of distinct solutions
                break

        # Extract chosen picks
        chosen_indices = [i for i in range(len(df)) if picks[i].solution_value() > 0.5]
        forbidden_solutions.append(chosen_indices)

        # Build PickResult list for this solution
        pick_results = []
        for i in sorted(chosen_indices, key=lambda x: df.iloc[x]["Week_Num"]):
            row = df.iloc[i]
            pick_results.append(PickResult(
                week=int(row["Week_Num"]),
                team=str(row["Team"]),
                ev=round(float(row.get("EV", 0) or 0), 4),
                win_pct=round(float(row.get("Win Pct", 0) or 0), 4),
                pick_pct=round(float(row.get("Expected Pick Percent", 0) or 0), 4),
                home_or_away="Away" if bool(row.get("Team Is Away", False)) else "Home",
                opponent=str(row.get("Opponent", "")),
                spread=round(float(row.get("Sportsbook Spread", 0) or 0), 1),
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

    # ── EV solutions ──
    ev_solutions, ev_feasible, ev_message = run_solver(df, request, maximize_ev=True)

    # ── Win % solutions ──
    rank_solutions, rank_feasible, rank_message = run_solver(df, request, maximize_ev=False)

    feasible = ev_feasible or rank_feasible
    message_parts = []
    if ev_message:
        message_parts.append(f"EV: {ev_message}")
    if rank_message:
        message_parts.append(f"Win%: {rank_message}")

    # Aggregate totals from the top EV solution
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
