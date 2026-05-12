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
