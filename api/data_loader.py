# api/data_loader.py

import os
import glob
import calendar
import pandas as pd
from datetime import datetime, timedelta


def get_target_year(today: datetime) -> int:
    """Jan–May = finishing previous season. Jun+ = current season."""
    return today.year - 1 if today.month < 6 else today.year


def get_upcoming_week(today: datetime, target_year: int, data_dir: str) -> int:
    """
    Derives the upcoming Circa contest week from today's date,
    mirroring the logic in your NFL and EV scripts exactly.
    """
    schedule_path = os.path.join(data_dir, f"nfl-schedules/schedule_{target_year}.csv")

    if not os.path.exists(schedule_path):
        raise FileNotFoundError(f"Schedule file not found: {schedule_path}")

    schedule_df = pd.read_csv(schedule_path)
    schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])
    first_game_date = schedule_df['Date'].min()

    # Calculate holiday dates
    c = calendar.monthcalendar(target_year, 11)
    thursdays = [row[calendar.THURSDAY] for row in c if row[calendar.THURSDAY] != 0]
    thanksgiving = datetime(target_year, 11, thursdays[3])
    black_friday = thanksgiving + timedelta(days=1)
    boxing_day = datetime(target_year, 12, 26)

    if today <= first_game_date:
        return 1

    week_end_dates = schedule_df.groupby('Week')['Date'].max()
    completed_weeks = week_end_dates[week_end_dates <= today]

    if completed_weeks.empty:
        return 1

    standard_nfl_week = int(completed_weeks.index.max())
    upcoming_week = standard_nfl_week + 1

    # Circa holiday adjustments — matches your script exactly
    if today >= black_friday:
        upcoming_week += 1
    if today >= boxing_day:
        upcoming_week += 1

    return min(upcoming_week, 20)


def get_simulation_file(data_dir: str, upcoming_week: int, target_year: int) -> str:
    """
    Returns the path to the correct simulation results CSV for the
    upcoming week and year. Falls back to the closest available week
    if the exact file doesn't exist yet (e.g. script hasn't run today).
    """
    ratings_dir = os.path.join(data_dir, "nfl-power-ratings")

    # Try exact match first
    exact = os.path.join(
        ratings_dir,
        f"final_sim_results_with_variance_week_{upcoming_week}_{target_year}.csv"
    )
    if os.path.exists(exact):
        return exact

    # Fall back to the highest available week for this year
    pattern = os.path.join(
        ratings_dir,
        f"final_sim_results_with_variance_week_*_{target_year}.csv"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No simulation results found for {target_year} in {ratings_dir}"
        )

    # Extract week numbers and pick the closest to upcoming_week
    def extract_week(path):
        base = os.path.basename(path)
        try:
            return int(base.split("_week_")[1].split(f"_{target_year}")[0])
        except (IndexError, ValueError):
            return 0

    matches_with_weeks = [(extract_week(p), p) for p in matches]
    # Prefer the most recent week that doesn't exceed upcoming_week
    valid = [(w, p) for w, p in matches_with_weeks if w <= upcoming_week]
    if valid:
        return max(valid, key=lambda x: x[0])[1]

    # If nothing valid, just return the highest available
    return max(matches_with_weeks, key=lambda x: x[0])[1]


def load_current_data(data_dir: str, model: str = "consensus") -> dict:
    today = datetime.now()
    target_year = get_target_year(today)
    upcoming_week = get_upcoming_week(today, target_year, data_dir)
    sim_file = get_simulation_file(data_dir, upcoming_week, target_year)

    sim_df = pd.read_csv(sim_file)

    # EV columns are embedded in the sim file (written by daily_3_calculate_ev.py)
    # No separate EV file needed anymore
    # Filter to the model requested for the ev_df view
    ev_home_col = f"{model}_Home_EV"
    ev_away_col = f"{model}_Away_EV"

    if ev_home_col in sim_df.columns:
        ev_df = sim_df.copy()
    else:
        ev_df = pd.DataFrame()

    return {
        "upcoming_week": upcoming_week,
        "target_year": target_year,
        "sim_file": sim_file,
        "ev_file": sim_file,  # same file
        "sim_df": sim_df,
        "ev_df": ev_df,
    }
