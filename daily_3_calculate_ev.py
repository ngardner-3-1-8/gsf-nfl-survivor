import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import pytz
from dateutil.parser import parse
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from ortools.linear_solver import pywraplp
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import itertools
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import os
import json
import sqlite3
import polars as pl 
import nflreadpy as nfl
import random
import csv
from typing import Optional
from typing import Dict, List, Any
from sklearn.feature_selection import RFE
from scipy.stats import percentileofscore
import warnings
import calendar
    
def loop_through_ev(date_str):   
    # 1. Get current date
    today = pd.to_datetime(date_str)  
    # 1. Get current date
    current_cal_year = today.year 
    
    # 2. Initial Year Logic based on Month (User Rule)
    # If Jan-May (< 6), assume we are finishing the previous season.
    target_year = current_cal_year - 1 if today.month < 6 else current_cal_year
    
    schedule_df = pd.read_csv(f"nfl-schedules/schedule_{target_year}.csv")
    
    schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])
    
    first_game_date = schedule_df['Date'].min()
    
    # 3. Calculate Important Dates Automatically
    def get_thanksgiving(year):
        # 4th Thursday in November
        c = calendar.monthcalendar(year, 11)
        thursdays = [row[calendar.THURSDAY] for row in c if row[calendar.THURSDAY] != 0]
        return datetime(year, 11, thursdays[3])
    
    
    
    thanksgiving_date = get_thanksgiving(target_year)
    black_friday = thanksgiving_date + timedelta(days=1)
    christmas_day = datetime(target_year, 12, 25)
    boxing_day = datetime(target_year, 12, 26)
    
    thanksgiving_week = int((thanksgiving_date - first_game_date).days/7) + 1 ## +1 because the first game date is technically week 1, not week 0
    christmas_week = int((christmas_day - first_game_date).days/7) + 2 ## +2 because the first game date is technically week 1, not week 0, and the addition of thanksgiving_week
    
    if today <= first_game_date:
        starting_week = 1
        upcoming_week = starting_week
    else:
        # 1. Find the final game date for every week in the season
        # This creates a Series where index = Week, value = Latest Game Date for that week
        week_end_dates = schedule_df.groupby('Week')['Date'].max()
        # 2. Filter for weeks where the LAST game of that week has already occurred
        completed_weeks = week_end_dates[week_end_dates <= today]
        if not completed_weeks.empty:
            # The "standard_nfl_week" is now the last FULLY completed week
            standard_nfl_week = int(completed_weeks.index.max())            
            # 3. Your starting point for simulations is the next week (the one in progress or upcoming)
            starting_week = standard_nfl_week + 1
            upcoming_week = starting_week
            # --- ADJUST FOR CIRCA SPECIAL WEEKS ---
            # Using your existing logic for Thanksgiving/Christmas shifts
            if today >= black_friday:
                starting_week += 0
                upcoming_week += 1
            if today >= boxing_day:
                starting_week += 0
                upcoming_week += 1
            # Bound check: Cap at 19 (or your season max)
            if starting_week > 18: 
                starting_week = 18
        else:
            # If no week is fully completed yet, we are still in Week 1
            starting_week = 1
    
    
    # 5. Final Assignment to your variables
    current_year = target_year
    starting_year = target_year
    
    current_year_plus_1 = current_year + 1
    season_start_date = first_game_date - timedelta(days=1)
    
    thanksgiving_reset_date = black_friday + timedelta(days=1) #THIS DATE IS INCLUDED IN THE RESET. SO IF THERE ARE GAMES ON THIS DATE, THEY WILL HAVE A WEEK ADDED
    christmas_reset_date = boxing_day
    
    NUM_WEEKS_TO_KEEP = starting_week - 1
    current_year_plus_1 = current_year + 1 #current_year + 1
    
    main_file_path = f"nfl-power-ratings/final_sim_results_with_variance_week_{upcoming_week}_{target_year}.csv"
    INPUT_FILE = pd.read_csv(main_file_path)
    
    df = INPUT_FILE
    
    
    
    def calculate_ev(df, config: dict, use_cache=False):
        start_w = upcoming_week
    
        # 1. Enforce team abbreviation standardization directly on main df
        replace_dict = {'JAC': 'JAX', 'LAR': 'LA'}
        df['Away Team'] = df['Away Team'].replace(replace_dict)
        df['Home Team'] = df['Home Team'].replace(replace_dict)
    
        # Find ending week bounds based on the full file
        end_w = int(df['Week_x'].max()) + 1
    
        probability_scenarios = {
            "sportsbook": {
                "away_col": "Away Team Sportsbook Fair Odds",
                "home_col": "Home Team Sportsbook Fair Odds",
                "prefix": "sportsbook"
            },
            "mp": {
                "away_col": "Away Team Massey-Peabody Fair Odds",
                "home_col": "Home Team Massey-Peabody Fair Odds",
                "prefix": "mp"
            },
            "gsf": {
                "away_col": "Away Team Generic Sports Fan Fair Odds",
                "home_col": "Home Team Generic Sports Fan Fair Odds",
                "prefix": "gsf"
            },
            "sim": {
                "away_col": "Sim_Away_Win_Pct",
                "home_col": "Sim_Home_Win_Pct",
                "prefix": "sim"
            },
            "consensus": {
                "away_col": "Consensus Away Win Pct",
                "home_col": "Consensus Home Win Pct",
                "prefix": "consensus"
            }
        }
    
        def calculate_all_scenarios(week_df, away_prob_col, home_prob_col):
            """
            EV(team) = P(team wins) / E[total survivors this week]
            E[survivors] = sum of (each team's pick% * their win probability)
            
            Teams with 0% availability (pick% == 0) are excluded from the
            expected survivors calculation — they're not pickable so they
            don't affect EV for the remaining eligible teams.
            """
            home_probs = week_df[home_prob_col].values
            away_probs = week_df[away_prob_col].values
            home_picks = week_df['Home Pick %'].values
            away_picks = week_df['Away Pick %'].values
        
            # Only include teams with non-zero availability in the denominator
            home_eligible = home_picks > 0
            away_eligible = away_picks > 0
        
            expected_survivors = (
                np.sum(home_probs[home_eligible] * home_picks[home_eligible]) +
                np.sum(away_probs[away_eligible] * away_picks[away_eligible])
            )
        
            ev_results = {}
            for i, row in week_df.iterrows():
                if expected_survivors > 0:
                    # Home team — only assign EV if eligible
                    if row['Home Pick %'] > 0:
                        ev_results[row['Home Team']] = row[home_prob_col] / expected_survivors
                    else:
                        ev_results[row['Home Team']] = 0
        
                    # Away team — only assign EV if eligible
                    if row['Away Pick %'] > 0:
                        ev_results[row['Away Team']] = row[away_prob_col] / expected_survivors
                    else:
                        ev_results[row['Away Team']] = 0
                else:
                    ev_results[row['Home Team']] = 0
                    ev_results[row['Away Team']] = 0
        
            return ev_results
    
        for scenario_name, scenario_config in probability_scenarios.items():
            away_prob_col = scenario_config["away_col"]
            home_prob_col = scenario_config["home_col"]
            prefix = scenario_config["prefix"]
            
            # Create dynamically named columns for the scenario on the main DataFrame
            home_ev_col = f"{prefix}_Home_EV"
            away_ev_col = f"{prefix}_Away_EV"
            
            # Initialize with default value for past weeks or empty rows
            df[home_ev_col] = 0.0
            df[away_ev_col] = 0.0
    
            for week in tqdm(range(start_w, end_w), desc=f"Processing {prefix.upper()} EV", leave=False):
                # Filter rows for the current week being processed
                week_df = df[df['Week_x'] == week].copy()
    
                if week_df.empty:
                    continue
    
                ev_results = calculate_all_scenarios(
                    week_df,
                    away_prob_col=away_prob_col,
                    home_prob_col=home_prob_col
                )
    
                # Write results back to the MAIN df
                for team in week_df['Home Team'].unique():
                    df.loc[
                        (df['Week_x'] == week) & (df['Home Team'] == team),
                        home_ev_col
                    ] = ev_results.get(team, 0)
    
                for team in week_df['Away Team'].unique():
                    df.loc[
                        (df['Week_x'] == week) & (df['Away Team'] == team),
                        away_ev_col
                    ] = ev_results.get(team, 0)
    
        # Save the updated main dataframe overwriting the original input file
        df.to_csv(main_file_path, index=False)
        print(f"\nSuccessfully appended all EV columns and saved to: {main_file_path}")
    
    calculate_ev(df, config={})

if __name__ == "__main__":
    formatted_date = datetime.now().strftime("%m/%d/%Y")
    week_starting_dates = [
#        "09/03/2025", 
#        "09/10/2025"#, 
#        "09/17/2025",
#        "09/24/2025", 
        "10/01/2025",
#        "10/08/2025", 
#        "10/15/2025", 
#        "10/22/2025", 
#        "10/29/2025", 
#        "11/05/2025", 
#        "11/12/2025", 
#        "11/19/2025", 
#        "11/26/2025", 
#        "11/29/2025", 
#        "12/03/2025", 
#        "12/10/2025", 
#        "12/17/2025", 
#        "12/24/2025", 
#        "12/26/2025", 
#        "12/31/2025",
        
#        "09/04/2024", 
#        "09/11/2024", 
#        "09/18/2024", 
#        "09/25/2024", 
#        "10/02/2024", 
#        "10/09/2024", 
#        "10/16/2024", 
#        "10/23/2024", 
#        "10/30/2024", 
#        "11/06/2024", 
#        "11/13/2024", 
#        "11/20/2024", 
#        "11/27/2024", 
#        "11/30/2024", 
#        "12/04/2024", 
#        "12/11/2024", 
#        "12/18/2024", 
#        "12/24/2024", 
#        "12/27/2024", 
#        "01/01/2025",
        
#        "09/06/2023", 
#        "09/13/2023", 
#        "09/20/2023", 
#        "09/27/2023", 
#        "10/04/2023", 
#        "10/11/2023", 
#        "10/18/2023", 
#        "10/25/2023", 
#        "11/01/2023", 
#        "11/08/2023", 
#        "11/15/2023", 
#        "11/22/2023", 
#        "11/29/2023", 
#        "12/02/2023", 
#        "12/06/2023", 
#        "12/13/2023", 
#        "12/20/2023", 
#        "12/24/2023", 
#        "12/27/2023", 
#        "01/03/2024",
        
#        "09/07/2022", 
#        "09/14/2022", 
#        "09/21/2022", 
#        "09/28/2022", 
#        "10/05/2022", 
#        "10/12/2022", 
#        "10/19/2022", 
#        "10/26/2022", 
#        "11/02/2022", 
#        "11/09/2022", 
#        "11/16/2022", 
#        "11/23/2022", 
#        "11/26/2022", 
#        "11/30/2022", 
#        "12/07/2022", 
#        "12/14/2022", 
#        "12/21/2022", 
#        "12/25/2022", 
#        "12/28/2023",
#        "01/04/2023",
        
#        "09/08/2021", 
#        "09/15/2021", 
#        "09/22/2021", 
#        "09/29/2021", 
#        "10/06/2021", 
#        "10/13/2021", 
#        "10/20/2021", 
#        "10/27/2021", 
#        "11/03/2021", 
#        "11/10/2021", 
#        "11/17/2021", 
#        "11/24/2021", 
#        "11/27/2021", 
#        "12/01/2021", 
#        "12/08/2021", 
#        "12/15/2021", 
#        "12/22/2021", 
#        "12/26/2021", 
#        "12/29/2022", 
#        "01/05/2022",
        
#        "09/09/2020", 
#        "09/16/2020", 
#        "09/23/2020", 
#        "09/30/2020", 
#        "10/07/2020", 
#        "10/14/2020", 
#        "10/21/2020", 
#        "10/28/2020", 
#        "11/04/2020", 
#        "11/11/2020", 
#        "11/18/2020", 
#        "11/25/2020",
#        "11/28/2020", 
#        "12/02/2020", 
#        "12/09/2020", 
#        "12/16/2020", 
#        "12/23/2020", 
#        "12/30/2020"
        
#        formatted_date
    ]

    for date in week_starting_dates:
        loop_through_ev(date)
