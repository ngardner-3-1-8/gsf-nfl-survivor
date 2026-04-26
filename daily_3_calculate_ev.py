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
from selenium.webdriver.chrome.options import Options # Make sure this is present!
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
    
    INPUT_FILE = pd.read_csv(f"nfl-power-ratings/final_sim_results_with_variance_week_{upcoming_week}_{target_year}.csv")
    
    df = INPUT_FILE
    
    
    
    def calculate_ev(df, config: dict, use_cache=False):
        start_w = upcoming_week
        end_w = 20
    
        # 1. Enforce team abbreviation standardization
        replace_dict = {'JAC': 'JAX', 'LAR': 'LA'}
        df['Away Team'] = df['Away Team'].replace(replace_dict)
        df['Home Team'] = df['Home Team'].replace(replace_dict)
        
        # 2. Filter for upcoming weeks
        # Assuming upcoming_week and target_year are defined earlier in your script
        df_future = df[df['Week_x'] >= upcoming_week].copy()
        end_w = int(df_future['Week_x'].max()) + 1
        
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
    
        # Update 1: Add the dynamic columns to the arguments
        def calculate_all_scenarios(week_df, away_prob_col, home_prob_col):
            """
            Calculates EV for each team using the direct formula instead of
            enumerating all 2^N outcome combinations.
            
            EV(team) = P(team wins) / E[total survivors this week]
            
            E[survivors] = sum of (each team's pick% * their win probability)
            This is the expected fraction of the pool that advances.
            """
            home_probs = week_df[home_prob_col].values
            away_probs = week_df[away_prob_col].values
            home_picks = week_df['Home Pick %'].values
            away_picks = week_df['Away Pick %'].values
        
            # Expected total survivor share (sum of pick% * win_prob across all teams)
            expected_survivors = np.sum(home_probs * home_picks) + np.sum(away_probs * away_picks)
        
            ev_results = {}
            for i, row in week_df.iterrows():
                if expected_survivors > 0:
                    ev_results[row['Home Team']] = row[home_prob_col] / expected_survivors
                    ev_results[row['Away Team']] = row[away_prob_col] / expected_survivors
                else:
                    ev_results[row['Home Team']] = 0
                    ev_results[row['Away Team']] = 0
        
            # Return signature matches original to avoid changing the caller
            return ev_results, None, None
    
        all_weeks_ev = {} #Store the EV values for each week
    
        for scenario_name, scenario_config in probability_scenarios.items():
            away_prob_col = scenario_config["away_col"]
            home_prob_col = scenario_config["home_col"]
            prefix = scenario_config["prefix"]
        
            # Create a fresh copy of the future schedule for this specific scenario
            scenario_df = df_future.copy()
            
            # Optional: add scenario name to tqdm for better console tracking
        ev_results, _, _ = calculate_all_scenarios(
            week_df,
            away_prob_col=away_prob_col,
            home_prob_col=home_prob_col
        )
        
        for team in week_df['Home Team'].unique():
            scenario_df.loc[(scenario_df['Week_x'] == week) & (scenario_df['Home Team'] == team), 'Home Team EV'] = ev_results.get(team, 0)
        for team in week_df['Away Team'].unique():
            scenario_df.loc[(scenario_df['Week_x'] == week) & (scenario_df['Away Team'] == team), 'Away Team EV'] = ev_results.get(team, 0)
        
                # Assign EV values back to the scenario dataframe
                for team in week_df['Home Team'].unique():
                    scenario_df.loc[(scenario_df['Week_x'] == week) & (scenario_df['Home Team'] == team), 'Home Team EV'] = weighted_avg_ev.get(team, 0)
                for team in week_df['Away Team'].unique():
                    scenario_df.loc[(scenario_df['Week_x'] == week) & (scenario_df['Away Team'] == team), 'Away Team EV'] = weighted_avg_ev.get(team, 0)
        
            # Export to CSV with the requested naming convention
            output_filename = f"circa-survivor-ev/{prefix}_team_ev_week_{upcoming_week}_{target_year}.csv"
            
            # You can choose to export the whole scenario_df, or just the relevant subset.
            scenario_df.to_csv(output_filename, index=False)
            print(f"Successfully exported: {output_filename}")
    calculate_ev(df, config={})
if __name__ == "__main__":
    formatted_date = datetime.now().strftime("%m/%d/%Y")
    week_starting_dates = [
        "09/03/2025" 
#        "09/10/2025", 
#        "09/17/2025", 
#        "09/24/2025", 
#        "10/01/2025", 
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
#        "12/24/2025" 
#        "12/26/2025", 
#        "12/31/2025"
#        formatted_date
    ]

    for date in week_starting_dates:
        loop_through_ev(date)
