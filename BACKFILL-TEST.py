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
    
def loop_through_simulations(date_str):   
    # 1. Get current date
    today = pd.to_datetime(date_str)
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
    
    circa_2020_entries = 1373
    circa_2021_entries = 4071
    circa_2022_entries = 6106
    circa_2023_entries = 9234
    circa_2024_entries = 14221
    circa_2025_entries = 18718
    circa_2026_entries = 24000
    
    circa_total_entries = 18718
    splash_big_splash_total_entries = 16337
    splash_4_for_4_total_entries = 10000
    splash_for_the_fans_total_entries = 8382
    splash_ship_it_nation_total_entries = 10114
    splash_high_roller_total_entries = 1004
    splash_rotowire_total_entries = 9048
    splash_walkers_25_total_entries = 36501
    splash_bloody_total_entries = 5000
    dk_total_entries = 20000
    
    MP_PRESEASON_RANKS = {
        'Arizona Cardinals': 0.075,
        'Atlanta Falcons': -0.71,
        'Baltimore Ravens': 6.69,
        'Buffalo Bills': 4.795,
        'Carolina Panthers': -5.25,
        'Chicago Bears': -1.575,
        'Cincinnati Bengals': 1.31,
        'Cleveland Browns': -4.705,
        'Dallas Cowboys': -0.615,
        'Denver Broncos': 2.05,
        'Detroit Lions': 4.305,
        'Green Bay Packers': 3.535,
        'Houston Texans': 0.035,
        'Indianapolis Colts': -2.265,
        'Jacksonville Jaguars': -1.825,
        'Kansas City Chiefs': 4.395,
        'Las Vegas Raiders': -2.35,
        'Los Angeles Chargers': 0.935,
        'Los Angeles Rams': 1.29,
        'Miami Dolphins': 0.66,
        'Minnesota Vikings': 0.27,
        'New England Patriots': -1.995,
        'New Orleans Saints': -6.145,
        'New York Giants': -2.84,
        'New York Jets': -3.725,
        'Philadelphia Eagles': 4.905,
        'Pittsburgh Steelers': -0.565,
        'San Francisco 49ers': 3.325,
        'Seattle Seahawks': -0.13,
        'Tampa Bay Buccaneers': 1.025,
        'Tennessee Titans': -4.36,
        'Washington Commanders': 1.45
    }
    
    GSF_PRESEASON_RANKS = {
        'Arizona Cardinals': 0.075,
        'Atlanta Falcons': -0.71,
        'Baltimore Ravens': 6.69,
        'Buffalo Bills': 4.795,
        'Carolina Panthers': -5.25,
        'Chicago Bears': -1.575,
        'Cincinnati Bengals': 1.31,
        'Cleveland Browns': -4.705,
        'Dallas Cowboys': -0.615,
        'Denver Broncos': 2.05,
        'Detroit Lions': 4.305,
        'Green Bay Packers': 3.535,
        'Houston Texans': 0.035,
        'Indianapolis Colts': -2.265,
        'Jacksonville Jaguars': -1.825,
        'Kansas City Chiefs': 4.395,
        'Las Vegas Raiders': -2.35,
        'Los Angeles Chargers': 0.935,
        'Los Angeles Rams': 1.29,
        'Miami Dolphins': 0.66,
        'Minnesota Vikings': 0.27,
        'New England Patriots': -1.995,
        'New Orleans Saints': -6.145,
        'New York Giants': -2.84,
        'New York Jets': -3.725,
        'Philadelphia Eagles': 4.905,
        'Pittsburgh Steelers': -0.565,
        'San Francisco 49ers': 3.325,
        'Seattle Seahawks': -0.13,
        'Tampa Bay Buccaneers': 1.025,
        'Tennessee Titans': -4.36,
        'Washington Commanders': 1.45
    }
    
    mp_current_ranks = {
        'Arizona Cardinals' : -5.6,
        'Atlanta Falcons' : -1.61,
        'Baltimore Ravens' : 5,
        'Buffalo Bills' : 5.24,
        'Carolina Panthers' : -5.07,
        'Chicago Bears' : -1.68,
        'Cincinnati Bengals' : -6.02,
        'Cleveland Browns' : -8.97,
        'Dallas Cowboys' : 0.33,
        'Denver Broncos' : 3.31,
        'Detroit Lions' : 4.55,
        'Green Bay Packers' : 4.69,
        'Houston Texans' : -1.23,
        'Indianapolis Colts' : 3.95,
        'Jacksonville Jaguars' : 1.02,
        'Kansas City Chiefs' : 6.47,
        'Las Vegas Raiders' : -5.91,
        'Los Angeles Chargers' : 0.68,
        'Los Angeles Rams' : 7.26,
        'Miami Dolphins' : -1.34,
        'Minnesota Vikings' : -0.87,
        'New England Patriots' : 0.28,
        'New Orleans Saints' : -7.09,
        'New York Giants' : -5.86,
        'New York Jets' : -3.07,
        'Philadelphia Eagles' : 5.73,
        'Pittsburgh Steelers' : 1.1,
        'San Francisco 49ers' : 4.49,
        'Seattle Seahawks' : 8.34,
        'Tampa Bay Buccaneers' : 1.3,
        'Tennessee Titans' : -7.24,
        'Washington Commanders' : -2.04
    }
    
    # 1. Define the file path based on your existing variables
    ratings_file = f"nfl-power-ratings/nfl_power_ratings_blended_week_{starting_week}_{target_year}.csv"
    
    # 2. Check if the file exists before trying to read it
    if os.path.exists(ratings_file):
        print(f"Loading ratings from {ratings_file}")
        ratings_df = pd.read_csv(ratings_file)
        
        # 3. Create a helper function to get the rating safely
        def get_mp_team_rating(team_abbr):
            # Look for the team in the 'team' or 'off_team' column (check your CSV header)
            # We use .iloc[0] to get the value from the matching row
            try:
                # Change 'team' to 'off_team' if that is the name of your team column
                rating = ratings_df.loc[ratings_df['Team'] == team_abbr, 'MP_Rating'].values[0]
                return float(rating)
            except (IndexError, KeyError):
                print(f"Warning: Could not find rating for {team_abbr}. Defaulting to 0.")
                return 0.0
    	    # 3. Create a helper function to get the rating safely
        def get_gsf_team_rating(team_abbr):
            # Look for the team in the 'team' or 'off_team' column (check your CSV header)
            # We use .iloc[0] to get the value from the matching row
            try:
                # Change 'team' to 'off_team' if that is the name of your team column
                rating = ratings_df.loc[ratings_df['Team'] == team_abbr, 'Power Rating'].values[0]
                return float(rating)
            except (IndexError, KeyError):
                print(f"Warning: Could not find rating for {team_abbr}. Defaulting to 0.")
                return 0.0
    
        # 4. Build your dictionary dynamically
        mp_current_ranks = {
            'Arizona Cardinals' : get_mp_team_rating("ARI"),
            'Atlanta Falcons' : get_mp_team_rating("ATL"),
            'Baltimore Ravens' : get_mp_team_rating("BAL"),
            'Buffalo Bills' : get_mp_team_rating("BUF"),
            'Carolina Panthers' : get_mp_team_rating("CAR"),
            'Chicago Bears' : get_mp_team_rating("CHI"),
            'Cincinnati Bengals' : get_mp_team_rating("CIN"),
            'Cleveland Browns' : get_mp_team_rating("CLE"),
            'Dallas Cowboys' : get_mp_team_rating("DAL"),
            'Denver Broncos' : get_mp_team_rating("DEN"),
            'Detroit Lions' : get_mp_team_rating("DET"),
            'Green Bay Packers' : get_mp_team_rating("GB"),
            'Houston Texans' : get_mp_team_rating("HOU"),
            'Indianapolis Colts' : get_mp_team_rating("IND"),
            'Jacksonville Jaguars' : get_mp_team_rating("JAX"),
            'Kansas City Chiefs' : get_mp_team_rating("KC"),
            'Las Vegas Raiders' : get_mp_team_rating("LV"),
            'Los Angeles Chargers' : get_mp_team_rating("LAC"),
            'Los Angeles Rams' : get_mp_team_rating("LA"),
            'Miami Dolphins' : get_mp_team_rating("MIA"),
            'Minnesota Vikings' : get_mp_team_rating("MIN"),
            'New England Patriots' : get_mp_team_rating("NE"),
            'New Orleans Saints' : get_mp_team_rating("NO"),
            'New York Giants' : get_mp_team_rating("NYG"),
            'New York Jets' : get_mp_team_rating("NYJ"),
            'Philadelphia Eagles' : get_mp_team_rating("PHI"),
            'Pittsburgh Steelers' : get_mp_team_rating("PIT"),
            'San Francisco 49ers' : get_mp_team_rating("SF"),
            'Seattle Seahawks' : get_mp_team_rating("SEA"),
            'Tampa Bay Buccaneers' : get_mp_team_rating("TB"),
            'Tennessee Titans' : get_mp_team_rating("TEN"),
            'Washington Commanders' : get_mp_team_rating("WAS")
        }
    	
        # 4. Build your dictionary dynamically
        gsf_current_ranks = {
            'Arizona Cardinals' : get_gsf_team_rating("ARI"),
            'Atlanta Falcons' : get_gsf_team_rating("ATL"),
            'Baltimore Ravens' : get_gsf_team_rating("BAL"),
            'Buffalo Bills' : get_gsf_team_rating("BUF"),
            'Carolina Panthers' : get_gsf_team_rating("CAR"),
            'Chicago Bears' : get_gsf_team_rating("CHI"),
            'Cincinnati Bengals' : get_gsf_team_rating("CIN"),
            'Cleveland Browns' : get_gsf_team_rating("CLE"),
            'Dallas Cowboys' : get_gsf_team_rating("DAL"),
            'Denver Broncos' : get_gsf_team_rating("DEN"),
            'Detroit Lions' : get_gsf_team_rating("DET"),
            'Green Bay Packers' : get_gsf_team_rating("GB"),
            'Houston Texans' : get_gsf_team_rating("HOU"),
            'Indianapolis Colts' : get_gsf_team_rating("IND"),
            'Jacksonville Jaguars' : get_gsf_team_rating("JAX"),
            'Kansas City Chiefs' : get_gsf_team_rating("KC"),
            'Las Vegas Raiders' : get_gsf_team_rating("LV"),
            'Los Angeles Chargers' : get_gsf_team_rating("LAC"),
            'Los Angeles Rams' : get_gsf_team_rating("LA"),
            'Miami Dolphins' : get_gsf_team_rating("MIA"),
            'Minnesota Vikings' : get_gsf_team_rating("MIN"),
            'New England Patriots' : get_gsf_team_rating("NE"),
            'New Orleans Saints' : get_gsf_team_rating("NO"),
            'New York Giants' : get_gsf_team_rating("NYG"),
            'New York Jets' : get_gsf_team_rating("NYJ"),
            'Philadelphia Eagles' : get_gsf_team_rating("PHI"),
            'Pittsburgh Steelers' : get_gsf_team_rating("PIT"),
            'San Francisco 49ers' : get_gsf_team_rating("SF"),
            'Seattle Seahawks' : get_gsf_team_rating("SEA"),
            'Tampa Bay Buccaneers' : get_gsf_team_rating("TB"),
            'Tennessee Titans' : get_gsf_team_rating("TEN"),
            'Washington Commanders' : get_gsf_team_rating("WAS")
        }
    else:
        print(f"Error: {ratings_file} not found. Hardcoded fallback or empty dict required.")
        team_ratings_dict = {}
    
    print("Dynamic Team Ratings Loaded Successfully.")
    
    
    
    CUSTOM_RANKS = {
        'Arizona Cardinals' : 0,
        'Atlanta Falcons' : 0,
        'Baltimore Ravens' : 0,
        'Buffalo Bills' : 0,
        'Carolina Panthers' : 0,
        'Chicago Bears' : 0,
        'Cincinnati Bengals' : 0,
        'Cleveland Browns' : 0,
        'Dallas Cowboys' : 0,
        'Denver Broncos' : 0,
        'Detroit Lions' : 0,
        'Green Bay Packers' : 0,
        'Houston Texans' : 0,
        'Indianapolis Colts' : 0,
        'Jacksonville Jaguars' : 0,
        'Kansas City Chiefs' : 0,
        'Las Vegas Raiders' : 0,
        'Los Angeles Chargers' : 0,
        'Los Angeles Rams' : 0,
        'Miami Dolphins' : 0,
        'Minnesota Vikings' : 0,
        'New England Patriots' : 0,
        'New Orleans Saints' : 0,
        'New York Giants' : 0,
        'New York Jets' : 0,
        'Philadelphia Eagles' : 0,
        'Pittsburgh Steelers' : 0,
        'San Francisco 49ers' : 0,
        'Seattle Seahawks' : 0,
        'Tampa Bay Buccaneers' : 0,
        'Tennessee Titans' : 0,
        'Washington Commanders' : 0
    }
        
    # 1. Define the file path based on your existing variables
    hfa_file = f"nfl-power-ratings/nfl_hfa_ratings.csv"
    
    # 2. Check if the file exists before trying to read it
    if os.path.exists(hfa_file):
        print(f"Loading ratings from {hfa_file}")
        hfa_df = pd.read_csv(hfa_file)
        
        # 3. Create a helper function to get the rating safely
        def get_home_advantage(team_abbr):
            # Look for the team in the 'team' or 'off_team' column (check your CSV header)
            # We use .iloc[0] to get the value from the matching row
            try:
                # Change 'team' to 'off_team' if that is the name of your team column
                hfa = hfa_df.loc[hfa_df['Team'] == team_abbr, 'HFA (Points)'].values[0]
                return float(hfa)
            except (IndexError, KeyError):
                print(f"Warning: Could not find rating for {team_abbr}. Defaulting to 0.")
                return 0.0
    
        # 4. Build your dictionary dynamically
        DEFAULT_HOME_ADVANTAGE = {
            'Arizona Cardinals' : get_home_advantage("ARI")/2,
            'Atlanta Falcons' : get_home_advantage("ATL")/2,
            'Baltimore Ravens' : get_home_advantage("BAL")/2,
            'Buffalo Bills' : get_home_advantage("BUF")/2,
            'Carolina Panthers' : get_home_advantage("CAR")/2,
            'Chicago Bears' : get_home_advantage("CHI")/2,
            'Cincinnati Bengals' : get_home_advantage("CIN")/2,
            'Cleveland Browns' : get_home_advantage("CLE")/2,
            'Dallas Cowboys' : get_home_advantage("DAL")/2,
            'Denver Broncos' : get_home_advantage("DEN")/2,
            'Detroit Lions' : get_home_advantage("DET")/2,
            'Green Bay Packers' : get_home_advantage("GB")/2,
            'Houston Texans' : get_home_advantage("HOU")/2,
            'Indianapolis Colts' : get_home_advantage("IND")/2,
            'Jacksonville Jaguars' : get_home_advantage("JAX")/2,
            'Kansas City Chiefs' : get_home_advantage("KC")/2,
            'Las Vegas Raiders' : get_home_advantage("LV")/2,
            'Los Angeles Chargers' : get_home_advantage("LAC")/2,
            'Los Angeles Rams' : get_home_advantage("LA")/2,
            'Miami Dolphins' : get_home_advantage("MIA")/2,
            'Minnesota Vikings' : get_home_advantage("MIN")/2,
            'New England Patriots' : get_home_advantage("NE")/2,
            'New Orleans Saints' : get_home_advantage("NO")/2,
            'New York Giants' : get_home_advantage("NYG")/2,
            'New York Jets' : get_home_advantage("NYJ")/2,
            'Philadelphia Eagles' : get_home_advantage("PHI")/2,
            'Pittsburgh Steelers' : get_home_advantage("PIT")/2,
            'San Francisco 49ers' : get_home_advantage("SF")/2,
            'Seattle Seahawks' : get_home_advantage("SEA")/2,
            'Tampa Bay Buccaneers' : get_home_advantage("TB")/2,
            'Tennessee Titans' : get_home_advantage("TEN")/2,
            'Washington Commanders' : get_home_advantage("WAS")/2
        }
    
    # --------------------------------------------------------------------------
    # --- 4. AWAY ADJUSTMENT (STATIC DEFAULTS) ---
    # Used if the user selects 'Default' in the UI for away adjustment.
    # These values are divided by 2 from the input as they appear to be half-points.
    # --------------------------------------------------------------------------
        DEFAULT_AWAY_ADJ = {
            'Arizona Cardinals' : -1 * get_home_advantage("ARI")/2,
            'Atlanta Falcons' : -1 * get_home_advantage("ATL")/2,
            'Baltimore Ravens' : -1 * get_home_advantage("BAL")/2,
            'Buffalo Bills' : -1 * get_home_advantage("BUF")/2,
            'Carolina Panthers' : -1 * get_home_advantage("CAR")/2,
            'Chicago Bears' : -1 * get_home_advantage("CHI")/2,
            'Cincinnati Bengals' : -1 * get_home_advantage("CIN")/2,
            'Cleveland Browns' : -1 * get_home_advantage("CLE")/2,
            'Dallas Cowboys' : -1 * get_home_advantage("DAL")/2,
            'Denver Broncos' : -1 * get_home_advantage("DEN")/2,
            'Detroit Lions' : -1 * get_home_advantage("DET")/2,
            'Green Bay Packers' : -1 * get_home_advantage("GB")/2,
            'Houston Texans' : -1 * get_home_advantage("HOU")/2,
            'Indianapolis Colts' : -1 * get_home_advantage("IND")/2,
            'Jacksonville Jaguars' : -1 * get_home_advantage("JAX")/2,
            'Kansas City Chiefs' : -1 * get_home_advantage("KC")/2,
            'Las Vegas Raiders' : -1 * get_home_advantage("LV")/2,
            'Los Angeles Chargers' : -1 * get_home_advantage("LAC")/2,
            'Los Angeles Rams' : -1 * get_home_advantage("LA")/2,
            'Miami Dolphins' : -1 * get_home_advantage("MIA")/2,
            'Minnesota Vikings' : -1 * get_home_advantage("MIN")/2,
            'New England Patriots' : -1 * get_home_advantage("NE")/2,
            'New Orleans Saints' : -1 * get_home_advantage("NO")/2,
            'New York Giants' : -1 * get_home_advantage("NYG")/2,
            'New York Jets' : -1 * get_home_advantage("NYJ")/2,
            'Philadelphia Eagles' : -1 * get_home_advantage("PHI")/2,
            'Pittsburgh Steelers' : -1 * get_home_advantage("PIT")/2,
            'San Francisco 49ers' : -1 * get_home_advantage("SF")/2,
            'Seattle Seahawks' : -1 * get_home_advantage("SEA")/2,
            'Tampa Bay Buccaneers' : -1 * get_home_advantage("TB")/2,
            'Tennessee Titans' : -1 * get_home_advantage("TEN")/2,
            'Washington Commanders' : -1 * get_home_advantage("WAS")/2
        }
        ABBR_TO_FULL = {
            "ARI": "Arizona Cardinals",
            "ATL": "Atlanta Falcons",
            "BAL": "Baltimore Ravens",
            "BUF": "Buffalo Bills",
            "CAR": "Carolina Panthers",
            "CHI": "Chicago Bears",
            "CIN": "Cincinnati Bengals",
            "CLE": "Cleveland Browns",
            "DAL": "Dallas Cowboys",
            "DEN": "Denver Broncos",
            "DET": "Detroit Lions",
            "GB": "Green Bay Packers",
            "HOU": "Houston Texans",
            "IND": "Indianapolis Colts",
            "JAX": "Jacksonville Jaguars",
    		"JAC": "Jacksonville Jaguars",
            "KC": "Kansas City Chiefs",
            "LV": "Las Vegas Raiders",
            "LAC": "Los Angeles Chargers",
            "LAR": "Los Angeles Rams",
            "LA": "Los Angeles Rams",
            "MIA": "Miami Dolphins",
            "MIN": "Minnesota Vikings",
            "NE": "New England Patriots",
            "NO": "New Orleans Saints",
            "NYG": "New York Giants",
            "NYJ": "New York Jets",
            "PHI": "Philadelphia Eagles",
            "PIT": "Pittsburgh Steelers",
            "SF": "San Francisco 49ers",
            "SEA": "Seattle Seahawks",
            "TB": "Tampa Bay Buccaneers",
            "TEN": "Tennessee Titans",
            "WAS": "Washington Commanders",
    		"WSH": "Washington Commanders"
        }
    
    STADIUM_INFO = {
        'Arizona Cardinals': ['State Farm Stadium', 33.5277, -112.262608, 'America/Denver', 'NFC West'],
        'Atlanta Falcons': ['Mercedes-Benz Stadium', 33.7489, -84.3880, 'America/New_York', 'NFC South'],
        'Baltimore Ravens': ['M&T Bank Stadium', 39.2789, -76.6228, 'America/New_York', 'AFC North'],
        'Buffalo Bills': ['Highmark Stadium', 42.7725, -78.7877, 'America/New_York', 'AFC East'],
        'Carolina Panthers': ['Bank of America Stadium', 35.2258, -80.8528, 'America/New_York', 'NFC South'],
        'Chicago Bears': ['Soldier Field', 41.8623, -87.6167, 'America/Chicago', 'NFC North'],
        'Cincinnati Bengals': ['Paycor Stadium', 39.0955, -84.5165, 'America/New_York', 'AFC North'],
        'Cleveland Browns': ['FirstEnergy Stadium', 41.5061, -81.6994, 'America/New_York', 'AFC North'],
        'Dallas Cowboys': ['AT&T Stadium', 32.7369, -97.0826, 'America/Chicago', 'NFC East'],
        'Denver Broncos': ['Empower Field at Mile High', 39.7648, -105.0076, 'America/Denver', 'AFC West'],
        'Detroit Lions': ['Ford Field', 42.3395, -83.0450, 'America/Detroit', 'NFC North'],
        'Green Bay Packers': ['Lambeau Field', 44.5013, -88.0622, 'America/Chicago', 'NFC North'],
        'Houston Texans': ['NRG Stadium', 29.6847, -95.4093, 'America/Chicago', 'AFC South'],
        'Indianapolis Colts': ['Lucas Oil Stadium', 39.7601, -86.1638, 'America/Indiana/Indianapolis', 'AFC South'],
        'Jacksonville Jaguars': ['TIAA Bank Field', 30.3239, -81.6554, 'America/New_York', 'AFC South'],
        'Kansas City Chiefs': ['GEHA Field at Arrowhead Stadium', 39.0489, -94.4839, 'America/Chicago', 'AFC West'],
        'Las Vegas Raiders': ['Allegiant Stadium', 36.1080, -115.1578, 'America/Los_Angeles', 'AFC West'],
        'Los Angeles Chargers': ['SoFi Stadium', 33.9535, -118.3395, 'America/Los_Angeles', 'AFC West'],
        'Los Angeles Rams': ['SoFi Stadium', 33.9535, -118.3395, 'America/Los_Angeles', 'NFC West'],
        'Miami Dolphins': ['Hard Rock Stadium', 25.9602, -80.2384, 'America/New_York', 'AFC East'],
        'Minnesota Vikings': ['U.S. Bank Stadium', 44.9738, -93.2575, 'America/Chicago', 'NFC North'],
        'New England Patriots': ['Gillette Stadium', 42.0628, -71.2687, 'America/New_York', 'AFC East'],
        'New Orleans Saints': ['Caesars Superdome', 29.9507, -90.0813, 'America/Chicago', 'NFC South'],
        'New York Giants': ['MetLife Stadium', 40.8136, -74.0744, 'America/New_York', 'NFC East'],
        'New York Jets': ['MetLife Stadium', 40.8136, -74.0744, 'America/New_York', 'AFC East'],
        'Philadelphia Eagles': ['Lincoln Financial Field', 39.9008, -75.1675, 'America/New_York', 'NFC East'],
        'Pittsburgh Steelers': ['Acrisure Stadium', 40.4468, -80.0158, 'America/New_York', 'AFC North'],
        'San Francisco 49ers': ['Levi\'s Stadium', 37.4031, -121.9702, 'America/Los_Angeles', 'NFC West'],
        'Seattle Seahawks': ['Lumen Field', 47.5952, -122.3316, 'America/Los_Angeles', 'NFC West'],
        'Tampa Bay Buccaneers': ['Raymond James Stadium', 27.9759, -82.5033, 'America/New_York', 'NFC South'],
        'Tennessee Titans': ['Nissan Stadium', 36.1664, -86.7716, 'America/Chicago', 'AFC South'],
        'Washington Commanders': ['FedExField', 38.9077, -76.8645, 'America/New_York', 'NFC East']
    }
    
    ALL_TEAMS = list(STADIUM_INFO.keys())
    
    
    def collect_schedule_travel_ranking_data(schedule_df):
    # Get the user's custom rankings from the config
    
        stadiums = {}
        for team, info in STADIUM_INFO.items():
            # info = [Stadium Name, Lat, Lon, Timezone, Division]
            
            # 1. Get Preseason Rank (from global static dict)
            mp_preseason_rank = MP_PRESEASON_RANKS.get(team, 0)
            gsf_preseason_rank = GSF_PRESEASON_RANKS.get(team, 0)
    
            mp_current_rank = mp_current_ranks.get(team, 0)
            gsf_current_rank = gsf_current_ranks.get(team, 0)
            
            # 2. Get Current/Custom Rank (from config or default)
            user_rank = CUSTOM_RANKS.get(team, 0)
            
            # 3. Get Home Advantage (from global static dict)
            #    (Your config doesn't store this, so we use default)
            home_adv = DEFAULT_HOME_ADVANTAGE.get(team, 0)
            
            # 4. Get Away Adjustment (from global static dict)
            #    (Your config doesn't store this, so we use default)
            away_adj = DEFAULT_AWAY_ADJ.get(team, 0)
            
            # Build the list in the format your code expects [cite: 25-28, 116]
            stadiums[team] = [
                info[0], # Stadium Name
                info[1], # Lat
                info[2], # Lon
                info[3], # Timezone
                info[4], # Division
                mp_preseason_rank,  # 5: Preseason Rank
                mp_current_rank,    # 6: Current Rank
                gsf_preseason_rank,   #7
                gsf_current_rank,    #8
                home_adv,        # 9: Home Advantage
                away_adj         # 10: Away Adjustment			
            ]
        data = []
        # Initialize a variable to hold the last valid date and week
        last_date = None
        start_date = pd.to_datetime(season_start_date)
        week = 1
        # Initialize a dictionary to store the last game date for each team
        last_game = {}
        last_away_game = {}
        # Initialize dictionaries to store cumulative rest advantage for each team
        cumulative_advantage = {}
        # 0: Stadium | 1: Lattitude | 2: Longitude | 3: Timezone | 4: Division | 5: Preseason Average points better than Average Team (Used for Spread and Odds Calculation) | 6: Current Average points better than Average Team (Used for Spread and Odds Calculation) | 7: Home Advantage | 8: Reduction of Home Advantage when Away Team #Calculated here: https://nfllines.com/nfl-2023-home-field-advantage-values/
    # default_data.py
    # Contains static, non-user-configurable data for NFL teams and stadiums.
    
    
        def haversine(lat1, lon1, lat2, lon2):
    	    # Convert degrees to radians
    	    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    	    # Differences
    	    dlat = lat2 - lat1
    	    dlon = lon2 - lon1
    	    # Haversine formula
    	    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    	    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    	    r = 3956 # Radius in miles
    	    return c * r
    	
        def calculate_hours_difference(tz1, tz2):
    	    try:
    	        tz1_offset = pytz.timezone(tz1).utcoffset(pd.to_datetime(date_str)).total_seconds() / 3600
    	        tz2_offset = pytz.timezone(tz2).utcoffset(pd.to_datetime(date_str)).total_seconds() / 3600
    	        return tz1_offset - tz2_offset
    	    except:
    	        return 0
    			
        df = schedule_df
    	
    	# 2. Pre-processing: Convert date column and sort to ensure chronological order
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date', 'Time'])
    	
    	# 3. Initialize tracking variables
        last_game = {}          # Stores the date of the last game for each team
        last_away_game = {}     # Stores the week of the last away game for each team
        cumulative_advantage = {} # Stores running total of rest advantage
        data = []
    	
        for index, row in df.iterrows():
    	    # 1. Use the row itself for the base data
    	    game_id = row['game_id']
    	    season = row['Season']
    	    week = row['Week']
    	    last_date = row['Date']
    	    gametime = row['Time']
    	    away_team = row['Away Team']
    	    home_team = row['Home Team']
    	    location = row['Location']
    	    away_qb = row['Away QB']
    	    home_qb = row['Home QB']
    	    away_qb_id = row['away_qb_id']
    	    home_qb_id = row['home_qb_id']
    
    
    	
    	    # 2. Calculate rest (Logic remains the same)
    	    away_rest_days = (last_date - last_game[away_team]).days if away_team in last_game else 0
    	    home_rest_days = (last_date - last_game[home_team]).days if home_team in last_game else 0
    	    
    	    away_advantage = away_rest_days - home_rest_days
    	    home_advantage = home_rest_days - away_rest_days
    	
    	    cumulative_advantage[away_team] = cumulative_advantage.get(away_team, 0) + away_advantage
    	    cumulative_advantage[home_team] = cumulative_advantage.get(home_team, 0) + home_advantage
    	
    	    # 3. Handle Back-to-Back Logic
    	    back_to_back_away = (away_team in last_away_game and last_away_game[away_team] == week - 1)
    	    last_away_game[away_team] = week
    	    last_game[away_team] = last_date
    	    last_game[home_team] = last_date
    	
    	    # 4. STORE AS DICTIONARY (Much safer than a list)
    	    # This maps specific values to specific column names immediately
    	    new_row = {
    	        'Game ID': game_id,
    	        'Season': season,
    	        'Week': week,
    	        'Date': last_date,
    	        'Time': gametime,
    	        'Away Team': away_team,
    	        'Home Team': home_team,
    	        'Location': location,
    	        'Away QB': away_qb,
    	        'Home QB': home_qb,
    	        'Away QB ID': away_qb_id,
    	        'Home QB ID': home_qb_id,
    	        'Away Team Weekly Rest': away_rest_days,
    	        'Home Team Weekly Rest': home_rest_days,
    	        'Weekly Away Rest Advantage': away_advantage,
    	        'Weekly Home Rest Advantage': home_advantage,
    	        'Away Cumulative Rest Advantage': cumulative_advantage[away_team],
    	        'Home Cumulative Rest Advantage': cumulative_advantage[home_team],
    	        'Actual Stadium': row['Stadium'],
    	        'Back to Back Away Games': back_to_back_away
    	    }
    	    data.append(new_row)
    	
    	# 5. Create the final DataFrame
    	# Because 'data' is a list of dicts, pandas automatically matches the keys to column names
        df = pd.DataFrame(data)
    
        df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
        # Adjust January games to 2025 in the DataFrame
        df.loc[df['Date'] >= pd.to_datetime(thanksgiving_reset_date), 'Week'] += 1
        df.loc[df['Date'] >= pd.to_datetime(christmas_reset_date), 'Week'] += 1
    
    
        # Convert 'Week' back to string format if needed
        df['Away Team Current Week Cumulative Rest Advantage'] = pd.to_numeric(df['Away Cumulative Rest Advantage'], errors='coerce').fillna(0) - pd.to_numeric(df['Home Cumulative Rest Advantage'], errors='coerce').fillna(0)
        df['Home Team Current Week Cumulative Rest Advantage'] = pd.to_numeric(df['Home Cumulative Rest Advantage'], errors='coerce').fillna(0) - pd.to_numeric(df['Away Cumulative Rest Advantage'], errors='coerce').fillna(0)
        df['Away Team Division'] = df['Away Team'].map(lambda team: stadiums[team][4] if team in stadiums else 'NA')
        df['Away Stadium'] = df['Away Team'].map(lambda team: stadiums[team][0] if team in stadiums else 'NA')
        df['Away Stadium Latitude'] = df['Away Team'].map(lambda team: stadiums[team][1] if team in stadiums else 'NA')
        df['Away Stadium Longitude'] = df['Away Team'].map(lambda team: stadiums[team][2] if team in stadiums else 'NA')
        df['Away Stadium TimeZone'] = df['Away Team'].map(lambda team: stadiums[team][3] if team in stadiums else 'NA')
    
        df['Home Team Division'] = df['Home Team'].map(lambda team: stadiums[team][4] if team in stadiums else 'NA')
        df['Home Stadium'] = df['Home Team'].map(lambda team: stadiums[team][0] if team in stadiums else 'NA')
        df['Home Stadium Latitude'] = df['Home Team'].map(lambda team: stadiums[team][1] if team in stadiums else 'NA')
        df['Home Stadium Longitude'] = df['Home Team'].map(lambda team: stadiums[team][2] if team in stadiums else 'NA')
        df['Home Stadium TimeZone'] = df['Home Team'].map(lambda team: stadiums[team][3] if team in stadiums else 'NA')
        df.loc[df['Actual Stadium'] == '', 'Actual Stadium'] = df['Home Stadium']
    
        df['Away Team Previous Opponent'] = 'BYE'
        df['Home Team Previous Opponent'] = 'BYE'
        df['Away Team Previous Location'] = 'BYE'
        df['Home Team Previous Location'] = 'BYE'
        df['Away Team Next Opponent'] = 'BYE'
        df['Home Team Next Opponent'] = 'BYE'
        df['Away Team Next Location'] = 'BYE'
        df['Home Team Next Location'] = 'BYE'
    
        team_last_opponent = {}
        team_last_location = {}
    
        for index, row in df.iterrows():
            away_team = row['Away Team']
            home_team = row['Home Team']
            week_num = row['Week']
            away_stadium = row['Actual Stadium']
            home_stadium = row['Actual Stadium']
            
            # Check if its not the first week
            if week_num > 1:
                # Get the previous opponents from the dictionary
                if away_team in team_last_opponent:
                    df.loc[index, 'Away Team Previous Opponent'] = team_last_opponent[away_team]
                if home_team in team_last_opponent:
                     df.loc[index, 'Home Team Previous Opponent'] = team_last_opponent[home_team]
                
                # Get the previous locations from the dictionary
                if away_team in team_last_location:
                    df.loc[index, 'Away Team Previous Location'] = team_last_location[away_team]
                if home_team in team_last_location:
                     df.loc[index, 'Home Team Previous Location'] = team_last_location[home_team]
            elif week_num == 1:
                df.loc[index, 'Away Team Previous Opponent'] = 'Preseason'
                df.loc[index, 'Home Team Previous Opponent'] = 'Preseason'
                df.loc[index, 'Away Team Previous Location'] = 'Preseason'
                df.loc[index, 'Home Team Previous Location'] = 'Preseason'
        
            # Update team last opponent dictionary
            team_last_opponent[home_team] = away_team
            team_last_opponent[away_team] = home_team
           
            # Update team last location dictionary
            team_last_location[home_team] = home_stadium
            team_last_location[away_team] = away_stadium
        
    
        team_next_opponent = {}
        team_next_location = {}
    
        # Iterate through the DataFrame in reverse order
        for index in reversed(df.index):
            row = df.loc[index]
            away_team = row['Away Team']
            home_team = row['Home Team']
            week_num = row['Week']
            away_stadium = row['Actual Stadium']
            home_stadium = row['Actual Stadium']
            
            # Check if its not the last week
            if week_num < df['Week'].max():
                # Get the previous opponents from the dictionary
                if away_team in team_next_opponent:
                    df.loc[index, 'Away Team Next Opponent'] = team_next_opponent[away_team]
                if home_team in team_next_opponent:
                     df.loc[index, 'Home Team Next Opponent'] = team_next_opponent[home_team]
                
                # Get the previous locations from the dictionary
                if away_team in team_next_location:
                    df.loc[index, 'Away Team Next Location'] = team_next_location[away_team]
                if home_team in team_next_location:
                     df.loc[index, 'Home Team Next Location'] = team_next_location[home_team]
            else:
                df.loc[index, 'Away Team Next Opponent'] = "Playoffs?"
                df.loc[index, 'Away Team Next Opponent'] = "Playoffs?"
                df.loc[index, 'Away Team Next Location'] = "Playoffs?"
                df.loc[index, 'Home Team Next Location'] = "Playoffs?"
    
            # Update team next opponent dictionary
            team_next_opponent[home_team] = away_team
            team_next_opponent[away_team] = home_team
           
            # Update team next location dictionary
            team_next_location[home_team] = home_stadium
            team_next_location[away_team] = away_stadium
        #df['Home Team'] = df['Home Team'].str.replace(' *', '')
        #df.to_csv('test.csv', index=False)
    
    
        # Add new columns to the DataFrame
        df['Actual Stadium Latitude'] = np.where(df['Actual Stadium'] == 'London, UK', 51.555973, df['Home Stadium Latitude'])
        df['Actual Stadium Longitude'] = np.where(df['Actual Stadium'] == 'London, UK', -0.279672, df['Home Stadium Longitude'])
        df['Actual Stadium TimeZone'] = np.where(df['Actual Stadium'] == 'London, UK', 'Europe/London', df['Home Stadium TimeZone'])
    
        df['Away Stadium Latitude'] = pd.to_numeric(df['Away Stadium Latitude'])
        df['Away Stadium Longitude'] = pd.to_numeric(df['Away Stadium Longitude'])
        df['Actual Stadium Latitude'] = pd.to_numeric(df['Actual Stadium Latitude'])
        df['Actual Stadium Longitude'] = pd.to_numeric(df['Actual Stadium Longitude'])
        df['Home Stadium Latitude'] = pd.to_numeric(df['Home Stadium Latitude'])
        df['Home Stadium Longitude'] = pd.to_numeric(df['Home Stadium Longitude'])
    
        df['Away Travel Distance'] = df.apply(lambda row: round(haversine(row['Away Stadium Latitude'], row['Away Stadium Longitude'], row['Actual Stadium Latitude'], row['Actual Stadium Longitude'])), axis=1)
        df['Home Travel Distance'] = df.apply(lambda row: round(haversine(row['Home Stadium Latitude'], row['Home Stadium Longitude'], row['Actual Stadium Latitude'], row['Actual Stadium Longitude'])), axis=1)
    
        df['Away Travel Advantage'] =  df['Home Travel Distance'] - df['Away Travel Distance']
        df['Home Travel Advantage'] =  df['Away Travel Distance'] - df['Home Travel Distance']
    
        # Apply the function to your DataFrame
        df['Away Timezone Change'] = df.apply(lambda row: calculate_hours_difference(row['Away Stadium TimeZone'], row['Actual Stadium TimeZone']), axis=1)
        df['Home Timezone Change'] = df.apply(lambda row: calculate_hours_difference(row['Home Stadium TimeZone'], row['Actual Stadium TimeZone']), axis=1)
    
        # Initialize empty lists for storing last game timezones
        last_game_timezones_away = []
        last_game_timezones_home = []
    
        # Initialize dictionary for storing last game timezone for each team
        last_game_timezone = {}
    
        # Iterate over DataFrame rows
        for i, row in df.iterrows():
            # Get current away team, home team and actual stadium timezone
            away_team = row['Away Team']
            home_team = row['Home Team']
            actual_stadium_timezone = row['Actual Stadium TimeZone']
    
            # Check if this is not the away team's first game
            if away_team in last_game_timezone:
                # If not, append last game's actual stadium timezone to list
                last_game_timezones_away.append(last_game_timezone[away_team])
            else:
                # If it is, append None (or any other value indicating no previous game)
                last_game_timezones_away.append(None)
    
            # Check if this is not the home team's first game
            if home_team in last_game_timezone:
                # If not, append last game's actual stadium timezone to list
                last_game_timezones_home.append(last_game_timezone[home_team])
            else:
                # If it is, append None (or any other value indicating no previous game)
                last_game_timezones_home.append(None)
    
            # Update last game's actual stadium timezone for current away and home teams
            last_game_timezone[away_team] = actual_stadium_timezone
            last_game_timezone[home_team] = actual_stadium_timezone
    
        # Add new columns to DataFrame
        df['Away Previous Game Actual Stadium TimeZone'] = last_game_timezones_away
        df['Home Previous Game Actual Stadium TimeZone'] = last_game_timezones_home
    
        # Add new column to DataFrame
        df['Home Previous Game Actual Stadium TimeZone'] = last_game_timezones_home
        df['Away Weekly Timezone Difference'] = df.apply(lambda row: calculate_hours_difference(row['Away Previous Game Actual Stadium TimeZone'], row['Actual Stadium TimeZone']) if pd.notnull(row['Away Previous Game Actual Stadium TimeZone']) and row['Away Previous Game Actual Stadium TimeZone'].strip() != '' else None, axis=1)
        df['Home Weekly Timezone Difference'] = df.apply(lambda row: calculate_hours_difference(row['Home Previous Game Actual Stadium TimeZone'], row['Actual Stadium TimeZone']) if pd.notnull(row['Home Previous Game Actual Stadium TimeZone']) and row['Home Previous Game Actual Stadium TimeZone'].strip() != '' else None, axis=1)
    
        df['Adjusted Away Timezone Change'] = df.apply(lambda row: 0 if row['Away Previous Game Actual Stadium TimeZone'] == row['Actual Stadium TimeZone'] and row['Actual Stadium'] != row['Away Stadium'] else calculate_hours_difference(row['Away Stadium TimeZone'], row['Actual Stadium TimeZone']), axis=1)
        df['Adjusted Home Timezone Change'] = df.apply(lambda row: 0 if row['Home Previous Game Actual Stadium TimeZone'] == row['Actual Stadium TimeZone'] and row['Actual Stadium'] != row['Home Stadium'] else calculate_hours_difference(row['Home Stadium TimeZone'], row['Actual Stadium TimeZone']), axis=1)
    
        df['Away Timezone Advantage'] = df.apply(lambda row: 0 if row['Adjusted Away Timezone Change'] == 0 else row['Adjusted Away Timezone Change'] - row['Adjusted Home Timezone Change'], axis=1)
        df['Home Timezone Advantage'] = df.apply(lambda row: 0 if row['Adjusted Home Timezone Change'] == 0 else row['Adjusted Home Timezone Change'] - row['Adjusted Away Timezone Change'], axis=1)
    
        #df['Away Timezone Advantage'] = (df['Away Timezone Change'] - df['Home Timezone Change'])
        #df['Home Timezone Advantage'] = (df['Home Timezone Change'] - df['Away Timezone Change'])
    
        df['Away Team Massey-Peabody Preseason Rank'] = df['Away Team'].map(lambda team: stadiums[team][5] if team in stadiums else 'NA')
        df['Home Team Massey-Peabody Preseason Rank'] = df['Home Team'].map(lambda team: stadiums[team][5] if team in stadiums else 'NA')
    
        df['Away Team Generic Sports Fan Preseason Rank'] = df['Away Team'].map(lambda team: stadiums[team][7] if team in stadiums else 'NA')
        df['Home Team Generic Sports Fan Preseason Rank'] = df['Home Team'].map(lambda team: stadiums[team][7] if team in stadiums else 'NA')
    
        df['Massey-Peabody Preseason Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Massey-Peabody Preseason Rank'] > row['Home Team Massey-Peabody Preseason Rank'] else (row['Home Team'] if row['Away Team Massey-Peabody Preseason Rank'] < row['Home Team Massey-Peabody Preseason Rank'] else 'Tie'), axis=1)
        df['Massey-Peabody Preseason Difference'] = abs(df['Away Team Massey-Peabody Preseason Rank'] - df['Home Team Massey-Peabody Preseason Rank'])
    
        df['Generic Sports Fan Preseason Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Generic Sports Fan Preseason Rank'] > row['Home Team Generic Sports Fan Preseason Rank'] else (row['Home Team'] if row['Away Team Generic Sports Fan Preseason Rank'] < row['Home Team Generic Sports Fan Preseason Rank'] else 'Tie'), axis=1)
        df['Generic Sports Fan Preseason Difference'] = abs(df['Away Team Generic Sports Fan Preseason Rank'] - df['Home Team Generic Sports Fan Preseason Rank'])
    
        df['Away Team MP + GSF Average Preseason Rank'] = (df['Away Team Massey-Peabody Preseason Rank'] + df['Away Team Generic Sports Fan Preseason Rank'])/2
        df['Home Team MP + GSF Average Preseason Rank'] = (df['Home Team Massey-Peabody Preseason Rank'] + df['Home Team Generic Sports Fan Preseason Rank'])/2
        df['MP + GSF Average Preseason Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team MP + GSF Average Preseason Rank'] > row['Home Team MP + GSF Average Preseason Rank'] else (row['Home Team'] if row['Away Team MP + GSF Average Preseason Rank'] < row['Home Team MP + GSF Average Preseason Rank'] else 'Tie'), axis=1)
        df['MP + GSF Average Preseason Difference'] = abs(df['Away Team MP + GSF Average Preseason Rank'] - df['Home Team MP + GSF Average Preseason Rank'])
    
        THREE_IN_TEN_PENALTY = -0.5
        FOUR_IN_SEVENTEEN_PENALTY = -0.25
        SHORT_REST_PENALTY = -.5
        
        # 2. Create a Non-Linear Timezone Penalty Function
        def calculate_timezone_penalty(tz_advantage):
            # Only penalize if the timezone shift is 2 or more hours
            if abs(tz_advantage) >= 2:
                return (abs(tz_advantage) ** 1) * 0.15 * (1 if tz_advantage > 0 else -.5)
            return 0
        df['Away NonLinear TZ'] = df['Away Timezone Advantage'].apply(calculate_timezone_penalty)
        df['Home NonLinear TZ'] = df['Home Timezone Advantage'].apply(calculate_timezone_penalty)
    
    
        # Create "HT 3 games in 10 days" and "AT 3 games in 10 Days" columns with default "No"
        df['Home Team 3 games in 10 days'] = 'No'
        df['Away Team 3 games in 10 days'] = 'No'
    
        # Convert 'Date' column to datetime objects
        df['Date'] = pd.to_datetime(df['Date'])
    
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Get the home and away teams
            home_team = row['Home Team']
            away_team = row['Away Team']
            game_date = row['Date']
    
            # Calculate the start date of the 10-day window
            ten_days_ago = game_date - pd.Timedelta(days=10)
    
            # Get the previous 10 days of games for the home team (regardless of home/away)
            home_team_games = df[
                ((df['Home Team'] == home_team) | (df['Away Team'] == home_team)) &
                (df['Date'] >= ten_days_ago) & (df['Date'] <= game_date) 
            ].sort_values('Date', ascending=False).head(10)
    
            # Get the previous 10 days of games for the away team (regardless of home/away)
            away_team_games = df[
                ((df['Home Team'] == away_team) | (df['Away Team'] == away_team)) &
                (df['Date'] >= ten_days_ago) & (df['Date'] <= game_date)
            ].sort_values('Date', ascending=False).head(10)
    
            # Check if home team has played 3 games in the last 10 days (regardless of home/away)
            if len(home_team_games) >= 3:
                df.loc[index, 'Home Team 3 games in 10 days'] = 'Yes'
    
            # Check if away team has played 3 games in the last 10 days (regardless of home/away)
            if len(away_team_games) >= 3:
                df.loc[index, 'Away Team 3 games in 10 days'] = 'Yes'
    
        # Create "HT 4 games in 17 days" and "AT 4 games in 17 Days" columns with default "No"
        df['Home Team 4 games in 17 days'] = 'No'
        df['Away Team 4 games in 17 days'] = 'No'
    
        # Convert 'Date' column to datetime objects
        df['Date'] = pd.to_datetime(df['Date'])
    
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Get the home and away teams
            home_team = row['Home Team']
            away_team = row['Away Team']
            game_date = row['Date']
    
            # Calculate the start date of the 10-day window
            seventeen_days_ago = game_date - pd.Timedelta(days=17)
    
            # Get the previous 10 days of games for the home team (regardless of home/away)
            home_team_games = df[
                ((df['Home Team'] == home_team) | (df['Away Team'] == home_team)) &
                (df['Date'] >= seventeen_days_ago) & (df['Date'] <= game_date) 
            ].sort_values('Date', ascending=False).head(17)
    
            # Get the previous 10 days of games for the away team (regardless of home/away)
            away_team_games = df[
                ((df['Home Team'] == away_team) | (df['Away Team'] == away_team)) &
                (df['Date'] >= seventeen_days_ago) & (df['Date'] <= game_date)
            ].sort_values('Date', ascending=False).head(17)
    
            # Check if home team has played 3 games in the last 10 days (regardless of home/away)
            if len(home_team_games) >= 4:
                df.loc[index, 'Home Team 4 games in 17 days'] = 'Yes'
    
            # Check if away team has played 3 games in the last 10 days (regardless of home/away)
            if len(away_team_games) >= 4:
                df.loc[index, 'Away Team 4 games in 17 days'] = 'Yes'
    
    
        # Convert 'NA' to NaN
        df['Away Team Weekly Rest'] = df['Away Team Weekly Rest'].replace('NA', 0)
        df['Home Team Weekly Rest'] = df['Home Team Weekly Rest'].replace('NA', 0)
    
        # Convert to integers
        df['Away Team Weekly Rest'] = pd.to_numeric(df['Away Team Weekly Rest'], errors='coerce')
        df['Home Team Weekly Rest'] = pd.to_numeric(df['Home Team Weekly Rest'], errors='coerce')        
    
        df['Away Team Short Rest'] = 'No'
        df['Divisional Matchup?'] = (df['Home Team Division'] == df['Away Team Division']).astype(int)
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Get the home and away teams
            home_team_rest = row['Home Team Weekly Rest']
            away_team_rest = row['Away Team Weekly Rest']
            game_date = row['Date']
    
            # Check for short rest and rest disadvantage
            if (away_team_rest < 7) and (away_team_rest < home_team_rest):
                # Update the 'Away Team Short Rest' for the specific row
                df.loc[index, 'Away Team Short Rest'] = 'Yes'

        df['Away Team Massey-Peabody Current Rank'] = df['Away Team'].map(lambda team: stadiums[team][6] if team in stadiums else 'NA')
        df['Home Team Massey-Peabody Current Rank'] = df['Home Team'].map(lambda team: stadiums[team][6] if team in stadiums else 'NA')
    
        df['Away Team Generic Sports Fan Current Rank'] = df['Away Team'].map(lambda team: stadiums[team][8] if team in stadiums else 'NA')
        df['Home Team Generic Sports Fan Current Rank'] = df['Home Team'].map(lambda team: stadiums[team][8] if team in stadiums else 'NA')


            # 3. Calculate the new, highly-situational Away GSF Rank
        df['Away Team Adjusted Massey-Peabody Preseason Rank'] = (
            df['Away Team'].map(lambda team: stadiums[team][5]) # Base Rank
            + np.where((df['Away Travel Advantage'] < -400) & (df['Home Stadium'] == df['Actual Stadium']), -0, 0) # Extreme Travel
            - df['Away NonLinear TZ'] # NEW Non-Linear Timezone
            + pd.to_numeric(df['Weekly Away Rest Advantage'], errors='coerce').fillna(0) * .125 # Standard Rest
            + df['Away Team Current Week Cumulative Rest Advantage'] * .05 #.0625 # Cumulative Rest
            + np.where(df['Away Team Short Rest'] == 'Yes', SHORT_REST_PENALTY / 5, 0) # NEW Short Rest Penalty
            + np.where(df['Away Team 3 games in 10 days'] == 'Yes', THREE_IN_TEN_PENALTY# / 5, 0) # NEW 3-in-10 Penalty
            + np.where(df['Away Team 4 games in 17 days'] == 'Yes', FOUR_IN_SEVENTEEN_PENALTY# / 5, 0) # NEW 4-in-17 Penalty
            - np.where((df['Away Team'].map(lambda team: stadiums[team][0])) != df['Home Team'].map(lambda team: stadiums[team][0]), df['Away Team'].map(lambda team: stadiums[team][10]), 0) # Static Away Adj
        )
        
        # 4. Calculate the new Home GSF Rank
        df['Home Team Adjusted Massey-Peabody Preseason Rank'] = (
            df['Home Team'].map(lambda team: stadiums[team][5]) # Base Rank
            + np.where((df['Home Travel Advantage'] < -400) & (df['Home Stadium'] == df['Actual Stadium']), -0, 0) # Extreme Travel
            - df['Home NonLinear TZ'] # NEW Non-Linear Timezone
            + pd.to_numeric(df['Weekly Home Rest Advantage'], errors='coerce').fillna(0) * .125 # Standard Rest
            + df['Home Team Current Week Cumulative Rest Advantage'] * .05 #.0625 # Cumulative Rest
            + np.where(df['Home Team Short Rest'] == 'Yes', SHORT_REST_PENALTY / 5, 0) # NEW Short Rest Penalty
            + np.where(df['Home Team 3 games in 10 days'] == 'Yes', THREE_IN_TEN_PENALTY# / 5, 0) # NEW 3-in-10 Penalty
            + np.where(df['Home Team 4 games in 17 days'] == 'Yes', FOUR_IN_SEVENTEEN_PENALTY# / 5, 0) # NEW 4-in-17 Penalty
            - np.where((df['Home Team'].map(lambda team: stadiums[team][0])) != df['Away Team'].map(lambda team: stadiums[team][0]), df['Home Team'].map(lambda team: stadiums[team][10]), 0) # Static Away Adj
        )

        # 3. Calculate the new, highly-situational Away GSF Rank
        df['Away Team Adjusted Generic Sports Fan Preseason Rank'] = (
            df['Away Team'].map(lambda team: stadiums[team][7]) # Base Rank
            + np.where((df['Away Travel Advantage'] < -400) & (df['Home Stadium'] == df['Actual Stadium']), -0, 0) # Extreme Travel
            - df['Away NonLinear TZ'] # NEW Non-Linear Timezone
            + pd.to_numeric(df['Weekly Away Rest Advantage'], errors='coerce').fillna(0) * .125 # Standard Rest
            + df['Away Team Current Week Cumulative Rest Advantage'] * .05 #.0625 # Cumulative Rest
            + np.where(df['Away Team Short Rest'] == 'Yes', SHORT_REST_PENALTY / 5, 0) # NEW Short Rest Penalty
            + np.where(df['Away Team 3 games in 10 days'] == 'Yes', THREE_IN_TEN_PENALTY# / 5, 0) # NEW 3-in-10 Penalty
            + np.where(df['Away Team 4 games in 17 days'] == 'Yes', FOUR_IN_SEVENTEEN_PENALTY# / 5, 0) # NEW 4-in-17 Penalty
            - np.where((df['Away Team'].map(lambda team: stadiums[team][0])) != df['Home Team'].map(lambda team: stadiums[team][0]), df['Away Team'].map(lambda team: stadiums[team][10]), 0) # Static Away Adj
        )
        
        # 4. Calculate the new Home GSF Rank
        df['Home Team Adjusted Generic Sports Fan Preseason Rank'] = (
            df['Home Team'].map(lambda team: stadiums[team][7]) # Base Rank
            + np.where((df['Home Travel Advantage'] < -400) & (df['Home Stadium'] == df['Actual Stadium']), -0, 0) # Extreme Travel
            - df['Home NonLinear TZ'] # NEW Non-Linear Timezone
            + pd.to_numeric(df['Weekly Home Rest Advantage'], errors='coerce').fillna(0) * .125 # Standard Rest
            + df['Home Team Current Week Cumulative Rest Advantage'] * .05 #.0625 # Cumulative Rest
            + np.where(df['Home Team Short Rest'] == 'Yes', SHORT_REST_PENALTY / 5, 0) # NEW Short Rest Penalty
            + np.where(df['Home Team 3 games in 10 days'] == 'Yes', THREE_IN_TEN_PENALTY# / 5, 0) # NEW 3-in-10 Penalty
            + np.where(df['Home Team 4 games in 17 days'] == 'Yes', FOUR_IN_SEVENTEEN_PENALTY# / 5, 0) # NEW 4-in-17 Penalty
            - np.where((df['Home Team'].map(lambda team: stadiums[team][0])) != df['Away Team'].map(lambda team: stadiums[team][0]), df['Home Team'].map(lambda team: stadiums[team][10]), 0) # Static Away Adj
        )
        
        # 5. NEW: Divisional Game Compression
        # If it's a divisional game, reduce the absolute difference between the teams' ranks by 15%
        divisional_mask = df['Divisional Matchup?'] == 1
        gsf_ps_rank_diff = df['Home Team Adjusted Generic Sports Fan Preseason Rank'] - df['Away Team Adjusted Generic Sports Fan Preseason Rank']
        
        # Pull the favorite closer to the underdog in divisional games
        df.loc[divisional_mask, 'Home Team Adjusted Generic Sports Fan Preseason Rank'] -= (gsf_ps_rank_diff * 0.075)
        df.loc[divisional_mask, 'Away Team Adjusted Generic Sports Fan Preseason Rank'] += (gsf_ps_rank_diff * 0.075)

        mp_ps_rank_diff = df['Home Team Adjusted Massey-Peabody Preseason Rank'] - df['Away Team Adjusted Massey-Peabody Preseason Rank']
        
        # Pull the favorite closer to the underdog in divisional games
        df.loc[divisional_mask, 'Home Team Adjusted Massey-Peabody Preseason Rank'] -= (mp_ps_rank_diff * 0.075)
        df.loc[divisional_mask, 'Away Team Adjusted Massey-Peabody Preseason Rank'] += (mp_ps_rank_diff * 0.075)

        # 3. Calculate the new, highly-situational Away GSF Rank
        df['Away Team Adjusted Massey-Peabody Current Rank'] = (
            df['Away Team'].map(lambda team: stadiums[team][6]) # Base Rank
            + np.where((df['Away Travel Advantage'] < -400) & (df['Home Stadium'] == df['Actual Stadium']), -0, 0) # Extreme Travel
            - df['Away NonLinear TZ'] # NEW Non-Linear Timezone
            + pd.to_numeric(df['Weekly Away Rest Advantage'], errors='coerce').fillna(0) * .125 # Standard Rest
            + df['Away Team Current Week Cumulative Rest Advantage'] * .05 #.0625 # Cumulative Rest
            + np.where(df['Away Team Short Rest'] == 'Yes', SHORT_REST_PENALTY / 5, 0) # NEW Short Rest Penalty
            + np.where(df['Away Team 3 games in 10 days'] == 'Yes', THREE_IN_TEN_PENALTY# / 5, 0) # NEW 3-in-10 Penalty
            + np.where(df['Away Team 4 games in 17 days'] == 'Yes', FOUR_IN_SEVENTEEN_PENALTY# / 5, 0) # NEW 4-in-17 Penalty
            - np.where((df['Away Team'].map(lambda team: stadiums[team][0])) != df['Home Team'].map(lambda team: stadiums[team][0]), df['Away Team'].map(lambda team: stadiums[team][10]), 0) # Static Away Adj
        )
        
        # 4. Calculate the new Home GSF Rank
        df['Home Team Adjusted Massey-Peabody Current Rank'] = (
            df['Home Team'].map(lambda team: stadiums[team][6]) # Base Rank
            + np.where((df['Home Travel Advantage'] < -400) & (df['Home Stadium'] == df['Actual Stadium']), -0, 0) # Extreme Travel
            - df['Home NonLinear TZ'] # NEW Non-Linear Timezone
            + pd.to_numeric(df['Weekly Home Rest Advantage'], errors='coerce').fillna(0) * .125 # Standard Rest
            + df['Home Team Current Week Cumulative Rest Advantage'] * .05 #.0625 # Cumulative Rest
            + np.where(df['Home Team Short Rest'] == 'Yes', SHORT_REST_PENALTY / 5, 0) # NEW Short Rest Penalty
            + np.where(df['Home Team 3 games in 10 days'] == 'Yes', THREE_IN_TEN_PENALTY# / 5, 0) # NEW 3-in-10 Penalty
            + np.where(df['Home Team 4 games in 17 days'] == 'Yes', FOUR_IN_SEVENTEEN_PENALTY# / 5, 0) # NEW 4-in-17 Penalty
            - np.where((df['Home Team'].map(lambda team: stadiums[team][0])) != df['Away Team'].map(lambda team: stadiums[team][0]), df['Home Team'].map(lambda team: stadiums[team][10]), 0) # Static Away Adj
        )

        # 3. Calculate the new, highly-situational Away GSF Rank
        df['Away Team Adjusted Generic Sports Fan Current Rank'] = (
            df['Away Team'].map(lambda team: stadiums[team][8]) # Base Rank
            + np.where((df['Away Travel Advantage'] < -400) & (df['Home Stadium'] == df['Actual Stadium']), -0, 0) # Extreme Travel
            - df['Away NonLinear TZ'] # NEW Non-Linear Timezone
            + pd.to_numeric(df['Weekly Away Rest Advantage'], errors='coerce').fillna(0) * .125 # Standard Rest
            + df['Away Team Current Week Cumulative Rest Advantage'] * .05 #.0625 # Cumulative Rest
            + np.where(df['Away Team Short Rest'] == 'Yes', SHORT_REST_PENALTY / 5, 0) # NEW Short Rest Penalty
            + np.where(df['Away Team 3 games in 10 days'] == 'Yes', THREE_IN_TEN_PENALTY# / 5, 0) # NEW 3-in-10 Penalty
            + np.where(df['Away Team 4 games in 17 days'] == 'Yes', FOUR_IN_SEVENTEEN_PENALTY# / 5, 0) # NEW 4-in-17 Penalty
            - np.where((df['Away Team'].map(lambda team: stadiums[team][0])) != df['Home Team'].map(lambda team: stadiums[team][0]), df['Away Team'].map(lambda team: stadiums[team][10]), 0) # Static Away Adj
        )
        
        # 4. Calculate the new Home GSF Rank
        df['Home Team Adjusted Generic Sports Fan Current Rank'] = (
            df['Home Team'].map(lambda team: stadiums[team][8]) # Base Rank
            + np.where((df['Home Travel Advantage'] < -400) & (df['Home Stadium'] == df['Actual Stadium']), -0, 0) # Extreme Travel
            - df['Home NonLinear TZ'] # NEW Non-Linear Timezone
            + pd.to_numeric(df['Weekly Home Rest Advantage'], errors='coerce').fillna(0) * .125 # Standard Rest
            + df['Home Team Current Week Cumulative Rest Advantage'] * .05 #.0625 # Cumulative Rest
            + np.where(df['Home Team Short Rest'] == 'Yes', SHORT_REST_PENALTY / 5, 0) # NEW Short Rest Penalty
            + np.where(df['Home Team 3 games in 10 days'] == 'Yes', THREE_IN_TEN_PENALTY# / 5, 0) # NEW 3-in-10 Penalty
            + np.where(df['Home Team 4 games in 17 days'] == 'Yes', FOUR_IN_SEVENTEEN_PENALTY# / 5, 0) # NEW 4-in-17 Penalty
            - np.where((df['Home Team'].map(lambda team: stadiums[team][0])) != df['Away Team'].map(lambda team: stadiums[team][0]), df['Home Team'].map(lambda team: stadiums[team][10]), 0) # Static Away Adj
        )
        
        # 5. NEW: Divisional Game Compression
        # If it's a divisional game, reduce the absolute difference between the teams' ranks by 15%
        divisional_mask = df['Divisional Matchup?'] == 1
        rank_diff = df['Home Team Adjusted Generic Sports Fan Current Rank'] - df['Away Team Adjusted Generic Sports Fan Current Rank']
        
        # Pull the favorite closer to the underdog in divisional games
        df.loc[divisional_mask, 'Home Team Adjusted Generic Sports Fan Current Rank'] -= (rank_diff * 0.075)
        df.loc[divisional_mask, 'Away Team Adjusted Generic Sports Fan Current Rank'] += (rank_diff * 0.075)

        mp_rank_diff = df['Home Team Adjusted Massey-Peabody Current Rank'] - df['Away Team Adjusted Massey-Peabody Current Rank']
        
        # Pull the favorite closer to the underdog in divisional games
        df.loc[divisional_mask, 'Home Team Adjusted Massey-Peabody Current Rank'] -= (mp_rank_diff * 0.075)
        df.loc[divisional_mask, 'Away Team Adjusted Massey-Peabody Current Rank'] += (mp_rank_diff * 0.075)

        df['Adjusted Massey-Peabody Preseason Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Adjusted Massey-Peabody Preseason Rank'] > row['Home Team Adjusted Massey-Peabody Preseason Rank'] else (row['Home Team'] if row['Away Team Adjusted Massey-Peabody Preseason Rank'] < row['Home Team Adjusted Massey-Peabody Preseason Rank'] else 'Tie'), axis=1)
        df['Adjusted Massey-Peabody Preseason Difference'] = abs(df['Away Team Adjusted Massey-Peabody Preseason Rank'] - df['Home Team Adjusted Massey-Peabody Preseason Rank'])
    
        df['Adjusted Generic Sports Fan Preseason Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Adjusted Generic Sports Fan Preseason Rank'] > row['Home Team Adjusted Generic Sports Fan Preseason Rank'] else (row['Home Team'] if row['Away Team Adjusted Generic Sports Fan Preseason Rank'] < row['Home Team Adjusted Generic Sports Fan Preseason Rank'] else 'Tie'), axis=1)
        df['Adjusted Generic Sports Fan Preseason Difference'] = abs(df['Away Team Adjusted Generic Sports Fan Preseason Rank'] - df['Home Team Adjusted Massey-Peabody Preseason Rank'])
    
        df['Away Team Adjusted MP + GSF Average Preseason Rank'] = (df['Away Team Adjusted Massey-Peabody Preseason Rank'] + df['Away Team Adjusted Generic Sports Fan Preseason Rank'])/2
        df['Home Team Adjusted MP + GSF Average Preseason Rank'] = (df['Home Team Adjusted Massey-Peabody Preseason Rank'] + df['Home Team Adjusted Generic Sports Fan Preseason Rank'])/2
        df['Adjusted MP + GSF Average Preseason Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Adjusted MP + GSF Average Preseason Rank'] > row['Home Team Adjusted MP + GSF Average Preseason Rank'] else (row['Home Team'] if row['Away Team Adjusted MP + GSF Average Preseason Rank'] < row['Home Team Adjusted MP + GSF Average Preseason Rank'] else 'Tie'), axis=1)
        df['Adjusted MP + GSF Average Preseason Difference'] = abs(df['Away Team Adjusted MP + GSF Average Preseason Rank'] - df['Home Team Adjusted MP + GSF Average Preseason Rank'])
    
    
        df['Away Team Massey-Peabody Current Rank'] = df['Away Team'].map(lambda team: stadiums[team][6] if team in stadiums else 'NA')
        df['Home Team Massey-Peabody Current Rank'] = df['Home Team'].map(lambda team: stadiums[team][6] if team in stadiums else 'NA')
    
        df['Away Team Generic Sports Fan Current Rank'] = df['Away Team'].map(lambda team: stadiums[team][8] if team in stadiums else 'NA')
        df['Home Team Generic Sports Fan Current Rank'] = df['Home Team'].map(lambda team: stadiums[team][8] if team in stadiums else 'NA')
    
        df['Massey-Peabody Current Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Massey-Peabody Current Rank'] > row['Home Team Massey-Peabody Current Rank'] else (row['Home Team'] if row['Away Team Massey-Peabody Current Rank'] < row['Home Team Massey-Peabody Current Rank'] else 'Tie'), axis=1)
        df['Massey-Peabody Current Difference'] = abs(df['Away Team Massey-Peabody Current Rank'] - df['Home Team Massey-Peabody Current Rank'])
    
        df['Generic Sports Fan Current Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Generic Sports Fan Current Rank'] > row['Home Team Generic Sports Fan Current Rank'] else (row['Home Team'] if row['Away Team Generic Sports Fan Current Rank'] < row['Home Team Generic Sports Fan Current Rank'] else 'Tie'), axis=1)
        df['Generic Sports Fan Current Difference'] = abs(df['Away Team Generic Sports Fan Current Rank'] - df['Home Team Generic Sports Fan Current Rank'])
    
        df['Away Team MP + GSF Average Current Rank'] = (df['Away Team Massey-Peabody Current Rank'] + df['Away Team Generic Sports Fan Current Rank'])/2
        df['Home Team MP + GSF Average Current Rank'] = (df['Home Team Massey-Peabody Current Rank'] + df['Home Team Generic Sports Fan Current Rank'])/2
        df['MP + GSF Average Current Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team MP + GSF Average Current Rank'] > row['Home Team MP + GSF Average Current Rank'] else (row['Home Team'] if row['Away Team MP + GSF Average Current Rank'] < row['Home Team MP + GSF Average Current Rank'] else 'Tie'), axis=1)
        df['MP + GSF Average Current Difference'] = abs(df['Away Team MP + GSF Average Current Rank'] - df['Home Team MP + GSF Average Current Rank'])
    
        df['Adjusted Massey-Peabody Current Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Adjusted Massey-Peabody Current Rank'] > row['Home Team Adjusted Massey-Peabody Current Rank'] else (row['Home Team'] if row['Away Team Adjusted Massey-Peabody Current Rank'] < row['Home Team Adjusted Massey-Peabody Current Rank'] else 'Tie'), axis=1)
        df['Adjusted Massey-Peabody Current Difference'] = abs(df['Away Team Adjusted Massey-Peabody Current Rank'] - df['Home Team Adjusted Massey-Peabody Current Rank'])
    
        df['Adjusted Generic Sports Fan Current Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Adjusted Generic Sports Fan Current Rank'] > row['Home Team Adjusted Generic Sports Fan Current Rank'] else (row['Home Team'] if row['Away Team Adjusted Generic Sports Fan Current Rank'] < row['Home Team Adjusted Generic Sports Fan Current Rank'] else 'Tie'), axis=1)
        df['Adjusted Generic Sports Fan Current Difference'] = abs(df['Away Team Adjusted Generic Sports Fan Current Rank'] - df['Home Team Adjusted Generic Sports Fan Current Rank'])
    
        df['Away Team Adjusted MP + GSF Average Current Rank'] = (df['Away Team Adjusted Massey-Peabody Current Rank'] + df['Away Team Adjusted Generic Sports Fan Current Rank'])/2
        df['Home Team Adjusted MP + GSF Average Current Rank'] = (df['Home Team Adjusted Massey-Peabody Current Rank'] + df['Home Team Adjusted Generic Sports Fan Current Rank'])/2
        df['Adjusted MP + GSF Average Current Winner'] = df.apply(lambda row: row['Away Team'] if row['Away Team Adjusted MP + GSF Average Current Rank'] > row['Home Team Adjusted MP + GSF Average Current Rank'] else (row['Home Team'] if row['Away Team Adjusted MP + GSF Average Current Rank'] < row['Home Team Adjusted MP + GSF Average Current Rank'] else 'Tie'), axis=1)
        df['Adjusted MP + GSF Average Current Difference'] = abs(df['Away Team Adjusted MP + GSF Average Current Rank'] - df['Home Team Adjusted MP + GSF Average Current Rank'])
    
        df['Massey-Peabody Bayesian Same Winner Across All Metrics'] = df.apply(lambda row: 'Same' if row['Massey-Peabody Preseason Winner'] == row['Adjusted Massey-Peabody Preseason Winner'] == row['Massey-Peabody Current Winner'] == row['Adjusted Massey-Peabody Current Winner'] else 'Different', axis=1)
        df['Generic Sports Fan Bayesian Same Adjusted Winner'] = df.apply(lambda row: 'Same' if row['Generic Sports Fan Preseason Winner'] == row['Adjusted Generic Sports Fan Preseason Winner'] == row['Generic Sports Fan Current Winner'] == row['Adjusted Generic Sports Fan Current Winner'] else 'Different', axis=1)
    
        df['Massey-Peabody Bayesian Same Current and Preseason Adjusted Winner'] = df.apply(lambda row: 'Same' if row['Adjusted Massey-Peabody Preseason Winner'] == row['Adjusted Massey-Peabody Current Winner'] else 'Different', axis=1)
        df['Generic Sports Fan Bayesian Current and Preseason Adjusted Winner'] = df.apply(lambda row: 'Same' if row['Adjusted Generic Sports Fan Preseason Winner'] == row['Adjusted Generic Sports Fan Current Winner'] else 'Different', axis=1)
    
        df['Massey-Peabody Bayesian Same Current and Adjusted Current Winner'] = df.apply(lambda row: 'Same' if row['Massey-Peabody Current Winner'] == row['Adjusted Massey-Peabody Current Winner'] else 'Different', axis=1)
        df['Generic Sports Fan Bayesian Same Current and Adjusted Current Winner'] = df.apply(lambda row: 'Same' if row['Generic Sports Fan Current Winner'] == row['Adjusted Generic Sports Fan Current Winner'] else 'Different', axis=1)
    
        
        df['Thursday Night Game'] = 'False'
        df["Thursday Night Game"] = df.apply(lambda row: 'True' if (row['Date'].weekday() == 3) and (row['Date'] != pd.to_datetime(thanksgiving_date)) and (row['Date'] != pd.to_datetime(boxing_day)) and (row['Date'] != pd.to_datetime(christmas_day)) else row["Thursday Night Game"], axis =1)
    
    
        df['Masey-Peabody Home Team Winner?'] = df.apply(lambda row: 'Home Team' if row['Adjusted Massey-Peabody Current Winner'] == row['Home Team'] else 'Away Team', axis=1)
        df['Generic Sports Fan Home Team Winner?'] = df.apply(lambda row: 'Home Team' if row['Adjusted Generic Sports Fan Current Winner'] == row['Home Team'] else 'Away Team', axis=1)
        #df['Divisional Matchup?'] = df.apply(lambda row: 'Divisional' if row['Home Team Division'] == row['Away Team Division'] else 'Non-divisional', axis=1)
        
        def get_backup_nfl_odds():
            """
            Fetches odds from nfl_data_py (nflverse) as a fallback.
            Useful for past games or when the main API is down.
            """
            try:
                print("Fetching backup odds from nflreadpy...")
                
                season = target_year
                
                # 2. Load Schedule and Team Data
                df_schedule = nfl.load_schedules([season])
                df_teams = nfl.load_teams()
                
                # Create a mapping from Abbreviation (KC) to Full Name (Kansas City Chiefs)
                # to match The Odds API format
                team_map = dict(zip(df_teams['team_abbr'], df_teams['team_name']))
                
                formatted_games = []
                
                # 3. Iterate and Format
                for index, row in df_schedule.iterrows():
                    # Skip games that don't have lines/odds yet
                    if pd.isna(row['home_moneyline']) or pd.isna(row['gametime']):
                        continue
        
                    # Format Time: nflreadpy times are typically strings in Eastern Time already
                    # Combine gameday (YYYY-MM-DD) and gametime (HH:MM)
                    game_time_str = f"{row['gameday']} {row['gametime']}"
                    try:
                        dt_obj = datetime.strptime(game_time_str, '%Y-%m-%d %H:%M')
                        # Format to your specific style: "8:20 pm ET"
                        formatted_time = dt_obj.strftime('%I:%M %p ET').replace('AM ET', 'am').replace('PM ET', 'pm').lstrip('0')
                    except ValueError:
                        formatted_time = row['gametime'] # Fallback if parsing fails
        
                    # Calculate Spreads
                    # nflreadpy 'spread_line' is usually the Home Team's spread
                    home_spread = row['spread_line']
                    # Away spread is typically the inverse
                    away_spread = -1 * home_spread if home_spread is not None else None
        
                    formatted_games.append({
                        'Time': formatted_time,
                        'Away Team': team_map.get(row['away_team'], row['away_team']),
                        'Away Odds': row['away_moneyline'], # nflreadpy already provides American odds
                        'Home Team': team_map.get(row['home_team'], row['home_team']),
                        'Home Odds': row['home_moneyline'], # nflreadpy already provides American odds
                        'Away Spread': away_spread,
                        'Home Spread': home_spread
                    })
        
                return pd.DataFrame(formatted_games)
        
            except Exception as e:
                print(f"Backup data fetch failed: {e}")
                return pd.DataFrame()
        
        def get_full_season_odds(api_key):
            """
            Generates a full season view:
            1. Fetches the ENTIRE season schedule from nflreadpy (Past & Future).
            2. Fetches LIVE odds from The Odds API.
            3. Merges them: Updates the nflreadpy schedule with live API data where available.
            """
            
            # ---------------------------------------------------------
            # STEP 1: Get the "Base" Schedule (Past + Future) from nflreadpy
            # ---------------------------------------------------------
            print("Fetching full season schedule from nflreadpy...")
            
            # Determine season (if currently Jan/Feb 2025, we want the 2024 season)
            now = pd.to_datetime(date_str)
            season = now.year if now.month > 3 else now.year - 1
            
            try:
                # 1. Load data (returns Polars DataFrame)
                df_schedule_polars = nfl.load_schedules([season])
                df_teams_polars = nfl.load_teams()
            
                # 2. Convert to Pandas to use .iterrows()
                df_schedule = df_schedule_polars.to_pandas()
                df_teams = df_teams_polars.to_pandas()
            except Exception as e:
                print(f"Error fetching nflreadpy data: {e}")
                return pd.DataFrame()
        
            # Create mapping: Abbr (KC) -> Full Name (Kansas City Chiefs) to match Odds API
            team_map = dict(zip(df_teams['team_abbr'], df_teams['team_name']))
            
            base_games = []
        
            for index, row in df_schedule.iterrows():
                # Map abbreviations to full names
                home_full = team_map.get(row['home_team'], row['home_team'])
                away_full = team_map.get(row['away_team'], row['away_team'])
                
                # Format Time
                try:
                    # Combine gameday and gametime
                    game_time_str = f"{row['gameday']} {row['gametime']}"
                    dt_obj = datetime.strptime(game_time_str, '%Y-%m-%d %H:%M')
                    formatted_time = dt_obj.strftime('%I:%M %p ET').replace('AM ET', 'am').replace('PM ET', 'pm').lstrip('0')
                except:
                    formatted_time = str(row['gameday']) # Fallback
        
                # Handle Spreads (nflreadpy is usually Home relative)
                # If Spread is -3.0, Home is favored by 3.
                home_spread = row['spread_line']
                away_spread = -1 * home_spread if home_spread is not None else None
        
                # Build the row
                base_games.append({
                    'Match_ID': f"{home_full} vs {away_full}", # Unique Key for merging
                    'Time': formatted_time,
                    'Away Team': away_full,
                    'Away Odds': row['away_moneyline'],
                    'Home Team': home_full,
                    'Home Odds': row['home_moneyline'],
                    'Away Spread': away_spread,
                    'Home Spread': home_spread,
                    'Source': 'Historical (nflreadpy)' # Tag source for debugging
                })
            
            df_base = pd.DataFrame(base_games)
        
            # ---------------------------------------------------------
            # STEP 2: Get the "Live" Data from The Odds API
            # ---------------------------------------------------------
            print("Fetching live odds from API...")
            
            live_games = []
            
            # API Config
            SPORT = 'americanfootball_nfl'
            REGIONS = 'us'
            MARKETS = 'h2h,spreads'
            ODDS_FORMAT = 'decimal'
            DATE_FORMAT = 'iso'
            url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={api_key}&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}&dateFormat={DATE_FORMAT}'
        
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    odds_data = response.json()
                    eastern_tz = pytz.timezone('America/New_York')
        
                    for event in odds_data:
                        home_team = event['home_team']
                        away_team = event['away_team']
                        
                        # Time Formatting
                        utc_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
                        east_time = utc_time.astimezone(eastern_tz)
                        formatted_time = east_time.strftime('%I:%M %p ET').replace('AM ET', 'am').replace('PM ET', 'pm').lstrip('0')
        
                        # Odds Aggregation
                        game_odds = {'home': [], 'away': [], 'home_spread': [], 'away_spread': []}
                        for bookmaker in event['bookmakers']:
                            for market in bookmaker['markets']:
                                if market['key'] == 'h2h':
                                    for outcome in market['outcomes']:
                                        if outcome['name'] == home_team: game_odds['home'].append(outcome['price'])
                                        elif outcome['name'] == away_team: game_odds['away'].append(outcome['price'])
                                elif market['key'] == 'spreads':
                                    for outcome in market['outcomes']:
                                        if outcome['name'] == home_team: game_odds['home_spread'].append(outcome['point'])
                                        elif outcome['name'] == away_team: game_odds['away_spread'].append(outcome['point'])
        
                        # Averages
                        avg_home = sum(game_odds['home'])/len(game_odds['home']) if game_odds['home'] else None
                        avg_away = sum(game_odds['away'])/len(game_odds['away']) if game_odds['away'] else None
                        avg_home_spread = sum(game_odds['home_spread'])/len(game_odds['home_spread']) if game_odds['home_spread'] else None
                        avg_away_spread = sum(game_odds['away_spread'])/len(game_odds['away_spread']) if game_odds['away_spread'] else None
        
                        # Convert Decimal to American
                        def dec_to_amer(dec):
                            if not dec: return None
                            if dec >= 2.0: return round((dec - 1) * 100)
                            else: return round(-100 / (dec - 1))
        
                        live_games.append({
                            'Match_ID': f"{home_team} vs {away_team}",
                            'Time': formatted_time,
                            'Away Team': away_team,
                            'Away Odds': dec_to_amer(avg_away),
                            'Home Team': home_team,
                            'Home Odds': dec_to_amer(avg_home),
                            'Away Spread': avg_away_spread,
                            'Home Spread': avg_home_spread,
                            'Source': 'Live API'
                        })
            except Exception as e:
                print(f"API failed ({e}), relying solely on backup data.")
        
            # ---------------------------------------------------------
            # STEP 3: Merge - Overwrite Base with Live Data
            # ---------------------------------------------------------
            
            if live_games:
                df_live = pd.DataFrame(live_games)
                
                # Iterate through live games and update the base dataframe
                # We match on "Match_ID" (Home vs Away)
                for index, row in df_live.iterrows():
                    match_id = row['Match_ID']
                    
                    # Find matching index in df_base
                    mask = df_base['Match_ID'] == match_id
                    
                    if mask.any():
                        # Update specific columns
                        cols_to_update = ['Time', 'Away Odds', 'Home Odds', 'Away Spread', 'Home Spread', 'Source']
                        df_base.loc[mask, cols_to_update] = row[cols_to_update].values
                    else:
                        # Optional: If for some reason the game isn't in nflreadpy (rare), append it
                        # df_base = pd.concat([df_base, pd.DataFrame([row])], ignore_index=True)
                        pass
        
            # Drop the Match_ID helper column before returning
            df_base = df_base.drop(columns=['Match_ID'])
            
            return df_base
        
        # ---------------------------------------------------------
        # Usage in Streamlit
        # ---------------------------------------------------------
        API_KEY = '34671f7aeaa8f4fbee2398163f2f45d3'# Replace with actual key
        
        if API_KEY != 'YOUR_API_KEY':
            # Fetch Data
            live_api_odds_df = get_full_season_odds(API_KEY)
            
            print("Full Season Odds (Historical + Live)")
            
            # Optional: Highlight the Source column so you see which are Live vs Historical
            print(live_api_odds_df)
        else:
            print("Please enter your API Key")
    	
        def add_odds_to_main_csv():
            """
            Adds odds data to the main DataFrame, prioritizing DraftKings data if available and complete.
            If DraftKings data is missing or incomplete for a game, it overrides with internal calculations.
        
            Args:
                df (pd.DataFrame): The main DataFrame to which odds will be added.
                live_api_odds_df (pd.DataFrame): DataFrame containing live odds scraped from DraftKings.
                # ... (all preseason_X_rank, X_rank, X_home_adv, X_away_adj parameters for each team)
        
            Returns:
                pd.DataFrame: The updated DataFrame with odds.
            """
        
            # 0: Spread | 1: Favorite Odds| 2: Underdog Odds
            odds = {
                0: [-110, -110], .5: [-116, -104], 1: [-122, 101], 1.5: [-128, 105], 2: [-131, 108],
                2.5: [-142, 117], 3: [-164, 135], 3.5: [-191, 156], 4: [-211, 171], 4.5: [-224, 181],
                5: [-234, 188], 5.5: [-244, 195], 6: [-261, 208], 6.5: [-282, 224], 7: [-319, 249],
                7.5: [-346, 268], 8: [-366, 282], 8.5: [-397, 302], 9: [-416, 314], 9.5: [-436, 327],
                10: [-483, 356], 10.5: [-538, 389], 11: [-567, 406], 11.5: [-646, 450], 12: [-660, 458],
                12.5: [-675, 466], 13: [-729, 494], 13.5: [-819, 539], 14: [-890, 573], 14.5: [-984, 615],
                15: [-1134, 677], 15.5: [-1197, 702], 16: [-1266, 728], 16.5: [-1267, 728], 17: [-1381, 769],
                17.5: [-1832, 906], 18: [-2149, 986], 18.5: [-2590, 1079], 19: [-3245, 1190], 19.5: [-4323, 1324],
                20: [-4679, 1359], 20.5: [-5098, 1396], 21: [-5597, 1434], 21.5: [-6000, 1500], 22: [-6500, 1600],
                22.5: [-7000, 1650], 23: [-7500, 1700], 23.5: [-8000, 1750], 24: [-8500, 1800], 24.5: [-9000, 1850],
                25: [-9500, 1900], 25.5: [-10000, 2000], 26: [-10000, 2000], 26.5: [-10000, 2000], 27: [-10000, 2000],
                27.5: [-10000, 2000], 28: [-10000, 2000], 28.5: [-10000, 2000], 29: [-10000, 2000], 29.5: [-10000, 2000],
                30: [-10000, 2000]
            }
        
            # Create a copy of the main DataFrame to work with, avoiding modification of the original
            csv_df = df.copy()
        
            # Initialize columns that will be populated by DraftKings data or overridden with internal data
            csv_df['Home Team Sportsbook Moneyline'] = np.nan
            csv_df['Away Team Sportsbook Moneyline'] = np.nan
            csv_df['Sportsbook Favorite'] = np.nan
            csv_df['Sportsbook Underdog'] = np.nan
            csv_df['Home Team Sportsbook Spread'] = np.nan
            csv_df['Away Team Sportsbook Spread'] = np.nan
            
            # Attempt to update CSV data with scraped odds from DraftKings
            # This block only executes if live_api_odds_df is not empty
            if not live_api_odds_df.empty:
                for index, row in csv_df.iterrows():
                    # Find a matching row in the scraped DraftKings data
                    matching_row = live_api_odds_df[
                        (live_api_odds_df['Away Team'] == row['Away Team']) & 
                        (live_api_odds_df['Home Team'] == row['Home Team'])
                    ]
                    if not matching_row.empty:
                        # If a match is found, apply DraftKings moneyline odds
                        csv_df.loc[index, 'Away Team Sportsbook Moneyline'] = matching_row.iloc[0]['Away Odds']
                        csv_df.loc[index, 'Home Team Sportsbook Moneyline'] = matching_row.iloc[0]['Home Odds']
                        csv_df.loc[index, 'Away Team Sportsbook Spread'] = matching_row.iloc[0]['Away Spread']
                        csv_df.loc[index, 'Home Team Sportsbook Spread'] = matching_row.iloc[0]['Home Spread']					
                        
                        # Determine Favorite/Underdog based on DraftKings odds
                        # Assuming odds <= -110 typically indicates the favorite
                        if matching_row.iloc[0]['Home Odds'] <= -110:
                            csv_df.loc[index, 'Sportsbook Favorite'] = csv_df.loc[index, 'Home Team']
                            csv_df.loc[index, 'Sportsbook Underdog'] = csv_df.loc[index, 'Away Team']
                        else:
                            csv_df.loc[index, 'Sportsbook Favorite'] = csv_df.loc[index, 'Away Team']
                            csv_df.loc[index, 'Sportsbook Underdog'] = csv_df.loc[index, 'Home Team']
        
    
    
        
            # Helper function to get moneyline based on calculated spread and internal odds dictionary
            def get_mp_moneyline(row, odds, team_type):
                """
                Calculates moneyline based on a team's adjusted spread and the predefined odds dictionary.
                Finds the closest spread in the dictionary if an exact match is not found.
                """
                spread = round(row['Adjusted Massey-Peabody Current Difference'] * 2) / 2
                
                # Find the closest spread in the odds dictionary to handle non-exact matches
                closest_spread = min(odds.keys(), key=lambda k: abs(k - spread))
                
                moneyline_tuple = odds[closest_spread] # Use the moneyline values for the closest spread
                
                # Determine which moneyline (favorite or underdog) applies to the current team
                if team_type == 'home':
                    if row['Adjusted Massey-Peabody Current Winner'] == row['Home Team']:
                        return moneyline_tuple[0] # Favorite odds
                    else:
                        return moneyline_tuple[1] # Underdog odds
                elif team_type == 'away':
                    if row['Adjusted Massey-Peabody Current Winner'] == row['Away Team']:
                        return moneyline_tuple[0] # Favorite odds
                    else:
                        return moneyline_tuple[1] # Underdog odds
                return np.nan # Should not be reached under normal circumstances
    	        # Helper function to get moneyline based on calculated spread and internal odds dictionary
            def get_gsf_moneyline(row, odds, team_type):
                """
                Calculates moneyline based on a team's adjusted spread and the predefined odds dictionary.
                Finds the closest spread in the dictionary if an exact match is not found.
                """
                spread = round(row['Adjusted Generic Sports Fan Current Difference'] * 2) / 2
                
                # Find the closest spread in the odds dictionary to handle non-exact matches
                closest_spread = min(odds.keys(), key=lambda k: abs(k - spread))
                
                moneyline_tuple = odds[closest_spread] # Use the moneyline values for the closest spread
                
                # Determine which moneyline (favorite or underdog) applies to the current team
                if team_type == 'home':
                    if row['Adjusted Generic Sports Fan Current Winner'] == row['Home Team']:
                        return moneyline_tuple[0] # Favorite odds
                    else:
                        return moneyline_tuple[1] # Underdog odds
                elif team_type == 'away':
                    if row['Adjusted Generic Sports Fan Current Winner'] == row['Away Team']:
                        return moneyline_tuple[0] # Favorite odds
                    else:
                        return moneyline_tuple[1] # Underdog odds
                return np.nan # Should not be reached under normal circumstances
        
            # Calculate internal moneyline values for all games
            csv_df['Massey-Peabody Home Team Moneyline'] = csv_df.apply(
                lambda row: get_mp_moneyline(row, odds, 'home'), axis=1
            )
            csv_df['Massey-Peabody Away Team Moneyline'] = csv_df.apply(
                lambda row: get_mp_moneyline(row, odds, 'away'), axis=1
            )
    
            # Calculate internal moneyline values for all games
            csv_df['Generic Sports Fan Home Team Moneyline'] = csv_df.apply(
                lambda row: get_gsf_moneyline(row, odds, 'home'), axis=1
            )
            csv_df['Generic Sports Fan Away Team Moneyline'] = csv_df.apply(
                lambda row: get_gsf_moneyline(row, odds, 'away'), axis=1
            )
    		
    
    #        st.subheader('Games with Unavailable Live Odds')
    #        st.write('This dataframe contains the games where live odds from the Live Odds API were unavailable. This will likely happen for lookahead lines and future weeks')
    #        st.write(overridden_games_df)
    
            csv_df['Massey-Peabody Home Team Spread'] = csv_df['Away Team Adjusted Massey-Peabody Current Rank'] - csv_df['Home Team Adjusted Massey-Peabody Current Rank']
            csv_df['Massey-Peabody Away Team Spread'] = csv_df['Home Team Adjusted Massey-Peabody Current Rank'] - csv_df['Away Team Adjusted Massey-Peabody Current Rank']
    
            csv_df['Generic Sports Fan Home Team Spread'] = csv_df['Away Team Adjusted Generic Sports Fan Current Rank'] - csv_df['Home Team Adjusted Generic Sports Fan Current Rank']
            csv_df['Generic Sports Fan Away Team Spread'] = csv_df['Home Team Adjusted Generic Sports Fan Current Rank'] - csv_df['Away Team Adjusted Generic Sports Fan Current Rank']
    		
            # Iterate through the DataFrame to apply overrides and calculate implied/fair odds
            for index, row in csv_df.iterrows():
                # Calculate Implied Odds for the final (potentially overridden) moneyline
                away_moneyline = csv_df.loc[index, 'Away Team Sportsbook Moneyline']
                home_moneyline = csv_df.loc[index, 'Home Team Sportsbook Moneyline']
        
                # Handle potential NaN values before calculating implied odds
                if pd.isna(away_moneyline):
                    csv_df.loc[index, 'Away Team Sportsbook Implied Odds to Win'] = np.nan
                elif away_moneyline > 0:
                    csv_df.loc[index, 'Away Team Sportsbook Implied Odds to Win'] = 100 / (away_moneyline + 100)
                else:
                    csv_df.loc[index, 'Away Team Sportsbook Implied Odds to Win'] = abs(away_moneyline) / (abs(away_moneyline) + 100)
                
                if pd.isna(home_moneyline):
                    csv_df.loc[index, 'Home Team Sportsbook Implied Odds to Win'] = np.nan
                elif home_moneyline > 0:
                    csv_df.loc[index, 'Home Team Sportsbook Implied Odds to Win'] = 100 / (home_moneyline + 100)
                else:
                    csv_df.loc[index, 'Home Team Sportsbook Implied Odds to Win'] = abs(home_moneyline) / (abs(home_moneyline) + 100)
    
    				
                away_mp_moneyline = csv_df.loc[index, 'Massey-Peabody Away Team Moneyline']
                home_mp_moneyline = csv_df.loc[index, 'Massey-Peabody Home Team Moneyline']
        
                # Handle potential NaN values before calculating implied odds
                if pd.isna(away_mp_moneyline):
                    csv_df.loc[index, 'Away Team Massey-Peabody Implied Odds to Win'] = np.nan
                elif away_mp_moneyline > 0:
                    csv_df.loc[index, 'Away Team Massey-Peabody Implied Odds to Win'] = 100 / (away_mp_moneyline + 100)
                else:
                    csv_df.loc[index, 'Away Team Massey-Peabody Implied Odds to Win'] = abs(away_mp_moneyline) / (abs(away_mp_moneyline) + 100)
                
                if pd.isna(home_mp_moneyline):
                    csv_df.loc[index, 'Home Team Massey-Peabody Implied Odds to Win'] = np.nan
                elif home_mp_moneyline > 0:
                    csv_df.loc[index, 'Home Team Massey-Peabody Implied Odds to Win'] = 100 / (home_mp_moneyline + 100)
                else:
                    csv_df.loc[index, 'Home Team Massey-Peabody Implied Odds to Win'] = abs(home_mp_moneyline) / (abs(home_mp_moneyline) + 100)
    
                away_gsf_moneyline = csv_df.loc[index, 'Generic Sports Fan Away Team Moneyline']
                home_gsf_moneyline = csv_df.loc[index, 'Generic Sports Fan Home Team Moneyline']
        
                # Handle potential NaN values before calculating implied odds
                if pd.isna(away_gsf_moneyline):
                    csv_df.loc[index, 'Away Team Generic Sports Fan Implied Odds to Win'] = np.nan
                elif away_gsf_moneyline > 0:
                    csv_df.loc[index, 'Away Team Generic Sports Fan Implied Odds to Win'] = 100 / (away_gsf_moneyline + 100)
                else:
                    csv_df.loc[index, 'Away Team Generic Sports Fan Implied Odds to Win'] = abs(away_gsf_moneyline) / (abs(away_gsf_moneyline) + 100)
                
                if pd.isna(home_gsf_moneyline):
                    csv_df.loc[index, 'Home Team Generic Sports Fan Implied Odds to Win'] = np.nan
                elif home_gsf_moneyline > 0:
                    csv_df.loc[index, 'Home Team Generic Sports Fan Implied Odds to Win'] = 100 / (home_gsf_moneyline + 100)
                else:
                    csv_df.loc[index, 'Home Team Generic Sports Fan Implied Odds to Win'] = abs(home_gsf_moneyline) / (abs(home_gsf_moneyline) + 100)
        
                # Calculate Fair Odds for the final (potentially overridden) moneyline
                away_implied_odds = csv_df.loc[index, 'Away Team Sportsbook Implied Odds to Win']
                home_implied_odds = csv_df.loc[index, 'Home Team Sportsbook Implied Odds to Win']
                
                # Ensure sum is not zero or NaN before division
                if pd.isna(away_implied_odds) or pd.isna(home_implied_odds) or (away_implied_odds + home_implied_odds) == 0:
                    csv_df.loc[index, 'Away Team Sportsbook Fair Odds'] = np.nan
                    csv_df.loc[index, 'Home Team Sportsbook Fair Odds'] = np.nan
                else:
                    csv_df.loc[index, 'Away Team Sportsbook Fair Odds'] = away_implied_odds / (away_implied_odds + home_implied_odds)
                    csv_df.loc[index, 'Home Team Sportsbook Fair Odds'] = home_implied_odds / (away_implied_odds + home_implied_odds)
        
                # Calculate Fair Odds for Internal Moneyline (always calculated)
                mp_away_implied_odds = csv_df.loc[index, 'Away Team Massey-Peabody Implied Odds to Win']
                mp_home_implied_odds = csv_df.loc[index, 'Home Team Massey-Peabody Implied Odds to Win']
                
                if pd.isna(mp_away_implied_odds) or pd.isna(mp_home_implied_odds) or (mp_away_implied_odds + mp_home_implied_odds) == 0:
                    csv_df.loc[index, 'Away Team Massey-Peabody Fair Odds'] = np.nan
                    csv_df.loc[index, 'Home Team Massey-Peabody Fair Odds'] = np.nan
                else:
                    csv_df.loc[index, 'Away Team Massey-Peabody Fair Odds'] = mp_away_implied_odds / (mp_away_implied_odds + mp_home_implied_odds)
                    csv_df.loc[index, 'Home Team Massey-Peabody Fair Odds'] = mp_home_implied_odds / (mp_away_implied_odds + mp_home_implied_odds)
    
                # Calculate Fair Odds for Internal Moneyline (always calculated)
                gsf_away_implied_odds = csv_df.loc[index, 'Away Team Generic Sports Fan Implied Odds to Win']
                gsf_home_implied_odds = csv_df.loc[index, 'Home Team Generic Sports Fan Implied Odds to Win']
                
                if pd.isna(gsf_away_implied_odds) or pd.isna(gsf_home_implied_odds) or (gsf_away_implied_odds + gsf_home_implied_odds) == 0:
                    csv_df.loc[index, 'Away Team Generic Sports Fan Fair Odds'] = np.nan
                    csv_df.loc[index, 'Home Team Generic Sports Fan Fair Odds'] = np.nan
                else:
                    csv_df.loc[index, 'Away Team Generic Sports Fan Fair Odds'] = gsf_away_implied_odds / (gsf_away_implied_odds + gsf_home_implied_odds)
                    csv_df.loc[index, 'Home Team Generic Sports Fan Fair Odds'] = gsf_home_implied_odds / (gsf_away_implied_odds + gsf_home_implied_odds)
        
                # Round all calculated odds to 4 decimal places
                for col in ['Away Team Massey-Peabody Implied Odds to Win', 'Home Team Massey-Peabody Implied Odds to Win', 'Away Team Sportsbook Implied Odds to Win',
    					   'Home Team Sportsbook Implied Odds to Win', 'Away Team Generic Sports Fan Implied Odds to Win', 'Home Team Generic Sports Fan Implied Odds to Win',
    					   'Away Team Sportsbook Fair Odds', 'Home Team Sportsbook Fair Odds', 'Away Team Massey-Peabody Fair Odds', 'Home Team Massey-Peabody Fair Odds',
    					   'Away Team Generic Sports Fan Fair Odds', 'Home Team Generic Sports Fan Fair Odds']:
                    if not pd.isna(csv_df.loc[index, col]): # Only round if not NaN
                        csv_df.loc[index, col] = round(csv_df.loc[index, col], 4)
        
            main_df_with_odds_df = csv_df
            return main_df_with_odds_df
        
        schedule_df_with_odds_df = add_odds_to_main_csv()
        
        df = schedule_df_with_odds_df
            
                
    
        df["Away Team Fair Odds"] = (
    	    df["Away Team Sportsbook Fair Odds"]
    	    .fillna(df["Away Team Massey-Peabody Fair Odds"])
    	    .fillna(df["Away Team Generic Sports Fan Fair Odds"])
    	)
    	
        df["Home Team Fair Odds"] = (
    	    df["Home Team Sportsbook Fair Odds"]
    	    .fillna(df["Home Team Massey-Peabody Fair Odds"])
    	    .fillna(df["Home Team Generic Sports Fan Fair Odds"])
    	)
    
        df["Away Team Expected Win Advantage"] = round(df["Away Team Fair Odds"] - 0.5, 4)
        df["Home Team Expected Win Advantage"] = round(df["Home Team Fair Odds"] - 0.5, 4)
        # Initialize an empty dictionary to store team information
        team_dict = {}
    
        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            week = row["Week"]
            away_team = row["Away Team"]
            home_team = row["Home Team"]    
            away_odds = row["Away Team Expected Win Advantage"]
            home_odds = row["Home Team Expected Win Advantage"]
    
            # Create a nested dictionary for each team if not already present
            if away_team not in team_dict:
                team_dict[away_team] = {}
            if home_team not in team_dict:
                team_dict[home_team] = {}
    
            # Populate the nested dictionary with game details and odds
            team_dict[away_team][week] = {"Opponent": home_team, "Home/Away": "Away", "Win Odds": away_odds}
            team_dict[home_team][week] = {"Opponent": away_team, "Home/Away": "Home", "Win Odds": home_odds}
    
        # Calculate cumulative win percentage for each team
        for team, games in team_dict.items():
            for week, details in games.items():
                opponent = details["Opponent"]
                home_away = details["Home/Away"]
                win_odds = details["Win Odds"]
    
                # Get the remaining weeks for the team
                remaining_weeks = [w for w in games.keys() if int(w) > int(week)]
    
                #print(remaining_weeks)
    
                # Calculate cumulative win percentage
                if remaining_weeks:
                    cumulative_win_odds = sum(team_dict[team][w]["Win Odds"] for w in remaining_weeks)
                    cumulative_win_percentage = cumulative_win_odds/len(remaining_weeks)
                else:
                    cumulative_win_percentage = 0  # Set to 0 for week 18
    
                # Add the cumulative win percentage to the dictionary
                team_dict[team][week]["Cumulative Win Percentage"] = cumulative_win_percentage
    
    
        # Initialize empty lists for cumulative win percentages
        away_cumulative_win_percentages = []
        home_cumulative_win_percentages = []
    
        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            week = row["Week"]
            away_team = row["Away Team"]
            home_team = row["Home Team"]
    
            # Get cumulative win percentages from your dictionary
            away_cumulative_win_percentage = team_dict.get(away_team, {}).get(week, {}).get("Cumulative Win Percentage", 0)
            home_cumulative_win_percentage = team_dict.get(home_team, {}).get(week, {}).get("Cumulative Win Percentage", 0)
    
            # Append to the lists
            away_cumulative_win_percentages.append(away_cumulative_win_percentage)
            home_cumulative_win_percentages.append(home_cumulative_win_percentage)
    
        # Add new columns to the DataFrame
        df["Away Team Cumulative Win Percentage"] = away_cumulative_win_percentages
        df["Home Team Cumulative Win Percentage"] = home_cumulative_win_percentages
    
    
        # Get unique week values
        unique_weeks = df["Week"].unique()
    
        # Calculate the maximum cumulative win percentage for each week
        max_cumulative_win_percentage = {}
        for week in unique_weeks:
            week_df = df[df["Week"] == week]
            # Calculate the maximum, using `0` as default if week_df is empty
            if week_df.empty:
                max_val = 0
            else:
                max_val = max(week_df["Away Team Cumulative Win Percentage"].max(),
                             week_df["Home Team Cumulative Win Percentage"].max())
    
            # Check if the calculated max_val is NaN and replace with 1 if so
            if pd.isna(max_val):
                max_cumulative_win_percentage[week] = 1
            else:
                max_cumulative_win_percentage[week] = max_val
    
        # Calculate the minimum cumulative win percentage for each week
        min_cumulative_win_percentage = {}
        for week in unique_weeks:
            week_df = df[df["Week"] == week]
            # Calculate the maximum, using `0` as default if week_df is empty
            if week_df.empty:
                min_val = 0
            else:
                min_val = min(week_df["Away Team Cumulative Win Percentage"].min(),
                             week_df["Home Team Cumulative Win Percentage"].min())
    
            # Check if the calculated max_val is NaN and replace with 1 if so
            if pd.isna(min_val):
                min_cumulative_win_percentage[week] = 0
            else:
                min_cumulative_win_percentage[week] = min_val
        
        # Calculate the range of cumulative win percentages for each week
        range_cumulative_win_percentage = {}
        for week in unique_weeks:
            range_cumulative_win_percentage[week] = max_cumulative_win_percentage[week] - min_cumulative_win_percentage[week]
            if range_cumulative_win_percentage[week] == 0:
                range_cumulative_win_percentage[week]=1
            if pd.isna(range_cumulative_win_percentage[week]):
                range_cumulative_win_percentage[week] = 1
                
        # Define a function to calculate the star rating
        def calculate_star_rating(cumulative_win_percentage, week):
            # Normalize the cumulative win percentage to a scale of 0 to 1
            if pd.isna(cumulative_win_percentage):
                cumulative_win_percentage = 0.0  # Return 0 for NaN inputs
                print("Cumulative Win % is error")
            if pd.isna(min_cumulative_win_percentage[week]):
                min_cumulative_win_percentage[week] = 0.0
                print("Minimum Cumulative Win % is error")
            if pd.isna(range_cumulative_win_percentage[week]):
                range_cumulative_win_percentage[week] = 1.0
                print("Range Cumulative Win % is error")
            try:
                normalized_percentage = (cumulative_win_percentage - min_cumulative_win_percentage[week]) / range_cumulative_win_percentage[week]
                # Assign stars linearly based on the normalized percentage
                return round(10 * normalized_percentage) / 2
            except ZeroDivisionError:
                return 0.0
    
        # Apply the function to create the new columns for each week
    
        # 1. Define Favorite/Underdog for the WHOLE DataFrame
        df["Favorite"] = (
            df["Sportsbook Favorite"]
            .fillna(df["Adjusted Massey-Peabody Current Winner"])
            .fillna(df["Adjusted Generic Sports Fan Current Winner"])
        )
    	
        df["Underdog"] = np.where(
            df["Favorite"] == df["Home Team"], 
            df["Away Team"], 
            df["Home Team"]
        )
    	
    	# 2. Identify Holiday Teams ONCE (Outside any loops)
    	# Using .unique() to get a set of teams for fast lookup
        tg_winners = set(df[df["Week"] == thanksgiving_week]["Favorite"].unique())
        tg_underdogs = set(df[df["Week"] == thanksgiving_week]["Underdog"].unique())
    	
        xm_winners = set(df[df["Week"] == christmas_week]["Favorite"].unique())
        xm_underdogs = set(df[df["Week"] == christmas_week]["Underdog"].unique())
    	
    	# 3. Create Holiday Columns using vectorized logic (No loop needed)
    	# Helper to check if a team is a Holiday Favorite
        def mark_holiday(team_col, week_col, holiday_week, team_set):
            # Returns 1 if week is <= holiday week AND team is in the set
            return ((week_col <= holiday_week) & (team_col.isin(team_set))).astype(int)
    	
    	# Apply Thanksgiving Flags
        df["Away Team Thanksgiving Favorite"] = mark_holiday(df["Away Team"], df["Week"], thanksgiving_week, tg_winners)
        df["Home Team Thanksgiving Favorite"] = mark_holiday(df["Home Team"], df["Week"], thanksgiving_week, tg_winners)
        df["Away Team Thanksgiving Underdog"] = mark_holiday(df["Away Team"], df["Week"], thanksgiving_week, tg_underdogs)
        df["Home Team Thanksgiving Underdog"] = mark_holiday(df["Home Team"], df["Week"], thanksgiving_week, tg_underdogs)
    	
    	# Apply Christmas Flags
        df["Away Team Christmas Favorite"] = mark_holiday(df["Away Team"], df["Week"], christmas_week, xm_winners)
        df["Home Team Christmas Favorite"] = mark_holiday(df["Home Team"], df["Week"], christmas_week, xm_winners)
        df["Away Team Christmas Underdog"] = mark_holiday(df["Away Team"], df["Week"], christmas_week, xm_underdogs)
        df["Home Team Christmas Underdog"] = mark_holiday(df["Home Team"], df["Week"], christmas_week, xm_underdogs)
    	
    	# 4. Pre-Holiday Logic (Vectorized)
        df['Away Team Pre Thanksgiving'] = ((df['Away Team Thanksgiving Favorite'] | df['Away Team Thanksgiving Underdog']) & (df['Week'] < thanksgiving_week)).astype(int)
        df['Home Team Pre Thanksgiving'] = ((df['Home Team Thanksgiving Favorite'] | df['Home Team Thanksgiving Underdog']) & (df['Week'] < thanksgiving_week)).astype(int)
    	
    	# 5. Divisional Matchup Boolean
        df["Divisional Matchup Boolean"] = (df["Divisional Matchup?"] == True).astype(int)
    
    
        unique_weeks = df["Week"].unique()
    	
    	# 6. ONLY loop for the Star Ratings (since that usually needs specialized logic)
        for week in unique_weeks:
            mask = df["Week"] == week
            df.loc[mask, "Away Team Star Rating"] = df.loc[mask, "Away Team Cumulative Win Percentage"].apply(lambda x: calculate_star_rating(x, week))
            df.loc[mask, "Home Team Star Rating"] = df.loc[mask, "Home Team Cumulative Win Percentage"].apply(lambda x: calculate_star_rating(x, week))
            def scrape_data(url):
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "lxml")
                table_rows = soup.find_all("tr")
            
                data = []
                for row in table_rows:
                    columns = row.find_all("td")
                    if len(columns) >= 5:
                        ev, win_pct, pick_pct, team, opponent = columns[:5]
                        rest = columns[5:]
                        future_value_cell = rest[-1] if rest else None
            
                        if future_value_cell:
                            div_tag = future_value_cell.find("div")
                            if div_tag and "style" in div_tag.attrs:
                                style_attr = div_tag["style"]
                                width_match = re.search(r"width:\s*(\d+)px", style_attr)
                                star_rating = int(width_match.group(1)) / 16 if width_match else 0
                            else:
                                star_rating = 0
                        else:
                            star_rating = 0
            
                        data.append({
                            "EV": ev.text,
                            "Win %": win_pct.text,
                            "Pick %": pick_pct.text,
                            "Team": team.text,
                            "Opponent": opponent.text,
                            "Future Value (Stars)": star_rating
                        })
            
                return data
        
        
        def scrape_all_data(starting_year, current_year_plus_1):
            all_data = []
            base_url = "https://www.survivorgrid.com/{year}/{week}"
        
            total_iterations = (current_year_plus_1 - starting_year) * 18
    
            start_week = starting_week
            completed = 0
            for year in range(starting_year, current_year_plus_1):
                for week in range(1, start_week + 1):
                    url = base_url.format(year=year, week=week)
                    print(f"🔄 Scraping data for {year} Week {week} ...")
                    week_data = scrape_data(url)
        
                    for row in week_data:
                        row["Year"] = year
                        row["Week"] = f"Week {week}"
                        all_data.append(row)
        
                    completed += 1
                    time.sleep(2)  # Delay between requests
        
            print("✅ Data scraping complete!")
        
            return all_data
        print("Collecting Live Public Pick Percentages...")
        all_data = scrape_all_data(starting_year, current_year_plus_1)
    
        print(f"Scraping complete! Retrieved {len(all_data)} rows.")
        
        # Convert the list of dictionaries to a DataFrame
        public_pick_df = pd.DataFrame(all_data)
        
        # Cleanup the scraped data
        public_pick_df['Team'] = public_pick_df['Team'].str.replace(r'\s\(L\)', '', regex=True)
        public_pick_df['Team'] = public_pick_df['Team'].str.replace(r'\s\(W\)', '', regex=True)
        public_pick_df['Opponent'] = public_pick_df['Opponent'].str.replace('@', '', regex=True)
        public_pick_df['Opponent'] = public_pick_df['Opponent'].str.replace(r'[\t\n\+\-]', '', regex=True)
        public_pick_df['Opponent'] = (
            public_pick_df['Opponent']
            .str.strip() # Strip whitespace
            .str[:3]      # Get the first 3 characters
            # Use regex to replace the 3rd character (index 2) with an empty string ('')
            # if the 3rd character is a digit (\d).
            .str.replace(r'^(.{2})\d$', r'\1', regex=True)
        )
        
        public_pick_df = public_pick_df[public_pick_df['Opponent'] != 'BYE']
        
        public_pick_df = public_pick_df.drop_duplicates()
        
        public_pick_df.to_csv(f"contest-historical-data/raw-public-pick-data{target_year}.csv", index = False)
        
        # ==============================================================================
        # SECTION 2: API DATA COLLECTION (REPLACED BY nflreadpy)
        # ==============================================================================
        
        print(f"\nFetching NFL schedule and game results using nflreadpy from {starting_year} to {current_year}...")
        
        # Load the schedule data.
        # The object returned here is a Polars DataFrame.
        schedule_data_pl = nfl.load_schedules(list(range(starting_year, current_year + 1)))
        # --- Data Processing using POLARS FILTERING ---
        
        # Filter 1: Exclude in-season future games (those with game_id ending in _XX)
        # Use the .filter() method and the Polars `~` (NOT) operator
        schedule_data_pl = schedule_data_pl.filter(
            ~pl.col('game_id').str.contains(r'\_[0-9]{2}$')
        )
        
        # Filter 2: Filter only Regular Season games
        schedule_data_pl = schedule_data_pl.filter(
            pl.col('game_type') == 'REG'
        )
        
        # CONVERT TO PANDAS DATAFRAME BEFORE PROCEEDING
        completed_games = schedule_data_pl.to_pandas()
        
        
        # --- Data Processing to Match Your Old API Output Structure (Now back in Pandas) ---
        
        # Prepare columns for Winner/Loser determination and abbreviation mapping
        # This part is now safe because `completed_games` is a Pandas DataFrame
        completed_games.rename(columns={
            'gameday': 'Calendar Date',
            'week': 'Week', 
            'home_team': 'Home Team',
            'away_team': 'Away Team',
            'home_score': 'Home Score',
            'away_score': 'Away Score'
        }, inplace=True)
        
        # Function to determine winner/loser
        def determine_result(row):
            home_score = row['Home Score']
            away_score = row['Away Score']
            if home_score > away_score:
                return row['Home Team'], row['Away Team'], home_score, away_score
            elif away_score > home_score:
                return row['Away Team'], row['Home Team'], away_score, home_score
            else:
                # Note: nflreadpy data handles ties by having equal scores
                return 'Tie', 'Tie', home_score, home_score
        
        # Apply the function
        results = completed_games.apply(determine_result, axis=1, result_type='expand')
        results.columns = ['Winner/tie', 'Loser/tie', 'PtsW', 'PtsL']
        
        # Merge the results back
        df_nflreadpy_schedule = pd.concat([completed_games, results], axis=1)
        
        # Select and reorder columns to match your original script's output
        df_api_schedule = df_nflreadpy_schedule[[
            'season', 'Week', 'Calendar Date', 'Home Team', 'Away Team', 'Winner/tie', 'Loser/tie', 'PtsW', 'PtsL'
        ]].copy()
        
        # Rename the season column to Year
        df_api_schedule.rename(columns={'season': 'Year'}, inplace=True)
        
        # Drop any rows with NaN in critical columns (e.g., games not fully recorded)
        df_api_schedule.dropna(subset=['Winner/tie', 'Loser/tie'], inplace=True)
        
        # Convert to string and clean up data types
        df_api_schedule['Week'] = df_api_schedule['Week'].astype(int)
        
        df_api_schedule['Calendar Date'] = pd.to_datetime(df_api_schedule['Calendar Date'], errors='coerce')
        df_api_schedule['Calendar Date'] = df_api_schedule['Calendar Date'].dt.strftime('%Y-%m-%d')
        
    
        df_api_schedule['Home Team'] = df_api_schedule['Home Team'].replace('LA', 'LAR')
        df_api_schedule['Home Team'] = df_api_schedule['Home Team'].replace('WSH', 'WAS')
        df_api_schedule['Away Team'] = df_api_schedule['Away Team'].replace('LA', 'LAR')
        df_api_schedule['Away Team'] = df_api_schedule['Away Team'].replace('WSH', 'WAS')
        
        df_api_schedule = df_api_schedule.drop_duplicates()
        
        df_api_schedule.to_csv("df_api_schedule.csv", index = False)
        # ==============================================================================
        # SECTION 3: DATA CLEANING AND MERGE (ADJUSTED FOR nflreadpy COLUMN NAMES)
        # ==============================================================================
        
        # Your 'teams' dictionary for mapping is now **redundant for the schedule data**
        # since nflreadpy already uses the abbreviations (e.g., ARI, BAL) that your
        # web-scraped data uses. This simplifies the code significantly!        
        
        # Existing cleanup of the scraped data
        public_pick_df = public_pick_df.replace(r'\u00A0\(W\)', '', regex=True)
        public_pick_df = public_pick_df.replace(r'\u00A0\(L\)', '', regex=True)
        public_pick_df = public_pick_df.replace(r'\u00A0\(tie\)', '', regex=True)
        public_pick_df = public_pick_df.replace(r'\u00A0\(PPD\)', '', regex=True)
        public_pick_df = public_pick_df.replace('--', '0.0%', regex=True)
        # Select the desired columns
        public_pick_df = public_pick_df[['EV', 'Win %', 'Pick %', 'Team', 'Opponent', 'Future Value (Stars)', 'Year', 'Week']]
        
        # Convert to numeric
        public_pick_df['Win %'] = pd.to_numeric(public_pick_df['Win %'].str.rstrip('%')) / 100
        public_pick_df['Pick %'] = pd.to_numeric(public_pick_df['Pick %'].str.rstrip('%')) / 100
        public_pick_df['Pick %'].fillna(0.0, inplace=True)
        public_pick_df['Public Pick %'] = public_pick_df['Pick %']
        
        # Convert 'Week' to integer representing the week number
        public_pick_df['Week'] = public_pick_df['Week'].str.replace('Week ', '').astype(int)
    
        # df['Week'] = pd.to_numeric(df['Week']) # This is now redundant after astype(int)
        
        # Use your existing 'teams' dictionary for *Division* mapping (still needed)
        teams2 = {
            # ... (Keep your original 'teams' dictionary here for Division mapping)
            'ARI': ['Arizona Cardinals', 'State Farm Stadium', 33.5277, -112.262608, 'America/Denver', 'NFC West'],
            'ATL': ['Atlanta Falcons', 'Mercedez-Benz Stadium', 33.757614, -84.400972, 'America/New_York', 'NFC South'],
            'BAL': ['Baltimore Ravens', 'M&T Stadium', 39.277969, -76.622767, 'America/New_York', 'AFC North'],
            'BUF': ['Buffalo Bills', 'Highmark Stadium', 42.773739, -78.786978, 'America/New_York', 'AFC East'],
            'CAR': ['Carolina Panthers', 'Bank of America Stadium', 35.225808, -80.852861, 'America/New_York', 'NFC South'],
            'CHI': ['Chicago Bears', 'Soldier Field', 41.862306, -87.616672, 'America/Chicago', 'NFC North'],
            'CIN': ['Cincinnati Bengals', 'Paycor Stadium', 39.095442, -84.516039, 'America/New_York', 'AFC North'],
            'CLE': ['Cleveland Browns', 'Cleveland Browns Stadium', 41.506022, -81.699564, 'America/New_York', 'AFC North'],
            'DAL': ['Dallas Cowboys', 'AT&T Stadium', 32.747778, -97.092778, 'America/Chicago', 'NFC East'],
            'DEN': ['Denver Broncos', 'Empower Field at Mile High', 39.743936, -105.020097, 'America/Denver', 'AFC West'],
            'DET': ['Detroit Lions', 'Ford Field', 42.340156, -83.045808, 'America/New_York', 'NFC North'],
            'GB': ['Green Bay Packers', 'Lambeau Field', 44.501306, -88.062167, 'America/Chicago', 'NFC North'],
            'HOU': ['Houston Texans', 'NRG Stadium', 29.684781, -95.410956, 'America/Chicago', 'AFC South'],
            'IND': ['Indianapolis Colts', 'Lucas Oil Stadium', 39.760056, -86.163806, 'America/New_York', 'AFC South'],
            'JAX': ['Jacksonville Jaguars', 'Everbank Stadium', 30.323925, -81.637356, 'America/New_York', 'AFC South'],
            'KC': ['Kansas City Chiefs', 'Arrowhead Stadium', 39.048786, -94.484566, 'America/Chicago', 'AFC West'],
            'LV': ['Las Vegas Raiders', 'Allegiant Stadium', 36.090794, -115.183952, 'America/Los_Angeles', 'AFC West'],
            'LAC': ['Los Angeles Chargers', 'SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'AFC West'],
            'LAR': ['Los Angeles Rams', 'SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'NFC West'],
    #        'LA': ['Los Angeles Rams', 'SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'NFC West'],
            'MIA': ['Miami Dolphins', 'Hard Rock Stadium', 25.957919, -80.238842, 'America/New_York', 'AFC East'],
            'MIN': ['Minnesota Vikings', 'U.S Bank Stadium', 44.973881, -93.258094, 'America/Chicago', 'NFC North'],
            'NE': ['New England Patriots', 'Gillette Stadium', 42.090925, -71.26435, 'America/New_York', 'AFC East'],
            'NO': ['New Orleans Saints', 'Caesars Superdome', 29.950931, -90.081364, 'America/Chicago', 'NFC South'],
            'NYG': ['New York Giants', 'MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'NFC East'],
            'NYJ': ['New York Jets', 'MetLife Stadium', 40.812194, -74.076983, 'America/New_York', 'AFC East'],
            'PHI': ['Philadelphia Eagles', 'Lincoln Financial Field', 39.900775, -75.167453, 'America/New_York', 'NFC East'],
            'PIT': ['Pittsburgh Steelers', 'Acrisure Stadium', 40.446786, -80.015761, 'America/New_York', 'AFC North'],
            'SF': ['San Francisco 49ers', 'Levi\'s Stadium', 37.713486, -122.386256, 'America/Los_Angeles', 'NFC West'],
            'SEA': ['Seattle Seahawks', 'Lumen Field', 47.595153, -122.331625, 'America/Los_Angeles', 'NFC West'],
            'TB': ['Tampa Bay Buccaneers', 'Raymomd James Stadium', 27.975967, -82.50335, 'America/New_York', 'NFC South'],
            'TEN': ['Tennessee Titans', 'Nissan Stadium', 36.166461, -86.771289, 'America/Chicago', 'AFC South'],
            'WAS': ['Washington Commanders', 'FedExField', 38.907697, -76.864517, 'America/New_York', 'NFC East'],
    #        'WSH': ['Washington Commanders', 'FedExField', 38.907697, -76.864517, 'America/New_York', 'NFC East']
        }
        
        # Division mapping
        public_pick_df['Team'] = public_pick_df['Team'].replace('WSH', 'WAS')
        public_pick_df['Opponent'] = public_pick_df['Opponent'].replace('WSH', 'WAS')
        public_pick_df['Team Division'] = public_pick_df['Team'].map(lambda team: teams2.get(team, ['', '', '', '', '', ''])[5])
        public_pick_df['Opponent Division'] = public_pick_df['Opponent'].map(lambda opponent: teams2.get(opponent, ['', '', '', '', '', ''])[5])
        public_pick_df['Divisional Matchup?'] = (public_pick_df['Team Division'] == public_pick_df['Opponent Division']).astype(int)
    
    
        # Load the historical data from the file created by nflreadpy
        away_data_df = df_api_schedule
        away_data_df['Calendar Date'] = pd.to_datetime(away_data_df['Calendar Date'])
        
        # Initialization of new columns
        public_pick_df['Away Team'] = 0
        public_pick_df[['Availability', 'Calculated Current Week Alive Entries', 'Calculated Current Week Picks', 'Winning Team']] = [0,0,0,0]
        public_pick_df['Calendar Date'] = pd.NaT
        
        # Merge the dataframes directly (replacing the slow apply/lambda functions)
        
        # 1. Merge to get HOME/AWAY/WINNER
        merged_schedule = pd.merge(
            public_pick_df,
            away_data_df[['Year', 'Week', 'Home Team', 'Away Team', 'Winner/tie']],
            left_on=['Year', 'Week', 'Team'],
            right_on=['Year', 'Week', 'Home Team'],
            how='left',
            suffixes=('', '_home') # Suffix for Home/Away columns when 'Team' is Home
        )
        
        # Rename the column from the first merge to avoid a name conflict
        merged_schedule = merged_schedule.rename(columns={'Away Team_home': 'Opponent_from_home_merge'})
        
        
        # Merge again for when 'Team' is the Away Team
        merged_schedule = pd.merge(
            merged_schedule,
            away_data_df[['Year', 'Week', 'Home Team', 'Away Team', 'Winner/tie']],
            left_on=['Year', 'Week', 'Team'],
            right_on=['Year', 'Week', 'Away Team'],
            how='left',
            suffixes=('_home', '_away') # Suffix for Home/Away columns when 'Team' is Away
        )
        
        merged_schedule = merged_schedule.drop_duplicates(
            subset=['Year', 'Week', 'Team'],
            keep='first'
        ).reset_index(drop=True)
        
        
        # Populate 'Away Team' (binary) and 'Winning Team' (binary)
        public_pick_df['Away Team'] = (
            merged_schedule['Away Team_away'].notna()
        ).astype(int).values
        
        
        # Winning Team Logic:
        # The team is the winner if it matches the 'Winner/tie' column from either merge
        public_pick_df['Winning Team'] = (
            (merged_schedule['Winner/tie_home'] == merged_schedule['Team']) | 
            (merged_schedule['Winner/tie_away'] == merged_schedule['Team'])
        ).fillna(0).astype(int).values
        
        # 2. Merge to get Calendar Date (using the cleaner merge logic from your original script)
        home_dates = away_data_df[['Year', 'Week', 'Home Team', 'Calendar Date']].copy()
        home_dates.rename(columns={'Home Team': 'Team_schedule', 'Calendar Date': 'Matched_Date'}, inplace=True)
        away_dates = away_data_df[['Year', 'Week', 'Away Team', 'Calendar Date']].copy()
        away_dates.rename(columns={'Away Team': 'Team_schedule', 'Calendar Date':'Matched_Date'}, inplace=True)
        
        
        
        schedule_lookup = pd.concat([home_dates, away_dates]).drop_duplicates(
            subset=['Year', 'Week', 'Team_schedule']
        ).reset_index(drop=True)
        
        schedule_lookup['Team_schedule'] = schedule_lookup['Team_schedule'].replace('LA', 'LAR')
        # Merge with the lookup table for the date
        merged_for_calendar_date = pd.merge(
            public_pick_df.reset_index(), # Reset index to avoid merge issues
            schedule_lookup,
            left_on=['Year', 'Week', 'Team'],
            right_on=['Year', 'Week', 'Team_schedule'],
            how='left'
        )
        public_pick_df['Calendar Date'] = merged_for_calendar_date.set_index('index')['Matched_Date'].values
        # Assuming your conversion worked, or you fix it like we discussed:
        public_pick_df['Calendar Date'] = pd.to_datetime(public_pick_df['Calendar Date'], format='%Y-%m-%d')
        #df['Calendar Date_String'] = df['Calendar Date'].dt.strftime('%m/%d/%Y')
        
        # Drop rows where 'Team Division' or 'Opponent Division' is an empty string
        public_pick_df = public_pick_df[public_pick_df['Team Division'] != '']
        public_pick_df = public_pick_df[public_pick_df['Opponent Division'] != '']
        
        public_pick_df = public_pick_df[public_pick_df['Year'] == target_year]
        
        public_pick_df = public_pick_df.drop_duplicates()
        
        public_pick_df['Calendar Date'] = pd.to_datetime(public_pick_df['Calendar Date'], format='%Y-%m-%d')
        
        # ... (The final date manipulation logic remains the same)
        pre_circa_dates = {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019}
        is_not_in_pre_circa = ~public_pick_df['Year'].isin(pre_circa_dates)
        public_pick_df = public_pick_df[is_not_in_pre_circa]
        
        # Final date manipulation (e.g., correcting Thanksgiving/Christmas week numbers)
        # NOTE: The df.loc assignments must be run *after* the Calendar Date is populated.
    
        condition_2026_date = (public_pick_df['Year'] == 2026) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2026-11-28'))
        public_pick_df.loc[condition_2026_date, 'Week'] += 1
        condition_2026_week = (public_pick_df['Year'] == 2026) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2025-12-26'))
        public_pick_df.loc[condition_2026_week, 'Week'] += 1
        
        # For Year 2025
        condition_2025_date = (public_pick_df['Year'] == 2025) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2025-11-29'))
        public_pick_df.loc[condition_2025_date, 'Week'] += 1
        condition_2025_week = (public_pick_df['Year'] == 2025) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2025-12-26'))
        public_pick_df.loc[condition_2025_week, 'Week'] += 1
        
        # For Year 2024
        condition_2024_date = (public_pick_df['Year'] == 2024) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2024-11-30'))
        public_pick_df.loc[condition_2024_date, 'Week'] += 1
        condition_2024_week = (public_pick_df['Year'] == 2024) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2024-12-27'))
        public_pick_df.loc[condition_2024_week, 'Week'] += 1
        
        # For Year 2023
        condition_2023_date = (public_pick_df['Year'] == 2023) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2023-11-25'))
        public_pick_df.loc[condition_2023_date, 'Week'] += 1
        condition_2023_week = (public_pick_df['Year'] == 2023) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2023-12-25'))
        public_pick_df.loc[condition_2023_week, 'Week'] += 1
        
        # For Year 2022
        condition_2022_date = (public_pick_df['Year'] == 2022) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2022-11-25'))
        public_pick_df.loc[condition_2022_date, 'Week'] += 1
        condition_2022_week = (public_pick_df['Year'] == 2022) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2022-12-25'))
        public_pick_df.loc[condition_2022_week, 'Week'] += 1
        
        # For Year 2021
        condition_2021_date = (public_pick_df['Year'] == 2021) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2021-11-26'))
        public_pick_df.loc[condition_2021_date, 'Week'] += 1
        
        condition_2021_week = (public_pick_df['Year'] == 2021) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2021-12-26'))
        public_pick_df.loc[condition_2021_week, 'Week'] += 1
        
        # For Year 2020
        condition_2020_date = (public_pick_df['Year'] == 2020) & (public_pick_df['Calendar Date'] >= pd.to_datetime('2020-11-27'))
        public_pick_df.loc[condition_2020_date, 'Week'] += 1
        
        public_pick_df['EV'] = 0
      
        public_pick_df = public_pick_df.drop_duplicates()
    
        
        # ==============================================================================
        # SECTION 4: POPULATE week_df WITH PUBLIC PICK DATA
        # ==============================================================================
        
        # This assumes 'week_df' already exists in your environment, as mentioned.
        
        print("Creating reverse team map for lookup...")
        # Create a reverse map: {"Carolina Panthers": "CAR", "Chicago Bears": "CHI", ...}
        # This is VITAL for linking week_df (full names) to public_pick_df (abbreviations)
        try:
            team_name_to_abbr_map = {details[0]: abbr for abbr, details in teams2.items()}
        except NameError:
            print("CRITICAL ERROR: 'teams' dictionary not defined. Cannot create lookup map.")
            # Handle this error, perhaps by exiting
            team_name_to_abbr_map = {}
        
        def get_public_pick_percent(row, team_type):
            """
            Looks up the public pick percentage from 'public_pick_df' for a team.
            
            'row' is a row from week_df.
            'team_type' is either 'home' or 'away'.
            """
            
            # 1. Get week number (e.g., "Week 10" -> 10)
            week_num = row["Week"]
            
            # 2. Get the full team name and identify if we seek a home or away team
            if team_type == 'home':
                team_name = row["Home Team"]
                is_away_flag = 0 # The 'Away Team' flag in public_pick_df should be 0
            elif team_type == 'away':
                team_name = row["Away Team"]
                is_away_flag = 1 # The 'Away Team' flag in public_pick_df should be 1
            else:
                return np.nan # Invalid team_type
    
            # 3. Convert the full team name ("Carolina Panthers") to its abbreviation ("CAR")
            team_abbr = team_name_to_abbr_map.get(team_name)
            
            if not team_abbr:
                # print(f"Warning: Could not find abbreviation for {team_name}")
                return np.nan # Team name not in our map
    
            # 4. Find the matching row in public_pick_df
            # We filter by the integer week, the team abbreviation, and the home/away flag
            match = public_pick_df[
                (public_pick_df["Week"] == week_num) &
                (public_pick_df["Team"] == team_abbr) &
                (public_pick_df["Away Team"] == is_away_flag)
            ]
    
            # 5. Return the value if found, otherwise return NaN
            if not match.empty:
                # .values[0] gets the first (and should be only) matching value
                return match["Public Pick %"].values[0]
            else:
                # No match found in public_pick_df for this team/week
                return np.nan
    
        
        print("Populating 'Away Team Public Pick %' in week_df...")
        df["Away Team Public Pick %"] = df.apply(
            lambda row: get_public_pick_percent(row, 'away'),
            axis=1
        )
        
        print("Populating 'Home Team Public Pick %' in week_df...")
        df["Home Team Public Pick %"] = df.apply(
            lambda row: get_public_pick_percent(row, 'home'),
            axis=1
        )
    
        print("Finished populating public pick percentages.")
    
        # Save the consolidated DataFrame to a single CSV file
    
        consolidated_csv_file = "nfl-schedules/nfl_schedule_rankings_travel_odds_circa.csv"
        schedule_df = df
        df.to_csv(consolidated_csv_file, index=False)    
        collect_schedule_travel_ranking_data_nfl_schedule_df = df
    
        return collect_schedule_travel_ranking_data_nfl_schedule_df
        
    
    collect_schedule_travel_ranking_data_df = collect_schedule_travel_ranking_data(schedule_df)
    
    # --- CONFIGURATION ---
    SIMULATIONS = 1000
    HISTORY_DAYS = 840
    CURRENT_SEASON = target_year
    DECAY_RATE = 0.00475
    GARBAGE_MIN = 0.05
    GARBAGE_MAX = 0.95
    
    # Context
    WIND_THRESHOLD = 15
    WIND_PASS_IMPACT = 0.85
    HFA_DEFENSE_BOOST_DEFAULT = 0.03
    
    TEAM_MAP = {
        'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU',
        'LAR': 'LA', 'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'
    }
    
    def weighted_avg_and_std(values, weights):
        if len(values) == 0: return 0.0, 0.0
        average = np.average(values, weights=weights)
        variance = np.average((values-average)**2, weights=weights)
        return average, np.sqrt(variance)
    
    def format_clock(seconds, phase="REG"):
        seconds = max(0, seconds)
        m, s = divmod(int(seconds), 60)
        if phase == "OT": return f"OT {m:02d}:{s:02d}"
        if seconds > 1800: return f"1H {m:02d}:{s:02d}"
        return f"2H {m:02d}:{s:02d}"
    
    def format_field(yardline, possession):
        if yardline <= 50: return f"{possession} {int(yardline)}"
        return f"Opp {int(100-yardline)}"
    
    class AdvancedNFLSimulator:
        def __init__(self):
            self.pbp = pd.DataFrame()
            self.profiles = {}
            self.def_mults = {}
            self.hfa_map = {} 
            self.league_avgs = {}
        
        def load_data(self, hfa_file="nfl-power-ratings/nfl_hfa_ratings.csv"):
            print("--- Loading Data & Calculating Advanced Profiles ---")
            try:
                hfa_df = pd.read_csv(hfa_file)
                self.hfa_map = hfa_df.set_index('Team')['HFA SR'].to_dict()
            except FileNotFoundError:
                self.hfa_map = {}
    
            seasons = [CURRENT_SEASON-2, CURRENT_SEASON-1, CURRENT_SEASON]
            try:
                df = nfl.load_pbp(seasons=seasons).to_pandas()
            except:
                print("CRITICAL ERROR: Could not load PBP data.")
                return
    
            df['game_date'] = pd.to_datetime(df['game_date'])
            cutoff = pd.to_datetime(date_str) - timedelta(days=HISTORY_DAYS)
            df = df[df['game_date'] >= cutoff].copy()
            df = df[(df['wp'] >= GARBAGE_MIN) & (df['wp'] <= GARBAGE_MAX)]
            
            current_date = pd.to_datetime(date_str)
            df['days_ago'] = (current_date - df['game_date']).dt.days.clip(lower=0)
            df['time_weight'] = np.exp(-DECAY_RATE * df['days_ago'])
            
            df = df[df['play_type'].isin(['run', 'pass', 'punt', 'field_goal', 'no_play'])]
            self.pbp = df
            
            self._build_profiles()
    
        def _build_profiles(self):
            print("--- Building Profiles ---")
            
            # 1. DISTANCE & CONTEXT
            def get_dist_bucket(dist):
                if dist <= 3: return 'short'
                if dist <= 7: return 'med'
                return 'long'
            self.pbp['dist_bucket'] = self.pbp['ydstogo'].apply(get_dist_bucket)
            
            self.pbp['score_diff'] = self.pbp['posteam_score'] - self.pbp['defteam_score']
            def get_context(row):
                if row['score_diff'] > 8: return 'leading'
                if row['score_diff'] < -8: return 'trailing'
                return 'neutral'
            self.pbp['context'] = self.pbp.apply(get_context, axis=1)
    
            # 2. PLAY CALLING
            league_groups = self.pbp.groupby(['down', 'dist_bucket', 'context'])
            self.league_pass_rates = {}
            for name, group in league_groups:
                is_pass = (group['play_type'] == 'pass').astype(int)
                self.league_pass_rates[name] = np.average(is_pass, weights=group['time_weight'])
    
            playcalling_dict = {}
            team_groups = self.pbp.groupby(['posteam', 'down', 'dist_bucket', 'context'])
            for name, group in team_groups:
                is_pass = (group['play_type'] == 'pass').astype(int)
                playcalling_dict[name] = np.average(is_pass, weights=group['time_weight'])
                
            # 3. PACE & CLOCK LOGIC
            self.pbp['next_snap_time'] = self.pbp.groupby(['game_id', 'drive'])['game_seconds_remaining'].shift(-1)
            self.pbp['seconds_consumed'] = self.pbp['game_seconds_remaining'] - self.pbp['next_snap_time']
            
            def get_pace_type(row):
                if row['play_type'] == 'run': return 'run'
                if row['play_type'] == 'pass':
                    return 'pass_complete' if row['complete_pass'] == 1 else 'pass_incomplete'
                return 'other'
            self.pbp['pace_type'] = self.pbp.apply(get_pace_type, axis=1)
            valid_pace = self.pbp[(self.pbp['seconds_consumed'] >= 0) & (self.pbp['seconds_consumed'] < 60)]
            
            pace_stats = valid_pace.groupby(['posteam', 'pace_type']).apply(
                lambda x: np.average(x['seconds_consumed'], weights=x['time_weight'])
            )
    
            oob_plays = self.pbp[self.pbp['play_type'].isin(['run', 'pass'])]
            oob_rates = {}
            for team, group in oob_plays.groupby('posteam'):
                 oob_rates[team] = np.average(group['out_of_bounds'].fillna(0), weights=group['time_weight'])
            
            incomplete_plays = valid_pace[valid_pace['pace_type'] == 'pass_incomplete']
            avg_play_duration = incomplete_plays['seconds_consumed'].mean()
            if np.isnan(avg_play_duration): avg_play_duration = 6.0
    
            # 4. EFFICIENCY
            self.pbp['field_zone'] = np.where(self.pbp['yardline_100'] <= 20, 'redzone', 'open')
            efficiency_dict = {}
            eff_plays = self.pbp[self.pbp['play_type'].isin(['run', 'pass'])]
            
            for (team, zone), team_group in eff_plays.groupby(['posteam', 'field_zone']):
                # RUN
                runs = team_group[team_group['play_type'] == 'run']
                if len(runs) > 0:
                    r_mu, r_sigma = weighted_avg_and_std(runs['yards_gained'].fillna(0).values, runs['time_weight'].values)
                    r_fumble = np.average(runs['fumble_lost'], weights=runs['time_weight'])
                else:
                    r_mu, r_sigma, r_fumble = 3.5, 3.0, 0.01
                efficiency_dict[(team, zone, 'run')] = {'mu': r_mu, 'sigma': r_sigma, 'fumble': r_fumble}
    
                # PASS
                passes = team_group[team_group['play_type'] == 'pass']
                if len(passes) > 0:
                    sack_rate = np.average(passes['sack'], weights=passes['time_weight'])
                    non_sacks = passes[passes['sack'] == 0]
                    if len(non_sacks) > 0:
                        comp_rate = np.average(non_sacks['complete_pass'], weights=non_sacks['time_weight'])
                        int_rate = np.average(non_sacks['interception'], weights=non_sacks['time_weight'])
                        completions = non_sacks[non_sacks['complete_pass'] == 1]
                        if len(completions) > 0:
                            p_mu, p_sigma = weighted_avg_and_std(completions['yards_gained'].values, completions['time_weight'].values)
                            p_fumble = np.average(completions['fumble_lost'], weights=completions['time_weight'])
                        else:
                            p_mu, p_sigma, p_fumble = 10.0, 5.0, 0.01
                    else:
                        comp_rate, int_rate, p_fumble, p_mu, p_sigma = 0.6, 0.03, 0.01, 7.0, 5.0
                else:
                    sack_rate, comp_rate, int_rate, p_fumble, p_mu, p_sigma = 0.07, 0.6, 0.03, 0.01, 7.0, 5.0
    
                efficiency_dict[(team, zone, 'pass')] = {
                    'mu': p_mu, 'sigma': p_sigma, 'fumble': p_fumble, 
                    'intercept': int_rate, 'complete': comp_rate, 'sack': sack_rate
                }
    
            # 5. DEFENSE MULTS
            self.def_mults = {}
            league_run = np.average(eff_plays[eff_plays['play_type']=='run']['yards_gained'], weights=eff_plays[eff_plays['play_type']=='run']['time_weight'])
            league_pass = np.average(eff_plays[(eff_plays['play_type']=='pass') & (eff_plays['complete_pass']==1)]['yards_gained'], 
                                     weights=eff_plays[(eff_plays['play_type']=='pass') & (eff_plays['complete_pass']==1)]['time_weight'])
            
            for team, group in eff_plays.groupby('defteam'):
                self.def_mults[team] = {}
                tr = group[group['play_type']=='run']
                raw_run_mult = (np.average(tr['yards_gained'], weights=tr['time_weight']) / league_run) if len(tr)>0 else 1.0
                self.def_mults[team]['run'] = (raw_run_mult * 0.8) + (1.0 * 0.2) # Regress 20% to League Avg
                tp = group[(group['play_type']=='pass') & (group['complete_pass']==1)]
                raw_pass_mult = (np.average(tp['yards_gained'], weights=tp['time_weight']) / league_pass) if len(tp)>0 else 1.0
                self.def_mults[team]['pass'] = (raw_pass_mult * 0.8) + (1.0 * 0.2) # Regress 20% to League Avg
    
            # 6. PENALTIES
            pen_dict = {}
            for team, group in self.pbp.groupby('posteam'):
                off_pen = group[(group['penalty'] == 1) & (group['penalty_team'] == team)]
                pen_dict[(team, 'off')] = np.sum(off_pen['time_weight']) / group['time_weight'].sum()
            
            def_pen_stats = {}
            for team, group in self.pbp.groupby('defteam'):
                def_pen_plays = group[(group['penalty'] == 1) & (group['penalty_team'] == team)]
                total_rate = np.sum(def_pen_plays['time_weight']) / group['time_weight'].sum()
                pen_dict[(team, 'def')] = total_rate
                
                if len(def_pen_plays) > 0:
                    is_dpi = def_pen_plays['penalty_type'].str.contains('Pass Interference', na=False, case=False)
                    is_major = (def_pen_plays['penalty_yards'] == 15) & (~is_dpi)
                    w = def_pen_plays['time_weight']
                    dpi_weight = w[is_dpi].sum()
                    major_weight = w[is_major].sum()
                    total_weight = w.sum()
                    dpi_share = dpi_weight / total_weight
                    major_share = major_weight / total_weight
                    dpi_yards = def_pen_plays[is_dpi]['penalty_yards']
                    if len(dpi_yards) > 0:
                        d_mu = np.average(dpi_yards, weights=w[is_dpi])
                        d_std = np.sqrt(np.average((dpi_yards - d_mu)**2, weights=w[is_dpi]))
                    else:
                        d_mu, d_std = 15.0, 10.0
                    def_pen_stats[team] = {'dpi_share': dpi_share, 'major_share': major_share, 'dpi_mu': d_mu, 'dpi_std': d_std}
                else:
                    def_pen_stats[team] = {'dpi_share': 0.1, 'major_share': 0.15, 'dpi_mu': 15.0, 'dpi_std': 10.0}
    
            # 7. PUNTING
            punt_stats = {}
            punts = self.pbp[self.pbp['play_type'] == 'punt'].copy()
            punts['net_yards'] = punts['kick_distance'] - punts['return_yards'].fillna(0)
            for team, group in punts.groupby('posteam'):
                 p_mu = np.average(group['net_yards'].fillna(40), weights=group['time_weight'])
                 p_std = np.sqrt(np.average((group['net_yards'].fillna(40) - p_mu)**2, weights=group['time_weight']))
                 punt_stats[team] = {'mu': p_mu, 'sigma': p_std}
                 
            # 8. KICKING
            kicking_stats = {}
            fgs = self.pbp[self.pbp['play_type'] == 'field_goal'].copy()
            for team, group in fgs.groupby('posteam'):
                made_fgs = group[group['field_goal_result'] == 'made']
                max_made = made_fgs['kick_distance'].max()
                if np.isnan(max_made): max_made = 50.0
                
                short_try = group[group['kick_distance'] < 40]
                short_acc = np.average((short_try['field_goal_result']=='made'), weights=short_try['time_weight']) if len(short_try)>0 else 0.98
                
                med_try = group[(group['kick_distance'] >= 40) & (group['kick_distance'] < 50)]
                med_acc = np.average((med_try['field_goal_result']=='made'), weights=med_try['time_weight']) if len(med_try)>0 else 0.85
    
                long_try = group[group['kick_distance'] >= 50]
                long_acc = np.average((long_try['field_goal_result']=='made'), weights=long_try['time_weight']) if len(long_try)>0 else 0.65
                
                kicking_stats[team] = {'max_made': max_made, 'short_acc': short_acc, 'med_acc': med_acc, 'long_acc': long_acc}
    
            # 9. BREAKAWAY RUN RATES
            # Define a breakaway as a run of 15+ yards
            run_plays = self.pbp[self.pbp['play_type'] == 'run']
            breakaway_plays = run_plays[run_plays['yards_gained'] >= 15]
            
            # Calculate League Average Rate first
            if len(run_plays) > 0:
                league_bk_rate = len(breakaway_plays) / len(run_plays)
            else:
                league_bk_rate = 0.035 # Default fallback (3.5%)
    
            breakaway_stats = {}
            for team, group in run_plays.groupby('posteam'):
                n_runs = len(group)
                n_breakaways = len(group[group['yards_gained'] >= 15])
                
                # REGRESSION TO THE MEAN:
                # We add 50 "league average runs" to the team's sample.
                # This prevents a team with few runs from having a wild 0% or 10% rate.
                regressed_rate = (n_breakaways + (50 * league_bk_rate)) / (n_runs + 50)
                breakaway_stats[team] = regressed_rate
    
            # Store in profiles (Add this to your self.profiles dictionary below)
            self.profiles['breakaway_run'] = breakaway_stats
            self.profiles['league_breakaway_run'] = league_bk_rate        
            
            
            self.profiles = {
                'efficiency': efficiency_dict,
                'pace': pace_stats.to_dict(),
                'penalties': pen_dict,
                'penalty_details': def_pen_stats,
                'punting': punt_stats,
                'kicking': kicking_stats,
                'playcalling': playcalling_dict,
                'oob_rates': oob_rates,
                'play_duration': avg_play_duration,
                'breakaway_run': breakaway_stats,
                'league_breakaway_run': league_bk_rate
            }
    
        def _resolve_play_outcome(self, off, def_, zone, ptype, stats, def_mult, hfa_impact, 
                                wind_speed, temp, is_rain, is_snow, is_dome, verbose):
            """Calculates the result of a play, injecting 'Breakaway' logic to fix low totals.
            Returns: (yards, is_complete, is_turnover, desc_tag)
            """
    
    # --- APPLY WEATHER PHYSICS ---
#####            if not is_dome:
                # 1. WIND EFFECTS
#####                if wind_speed > 15:
                    # Harder to throw accurate deep balls
#####                    if ptype == 'pass':
#####                        stats['complete'] -= 0.05
#####                        stats['mu'] -= 1.0 # Average depth of target drops
#####                if wind_speed > 25:
#####                    if ptype == 'pass':
#####                        stats['complete'] -= 0.12
#####                        stats['intercept'] += 0.01 # Tips/Overthrows
                
                # 2. PRECIPITATION EFFECTS (Ball Security & Catching)
#####                if is_rain:
                    # "Slick Ball"
#####                    stats['fumble'] *= 1.3      # 50% increase in fumble risk
#####                    if ptype == 'pass':
#####                        stats['complete'] -= 0.06 # Drops
#####                        stats['mu'] *= 0.9        # Players slip, less YAC
#####                    else:
#####                        stats['mu'] *= 0.95       # Slower footing
                
#####                elif is_snow:
                    # "Chaos" Factor
#####                    stats['fumble'] *= 1.4
#####                    if ptype == 'pass':
#####                        stats['complete'] -= 0.08 # Visibility/Tracking issues
#####                    else:
                        # OFFENSIVE ADVANTAGE in Snow (Run Game)
                        # Defenders react slower and slip.
                        # RBs know where they are going.
#####                        stats['mu'] += 0.4 
    
                # 3. TEMPERATURE EFFECTS (The "Rock")
#####                if temp < 20:
#####                    stats['fumble'] *= 1.25 # Hits hurt more, ball is hard
#####                    if ptype == 'pass':
#####                        stats['complete'] -= 0.04 # Hard to grip/catch
    
            yards = 0
            is_complete = True
            is_turnover = False
            desc_tag = ""

            # Apply Defensive Multiplier & HFA to base efficiency
            # If defense is good (mult < 1.0), they reduce yardage.
            adjusted_mu = stats['mu'] * def_mult
        
            # --- RUN LOGIC ---
            if ptype == 'run':
            # 1. Check Fumble
                if np.random.random() < stats['fumble']:
                    is_turnover = True
                    yards = 0 # Fumbles usually happen at LOS or slight gain, simplifying to 0 for sim
                    desc_tag = "FUMBLE"
            
                # 2. Check BREAKAWAY (Team Specific Rate)
                else:
                    # RETRIEVE TEAM RATE HERE
                    # Fallback to league average if team not found
                    league_avg = self.profiles.get('league_breakaway_run', 0.035)
                    bk_prob = self.profiles['breakaway_run'].get(off, league_avg)
                
                    if np.random.random() < bk_prob:
                        # Log-normal distribution for breakaway yards
                        raw_yards = np.random.lognormal(3.0, 0.6) 
                        yards = int(max(15, raw_yards))
                        yards = min(yards, 99)
                        desc_tag = "BREAKAWAY RUN"
            
                # 3. Standard Run
                    else:
                        # Shift the distribution so 0 in Gamma = -3 yards in real life
                        shift = 3.0
                        shifted_mu = adjusted_mu + shift
                        
                        # Safety catch to prevent math errors if a team's mu is horribly negative
                        if shifted_mu <= 0.1: shifted_mu = 0.1 
                        
                        shape = (shifted_mu ** 2) / (stats['sigma'] ** 2)
                        scale = (stats['sigma'] ** 2) / shifted_mu
                        
                        raw_yards = np.random.gamma(shape, scale)
                        
                        # Subtract the shift back out
                        yards = int(raw_yards) - int(shift)

            # --- PASS LOGIC ---
            else:
                # 1. Check Sack
                if np.random.random() < stats['sack']:
                    yards = -7
                    is_complete = False
                    desc_tag = "SACK"
                    # Small chance of strip-sack
                    if np.random.random() < 0.015: 
                        is_turnover = True
                        desc_tag += " / FUMBLE"

                # 2. Check Interception
                elif np.random.random() < stats['intercept']:
                    is_turnover = True
                    is_complete = False # Technically incomplete stats-wise for yardage calc
                    yards = 0
                    desc_tag = "INTERCEPTION"

                # 3. Check Completion
                elif np.random.random() > stats['complete']:
                    is_complete = False
                    yards = 0
                    desc_tag = "INCOMPLETE"

                # 4. COMPLETED PASS
                else:
                    # Check BREAKAWAY (The Fix for Totals)
                    # ~7% of completions go for big yardage
                    if np.random.random() < 0.07:
                        # Normal dist centered on 35 yards, high variance
                        raw_yards = np.random.normal(35, 12)
                        yards = int(max(20, raw_yards)) # Minimum 20 yards for a "breakaway"
                        yards = min(yards, 99)
                        desc_tag = "DEEP BALL"
                    
                        # Add fumble chance on long run after catch
                        if np.random.random() < 0.01:
                            is_turnover = True
                            desc_tag += " / FUMBLE"
                    # 3. Standard Run (and apply similar to Standard Pass)
                    else:
                        # Shift the distribution so 0 in Gamma = -3 yards in real life
                        shift = 3.0
                        shifted_mu = adjusted_mu + shift
                        
                        # Safety catch to prevent math errors if a team's mu is horribly negative
                        if shifted_mu <= 0.1: shifted_mu = 0.1 
                        
                        shape = (shifted_mu ** 2) / (stats['sigma'] ** 2)
                        scale = (stats['sigma'] ** 2) / shifted_mu
                        
                        raw_yards = np.random.gamma(shape, scale)
                        
                        # Subtract the shift back out
                        yards = int(raw_yards) - int(shift)
                        # Standard fumble chance
                        if np.random.random() < stats['fumble']:
                            is_turnover = True
                            desc_tag = "FUMBLE"

            return yards, is_complete, is_turnover, desc_tag
    
        def _get_kickoff_start(self, team):
            # NFL Kickoff Return Distribution (Approximate)
            roll = np.random.random()
            
            if roll < 0.40: 
                return 35 # Standard Touchback (New Rules is 30)
            elif roll < 0.60:
                # Poor/Normal return
                # FIX: Changed sigma from 25 to 4
                return int(np.random.normal(18, 4)) 
            elif roll < 0.85:
                # Good return
                # FIX: Changed sigma from 35 to 5
                return int(np.random.normal(26, 5))
            elif roll < 0.95:
                # Great return
                return int(np.random.randint(35, 50))
            elif roll < 0.995:
                # Explosive return into opponent territory
                # Return yardline (e.g. 80 means own 80, which is opp 20)
                return int(np.random.randint(50, 85)) 
            else:
                # KICKOFF RETURN TOUCHDOWN (0.5% chance)
                return 100
    
        def simulate_matchup(self, home, away, wind_speed=0, temp=70, precip=0, is_dome=False, print_sample_game=False):
            results = []
    
            is_snow = (temp <= 32 and precip > 0)
            is_rain = (temp > 32 and precip > 0)
            
            h_lookup = TEAM_MAP.get(home, home)
            hfa_impact = self.hfa_map.get(h_lookup, HFA_DEFENSE_BOOST_DEFAULT)
            
            print(f"Simulating {home} vs {away} | HFA: {hfa_impact:.1%} | Wind: {wind_speed}mph")
            
            if print_sample_game:
                print(f"\n{'='*60}\nSAMPLE GAME LOG ({away} @ {home})\n{'='*60}")
                self._play_game(home, away, wind_speed, temp, is_rain, is_snow, is_dome, hfa_impact, verbose=True)
                print(f"{'='*60}\nEND SAMPLE LOG\n{'='*60}\n")
    
            for _ in range(SIMULATIONS):
                res = self._play_game(home, away, wind_speed, temp, is_rain, is_snow, is_dome, hfa_impact, verbose=False)
                results.append(res)
                
            return pd.DataFrame(results)
    
        def _attempt_pat(self, off, def_, scores, clock, phase, wind_speed, verbose):
            diff = scores[off] - scores[def_] 
            go_for_2 = False
            minutes_left = clock / 60.0
            is_late = (phase == 'REG' and minutes_left < 10) or (phase == 'OT')
            
            if is_late:
                if diff == -2: go_for_2 = True
                elif diff == -5: go_for_2 = True
                elif diff == -1: 
                    if minutes_left < 2: go_for_2 = True
                elif diff == 1: go_for_2 = True
                elif diff == 5: go_for_2 = True
            
            points_added = 0
            desc = ""
            
            if go_for_2:
                success = np.random.random() < 0.48
                if success:
                    points_added = 2
                    desc = "2PT GOOD"
                else:
                    desc = "2PT FAILED"
            else:
                pat_prob = 0.94
                if wind_speed > 15: pat_prob = 0.90
                success = np.random.random() < pat_prob
                if success:
                    points_added = 1
                    desc = "XP GOOD"
                else:
                    desc = "XP MISS"
                    
            scores[off] += points_added
            if verbose: print(f"   >>> {desc} ({off} {scores[off]} - {def_} {scores[def_]})")
            return
    
        def _play_game(self, home, away, wind_speed, temp, is_rain, is_snow, is_dome, hfa_impact, verbose=False):
            clock = 3600
            phase = 'REG' 
            scores = {home: 0, away: 0}
            timeouts = {home: 3, away: 3}
            halftime_processed = False
            
            # --- OPENING COIN TOSS & KICKOFF ---
            possession = np.random.choice([home, away])
            opponent = away if possession == home else home # Define opponent early for PAT logic
            
            # Calculate the opening field position
            start_yard = self._get_kickoff_start(possession)
            
            if start_yard >= 100:
                # OPENING KICKOFF RETURN TD!
                scores[possession] += 6
                if verbose: print(f"[{format_clock(clock, phase)}] OPENING KICKOFF RETURN TOUCHDOWN {possession}!")
                
                # Attempt PAT (Use 'opponent' since 'def_' isn't defined yet)
                self._attempt_pat(possession, opponent, scores, clock, phase, wind_speed, verbose)
                
                # Since they scored, they kick off to the opponent.
                # The opponent gets the ball for the first drive of the loop.
                possession = opponent
                
                # For simplicity, we assume the next kickoff is a standard return 
                # (to avoid infinite recursion of return TDs at 0:00)
                yardline = self._get_kickoff_start(possession)
                if yardline >= 100: yardline = 25 # Safety valve: Force touchback if back-to-back return TDs
                
            else:
                # Normal Start
                yardline = start_yard
    
            # Standard Drive Setup
            down, dist = 1, 10
            ot_drive_count = 0
            game_active = True
            
            while game_active:
                # --- HALFTIME RESET ---
                if phase == 'REG' and clock <= 1800 and not halftime_processed:
                    timeouts = {home: 3, away: 3}
                    halftime_processed = True
                    clock_running = False 
                    if verbose: print(f"[{format_clock(clock, phase)}] --- HALFTIME (Timeouts Reset) ---")
    
                # --- PHASE TRANSITION ---
                if clock <= 0:
                    if phase == 'REG' and scores[home] == scores[away]:
                        phase = 'OT'
                        clock = 600
                        possession = np.random.choice([home, away])
                        timeouts = {home: 2, away: 2} # Reset to 2 for OT
                        yardline = 32
                        down, dist = 1, 10
                        ot_drive_count = 0
                        clock_running = False
                        if verbose: print(f"\n[{format_clock(clock, phase)}] --- OVERTIME: {possession} wins toss ---")
                    else:
                        game_active = False
                        break
    
                off = possession
                def_ = away if off == home else home
                
                # Context
                diff = scores[off] - scores[def_]
                if diff > 8: ctx = 'leading'
                elif diff < -8: ctx = 'trailing'
                else: ctx = 'neutral'
                
                if dist <= 3: d_bucket = 'short'
                elif dist <= 7: d_bucket = 'med'
                else: d_bucket = 'long'
                
                zone = 'redzone' if yardline >= 80 else 'open'
                time_left_in_half = clock - 1800 if clock > 1800 else clock
    
                # --- PLAY CALL ---
                pass_prob = self.profiles['playcalling'].get((off, down, d_bucket, ctx))
                if pass_prob is None: pass_prob = self.league_pass_rates.get((down, d_bucket, ctx), 0.55)
    
####                # WIND: severe penalty if over 20mph
####                if not is_dome and wind_speed > 20:
####                    pass_prob -= 0.15  # Heavy shift to run
####                elif not is_dome and wind_speed > 15:
####                    pass_prob -= 0.05
                
                # RAIN/SNOW: slight shift to run to avoid drops/tips
####                if not is_dome and (is_rain or is_snow):
####                    pass_prob -= 0.05
                
                # Standard Adjustments
                if phase == 'REG' and clock < 300:
                    if diff > 0: pass_prob -= 0.4
                    if diff < 0: pass_prob += 0.4
                
                # --- NEW 3RD/4TH DOWN LOGIC OVERRIDE ---
                if down == 3 or down == 4:
                    if dist <= 2:
                        pass_prob = 0.50
                    elif dist <= 4:
                        pass_prob = 0.85
                    else:
                        pass_prob = 1.0
    
                pass_prob = np.clip(pass_prob, 0.01, 1.0)
    
                
                # --- DEFENSIVE PENALTY ---
                if np.random.random() < self.profiles['penalties'].get((def_, 'def'), 0.015):
                    pen_stats = self.profiles['penalty_details'].get(def_, {'dpi_share': 0.1, 'major_share': 0.15})
                    roll = np.random.random()
                    
                    if roll < pen_stats['dpi_share']:
                        raw_dpi = np.random.normal(pen_stats.get('dpi_mu', 15), pen_stats.get('dpi_std', 10))
                        p_yards = max(1, int(raw_dpi))
                        dist_to_goal = 100 - yardline
                        p_yards = min(p_yards, dist_to_goal - 1)
                        p_yards = max(1, p_yards)
                        if verbose: print(f"[{format_clock(clock, phase)}] {def_} PENALTY: Pass Interference ({p_yards} yds)")
    
                    elif roll < (pen_stats['dpi_share'] + pen_stats['major_share']):
                        p_yards = 15
                        dist_to_goal = 100 - yardline
                        if dist_to_goal < 30: 
                            p_yards = int(dist_to_goal / 2)
                            p_yards = max(1, p_yards)
                        if verbose: print(f"[{format_clock(clock, phase)}] {def_} PENALTY: Major/Unnecessary Roughness ({p_yards} yds)")
                        
                    else:
                        p_yards = 5
                        dist_to_goal = 100 - yardline
                        if dist_to_goal < 10:
                            p_yards = int(dist_to_goal / 2)
                            p_yards = max(1, p_yards)
                        if verbose: print(f"[{format_clock(clock, phase)}] {def_} PENALTY: Defensive Holding/Offsides ({p_yards} yds)")
    
                    yardline += p_yards
                    down, dist = 1, 10
                    if yardline >= 100: 
                        yardline = 99
                    clock_running = False 
                    continue
    
                # --- OFFENSIVE PENALTY ---
                if np.random.random() < self.profiles['penalties'].get((off, 'off'), 0.055):
                    yardline = max(1, yardline - 10)
                    clock -= 5
                    dist += 10
                    if verbose: print(f"[{format_clock(clock, phase)}] {off} OFFENSIVE PENALTY")
                    continue
    
                # --- 4TH DOWN DECISIONS ---
                if down == 4:
                    minutes = clock / 60.0
                    deficit = -diff if diff < 0 else 0
                    is_4q_or_ot = (phase == 'OT' or minutes < 15)
    
                    must_go_punt_range = False
                    if phase == 'REG':
                        if (9 <= deficit <= 16 and minutes < 4) or (1 <= deficit <= 8 and minutes < 2):
                            must_go_punt_range = True
                    if phase == 'OT': 
                        if scores[def_] >= scores[off]: must_go_punt_range = True
    
                    must_go_fg_range = False
                    if phase == 'REG':
                        if (4 <= deficit <= 8 and minutes < 4) or (12 <= deficit <= 16 and minutes < 5):
                            must_go_fg_range = True
                    
                    if is_4q_or_ot and deficit > 3:
                        must_go_fg_range = True
    
                    aggressive_go = (dist <= 2 and yardline >= 50)
                    attempt_play = False
                    
                    # FG LOGIC
                    kick_dist = (100 - yardline) + 18
                    k_stats = self.profiles['kicking'].get(off, {'max_made': 55, 'short_acc': 0.95, 'med_acc': 0.85, 'long_acc': 0.60})
                    
                    weather_max_dist = k_stats['max_made']
                    weather_acc_mod = 1.0
                    if not is_dome and wind_speed > 0:
                        weather_max_dist -= (wind_speed / 3)
                        if wind_speed > 25: weather_acc_mod = 0.75
                        elif wind_speed > 15: weather_acc_mod = 0.90
                        
                        if temp < 15: weather_max_dist -= 10
                        elif temp < 30: weather_max_dist -= 5
                        elif temp < 40: weather_max_dist -= 2
                        
                    
                    in_fg_range = kick_dist <= (weather_max_dist + 2)
                    
                    if in_fg_range and kick_dist <= 65:
                        if must_go_fg_range:
                            attempt_play = True
                            if verbose: print(f"[{format_clock(clock, phase)}] {off} NEED TD: Going for it on 4th!")
                        else:
                            if kick_dist < 40: base_prob = k_stats['short_acc']
                            elif kick_dist < 50: base_prob = k_stats['med_acc']
                            else: base_prob = k_stats['long_acc']
    #WEATHER                        
                            final_prob = base_prob * weather_acc_mod
                            if kick_dist > (weather_max_dist - 3): final_prob *= 0.8 
                            made = np.random.random() < final_prob
                            
                            if verbose: print(f"[{format_clock(clock, phase)}] {off} {int(kick_dist)} yd FG Attempt... {'GOOD' if made else 'MISS'}")
                            clock -= 5
                            clock_running = False 
                            
                            if made:
                                scores[off] += 3
                                if phase == 'OT':
                                    if ot_drive_count == 0:
                                        if verbose: print(f"   >>> OT: {def_} must score.")
                                    else:
                                        if scores[off] > scores[def_]:
                                            game_active = False
                                            if verbose: print(f"   >>> OVERTIME WINNER: {off}!")
                                            break
                                        elif scores[def_] > scores[off]:
                                            game_active = False
                                            if verbose: print(f"   >>> OVERTIME WINNER: {def_}!")
                                            break
                                possession = def_
                                new_start = self._get_kickoff_start(possession)
                                
                                if new_start >= 100:
                                    # KICK RETURN TD!
                                    scores[possession] += 6
                                    if verbose: print(f"   >>> KICKOFF RETURN TOUCHDOWN {possession}!")
                                    self._attempt_pat(possession, off, scores, clock, phase, wind_speed, verbose)
                                    # Kick it right back to the other team
                                    possession = off 
                                    yardline = 30 
                                    continue # Skip to next iteration
                                
                                yardline = new_start
                                down, dist = 1, 10
                                if phase == 'OT': ot_drive_count += 1
                            else:
                                if phase == 'OT' and scores[def_] > scores[off]:
                                    game_active = False
                                    break
                                possession = def_
                                yardline = 100 - (yardline + 7)
                                if yardline < 0: yardline = 20
                                down, dist = 1, 10
                                if phase == 'OT': ot_drive_count += 1
                            continue
    
                    else: 
                        if must_go_punt_range:
                            attempt_play = True
                            if verbose: print(f"[{format_clock(clock, phase)}] {off} DESPERATION: Going for it!")
                        elif aggressive_go:
                            attempt_play = True
                            if verbose: print(f"[{format_clock(clock, phase)}] {off} ANALYTICS: Going for it (4th & {dist})!")
                        else:
                            p_stats = self.profiles['punting'].get(off, {'mu': 41.0, 'sigma': 4.0})
                            adj_mu = p_stats['mu'] - (wind_speed / 2.0)
                            dist_to_goal = 100 - yardline
                            
                            if adj_mu > dist_to_goal:
                                if verbose: print(f"[{format_clock(clock, phase)}] {off} PUNT (Pinning Attempt)")
                                new_start = np.random.randint(1, 21) 
                                yardline = new_start 
                            else:
                                if verbose: print(f"[{format_clock(clock, phase)}] {off} PUNT")
                                punt_dist = np.random.normal(adj_mu, p_stats['sigma'])
                                punt_dist = max(10, punt_dist)
                                new_yardline = 100 - (yardline + punt_dist)
                                if new_yardline <= 0:
                                    new_yardline = 20
                                    if verbose: print(f"   >>> Touchback")
                                yardline = new_yardline
    
                            clock_running = False 
                            
                            if phase == 'OT' and scores[def_] > scores[off]:
                                game_active = False
                                if verbose: print(f"   >>> OVERTIME WINNER: {def_} (Stop)!")
                                break
                            
                            possession = def_
                            down, dist = 1, 10
                            clock -= 40
                            if phase == 'OT': ot_drive_count += 1
                            continue
                    
                    # Fall through to execute
    
                
                # --- EXECUTE PLAY ---
                is_pass = np.random.random() < pass_prob
                ptype = 'pass' if is_pass else 'run'
    
                # Get INITIAL stats profile
                base_stats = self.profiles['efficiency'].get((off, zone, ptype), 
                        {'mu': 4.0, 'sigma': 4.0, 'complete': 0.6, 'intercept': 0.03, 'fumble': 0.01, 'sack': 0.07})
                
                # MUST COPY THE DICTIONARY SO WE DON'T PERMANENTLY CHANGE IT
                stats = base_stats.copy()

                # --- PREVENT DEFENSE LOGIC ---
                if ctx == 'trailing' and clock < 600 and ptype == 'pass':
                    stats['complete'] += 0.08  # Defenses give up underneath stuff
                    stats['mu'] -= 1.5         # But keep everything in front of them
                    stats['intercept'] -= 0.01 # Not taking risks jumping routes
    
                # OVERRIDE: Goal-to-Go Efficiency Boost
                # We only need a tiny nudge to convert ~1 extra drive per game from FG to TD
                is_goal_to_go = (100 - yardline) <= 10
                
    #            if is_goal_to_go:
    #                if ptype == 'run':
                        # PREVIOUS (Too Strong): mu += 1.3, sigma = 1.5 
                        # NEW (Marginal):
                        # Add 0.4 yards to the average surge (e.g., 3.5 -> 3.9)
    #                    stats['mu'] += 0.0  
                        
                        # Don't crush the variance (sigma). Keep it around 3.0.
                        # This allows for 1-yard gains or 0-yard stuffs, which forces 
                        # 3rd & 4th down decisions rather than automatic 1st downs.
    #                    stats['sigma'] = 3.0 
                        
    #                else:
                        # PREVIOUS (Too Strong): complete += 0.05
                        # NEW (Marginal):
                        # Tiny bump to completion % (2.5%) for short throws
    #                    stats['complete'] += 0.0025 
                        
                        # QBs are still careful, but picks happen (tipped balls).
                        # 1.5% is a realistic low floor.
    #                    stats['intercept'] = 0.03
                
                # (Deleted the duplicate 'stats =' line that was here)
    
                # Get Defense Adjustments
                def_mult = self.def_mults.get(def_, {}).get(ptype, 1.0)
                if def_ == home: def_mult *= (1 - hfa_impact)
                
                # --- CALL THE NEW HELPER FUNCTION ---
                yards, is_complete, is_turnover, desc_tag = self._resolve_play_outcome(
                    off, def_, zone, ptype, stats, def_mult, hfa_impact, 
                    wind_speed, temp, is_rain, is_snow, is_dome, verbose
                )
                
                # If verbose, append the specific tag (Deep Ball, Breakaway) to the printout later
                if verbose and desc_tag:
                    # We'll save this tag to print it in the verbose section below
                    pass
    
                # --- CHECK TURNOVER ON DOWNS ---
                if down == 4 and yards < dist:
                    is_turnover = True
                    if verbose: print(f"   >>> TURNOVER ON DOWNS!")
    
                # --- CLOCK LOGIC (OOB, Stoppage & TIMEOUTS) ---
                is_oob = False
                if ptype == 'run' or (ptype == 'pass' and is_complete):
                    oob_prob = self.profiles['oob_rates'].get(off, 0.15)
                    if np.random.random() < oob_prob: is_oob = True
                
                clock_stops = False
                if ptype == 'pass' and not is_complete and yards >= 0:
                    clock_stops = True 
                elif is_oob:
                    if (1800 < clock <= 1920) or (clock <= 300 and phase == 'REG') or (phase == 'OT'):
                        clock_stops = True
                
                clock_running = not clock_stops
                
                # --- TIMEOUT LOGIC ---
                is_two_minute = time_left_in_half <= 120
                
                if not clock_stops and is_two_minute:
                    if scores[def_] <= scores[off] and timeouts[def_] > 0:
                        timeouts[def_] -= 1
                        clock_stops = True
                        clock_running = False
                        if verbose: print(f"   >>> TIMEOUT {def_} ({timeouts[def_]} left)")
                    
                    elif scores[off] <= scores[def_] and timeouts[off] > 0:
                        timeouts[off] -= 1
                        clock_stops = True
                        clock_running = False
                        if verbose: print(f"   >>> TIMEOUT {off} ({timeouts[off]} left)")
    
                if clock_stops:
                    time_consumed = self.profiles.get('play_duration', 6.0)
                else:
                    pace_t = 'run'
                    if ptype == 'pass':
                        if is_complete: pace_t = 'pass_complete'
                        elif yards < 0: pace_t = 'sack'
                        else: pace_t = 'pass_incomplete'
                    
                    time_consumed = self.profiles['pace'].get((off, pace_t), 35.0)
                    
                    # FIX: Cap standard plays to prevent "huddle drift"
                    # If the data has a weird outlier (like an injury play taking 90 seconds), 
                    # it ruins the sim average.
                    if pace_t == 'run' or pace_t == 'pass_complete':
                        time_consumed = min(time_consumed, 40) # Cap at 40s (play clock)
                    elif pace_t == 'pass_incomplete' or is_oob:
                        time_consumed = min(time_consumed, 10) # Quick stoppage
                    if phase == 'REG' and clock < 300:
                        if diff < 0: time_consumed = min(time_consumed, 15)
                        if diff > 0: time_consumed = max(time_consumed, 40)
                    if phase == 'OT': time_consumed = min(time_consumed, 25)
    
                # HURRY UP LOGIC
                # If inside 2 mins of 2nd or 4th quarter and trailing or tied (or just wanting to score before half)
                is_end_of_half = (phase == 'REG' and 1800 < clock <= 1920) or (clock <= 120)
                trying_to_score = (scores[off] <= scores[def_] + 8) or (1800 < clock <= 1920) # Always try to score before half
    
                if is_end_of_half and trying_to_score and not clock_stops:
                    # In hurry up, plays take 12-15 seconds total, not 35
                    if is_complete or ptype == 'run':
                        time_consumed = min(time_consumed, 14) 
    
                clock -= time_consumed
    
                if verbose:
                    loc = format_field(yardline, off)
                    desc = f"Run {yards}" if ptype=='run' else (f"Pass {yards}" if is_complete else "Pass Inc")
                    if yards < 0 and ptype == 'pass' and not is_complete: desc = "SACK"
                    if is_turnover: desc += " TURNOVER"
                    if is_oob: desc += " (OOB)"
                    print(f"[{format_clock(clock, phase)}] {off} {down}&{dist} @ {loc} | {desc}")
    
                if is_turnover:
                     clock_running = False 
                     
                     # --- FIX: DEFENSIVE TOUCHDOWN LOGIC ---
                     # Approx 8% of turnovers result in a defensive score
                     if np.random.random() < 0.08:
                         scores[def_] += 6
                         if verbose: print(f"   >>> DEFENSIVE TOUCHDOWN (PICK-6/FUMBLE-6) {def_}!")
                         self._attempt_pat(def_, off, scores, clock, phase, wind_speed, verbose)
                         
                         # Kickoff logic
                         possession = off # Offense gets ball back
                         yardline = 32
                         down, dist = 1, 10
                         continue # Skip the rest, start new drive
                     # --------------------------------------
    
                     if phase == 'OT' and scores[off] == scores[def_]:
                          if verbose: print("   >>> OT: Turnover. Next score wins.")
                     
                     # Standard Turnover
                     possession = def_
                     yardline = 100 - (yardline + yards)
                     # Add variance to turnover return (sometimes they return it 20 yards)
                     return_yards = int(np.random.exponential(5)) # Avg 5 yard return
                     yardline += return_yards
                     yardline = min(yardline, 99) # Don't go past goal line
                     
                     down, dist = 1, 10
                     clock -= 10
                     if phase == 'OT': ot_drive_count += 1
                     continue
    
                yardline += yards
                dist -= yards
                
                if yardline >= 100:
                    scores[off] += 6 
                    if verbose: print(f"   >>> TOUCHDOWN {off}!")
                    
                    self._attempt_pat(off, def_, scores, clock, phase, wind_speed if not is_dome else 0, verbose)
                    clock_running = False 
    
                    if phase == 'OT':
                        if ot_drive_count == 0:
                            if verbose: print(f"   >>> OT: {def_} gets a chance to match!")
                        else:
                            if scores[off] > scores[def_]:
                                game_active = False
                                if verbose: print(f"   >>> OVERTIME WINNER: {off}!")
                                break
                            elif scores[off] == scores[def_]:
                                if verbose: print(f"   >>> OT: Game Tied. Next Score Wins!")
    
                    possession = def_
                    new_start = self._get_kickoff_start(possession)
                    
                    if new_start >= 100:
                        # KICK RETURN TD!
                        scores[possession] += 6
                        if verbose: print(f"   >>> KICKOFF RETURN TOUCHDOWN {possession}!")
                        self._attempt_pat(possession, off, scores, clock, phase, wind_speed, verbose)
                        # Kick it right back to the other team
                        possession = off 
                        yardline = 30 
                        continue # Skip to next iteration
                    
                    yardline = new_start
                    down, dist = 1, 10
                    if phase == 'OT': ot_drive_count += 1
                elif dist <= 0:
                    down = 1
                    dist = 10
                else:
                    down += 1
            
            return {'Home': home, 'Away': away, 'Home_Score': scores[home], 'Away_Score': scores[away], 
                    'Margin': scores[away] - scores[home]}
    
    
    # --- CONFIGURATION: DOMES & HISTORICAL AVERAGES ---
    # We keep this to know who plays indoors and what to default to if the game is months away.
    # (Latitude/Longitude are now pulled from your DF, so they are removed from here)
    STADIUM_CONFIG = {
        # DOMES (Weather is always 0 wind)
        'Allegiant Stadium': {'dome': True},
        'AT&T Stadium': {'dome': True},
        'Caesars Superdome': {'dome': True},
        'Ford Field': {'dome': True},
        'Lucas Oil Stadium': {'dome': True},
        'Mercedes-Benz Stadium': {'dome': True},
        'NRG Stadium': {'dome': True},
        'State Farm Stadium': {'dome': True},
        'U.S. Bank Stadium': {'dome': True},
        'SoFi Stadium': {'dome': True},
        
        # OUTDOORS (Historical Averages per month: Sept=9, Oct=10, etc.)
        # Format: Month: (Temp, WindSpeed)
        'Arrowhead Stadium': {'dome': False, 'defaults': {9: (75, 10), 10: (60, 12), 11: (45, 12), 12: (35, 15), 1: (30, 15)}},
        'M&T Bank Stadium': {'dome': False, 'defaults': {9: (70, 8), 10: (60, 10), 11: (50, 10), 12: (40, 12), 1: (35, 12)}},
        'Highmark Stadium': {'dome': False, 'defaults': {9: (65, 12), 10: (55, 15), 11: (40, 15), 12: (30, 20), 1: (25, 20)}},
        'Bank of America Stadium': {'dome': False, 'defaults': {9: (78, 5), 10: (68, 6), 11: (55, 6), 12: (48, 8), 1: (45, 8)}},
        'Soldier Field': {'dome': False, 'defaults': {9: (70, 12), 10: (58, 15), 11: (45, 15), 12: (32, 18), 1: (26, 18)}},
        'Paycor Stadium': {'dome': False, 'defaults': {9: (72, 8), 10: (62, 10), 11: (48, 10), 12: (38, 12), 1: (34, 12)}},
        'Cleveland Browns Stadium': {'dome': False, 'defaults': {9: (68, 12), 10: (58, 15), 11: (45, 15), 12: (35, 20), 1: (30, 20)}},
        'Empower Field at Mile High': {'dome': False, 'defaults': {9: (75, 8), 10: (60, 10), 11: (45, 10), 12: (35, 12), 1: (35, 12)}},
        'Lambeau Field': {'dome': False, 'defaults': {9: (65, 10), 10: (52, 12), 11: (38, 15), 12: (25, 15), 1: (20, 15)}},
        'Hard Rock Stadium': {'dome': False, 'defaults': {9: (88, 8), 10: (82, 10), 11: (75, 10), 12: (70, 10), 1: (68, 10)}},
        'Gillette Stadium': {'dome': False, 'defaults': {9: (68, 10), 10: (58, 12), 11: (45, 12), 12: (35, 15), 1: (30, 15)}},
        'MetLife Stadium': {'dome': False, 'defaults': {9: (72, 10), 10: (62, 12), 11: (48, 12), 12: (38, 15), 1: (34, 15)}},
        'Lincoln Financial Field': {'dome': False, 'defaults': {9: (74, 10), 10: (64, 12), 11: (50, 12), 12: (40, 15), 1: (36, 15)}},
        'Acrisure Stadium': {'dome': False, 'defaults': {9: (70, 10), 10: (58, 12), 11: (45, 12), 12: (35, 12), 1: (32, 12)}},
        'Lumen Field': {'dome': False, 'defaults': {9: (65, 8), 10: (55, 10), 11: (48, 10), 12: (42, 12), 1: (42, 12)}},
        'Raymond James Stadium': {'dome': False, 'defaults': {9: (88, 8), 10: (82, 8), 11: (75, 8), 12: (70, 10), 1: (68, 10)}},
        'Nissan Stadium': {'dome': False, 'defaults': {9: (78, 6), 10: (68, 8), 11: (55, 8), 12: (45, 10), 1: (40, 10)}},
        'FedExField': {'dome': False, 'defaults': {9: (75, 8), 10: (65, 10), 11: (52, 10), 12: (42, 12), 1: (38, 12)}},
        'EverBank Stadium': {'dome': False, 'defaults': {9: (85, 8), 10: (78, 10), 11: (68, 10), 12: (60, 12), 1: (58, 12)}},
        'Levi\'s Stadium': {'dome': False, 'defaults': {9: (75, 10), 10: (70, 10), 11: (60, 8), 12: (55, 8), 1: (55, 8)}}
    }
    
    def get_weather_for_game(lat, lon, date_str, stadium_name, row_temp=None, row_wind=None, row_desc=None):
        """
        Determines weather using a 4-step priority:
        1. Dome Check: Always 70F / 0mph.
        2. Open-Meteo API: Exact hourly data for recent past or near future (-10 to +10 days).
        3. Row Data Fallback: If API times out/fails on a past game, use nfl_read_py 'temp'/'wind'.
        4. Monthly Average: Final fallback for errors OR games >10 days in the future.
        """
        # 1. DOME CHECK (Fastest)
        config = STADIUM_CONFIG.get(stadium_name, {'dome': False})
        if config.get('dome', False):
            return 0.0, 0.0, 70.0, True, "Dome"
    
        # 2. PREPARE DEFAULTS & DATES
        try:
            clean_date_str = str(date_str)[:10]
            game_date = datetime.strptime(clean_date_str, "%Y-%m-%d")
            days_diff = (game_date - pd.to_datetime(date_str)).days
            month = game_date.month
        except:
            # If date is broken, rely immediately on row data or generic safe values
            if row_temp is not None and not pd.isna(row_temp):
                 return float(row_wind or 0), 0.0, float(row_temp), False, "nfl_read_py (Date Err)"
            return 10.0, 0.0, 60.0, False, "Error Default"
    
        # Retrieve Monthly Averages from STADIUM_CONFIG (The "Final Fallback")
        # Format: { Month_Int: (Temp, Wind) }
        month_defaults = config.get('defaults', {}).get(month, (60, 10))
        def_temp, def_wind = month_defaults
    
        # 3. API LOGIC (Only runs if within window)
        #    We purposely skip this for games > 10 days out.
        try:
            # A) HISTORICAL (Past Games > 10 days ago)
            if days_diff < -10:
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat, "longitude": lon,
                    "start_date": clean_date_str, "end_date": clean_date_str,
                    "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
                    "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "auto"
                }
                # 10s timeout to prevent hanging on history
                r = requests.get(url, params=params, timeout=10)
                data = r.json()
                if 'hourly' in data:
                    # Average indices 13-17 (approx 1 PM - 5 PM)
                    wind = np.mean(data['hourly']['wind_speed_10m'][13:17])
                    precip = np.sum(data['hourly']['precipitation'][13:17])
                    temp = np.mean(data['hourly']['temperature_2m'][13:17])
                    
                    # Check for NaNs (sometimes API returns nulls)
                    if np.isnan(wind): wind = def_wind
                    if np.isnan(temp): temp = def_temp
                    if np.isnan(precip): precip = 0.0
                    return wind, precip, temp, False, "Historical API"
    
            # B) LIVE FORECAST (Recent Past / Near Future: -10 to +10 days)
            elif -10 <= days_diff <= 10:
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": lat, "longitude": lon,
                    "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
                    "start_date": clean_date_str, "end_date": clean_date_str,
                    "wind_speed_unit": "mph", "temperature_unit": "fahrenheit", "timezone": "auto"
                }
                r = requests.get(url, params=params, timeout=5)
                data = r.json()
                if 'hourly' in data:
                    wind = np.mean(data['hourly']['wind_speed_10m'][13:17])
                    precip = np.sum(data['hourly']['precipitation'][13:17])
                    temp = np.mean(data['hourly']['temperature_2m'][13:17])
                    return wind, precip, temp, False, "Live Forecast"
    
        except Exception as e:
            # If API fails, we just print/pass and let it hit the fallbacks below
            # print(f"   [API Ignored] {stadium_name}: {e}") 
            pass
    
        # 4. FALLBACK LOGIC
        
        # Priority A: Row Data (Real recorded weather from nfl_read_py)
        # We use this if the API timed out but we have data in the dataframe.
        # We do NOT use this if it's a future game (row data will be NaN).
        if row_temp is not None and not pd.isna(row_temp):
            r_wind = float(row_wind) if (row_wind is not None and not pd.isna(row_wind)) else def_wind
            return r_wind, 0.0, float(row_temp), False, "nfl_read_py Fallback"
    
        # Priority B: Monthly Averages (Future games > 10 days out OR Missing data)
        return def_wind, 0.0, def_temp, False, "Historical Avg"
    
    
    NAME_MAP = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
    }
    
    NAME_TO_ABBR = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
    }
    
    # --- MAIN EXECUTION BLOCK ---
    if __name__ == "__main__":
        sim = AdvancedNFLSimulator()
        sim.load_data() 
        
        simulation_results = []
        print(f"\nStarting Simulations for {len(collect_schedule_travel_ranking_data_df)} games...")
        print(f"{'Game':<30} | {'Source':<15} | {'Wind':<5} | {'Spread':<6} | {'Spread Var':<10}")
        print("-" * 85)
    
        # 1. UPDATED THRESHOLDS: Based on your model's actual outputs (140-210 range)
        def get_variance_label(val, metric_type='combined'):
            if metric_type == 'combined':
                if val < 160: return "Low"
                if val < 170: return "Med-Low"
                if val < 180: return "Medium"
                if val < 195: return "Med-High"
                return "High"
            else:
                if val < 70:  return "Low"
                if val < 85:  return "Med-Low"
                if val < 100: return "Medium"
                if val < 115: return "Med-High"
                return "High"
    
        weather_df = pd.read_csv(f'nfl-schedules/schedule_{target_year}.csv')
        weather_lookup = weather_df[['game_id', 'Temperature', 'Wind Speed']].rename(columns={
    		'game_id': 'Game ID',
            'Temperature': 'temp',
            'Wind Speed': 'wind'
        })
    	
    	# 2. Merge this into your main dataframe on the 'game_id' key
    	# Using how='left' ensures you don't lose any games even if weather is missing
        collect_schedule_travel_ranking_data_df = collect_schedule_travel_ranking_data_df.merge(
            weather_lookup, 
            on='Game ID', 
            how='left'
        )
    
        for index, row in collect_schedule_travel_ranking_data_df.iterrows():
            try:
                # Extract Row Data
                away_full = row['Away Team']
                home_full = row['Home Team']
                away = NAME_MAP.get(away_full, away_full)
                home = NAME_MAP.get(home_full, home_full)
                stadium = row['Actual Stadium']
                date = pd.to_datetime(row['Date']) 
                lat = row['Actual Stadium Latitude']
                lon = row['Actual Stadium Longitude']
                sched_temp = row.get('temp') 
                sched_wind = row.get('wind')
                sched_desc = row.get('weather') # <--- ADD THIS LINE
                
                # 1. Get Weather
                raw_wind, precip, temp, is_dome, source = get_weather_for_game(
                    lat, lon, date, stadium, 
                    row_temp=sched_temp, 
                    row_wind=sched_wind,
                    row_desc=sched_desc
                )
                # 2. Run Simulation
                df_sim = sim.simulate_matchup(home, away, wind_speed=raw_wind, temp=temp, precip=precip, is_dome=is_dome)
                if not df_sim.empty:
                    # 3. Define the Series variables
                    margin = df_sim['Margin']
                    df_sim['Total'] = df_sim['Home_Score'] + df_sim['Away_Score']
                    total = df_sim['Total']
                    
                    # 4. Calculate Stats & Labels
                    # --- FIX: Define spread_var BEFORE using it in the function ---
                    spread_var = margin.var() 
                    vol_label = get_variance_label(spread_var, metric_type='combined')
                    
                    abs_margin = margin.abs()
                    prob_land_3 = (abs_margin == 3).mean()
                    prob_land_7 = (abs_margin == 7).mean()
                    
                    # 5. Build the Result Row
                    res = {
                        'Matchup_ID': index,
                        'Week': row.get('Week'),
                        'Date': date,
                        'Matchup': f"{away} @ {home}",
                        'Wind': raw_wind,
                        'Temperature': temp,
                        'Precipitation': precip,
                        'Sim_Weather_Source': source,
                        'Sim_Spread_Mean': margin.mean(),
                        'Sim_Spread_Median': margin.median(),
                        'Sim_Spread_Std': margin.std(),
                        'Sim_Spread_Variance': spread_var,
                        'Sim_Spread_Variance_Label': vol_label,
                        'Sim_Spread_25th': margin.quantile(0.25),
                        'Sim_Spread_75th': margin.quantile(0.75),
                        'Sim_Total_Mean': total.mean(),
                        'Sim_Total_Median': total.median(),
                        'Sim_Total_Std': total.std(),
                        'Sim_Total_10th_Floor': total.quantile(0.10),
                        'Sim_Total_90th_Ceiling': total.quantile(0.90),
                        'Sim_Home_Win_Pct': (margin < 0).mean(),
                        'Sim_Away_Win_Pct': (margin > 0).mean(),
                        'Sim_Prob_Land_3': prob_land_3,
                        'Sim_Prob_Land_7': prob_land_7
                    }
                    
                    simulation_results.append(res)
                    
                    # Progress Print
                    print(f"{away:>3} @ {home:<3} {date.strftime('%Y-%m-%d'):<10} | {source:<15} | {raw_wind:>4.1f} | {res['Sim_Spread_Mean']:>6.2f} | {spread_var:>8.2f}")
    
            except Exception as e:
                print(f"Error simulating {row.get('Away Team')} vs {row.get('Home Team')}: {e}")
                continue
    
        # --- SAVE TO DATAFRAME ---
        monte_carlo_df = pd.DataFrame(simulation_results)
        
        if not monte_carlo_df.empty:
            cols_to_round = ['Sim_Spread_Mean', 'Sim_Spread_Median', 'Sim_Total_Mean', 
                             'Sim_Total_Median', 'Sim_Spread_Variance', 'Sim_Spread_Std']
            monte_carlo_df[cols_to_round] = monte_carlo_df[cols_to_round].round(2)
            final_combined_df = collect_schedule_travel_ranking_data_df.merge(monte_carlo_df, left_index=True, right_on='Matchup_ID', how='left')
    
            
            print("\nSimulation Complete!")
            # Ensure directory exists or remove prefix if not needed
            final_combined_df.to_csv(f"nfl-power-ratings/final_sim_results_with_variance{upcoming_week}_TEST.csv", index=False)
            print(f"Results saved to 'nfl-power-ratings/final_sim_results_with_variance_{upcoming_week}_TEST.csv'")


if __name__ == "__main__":
    week_starting_dates = [
        "09/03/2025", 
        "09/10/2025", 
        "09/17/2025", 
        "09/24/2025", 
        "10/01/2025", 
        "10/08/2025", 
        "10/15/2025",
        "10/22/2025", 
        "10/29/2025"
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
#        "12/31/2025"
    ]

    for date in week_starting_dates:
        loop_through_simulations(date)
