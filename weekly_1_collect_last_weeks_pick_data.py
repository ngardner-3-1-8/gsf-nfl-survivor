## from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import random
import csv
import os
import numpy as np
import itertools
from tqdm import tqdm
import sys
from typing import Optional
from typing import Dict, List, Any
import polars as pl
import nflreadpy as nfl
from datetime import datetime, timedelta
import calendar

def loop_through_historical_final_data(date_str): 
    # 1. Get current date
    today = pd.to_datetime(date_str)
    
    current_cal_year = today.year 
    
    # 2. Initial Year Logic based on Month (User Rule)
    # If Jan-May (< 6), assume we are finishing the previous season.
    if today < datetime(current_cal_year, 5, 15, 0, 0):
        target_year = current_cal_year - 1
    
        # 3. Pre-Season Check (User Rule)
        # We need to see if the season has actually started yet.
        try:
            # Load the schedule for the target year
            schedule = nfl.load_schedules([target_year])
            print("schedule Loaded successfully")
            schedule = schedule.to_pandas() # Convert here!
            print("schedule converted successfully")
            # Now all the standard Pandas filtering works:
            reg_season_games = schedule[schedule['game_type'] == 'REG']
            print("reg season games filtered successfully")
            if not reg_season_games.empty:
                # Find the very first game date of the season
                first_game_date = pd.to_datetime(reg_season_games['gameday'].min())
                print(f"first game date: {first_game_date}")
                # Check if today is BEFORE the first game
                if pd.to_datetime(today) < first_game_date:
                    print(f"Today ({today.date()}) is before the first game ({first_game_date.date()}). dropping year by 1.")
                    # Reload schedule for the adjusted year so we can calculate the week correctly below
                    schedule = nfl.load_schedules([target_year])
            
            # 4. Calculate the Current Week
            # We find the latest game that has happened to determine "current" week
            games_played = schedule[
                pd.to_datetime(schedule['gameday']) <= pd.to_datetime(today)
            ]
            
            if not games_played.empty:
                # If games have been played, the "starting_week" for your script 
                # (which usually scrapes the *upcoming* week) should be the last played week + 1.
                last_played_week = int(games_played['week'].max())
                starting_week = last_played_week + 1
                print("last_played_week")
                # Bound check: If season is over (e.g. Week 22), cap it or handle as needed
                if starting_week > 19: 
                    starting_week = 19 
            else:
                # If we fell back a year but that season is fully over, or if no games played yet
                starting_week = 1
                last_played_week = 0
        
        except Exception as e:
            print(f"⚠️ Error in dynamic detection: {e}. Falling back to defaults.")
            # Fallback defaults to prevent crash
            target_year = 2025
            starting_week = 19
    else:
        target_year = current_cal_year
    
        # 3. Pre-Season Check (User Rule)
        # We need to see if the season has actually started yet.
        try:
            # Load the schedule for the target year
            schedule = nfl.load_schedules([target_year])
            print("schedule Loaded successfully")
            schedule = schedule.to_pandas() # Convert here!
            print("schedule converted successfully")
            # Now all the standard Pandas filtering works:
            reg_season_games = schedule[schedule['game_type'] == 'REG']
            print("reg season games filtered successfully")
            if not reg_season_games.empty:
                # Find the very first game date of the season
                first_game_date = pd.to_datetime(reg_season_games['gameday'].min())
                print(f"first game date: {first_game_date}")                
                # Check if today is BEFORE the first game
                if pd.to_datetime(today) < first_game_date:
                    print(f"Today ({today.date()}) is before the first game ({first_game_date.date()}). dropping year by 1.")
                    target_year -= 1
                    # Reload schedule for the adjusted year so we can calculate the week correctly below
                    schedule = nfl.load_schedules([target_year])
            
            # 4. Calculate the Current Week
            # We find the latest game that has happened to determine "current" week
            games_played = schedule[
                pd.to_datetime(schedule['gameday']) <= pd.to_datetime(today)
            ]
            
            if not games_played.empty:
                # If games have been played, the "starting_week" for your script 
                # (which usually scrapes the *upcoming* week) should be the last played week + 1.
                last_played_week = int(games_played['week'].max())
                starting_week = last_played_week + 1
                
                # Bound check: If season is over (e.g. Week 22), cap it or handle as needed
                if starting_week > 18: 
                    starting_week = 18
                    last_played_week = 17
            else:
                # If we fell back a year but that season is fully over, or if no games played yet
                starting_week = 1 
                last_played_week = 0

            print(starting_week)
            print(last_played_week)
        
        except Exception as e:
            print(f"⚠️ Error in dynamic detection: {e}. Falling back to defaults.")
            # Fallback defaults to prevent crash
            target_year = 2025
            starting_week = 18
            last_played_week = 17
    # 5. Final Assignment to your variables
    current_year = target_year
    starting_year = target_year
    
    current_year_plus_1 = current_year + 1
    
    print(f"✅ Final Configuration -> Year: {current_year} | Starting Week: {starting_week}")
    if target_year == 2026:
        MAX_PAGES = 225    
    elif target_year == 2025:
        MAX_PAGES = 187
    elif target_year == 2024:
        MAX_PAGES = 142
    elif target_year == 2023:
        MAX_PAGES = 92
    elif target_year == 2022:
        MAX_PAGES = 61
    elif target_year == 2021:
        MAX_PAGES = 40
    elif target_year == 2020:
        MAX_PAGES = 13
    
    circa_2020_entries = 1373
    circa_2021_entries = 4071
    circa_2022_entries = 6106
    circa_2023_entries = 9234
    circa_2024_entries = 14221
    circa_2025_entries = 18718
    circa_2026_entries = 22500
    # ==============================================================================
    # SECTION 1: SURVIVORGRID.COM SCRAPING (UNCHANGED - nflreadpy CANNOT DO THIS)
    # ==============================================================================
    
    # --- AUTOMATION FIX 3: Thanksgiving & Christmas Adjustment ---
    
    def get_thanksgiving(year):
        # Thanksgiving is the 4th Thursday in November
        nov1_weekday = calendar.weekday(year, 11, 1)
        days_to_first_thursday = (3 - nov1_weekday + 7) % 7
        thanksgiving_day = 1 + days_to_first_thursday + 21
        return datetime(year, 11, thanksgiving_day)
    
    # 1. Standard Logic
    NUM_WEEKS_TO_KEEP = starting_week - 1
    
    # 2. HOLIDAY CALCULATIONS BASED ON THE SEASON YEAR (current_year)
    # If it's Jan 2026, current_year is 2025. This ensures we look at 2025 holidays.
    thanksgiving_season = get_thanksgiving(current_year)
    two_days_after_thanksgiving = thanksgiving_season + timedelta(days=2)
    
    christmas_season_cutoff = datetime(current_year, 12, 26)
    
    # 3. Apply Adjustments
    # Use separate 'if' statements if you want them to be additive (+2 total)
    if today >= two_days_after_thanksgiving:
        print(f"Detected: Date is after {current_year} Thanksgiving. +1 to NUM_WEEKS_TO_KEEP.")
        NUM_WEEKS_TO_KEEP += 1
    
    if today >= christmas_season_cutoff:
        print(f"Detected: Date is after {current_year} Christmas. +1 to NUM_WEEKS_TO_KEEP.")
        NUM_WEEKS_TO_KEEP += 1
    
    print(f"Final NUM_WEEKS_TO_KEEP: {NUM_WEEKS_TO_KEEP}")
    
    def scrape_data(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "lxml")
        table_rows = soup.find_all("tr")
    
        data = []
        for row in table_rows:
            columns = row.find_all("td")
            if len(columns) >= 5:
                # Safely unpack columns, adjusting for potential length differences
                # EV, win_pct, pick_pct, team, opponent are the first 5 columns we care about
                if len(columns) < 5: continue # Skip if not enough columns
    
                ev, win_pct, pick_pct, team, opponent = columns[:5]
                rest = columns[5:] # Any columns after opponent
    
                # This logic assumes the 'Future Value' is in the last cell of the row
                future_value_cell = rest[-1] if rest else None
    
                if future_value_cell:
                    # Extract the width value from the style attribute
                    div_tag = future_value_cell.find("div")
                    if div_tag and "style" in div_tag.attrs:
                        style_attr = div_tag["style"]
                        width_match = re.search(r"width:\s*(\d+)px", style_attr)
                        if width_match:
                            width_px = int(width_match.group(1))
                            star_rating = width_px / 16  # Assuming each star is 16px wide
                        else:
                            star_rating = 0
                    else:
                        star_rating = 0
                else:
                    star_rating = 0  # No "Future Value" column/cell
    
                data.append({
                    "EV": ev.text,
                    "Win %": win_pct.text,
                    "Pick %": pick_pct.text,
                    "Team": team.text,
                    "Opponent": opponent.text,
                    "Future Value (Stars)": star_rating
                })
    
        return data
    
    
    # Create an empty list to store data
    all_data = []
    
    # Iterate through desired weeks and years
    base_url = "https://www.survivorgrid.com/{year}/{week}"
    # NOTE: The loop runs from 2025 up to (but not including) 2026 for the year.
    for year in tqdm(range(starting_year, current_year_plus_1), desc="Scraping data"):
    ###for year in tqdm(range(starting_year, current_year), desc="Scraping data"):
        # The week loop runs from 1 up to (but not including) starting_week (e.g., up to 6)
    ###    for week in tqdm(range(starting_week - 1, starting_week), desc=f"Year {year}"):
        for week in tqdm(range(1, starting_week), desc=f"Year {year}"):
    ###    for week in tqdm(range(1, 19), desc=f"Year {year}"):
            url = base_url.format(year=year, week=week)
            # print(url) # Uncomment for detailed progress
            week_data = scrape_data(url)
            for row in week_data:
                row["Year"] = year
                row["Week"] = week
                all_data.append(row)
            time.sleep(2)  # Pause for 2 seconds between requests
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_data)
    
    # Cleanup the scraped data
    df['Team'] = df['Team'].str.replace(r'\s\(L\)', '', regex=True)
    df['Team'] = df['Team'].str.replace(r'\s\(W\)', '', regex=True)
    df['Opponent'] = df['Opponent'].str.replace('@', '', regex=True)
    df['Opponent'] = df['Opponent'].str.replace(r'[\t\n\+\-]', '', regex=True)
    df['Opponent'] = (
        df['Opponent']
        .str.strip() # Strip whitespace
        .str[:3]      # Get the first 3 characters
        # Use regex to replace the 3rd character (index 2) with an empty string ('')
        # if the 3rd character is a digit (\d).
        .str.replace(r'^(.{2})\d$', r'\1', regex=True)
    )
    
    df = df[df['Opponent'] != 'BYE']
    
    ###
    df2 = df[df['Year'] != current_year]
    
    
    df = df[df['Year'] == current_year]
    # Save the initial scraped data
    df.to_csv(f'contest-historical-data/historical_pick_data_FV_week_{starting_week - 1}.csv', index=False)
    
    ###
    ###df.to_csv(f'contest-historical-data/historical_pick_data_FV_ALL_Years.csv', index=False)
    
    # ==============================================================================
    # SECTION 2: API DATA COLLECTION (REPLACED BY nflreadpy)
    # ==============================================================================
    
    # NEW REQUIRED IMPORT FOR POLARS EXPRESSIONS
    
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
    
    historical_home_df = pd.read_csv('contest-historical-data/Historical Home and Away data.csv')
    
    historical_home_df = historical_home_df.drop_duplicates()
    
    historical_home_df['Calendar Date'] = pd.to_datetime(
        historical_home_df['Calendar Date'],
        format='mixed',  # Instructs Pandas to infer the format for each string
        dayfirst=False,  # Assuming your dates are year-first or month-first
        errors='coerce'  # Highly recommended: turns unparsable strings into NaT
    )
    
    # Optional: To ensure the time component is consistently 00:00:00 after conversion
    historical_home_df['Calendar Date'] = historical_home_df['Calendar Date'].dt.normalize()
    
    historical_home_df['Calendar Date'] = historical_home_df['Calendar Date'].dt.strftime('%Y-%m-%d')
    
    df_api_schedule = pd.concat([historical_home_df, df_api_schedule], ignore_index=True)
    
    df_api_schedule = df_api_schedule.drop_duplicates()
    
    df_api_schedule['Home Team'] = df_api_schedule['Home Team'].replace('LA', 'LAR')
    df_api_schedule['Home Team'] = df_api_schedule['Home Team'].replace('WAS', 'WSH')
    df_api_schedule['Away Team'] = df_api_schedule['Away Team'].replace('LA', 'LAR')
    df_api_schedule['Away Team'] = df_api_schedule['Away Team'].replace('WAS', 'WSH')
    df_api_schedule['Winner/tie'] = df_api_schedule['Winner/tie'].replace('LA', 'LAR')
    df_api_schedule['Winner/tie'] = df_api_schedule['Winner/tie'].replace('WAS', 'WSH')
    df_api_schedule['Loser/tie'] = df_api_schedule['Loser/tie'].replace('LA', 'LAR')
    df_api_schedule['Loser/tie'] = df_api_schedule['Loser/tie'].replace('WAS', 'WSH')
    
    df_api_schedule = df_api_schedule.drop_duplicates()
    
    # Save the nflreadpy data
    df_api_schedule.to_csv('contest-historical-data/Historical Home and Away data.csv', index=False)
    
    print("nflreadpy data successfully fetched and saved to 'contest-historical-data/Historical Home and Away data.csv'.")
    
    
    # ==============================================================================
    # SECTION 3: DATA CLEANING AND MERGE (ADJUSTED FOR nflreadpy COLUMN NAMES)
    # ==============================================================================
    
    # Your 'teams' dictionary for mapping is now **redundant for the schedule data**
    # since nflreadpy already uses the abbreviations (e.g., ARI, BAL) that your
    # web-scraped data uses. This simplifies the code significantly!
    df = pd.read_csv('contest-historical-data/historical_pick_data_FV_ALL_Years.csv')
    df2 = pd.read_csv(f'contest-historical-data/historical_pick_data_FV_week_{starting_week - 1}.csv')
      
    #condition_to_remove = (
    #    (df['Year'] == current_year)
    #)
    #df = df[~condition_to_remove].copy()
    
    df = df[df['Year'] != current_year]
    
    
    ####condition_to_remove = (
    ####    (df2['Year'] == current_year) & 
    ####    (df2['Week'] == starting_week - 1)
    ####)
    ####df2 = df2[~condition_to_remove].copy()
    
    
    condition_to_remove = (
        (df2['Year'] == current_year) & 
        (df2['Week'] == starting_week)
    )
    df2 = df2[~condition_to_remove].copy()
    
    
    
    df = pd.concat([df, df2], ignore_index=True)
    
    df['Team'] = df['Team'].replace('WSH', 'WAS')
    df['Opponent'] = df['Opponent'].replace('WSH', 'WAS')
    
    df.drop_duplicates()
    
    df.to_csv('contest-historical-data/historical_pick_data_FV_ALL_Years.csv', index=False)
    
    # Existing cleanup of the scraped data
    df = df.replace(r'\u00A0\(W\)', '', regex=True)
    df = df.replace(r'\u00A0\(L\)', '', regex=True)
    df = df.replace(r'\u00A0\(tie\)', '', regex=True)
    df = df.replace(r'\u00A0\(PPD\)', '', regex=True)
    df = df.replace('--', '0.0%', regex=True)
    # Select the desired columns
    df = df[['EV', 'Win %', 'Pick %', 'Team', 'Opponent', 'Future Value (Stars)', 'Year', 'Week']]
    
    # Convert to numeric
    df['Win %'] = pd.to_numeric(df['Win %'].str.rstrip('%')) / 100
    df['Pick %'] = pd.to_numeric(df['Pick %'].str.rstrip('%')) / 100
    df['Pick %'].fillna(0.0, inplace=True)
    df['Public Pick %'] = df['Pick %']
    
    # Convert 'Week' to integer representing the week number
    #df['Week'] = df['Week'].str.replace('Week ', '').astype(int)
    # df['Week'] = pd.to_numeric(df['Week']) # This is now redundant after astype(int)
    
    # Use your existing 'teams' dictionary for *Division* mapping (still needed)
    teams = {
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
        'LA': ['Los Angeles Rams', 'SoFi Stadium', 33.953587, -118.33963, 'America/Los_Angeles', 'NFC West'],
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
        'WAS': ['Washington Commanders', 'FedExField', 38.907697, -76.864517, 'America/New_York', 'NFC East']
    }
    
    # Division mapping
    df['Team Division'] = df['Team'].map(lambda team: teams.get(team, ['', '', '', '', '', ''])[5])
    df['Opponent Division'] = df['Opponent'].map(lambda opponent: teams.get(opponent, ['', '', '', '', '', ''])[5])
    df['Divisional Matchup?'] = (df['Team Division'] == df['Opponent Division']).astype(int)
    
    # Load the historical data from the file created by nflreadpy
    away_data_df = pd.read_csv('contest-historical-data/Historical Home and Away data.csv')
    away_data_df['Calendar Date'] = pd.to_datetime(away_data_df['Calendar Date'])
    
    # Initialization of new columns
    df['Away Team'] = 0
    df[['Availability', 'Calculated Current Week Alive Entries', 'Calculated Current Week Picks', 'Winning Team']] = [0,0,0,0]
    df['Calendar Date'] = pd.NaT
    
    # Merge the dataframes directly (replacing the slow apply/lambda functions)
    
    # 1. Merge to get HOME/AWAY/WINNER
    merged_schedule = pd.merge(
        df,
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
    
    
    # Populate 'Away Team' (binary) and 'Winning Team' (binary)
    df['Away Team'] = (
        merged_schedule['Away Team_away'].notna()
    ).astype(int)
    
    # Winning Team Logic:
    # The team is the winner if it matches the 'Winner/tie' column from either merge
    df['Winning Team'] = (
        (merged_schedule['Winner/tie_home'] == merged_schedule['Team']) | 
        (merged_schedule['Winner/tie_away'] == merged_schedule['Team'])
    ).fillna(0).astype(int)
    
    
    # 2. Merge to get Calendar Date (using the cleaner merge logic from your original script)
    home_dates = away_data_df[['Year', 'Week', 'Home Team', 'Calendar Date']].copy()
    home_dates.rename(columns={'Home Team': 'Team_schedule', 'Calendar Date': 'Matched_Date'}, inplace=True)
    away_dates = away_data_df[['Year', 'Week', 'Away Team', 'Calendar Date']].copy()
    away_dates.rename(columns={'Away Team': 'Team_schedule', 'Calendar Date':'Matched_Date'}, inplace=True)
    
    
    schedule_lookup = pd.concat([home_dates, away_dates]).drop_duplicates(
        subset=['Year', 'Week', 'Team_schedule']
    ).reset_index(drop=True)
    
    schedule_lookup['Team_schedule'] = schedule_lookup['Team_schedule'].replace('LA', 'LAR')
    schedule_lookup['Team_schedule'] = schedule_lookup['Team_schedule'].replace('WSH', 'WAS')
    # Merge with the lookup table for the date
    merged_for_calendar_date = pd.merge(
        df.reset_index(), # Reset index to avoid merge issues
        schedule_lookup,
        left_on=['Year', 'Week', 'Team'],
        right_on=['Year', 'Week', 'Team_schedule'],
        how='left'
    )
    
    
    df['Calendar Date'] = merged_for_calendar_date.set_index('index')['Matched_Date'].values
    # Assuming your conversion worked, or you fix it like we discussed:
    df['Calendar Date'] = pd.to_datetime(df['Calendar Date'], format='%Y-%m-%d')
    #df['Calendar Date_String'] = df['Calendar Date'].dt.strftime('%m/%d/%Y')
    
    # Drop rows where 'Team Division' or 'Opponent Division' is an empty string
    df = df[df['Team Division'] != '']
    df = df[df['Opponent Division'] != '']
    
    df = df[df['Year'] == current_year]
    
    
    df2 = pd.read_csv("contest-historical-data/DK_historical_data.csv")
    df2 = df2[df2['Year'] != current_year]
    
    
    df = df.drop_duplicates()
    df.to_csv(f"contest-historical-data/DK_historical_data_{current_year}.csv", index=False)
    
    df = pd.concat([df2, df], ignore_index=True)
    
    #df = df.drop('Calendar Date_String', axis=1)
    
    df['Calendar Date'] = pd.to_datetime(df['Calendar Date'], format='%Y-%m-%d')
    
    ###
    df = df[~((df['Team'] == 'BUF') &
              (df['Opponent'] == 'CIN') &
              (df['Year'] == 2022) &
              (df['Week'] == 17))]
    
    df = df[~((df['Team'] == 'CIN') &
              (df['Opponent'] == 'BUF') &
              (df['Year'] == 2022) &
              (df['Week'] == 17))]
    ###
    home_df = df
    home_df = home_df.drop_duplicates()
    
    # ==============================================================================
    # SECTION 4: NEW FEATURE ENGINEERING (RANKS AND RELATIVE STATS)
    # ==============================================================================
    
    # Convert 'Calendar Date' to datetime objects to extract the day of the week
    # This assumes the format is YYYY-MM-DD
    home_df['Calendar Date'] = pd.to_datetime(home_df['Calendar Date'])
    
    home_df['Thursday Night Game'] = (
        (home_df['Calendar Date'].dt.dayofweek == 3)
    ).astype(int) # Convert boolean (True/False) to integer (1/0)
    
    # Home vs Away on Thursday
    home_df['Thursday_Home'] = (home_df['Thursday Night Game'] == 1) & (home_df['Away Team'] == 0)
    home_df['Thursday_Away'] = (home_df['Thursday Night Game'] == 1) & (home_df['Away Team'] == 1)
    
    # Favorite vs Underdog on Thursday
    home_df['Thursday_Favorite'] = (home_df['Thursday Night Game'] == 1) & (home_df['Win %'] > .5)
    home_df['Thursday_Underdog'] = (home_df['Thursday Night Game'] == 1) & (home_df['Win %'] <= .5)
    
    # Convert all to integers (1/0)
    cols_to_convert = ['Thursday_Home', 'Thursday_Away', 'Thursday_Favorite', 'Thursday_Underdog']
    home_df[cols_to_convert] = home_df[cols_to_convert].astype(int)
    
    
    
    print("\n⚙️ Starting Feature Engineering (Ranks and Relative Stats)...")
    
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    home_df['Week_Mean_WinPct'] = home_df.groupby(group_keys)['Win %'].transform('mean')
    home_df['Week_Max_WinPct'] = home_df.groupby(group_keys)['Win %'].transform('max')
    home_df['Week_Min_WinPct'] = home_df.groupby(group_keys)['Win %'].transform('min')
    home_df['Week_Std_WinPct'] = home_df.groupby(group_keys)['Win %'].transform('std')
    
    home_df['Week_Mean_FV'] = home_df.groupby(group_keys)['Future Value (Stars)'].transform('mean')
    home_df['Week_Max_FV'] = home_df.groupby(group_keys)['Future Value (Stars)'].transform('max')
    home_df['Week_Min_FV'] = home_df.groupby(group_keys)['Future Value (Stars)'].transform('min')
    home_df['Week_Std_FV'] = home_df.groupby(group_keys)['Future Value (Stars)'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    home_df['Week_Std_WinPct'] = home_df['Week_Std_WinPct'].fillna(0)
    
    # Fill NaN for Std on weeks with only one game (if any)
    home_df['Week_Std_FV'] = home_df['Week_Std_FV'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    home_df['Team_WinPct_RelativeToWeekMean'] = home_df['Win %'] - home_df['Week_Mean_WinPct']
    
    # 2. Calculate Team-Specific Relative Stats

    home_df['Team_FV_RelativeToWeekMean'] = home_df['Future Value (Stars)'] - home_df['Week_Mean_FV']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    home_df['Team_WinPct_RelativeToTopTeam'] = home_df['Win %'] / home_df['Week_Max_WinPct']
    home_df['Team_WinPct_RelativeToTopTeam'] = home_df['Team_WinPct_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)    
    # Handle potential division by zero if Max_Win is 0 (unlikely, but safe)
    home_df['Team_FV_RelativeToTopTeam'] = home_df['Future Value (Stars)'] / home_df['Week_Max_FV']
    home_df['Team_FV_RelativeToTopTeam'] = home_df['Team_FV_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                                  
    
    # 3. Calculate Ranks (Win % and Star Rating)
    # .rank(ascending=False) means the highest value gets rank 1 (e.g., "best")
    home_df['Win % Rank'] = home_df.groupby(group_keys)['Win %'].rank(ascending=False, method='min')
    home_df['Star Rating Rank'] = home_df.groupby(group_keys)['Future Value (Stars)'].rank(ascending=False, method='min')
    
    # 4. Calculate Rank Density
    # First, get the number of teams (games) in each week
    home_df['Num_Teams_This_Week'] = home_df.groupby(group_keys)['Team'].transform('count')
    
    # This normalizes the rank based on the number of available teams that week
    home_df['Rank_Density'] = home_df['Win % Rank'] / home_df['Num_Teams_This_Week']
    
    home_df['FV_Rank_Density'] = home_df['Star Rating Rank'] / home_df['Num_Teams_This_Week']
    
    
    home_df['Is_Top_In_Week'] = (home_df['Win %'] == home_df['Week_Max_WinPct']).astype(int)
    
    home_df['WinPct_Diff_From_Top'] = home_df['Win %'] - home_df['Week_Max_WinPct']
    
    def get_row_future_stats(row, full_df):
        """Calculates future stats for a specific team in a specific year/week."""
        # Filter for same year and FUTURE weeks only
        team_future = full_df[
            (full_df['Year'] == row['Year']) & 
            (full_df['Week'] > row['Week']) &
            (full_df['Team'] == row['Team'])
        ]
        
        # Calculate weeks remaining to avoid hardcoding "20"
        # Or keep your 20 if that is your specific league structure
        weeks_remaining = 18 - row['Week']
        
        if team_future.empty or weeks_remaining <= 0:
            return pd.Series([0.0, 0.0, 0.0, 0.0])
        
        # Calculations
        top_sum = team_future['Is_Top_In_Week'].sum() / weeks_remaining
        high_win = (team_future['Win %'] > 0.80).sum() / weeks_remaining
        mid_win = team_future['Win %'].between(0.70, 0.80).sum() / weeks_remaining
        low_win = team_future['Win %'].between(0.60, 0.70, inclusive='left').sum() / weeks_remaining
        
        return pd.Series([top_sum, high_win, mid_win, low_win])
    
    
    # Apply the logic to every row
    future_stats_cols = [
        'Future_Weeks_Top_Team', 
        'Future_Weeks_Over_80', 
        'Future_Weeks_70_80', 
        'Future_Weeks_60_70'
    ]
    
    home_df[future_stats_cols] = home_df.apply(lambda row: get_row_future_stats(row, home_df), axis=1)
    
    # 3. Add current week relative strength
    home_df['WinPct_Diff_From_Top'] = home_df['Win %'] - home_df.groupby(['Year', 'Week'])['Win %'].transform('max')
    
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    home_df['Week_Mean_80'] = home_df.groupby(group_keys)['Future_Weeks_Over_80'].transform('mean')
    home_df['Week_Max_80'] = home_df.groupby(group_keys)['Future_Weeks_Over_80'].transform('max')
    home_df['Week_Min_80'] = home_df.groupby(group_keys)['Future_Weeks_Over_80'].transform('min')
    home_df['Week_Std_80'] = home_df.groupby(group_keys)['Future_Weeks_Over_80'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    home_df['Week_Std_80'] = home_df['Week_Std_80'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    home_df['Team_80_RelativeToWeekMean'] = home_df['Future_Weeks_Over_80'] - home_df['Week_Mean_80']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    home_df['Team_80_RelativeToTopTeam'] = home_df['Future_Weeks_Over_80'] / home_df['Week_Max_80']
    home_df['Team_80_RelativeToTopTeam'] = home_df['Team_80_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                        
    
    # 3. Calculate Ranks (Win % and Star Rating)
    home_df['80_Rank'] = home_df.groupby(group_keys)['Future_Weeks_Over_80'].rank(ascending=False, method='min')
    
    # This normalizes the rank based on the number of available teams that week
    home_df['80_Rank_Density'] = home_df['80_Rank'] / home_df['Num_Teams_This_Week']
    
    
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    home_df['Week_Mean_70_80'] = home_df.groupby(group_keys)['Future_Weeks_70_80'].transform('mean')
    home_df['Week_Max_70_80'] = home_df.groupby(group_keys)['Future_Weeks_70_80'].transform('max')
    home_df['Week_Min_70_80'] = home_df.groupby(group_keys)['Future_Weeks_70_80'].transform('min')
    home_df['Week_Std_70_80'] = home_df.groupby(group_keys)['Future_Weeks_70_80'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    home_df['Week_Std_70_80'] = home_df['Week_Std_70_80'].fillna(0)
    
    
    home_df['Team_70_80_RelativeToWeekMean'] = home_df['Future_Weeks_70_80'] - home_df['Week_Mean_70_80']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    home_df['Team_70_80_RelativeToTopTeam'] = home_df['Future_Weeks_70_80'] / home_df['Week_Max_70_80']
    home_df['Team_70_80_RelativeToTopTeam'] = home_df['Team_70_80_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                        
    
    # 3. Calculate Ranks (Win % and Star Rating)
    home_df['70_80_Rank'] = home_df.groupby(group_keys)['Future_Weeks_70_80'].rank(ascending=False, method='min')
    
    # This normalizes the rank based on the number of available teams that week
    home_df['70_80_Rank_Density'] = home_df['70_80_Rank'] / home_df['Num_Teams_This_Week']
    
    
    
    
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    home_df['Week_Mean_60_70'] = home_df.groupby(group_keys)['Future_Weeks_60_70'].transform('mean')
    home_df['Week_Max_60_70'] = home_df.groupby(group_keys)['Future_Weeks_60_70'].transform('max')
    home_df['Week_Min_60_70'] = home_df.groupby(group_keys)['Future_Weeks_60_70'].transform('min')
    home_df['Week_Std_60_70'] = home_df.groupby(group_keys)['Future_Weeks_60_70'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    home_df['Week_Std_60_70'] = home_df['Week_Std_60_70'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    home_df['Team_60_70_RelativeToWeekMean'] = home_df['Future_Weeks_60_70'] - home_df['Week_Mean_60_70']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    home_df['Team_60_70_RelativeToTopTeam'] = home_df['Future_Weeks_60_70'] / home_df['Week_Max_60_70']
    home_df['Team_60_70_RelativeToTopTeam'] = home_df['Team_60_70_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                        
    
    # 3. Calculate Ranks (Win % and Star Rating)
    home_df['60_70_Rank'] = home_df.groupby(group_keys)['Future_Weeks_60_70'].rank(ascending=False, method='min')
    
    # This normalizes the rank based on the number of available teams that week
    home_df['60_70_Rank_Density'] = home_df['60_70_Rank'] / home_df['Num_Teams_This_Week']
    
    
    
    
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    home_df['Week_Mean_Top_Team'] = home_df.groupby(group_keys)['Future_Weeks_Top_Team'].transform('mean')
    home_df['Week_Max_Top_Team'] = home_df.groupby(group_keys)['Future_Weeks_Top_Team'].transform('max')
    home_df['Week_Min_Top_Team'] = home_df.groupby(group_keys)['Future_Weeks_Top_Team'].transform('min')
    home_df['Week_Std_Top_Team'] = home_df.groupby(group_keys)['Future_Weeks_Top_Team'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    home_df['Week_Std_Top_Team'] = home_df['Week_Std_Top_Team'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    home_df['Team_Top_Team_RelativeToWeekMean'] = home_df['Future_Weeks_Top_Team'] - home_df['Week_Mean_Top_Team']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    home_df['Team_Top_Team_RelativeToTopTeam'] = home_df['Future_Weeks_Top_Team'] / home_df['Week_Max_Top_Team']
    home_df['Team_Top_Team_RelativeToTopTeam'] = home_df['Team_Top_Team_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                        
    
    # 3. Calculate Ranks (Win % and Star Rating)
    home_df['Top_Team_Rank'] = home_df.groupby(group_keys)['Future_Weeks_Top_Team'].rank(ascending=False, method='min')
    
    # This normalizes the rank based on the number of available teams that week
    home_df['Top_Team_Rank_Density'] = home_df['Top_Team_Rank'] / home_df['Num_Teams_This_Week']
    
    
    
    
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    home_df['Week_Mean_Availability'] = home_df.groupby(group_keys)['Availability'].transform('mean')
    home_df['Week_Max_Availability'] = home_df.groupby(group_keys)['Availability'].transform('max')
    home_df['Week_Min_Availability'] = home_df.groupby(group_keys)['Availability'].transform('min')
    home_df['Week_Std_Availability'] = home_df.groupby(group_keys)['Availability'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    home_df['Week_Std_Availability'] = home_df['Week_Std_Availability'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    home_df['Team_Availability_RelativeToWeekMean'] = home_df['Availability'] - home_df['Week_Mean_Availability']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    home_df['Team_Availability_RelativeToTopTeam'] = home_df['Availability'] / home_df['Week_Max_Availability']
    home_df['Team_Availability_RelativeToTopTeam'] = home_df['Team_Availability_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                        
    
    # 3. Calculate Ranks (Win % and Star Rating)
    home_df['Availability_Rank'] = home_df.groupby(group_keys)['Availability'].rank(ascending=False, method='min')
    
    # This normalizes the rank based on the number of available teams that week
    home_df['Availability_Rank_Density'] = home_df['Availability_Rank'] / home_df['Num_Teams_This_Week']
    
    home_df.to_csv("contest-historical-data/DK_historical_data.csv", index=False)
    
    # ... (The final date manipulation logic remains the same)
    pre_circa_dates = {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019}
    is_not_in_pre_circa = ~home_df['Year'].isin(pre_circa_dates)
    df = home_df[is_not_in_pre_circa].copy()
    
    # Final date manipulation (e.g., correcting Thanksgiving/Christmas week numbers)
    # NOTE: The df.loc assignments must be run *after* the Calendar Date is populated.
    
    # Initialize columns
    df['christmas_week'] = 0
    df['thanksgiving_week'] = 0
    
    # Ensure 'Calendar Date' is datetime format
    df['Calendar Date'] = pd.to_datetime(df['Calendar Date'])
    
    # --- Year 2026 ---
    # Note: Fixed typo (you had 2023-11-28 for the second date)
    condition_2026_date = (df['Year'] == 2026) & (df['Calendar Date'] >= pd.to_datetime('2026-11-28'))
    df.loc[condition_2026_date, 'Week'] += 1
    
    condition_2026_week = (df['Year'] == 2026) & (df['Calendar Date'] >= pd.to_datetime('2025-12-26'))
    df.loc[condition_2026_week, 'Week'] += 1
    
    condition_2026_xmas = (df['Year'] == 2026) & (df['Calendar Date'] == pd.to_datetime('2026-12-25'))
    df.loc[condition_2026_xmas, 'christmas_week'] = 1
    
    condition_2026_thanksgiving = (df['Year'] == 2026) & (
        (df['Calendar Date'] == pd.to_datetime('2026-11-26')) | 
        (df['Calendar Date'] == pd.to_datetime('2026-11-27'))
    )
    df.loc[condition_2026_thanksgiving, 'thanksgiving_week'] = 1
    
    # --- Year 2025 ---
    # Note: Fixed typo (you had 2023-11-28 for the second date)
    condition_2025_date = (df['Year'] == 2025) & (df['Calendar Date'] >= pd.to_datetime('2025-11-29'))
    df.loc[condition_2025_date, 'Week'] += 1
    
    condition_2025_week = (df['Year'] == 2025) & (df['Calendar Date'] >= pd.to_datetime('2025-12-26'))
    df.loc[condition_2025_week, 'Week'] += 1
    
    condition_2025_xmas = (df['Year'] == 2025) & (df['Calendar Date'] == pd.to_datetime('2025-12-25'))
    df.loc[condition_2025_xmas, 'christmas_week'] = 1
    
    condition_2025_thanksgiving = (df['Year'] == 2025) & (
        (df['Calendar Date'] == pd.to_datetime('2025-11-27')) | 
        (df['Calendar Date'] == pd.to_datetime('2025-11-28'))
    )
    df.loc[condition_2025_thanksgiving, 'thanksgiving_week'] = 1
    
    # --- Year 2024 ---
    condition_2024_date = (df['Year'] == 2024) & (df['Calendar Date'] >= pd.to_datetime('2024-11-30'))
    df.loc[condition_2024_date, 'Week'] += 1
    
    condition_2024_week = (df['Year'] == 2024) & (df['Calendar Date'] >= pd.to_datetime('2024-12-27'))
    df.loc[condition_2024_week, 'Week'] += 1
    
    condition_2024_xmas = (df['Year'] == 2024) & (df['Calendar Date'] == pd.to_datetime('2024-12-25'))
    df.loc[condition_2024_xmas, 'christmas_week'] = 1
    
    condition_2024_thanksgiving = (df['Year'] == 2024) & (
        (df['Calendar Date'] == pd.to_datetime('2024-11-28')) | 
        (df['Calendar Date'] == pd.to_datetime('2024-11-29'))
    )
    df.loc[condition_2024_thanksgiving, 'thanksgiving_week'] = 1
    
    # --- Year 2023 ---
    condition_2023_date = (df['Year'] == 2023) & (df['Calendar Date'] >= pd.to_datetime('2023-11-25'))
    df.loc[condition_2023_date, 'Week'] += 1
    
    condition_2023_week = (df['Year'] == 2023) & (df['Calendar Date'] >= pd.to_datetime('2023-12-26'))
    df.loc[condition_2023_week, 'Week'] += 1
    
    condition_2023_xmas = (df['Year'] == 2023) & (df['Calendar Date'] == pd.to_datetime('2023-12-25'))
    df.loc[condition_2023_xmas, 'christmas_week'] = 1
    
    condition_2023_thanksgiving = (df['Year'] == 2023) & (
        (df['Calendar Date'] == pd.to_datetime('2023-11-23')) | 
        (df['Calendar Date'] == pd.to_datetime('2023-11-24'))
    )
    df.loc[condition_2023_thanksgiving, 'thanksgiving_week'] = 1
    
    # --- Year 2022 ---
    condition_2022_date = (df['Year'] == 2022) & (df['Calendar Date'] >= pd.to_datetime('2022-11-25'))
    df.loc[condition_2022_date, 'Week'] += 1
    
    condition_2022_week = (df['Year'] == 2022) & (df['Calendar Date'] >= pd.to_datetime('2022-12-26'))
    df.loc[condition_2022_week, 'Week'] += 1
    
    condition_2022_xmas = (df['Year'] == 2022) & (df['Calendar Date'] == pd.to_datetime('2022-12-25'))
    df.loc[condition_2022_xmas, 'christmas_week'] = 1
    
    condition_2022_thanksgiving = (df['Year'] == 2022) & (df['Calendar Date'] == pd.to_datetime('2022-11-24'))
    df.loc[condition_2022_thanksgiving, 'thanksgiving_week'] = 1
    
    # --- Year 2021 ---
    condition_2021_date = (df['Year'] == 2021) & (df['Calendar Date'] >= pd.to_datetime('2021-11-26'))
    df.loc[condition_2021_date, 'Week'] += 1
    
    condition_2021_week = (df['Year'] == 2021) & (df['Calendar Date'] >= pd.to_datetime('2021-12-26'))
    df.loc[condition_2021_week, 'Week'] += 1
    
    condition_2021_xmas = (df['Year'] == 2021) & (df['Calendar Date'] == pd.to_datetime('2021-12-25'))
    df.loc[condition_2021_xmas, 'christmas_week'] = 1
    
    condition_2021_thanksgiving = (df['Year'] == 2021) & (df['Calendar Date'] == pd.to_datetime('2021-11-25'))
    df.loc[condition_2021_thanksgiving, 'thanksgiving_week'] = 1
    
    # --- Year 2020 ---
    condition_2020_date = (df['Year'] == 2020) & (df['Calendar Date'] >= pd.to_datetime('2020-11-27'))
    df.loc[condition_2020_date, 'Week'] += 1
    
    condition_2020_thanksgiving = (df['Year'] == 2020) & (df['Calendar Date'] == pd.to_datetime('2020-11-26'))
    df.loc[condition_2020_thanksgiving, 'thanksgiving_week'] = 1
    
    
    ## 1. Initiate 6 Columns as 0
    new_cols = ['Thanksgiving Favorite', 'Thanksgiving Underdog',
                'Christmas Favorite', 'Christmas Underdog',
                'Pre Thanksgiving', 'Pre Christmas']
    df[new_cols] = 0
    
    
    ## 2. Define Variables (Using Dictionaries for Years)
    # Using strings for keys aligns with 'Year' column if it's an integer
    thanksgiving_weeks = {
        2020: 12, 2021: 12, 2022: 12, 2023: 12, 2024: 13, 2025: 13
    }
    christmas_weeks = {
        2021: 17, 2022: 18, 2023: 18, 2024: 18, 2025: 18 
    }
    
    
    ## 3. Define 22 Lists (Using Dictionaries for Years/Events)
    # You will populate these lists with the actual team names
    thanksgiving_favorites = {
        2020: ['HOU', 'DAL'], 2021: ['BUF', 'DAL', 'CHI'], 2022: ['DAL', 'MIN', 'BUF'], 2023: ['DAL', 'MIA', 'DET', 'SF'], 
        2024: ['KC', 'DET', 'GB', 'DAL'], 2025: ['DET', 'BAL', 'KC', 'PHI']
    }
    thanksgiving_underdogs = {
        2020: ['DET', 'WAS'], 2021: ['DET', 'LV', 'NO'], 2022: ['NE', 'DET', 'NYG'], 2023: ['SEA', 'GB', 'NYJ', 'WAS'], 
        2024: ['NYG', 'MIA', 'LV', 'CHI'], 2025: ['GB', 'DAL', 'CIN', 'CHI']
    }
    christmas_favorites = {
        2021: ['GB', 'SF', 'ARI'], 2022: ['TB', 'MIA', 'LAC', 'DEN'], 2023: ['KC', 'SF', 'PHI'], 2024: ['BAL', 'SEA', 'KC'], 2025: ['DAL', 'DET', 'KC']
    }
    christmas_underdogs = {
        2021: ['IND', 'TEN', 'CLE'], 2022: ['ARI', 'GB', 'LAR', 'IND'], 2023: ['LV', 'NYG', 'BAL'], 2024: ['PIT', 'CHI', 'HOU'], 2025: ['WAS', 'MIN', 'DEN']
    }
    
    
    ## 4. Apply Conditional Logic (The Main Update)
    
    # --- Thanksgiving Logic (2020-2025) ---
    for year in thanksgiving_weeks.keys():
        # Convert year to integer for comparison
        current_year = int(year) 
        
        # 1. Update 'Thanksgiving Favorite'
        fav_teams = thanksgiving_favorites.get(current_year, [])
        if fav_teams:
            condition_fav = (df['Year'] == current_year) & (df['Team'].isin(fav_teams))
            df.loc[condition_fav, 'Thanksgiving Favorite'] = 1
    
        # 2. Update 'Thanksgiving Underdog'
        underdog_teams = thanksgiving_underdogs.get(current_year, [])
        if underdog_teams:
            condition_udg = (df['Year'] == current_year) & (df['Team'].isin(underdog_teams))
            df.loc[condition_udg, 'Thanksgiving Underdog'] = 1
    
        # 3. Update 'Pre Thanksgiving'
        # Condition: Team is a favorite OR underdog AND it's before the holiday week
        holiday_week = thanksgiving_weeks[current_year]
        
        pre_teams = fav_teams + underdog_teams
        if pre_teams:
            condition_pre = (df['Year'] == current_year) & \
                            (df['Team'].isin(pre_teams)) & \
                            (df['Week'] < holiday_week)
            df.loc[condition_pre, 'Pre Thanksgiving'] = 1
    
    
    # --- Christmas Logic (2021-2025) ---
    for year in christmas_weeks.keys():
        current_year = int(year) 
    
        # 1. Update 'Christmas Favorite'
        fav_teams = christmas_favorites.get(current_year, [])
        if fav_teams:
            condition_fav = (df['Year'] == current_year) & (df['Team'].isin(fav_teams))
            df.loc[condition_fav, 'Christmas Favorite'] = 1
    
        # 2. Update 'Christmas Underdog'
        underdog_teams = christmas_underdogs.get(current_year, [])
        if underdog_teams:
            condition_udg = (df['Year'] == current_year) & (df['Team'].isin(underdog_teams))
            df.loc[condition_udg, 'Christmas Underdog'] = 1
    
        # 3. Update 'Pre Christmas'
        holiday_week = christmas_weeks[current_year]
        
        pre_teams = fav_teams + underdog_teams
        if pre_teams:
            condition_pre = (df['Year'] == current_year) & \
                            (df['Team'].isin(pre_teams)) & \
                            (df['Week'] < holiday_week)
            df.loc[condition_pre, 'Pre Christmas'] = 1
    
    # 1. Create lookup maps for the Win % on the actual holiday weeks
    # This isolates the team's strength specifically on the day of the holiday
    xmas_map = df[df['christmas_week'] == 1].set_index(['Year', 'Team'])['Win %']
    tgiving_map = df[df['thanksgiving_week'] == 1].set_index(['Year', 'Team'])['Win %']
    
    # 2. Map those holiday-specific Win percentages back to every row for that team/year
    # This allows the "Pre holiday" rows to "know" how strong the team is on the upcoming holiday
    df['christmas_win_pct'] = df.set_index(['Year', 'Team']).index.map(xmas_map).fillna(0)
    df['thanksgiving_win_pct'] = df.set_index(['Year', 'Team']).index.map(tgiving_map).fillna(0)
    
    # 3. Apply your interaction logic
    # This turns the 'Pre' binary flag into a continuous "Expectation" variable
    df['Pre Christmas'] = df['Pre Christmas'] * df['christmas_win_pct']
    df['Pre Thanksgiving'] = df['Pre Thanksgiving'] * df['thanksgiving_win_pct']
    
    # 4. Create the final aggregate feature
    df['Holiday Strength'] = df['Pre Thanksgiving'] + df['Pre Christmas']
    
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    print("  Calculating weekly Win % statistics (mean, max, min, std)...")
    df['Week_Mean_WinPct'] = df.groupby(group_keys)['Win %'].transform('mean')
    df['Week_Max_WinPct'] = df.groupby(group_keys)['Win %'].transform('max')
    df['Week_Min_WinPct'] = df.groupby(group_keys)['Win %'].transform('min')
    df['Week_Std_WinPct'] = df.groupby(group_keys)['Win %'].transform('std')
    
    print("  Calculating weekly Future Value statistics (mean, max, min, std)...")
    df['Week_Mean_FV'] = df.groupby(group_keys)['Future Value (Stars)'].transform('mean')
    df['Week_Max_FV'] = df.groupby(group_keys)['Future Value (Stars)'].transform('max')
    df['Week_Min_FV'] = df.groupby(group_keys)['Future Value (Stars)'].transform('min')
    df['Week_Std_FV'] = df.groupby(group_keys)['Future Value (Stars)'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    df['Week_Std_WinPct'] = df['Week_Std_WinPct'].fillna(0)
    
    # Fill NaN for Std on weeks with only one game (if any)
    df['Week_Std_FV'] = df['Week_Std_FV'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    df['Team_WinPct_RelativeToWeekMean'] = df['Win %'] - df['Week_Mean_WinPct']
    
    # 2. Calculate Team-Specific Relative Stats
    df['Team_FV_RelativeToWeekMean'] = df['Future Value (Stars)'] - df['Week_Mean_FV']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    df['Team_WinPct_RelativeToTopTeam'] = df['Win %'] / df['Week_Max_WinPct']
    df['Team_WinPct_RelativeToTopTeam'] = df['Team_WinPct_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Handle potential division by zero if Max_Win is 0 (unlikely, but safe)
    df['Team_FV_RelativeToTopTeam'] = df['Future Value (Stars)'] / df['Week_Max_FV']
    df['Team_FV_RelativeToTopTeam'] = df['Team_FV_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                                  
    
    # 3. Calculate Ranks (Win % and Star Rating)
    # .rank(ascending=False) means the highest value gets rank 1 (e.g., "best")
    df['Win % Rank'] = df.groupby(group_keys)['Win %'].rank(ascending=False, method='min')
    df['Star Rating Rank'] = df.groupby(group_keys)['Future Value (Stars)'].rank(ascending=False, method='min')
    
    # 4. Calculate Rank Density
    # First, get the number of teams (games) in each week
    df['Num_Teams_This_Week'] = df.groupby(group_keys)['Team'].transform('count')
    
    # This normalizes the rank based on the number of available teams that week
    df['Rank_Density'] = df['Win % Rank'] / df['Num_Teams_This_Week']
    
    df['FV_Rank_Density'] = df['Star Rating Rank'] / df['Num_Teams_This_Week']
     
    # ------------------------------------------------------------------------------
    # NEW SECTION: Future Value & Holiday Features
    # ------------------------------------------------------------------------------
    
    # B. Current Week Relative Strength
    # "Win Percentage of the team minus the win percentage of the Top team that week."
    # Note: 'Week_Max_WinPct' was calculated in Section 4
    df['Is_Top_In_Week'] = (df['Win %'] == df['Week_Max_WinPct']).astype(int)
    
    df['WinPct_Diff_From_Top'] = df['Win %'] - df['Week_Max_WinPct']
    
    def get_row_future_stats(row, full_df):
        # Filter for same year, same team, and FUTURE weeks
        team_future = full_df[
            (full_df['Year'] == row['Year']) & 
            (full_df['Week'] > row['Week']) &
            (full_df['Team'] == row['Team'])
        ]
        
        # Calculate weeks remaining to avoid hardcoding "20"
        # Or keep your 20 if that is your specific league structure
        weeks_remaining = 20 - row['Week']
        
        if team_future.empty or weeks_remaining <= 0:
            return pd.Series([0.0, 0.0, 0.0, 0.0])
        
        # Calculations
        top_sum = team_future['Is_Top_In_Week'].sum() / weeks_remaining
        high_win = (team_future['Win %'] > 0.80).sum() / weeks_remaining
        mid_win = team_future['Win %'].between(0.70, 0.80).sum() / weeks_remaining
        low_win = team_future['Win %'].between(0.60, 0.70, inclusive='left').sum() / weeks_remaining
        
        return pd.Series([top_sum, high_win, mid_win, low_win])
    
    # Apply the logic to every row
    future_stats_cols = [
        'Future_Weeks_Top_Team', 
        'Future_Weeks_Over_80', 
        'Future_Weeks_70_80', 
        'Future_Weeks_60_70'
    ]
    
    df[future_stats_cols] = df.apply(lambda row: get_row_future_stats(row, df), axis=1)
    
    # 3. Add current week relative strength
    df['WinPct_Diff_From_Top'] = df['Win %'] - df.groupby(['Year', 'Week'])['Win %'].transform('max')
    
    
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    df['Week_Mean_80'] = df.groupby(group_keys)['Future_Weeks_Over_80'].transform('mean')
    df['Week_Max_80'] = df.groupby(group_keys)['Future_Weeks_Over_80'].transform('max')
    df['Week_Min_80'] = df.groupby(group_keys)['Future_Weeks_Over_80'].transform('min')
    df['Week_Std_80'] = df.groupby(group_keys)['Future_Weeks_Over_80'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    df['Week_Std_80'] = df['Week_Std_80'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    df['Team_80_RelativeToWeekMean'] = df['Future_Weeks_Over_80'] - df['Week_Mean_80']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    df['Team_80_RelativeToTopTeam'] = df['Future_Weeks_Over_80'] / df['Week_Max_80']
    df['Team_80_RelativeToTopTeam'] = df['Team_80_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                        
    
    # 3. Calculate Ranks (Win % and Star Rating)
    df['80_Rank'] = df.groupby(group_keys)['Future_Weeks_Over_80'].rank(ascending=False, method='min')
    
    # This normalizes the rank based on the number of available teams that week
    df['80_Rank_Density'] = df['80_Rank'] / df['Num_Teams_This_Week']
    
    
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    df['Week_Mean_70_80'] = df.groupby(group_keys)['Future_Weeks_70_80'].transform('mean')
    df['Week_Max_70_80'] = df.groupby(group_keys)['Future_Weeks_70_80'].transform('max')
    df['Week_Min_70_80'] = df.groupby(group_keys)['Future_Weeks_70_80'].transform('min')
    df['Week_Std_70_80'] = df.groupby(group_keys)['Future_Weeks_70_80'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    df['Week_Std_70_80'] = df['Week_Std_70_80'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    df['Team_70_80_RelativeToWeekMean'] = df['Future_Weeks_70_80'] - df['Week_Mean_70_80']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    df['Team_70_80_RelativeToTopTeam'] = df['Future_Weeks_70_80'] / df['Week_Max_70_80']
    df['Team_70_80_RelativeToTopTeam'] = df['Team_70_80_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                        
    
    # 3. Calculate Ranks (Win % and Star Rating)
    df['70_80_Rank'] = df.groupby(group_keys)['Future_Weeks_70_80'].rank(ascending=False, method='min')
    
    # This normalizes the rank based on the number of available teams that week
    df['70_80_Rank_Density'] = df['70_80_Rank'] / df['Num_Teams_This_Week']
    
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    print("  Calculating weekly Win % statistics (mean, max, min, std)...")
    df['Week_Mean_60_70'] = df.groupby(group_keys)['Future_Weeks_60_70'].transform('mean')
    df['Week_Max_60_70'] = df.groupby(group_keys)['Future_Weeks_60_70'].transform('max')
    df['Week_Min_60_70'] = df.groupby(group_keys)['Future_Weeks_60_70'].transform('min')
    df['Week_Std_60_70'] = df.groupby(group_keys)['Future_Weeks_60_70'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    df['Week_Std_60_70'] = df['Week_Std_60_70'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    df['Team_60_70_RelativeToWeekMean'] = df['Future_Weeks_60_70'] - df['Week_Mean_60_70']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    df['Team_60_70_RelativeToTopTeam'] = df['Future_Weeks_60_70'] / df['Week_Max_60_70']
    df['Team_60_70_RelativeToTopTeam'] = df['Team_60_70_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                        
    
    # 3. Calculate Ranks (Win % and Star Rating)
    df['60_70_Rank'] = df.groupby(group_keys)['Future_Weeks_60_70'].rank(ascending=False, method='min')
    
    # This normalizes the rank based on the number of available teams that week
    df['60_70_Rank_Density'] = df['60_70_Rank'] / df['Num_Teams_This_Week']
    
    
    
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    df['Week_Mean_Top_Team'] = df.groupby(group_keys)['Future_Weeks_Top_Team'].transform('mean')
    df['Week_Max_Top_Team'] = df.groupby(group_keys)['Future_Weeks_Top_Team'].transform('max')
    df['Week_Min_Top_Team'] = df.groupby(group_keys)['Future_Weeks_Top_Team'].transform('min')
    df['Week_Std_Top_Team'] = df.groupby(group_keys)['Future_Weeks_Top_Team'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    df['Week_Std_Top_Team'] = df['Week_Std_Top_Team'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    df['Team_Top_Team_RelativeToWeekMean'] = df['Future_Weeks_Top_Team'] - df['Week_Mean_Top_Team']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    df['Team_Top_Team_RelativeToTopTeam'] = df['Future_Weeks_Top_Team'] / df['Week_Max_Top_Team']
    df['Team_Top_Team_RelativeToTopTeam'] = df['Team_Top_Team_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                        
    
    # 3. Calculate Ranks (Win % and Star Rating)
    df['Top_Team_Rank'] = df.groupby(group_keys)['Future_Weeks_Top_Team'].rank(ascending=False, method='min')
    
    # This normalizes the rank based on the number of available teams that week
    df['Top_Team_Rank_Density'] = df['Top_Team_Rank'] / df['Num_Teams_This_Week']
    
    print("✅ Feature engineering complete.")
    
    
    ###df['EV'] = 0 
    
    ###df = df[df['Year'] == current_year]
    
    df2 = pd.read_csv("contest-historical-data/Circa_historical_data.csv")
    df2 = df2[df2['Year'] != current_year]
    df = df[df['Year'] == current_year]
    
    
    # Convert 'Calendar Date' to datetime objects to extract the day of the week
    # This assumes the format is YYYY-MM-DD
    df['Calendar Date'] = pd.to_datetime(df['Calendar Date'])
    
    # Create the "Thursday Night Game" column
    # Logic:
    # 1. Day of week is Thursday (dt.dayofweek == 3; Monday is 0, Sunday is 6)
    # 2. christmas_week is 0
    # 3. thanksgiving_week is 0
    df['Thursday Night Game'] = 0
    
    df['Thursday Night Game'] = (
        (df['Calendar Date'].dt.dayofweek == 3) & 
        (df['christmas_week'] == 0) & 
        (df['thanksgiving_week'] == 0)
    ).astype(int) # Convert boolean (True/False) to integer (1/0)
    
    # Home vs Away on Thursday
    df['Thursday_Home'] = (df['Thursday Night Game'] == 1) & (df['Away Team'] == 0)
    df['Thursday_Away'] = (df['Thursday Night Game'] == 1) & (df['Away Team'] == 1)
    
    # Favorite vs Underdog on Thursday
    df['Thursday_Favorite'] = (df['Thursday Night Game'] == 1) & (df['Win %'] > .5)
    df['Thursday_Underdog'] = (df['Thursday Night Game'] == 1) & (df['Win %'] <= .5)
    
    # Convert all to integers (1/0)
    cols_to_convert = ['Thursday_Home', 'Thursday_Away', 'Thursday_Favorite', 'Thursday_Underdog']
    df[cols_to_convert] = df[cols_to_convert].astype(int)
    
    
    df = df.drop_duplicates()
    df2 = df2.drop_duplicates()
    #df2['Calendar Date'] = pd.to_datetime(df2['Calendar Date'], format='%Y-%m-%d %H:%M:%S')
    #df2['Calendar Date_String'] = df2['Calendar Date'].dt.strftime('%m/%d/%Y')
    
    df.to_csv(f"contest-historical-data/Circa_historical_data_{current_year}.csv", index=False)
    
    df = pd.concat([df2, df], ignore_index=True)
    
    
    #df.drop(columns=["Calculated Total Picks"], inplace=True)
    #df = df.drop('Calendar Date_String', axis=1)
    
    df['Calendar Date'] = pd.to_datetime(df['Calendar Date'], format='%Y-%m-%d')
    
    df = df.drop_duplicates()
    
    df.to_csv("contest-historical-data/Circa_historical_data.csv", index=False)
    df.to_csv(f"contest-historical-data/Circa_historical_data.csv", index=False)
    
    
    def scrape_circa_survivor_picks():
        """
        Scrapes Circa Survivor entry data up to a defined starting week and saves it to a CSV file.
        
        The script implements the following logic:
        1. Defines 'starting_week' (e.g., 10).
        2. Only includes weekly pick columns up to Week_(starting_week - 1).
        3. Any empty picks in the included columns are filled with 'ELIMINATED'.
        """
        
        BASE_URL = "https://poolgenius.teamrankings.com/circa-survivor-picks/entry-pick-history/"
        
    
        # List of ALL 20 potential week headers (used for mapping the full HTML table)
        ALL_20_WEEK_HEADERS = [
            "Week_1", "Week_2", "Week_3", "Week_4", "Week_5", "Week_6", 
            "Week_7", "Week_8", "Week_9", "Week_10", "Week_11", "Week_12", 
            "Week_13", # Corresponds to "Th" (Thanksgiving)
            "Week_14", # Corresponds to "13"
            "Week_15", # Corresponds to "14"
            "Week_16", # Corresponds to "15"
            "Week_17", # Corresponds to "16"
            "Week_18", # Corresponds to "Ch" (Christmas)
            "Week_19", # Corresponds to "17"
            "Week_20"  # Corresponds to "18"
        ]
    
        # The actual headers to use for the CSV output (Entry_Name, Total_Wins, and weeks up to starting_week - 1)
        WEEKS_TO_KEEP = ALL_20_WEEK_HEADERS########[:NUM_WEEKS_TO_KEEP] 
        FIXED_HEADERS = ["EntryName", "Total_Wins"]
########        COLUMN_HEADERS = FIXED_HEADERS + WEEKS_TO_KEEP
        COLUMN_HEADERS = FIXED_HEADERS + ALL_20_WEEK_HEADERS
        
        
        FIRST_PICK_CELL_INDEX = 2
########        LAST_PICK_CELL_INDEX = FIRST_PICK_CELL_INDEX + NUM_WEEKS_TO_KEEP # The index up to (but not including) which we scrape
        LAST_PICK_CELL_INDEX = FIRST_PICK_CELL_INDEX + len(ALL_20_WEEK_HEADERS) # The index up to (but not including) which we scrape
    
        all_entries_data = []
        
        print(f"Starting scrape from page 1 to {MAX_PAGES}...")
        print(f"Limiting output columns to Week_1 through Week_{NUM_WEEKS_TO_KEEP} (since starting_week={starting_week}).")
    
        # Iterate through all pages
        for page in range(1, MAX_PAGES + 1):
            url = f"{BASE_URL}?season={current_year}&page={page}"
            
            # Simple exponential backoff for retries
            for attempt in range(3):
                try:
                    #print(f"-> Scraping page {page}/{MAX_PAGES}: {url}")
                    
                    # Use a specific user-agent to look like a regular browser
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find the main data table
                    table = soup.find('table', class_='planner-grid')
                    
                    if not table:
                        print(f"!!! Warning: Could not find table on page {page}. Skipping.")
                        break # Break out of retry loop for this page
    
                    # Get all rows in the table body
                    rows = table.find('tbody').find_all('tr')
                    
                    # Process each entry row
                    for row in rows:
                        entry_data = {}
                        cells = row.find_all('td')
                        
                        if not cells:
                            continue
                            
                        # 1. Entry Name (Index 0 in cells, Index 0 in COLUMN_HEADERS)
                        entry_data[COLUMN_HEADERS[0]] = cells[0].text.strip()
                        
                        # 2. Total Wins (Index 1 in cells, Index 1 in COLUMN_HEADERS)
                        entry_data[COLUMN_HEADERS[1]] = cells[1].text.strip()
                        
                        # 3. Weekly Picks (starting from index 2 up to LAST_PICK_CELL_INDEX)
                        for i in range(FIRST_PICK_CELL_INDEX, LAST_PICK_CELL_INDEX):
                            cell_index = i
                            header_name = COLUMN_HEADERS[i] # Safely gets Week_1, Week_2, ..., Week_9
                            
                            # Check if the cell exists (important for potentially incomplete data)
                            if cell_index < len(cells):
                                cell = cells[cell_index]
                                
                                # The cell content is the team name (e.g., 'DEN').
                                team_name = cell.text.strip()
                                
                                # Record only the team name if a pick was made, otherwise empty string.
                                if team_name:
                                    entry_data[header_name] = team_name
                                else:
                                    entry_data[header_name] = "" # Empty if no pick
                            else:
                                # If the row is shorter than expected, fill remaining with empty string
                                entry_data[header_name] = ""
                                
                        all_entries_data.append(entry_data)
                    
                    #print(f"   -> Successfully scraped {len(rows)} entries from page {page}.")
                    
                    # Success, break retry loop
                    break 
    
                except requests.exceptions.HTTPError as e:
                    print(f"!!! HTTP Error on page {page} (Attempt {attempt + 1}): {e}")
                except requests.exceptions.ConnectionError as e:
                    print(f"!!! Connection Error on page {page} (Attempt {attempt + 1}): {e}")
                except requests.exceptions.Timeout:
                    print(f"!!! Timeout Error on page {page} (Attempt {attempt + 1}).")
                except Exception as e:
                    print(f"!!! An unexpected error occurred on page {page} (Attempt {attempt + 1}): {e}")
    
                # Wait longer for the next attempt, plus a random jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"   -> Waiting {wait_time:.2f} seconds before retrying page {page}...")
                time.sleep(wait_time)
                
            # Always be polite, even after success. Wait for 1 to 3 seconds between pages.
            if page < MAX_PAGES:
                delay = random.uniform(1, 3)
                # print(f"   -> Waiting {delay:.2f} seconds before next page...")
                time.sleep(delay)
    
        # ------------------
        # Data Compilation
        # ------------------
        if not all_entries_data:
            print("\nScraping complete, but no data was collected. Please check the URL and HTML structure.")
            return
            
        # --- STEP 4: Imputation of Eliminated Entries ---
        print(f"Applying 'ELIMINATED' status for empty picks in Week_1 through Week_{NUM_WEEKS_TO_KEEP}...")
        
        # Check only the weeks we are keeping
########        weeks_to_check = WEEKS_TO_KEEP 
        weeks_to_check = ALL_20_WEEK_HEADERS
    
        for entry in all_entries_data:
            for week_header in ALL_20_WEEK_HEADERS:
                # If the pick is empty (set as "" in the scraping loop), fill with "ELIMINATED"
                if entry.get(week_header) == "":
                    entry[week_header] = "ELIMINATED"
    
    
        # Convert the list of dictionaries into a Pandas DataFrame, using the truncated COLUMN_HEADERS
        df = pd.DataFrame(all_entries_data, columns=COLUMN_HEADERS)
        
        # Save the DataFrame to a CSV file
        output_filename = f"circa-pick-history/{current_year}_survivor_picks.csv"
        df.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8')
    
        print(f"\n✅ Scraping and processing finished!")
        print(f"Total entries collected: {len(df)}")
        print(f"Data saved to: {output_filename}")
    
    
    if __name__ == "__main__":
        # Ensure all Firebase variables are defined as this is a standalone script
        if 'getAuth' in globals(): # A check for the canvas environment
            print("Note: Running as a standalone Python script, not using Firebase initialization.")
        
        scrape_circa_survivor_picks()
    
    # --- User Provided Data ---
    team_dictionary = {
        "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
        "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
        "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GB",
        "HOU": "HOU", "IND": "IND", "JAX": "JAC", "KC": "KC",
        "LAC": "LAC", "LAR": "LAR", "LV": "LV", "MIA": "MIA",
        "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG",
        "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA",
        "SF": "SF", "TB": "TB", "TEN": "TEN", "WAS": "WSH"
    }
    
    # --- File Names ---
    historical_data_file = f"contest-historical-data/Circa_historical_data_{current_year}.csv"
    ####historical_data_file = "contest-historical-data/Circa_historical_data.csv"
    
    
    # Assuming the pick data CSVs are named like '2025_survivor_picks.csv'
    pick_data_base_path = "circa-pick-history/"
    
    ####output_file = f"contest-historical-data/Circa_historical_data.csv"
    output_file = f"contest-historical-data/Circa_historical_data_{current_year}.csv"
    
    # --- Correction: Use the exact string the scraper uses for eliminated entries ---
    ELIMINATED_MARKER = "ELIMINATED"
    
    def get_total_entries(year):
        """Helper to get the total starting entries for a given year."""
        map_ = {
            2020: circa_2020_entries, 2021: circa_2021_entries,
            2022: circa_2022_entries, 2023: circa_2023_entries,
            2024: circa_2024_entries, 2025: circa_2025_entries,
            2026: circa_2026_entries
        }
        return map_.get(year, 0)
    
    def update_historical_data():
        try:
            main_df = pd.read_csv(historical_data_file)
            print(f"Successfully loaded '{historical_data_file}'. Processing {len(main_df)} rows.")
        except FileNotFoundError:
            print(f"Error: Main historical data file not found at '{historical_data_file}'")
            return
        except Exception as e:
            print(f"Error loading '{historical_data_file}': {e}")
            return
    
        # Define Column Names
        alive_entries_col_name = "Calculated Current Week Alive Entries"
        prior_picks_col_name = "Calculated Prior Week Picks by Alive Entries" 
        weeks_team_picks_col_name = "Calculated Current Week Picks"
        availability_col_name = "Availability"
        pick_percentage_col_name = "Pick %"
        entry_remaining_percent_col_name = "Entry Remaining Percent"
    
        # Initialize New Columns
        main_df[alive_entries_col_name] = pd.NA
        main_df[prior_picks_col_name] = pd.NA
        main_df[weeks_team_picks_col_name] = pd.NA
        main_df[availability_col_name] = pd.NA
        main_df[pick_percentage_col_name] = pd.NA
        main_df[entry_remaining_percent_col_name] = pd.NA
        main_df = main_df.rename(columns={'Date': 'Week'})
        
        # Drop old column if present
        if "Calculated Total Picks" in main_df.columns:
            main_df = main_df.drop(columns=["Calculated Total Picks"])
        
        pick_dataframes_cache = {}
    
        for index, row in main_df.iterrows():
            team_abbr = row["Team"]
            try:
                year = int(row["Year"])
                current_week_num = int(row["Week"])
            except ValueError:
                print(f"Warning: Could not parse Year/Week for row {index}. Skipping calculations for this row.")
                continue
                
            full_team_name = team_dictionary.get(team_abbr)
            
            pick_df = None
            if year in pick_dataframes_cache:
                pick_df = pick_dataframes_cache[year]
            else:
                pick_file_path = os.path.join(pick_data_base_path, f"{year}_survivor_picks.csv")
                try:
                    # Assuming the pick files are named like '2025_survivor_picks.csv'
                    current_year_pick_df = pd.read_csv(pick_file_path)
                    
                    # Use EntryName as index for faster lookups
                    if "EntryName" in current_year_pick_df.columns:
                        # Drop=False is important to keep the column if needed elsewhere
                        current_year_pick_df = current_year_pick_df.set_index("EntryName", drop=False)
                    else:
                        current_year_pick_df.index.name = "EntryName"  
                    
                    pick_dataframes_cache[year] = current_year_pick_df
                    pick_df = current_year_pick_df
                except FileNotFoundError:
                    print(f"Warning: Row {index} (Year {year}): Pick data file not found. Calculations will be NA/0.")
                    main_df.loc[index, alive_entries_col_name] = 0
                    main_df.loc[index, prior_picks_col_name] = 0
                    main_df.loc[index, weeks_team_picks_col_name] = 0
                    continue
                except Exception as e:
                    print(f"Warning: Row {index} (Year {year}): Error loading pick data: {e}. Calculations will be NA/0.")
                    main_df.loc[index, alive_entries_col_name] = 0
                    main_df.loc[index, prior_picks_col_name] = 0
                    main_df.loc[index, weeks_team_picks_col_name] = 0
                    continue
            
            if pick_df is None or pick_df.empty:
                main_df.loc[index, alive_entries_col_name] = 0
                main_df.loc[index, prior_picks_col_name] = 0
                main_df.loc[index, weeks_team_picks_col_name] = 0
                continue
    
            # --- CRITICAL FIX 1: Determine entries ALIVE AT THE START of the current week ---
            
            total_alive_entries = 0
            alive_entries_mask = pd.Series(False, index=pick_df.index)
    
            # Week 1: Everyone is alive (use total entries for the year)
            if current_week_num == 1:
                total_alive_entries = get_total_entries(year)
                # All entries in the pick_df are considered "alive" for Week 1 calculations
                # Since we rely on pick_df for subsequent pick filtering, mask should match entries in pick_df
                alive_entries_mask = pd.Series(True, index=pick_df.index) 
            # Week 2+: Check survival using Total_Wins column
            else:
                # An entry is ALIVE at the start of Week X if they have survived (X - 1) weeks.
                # Total_Wins must be greater than or equal to (X - 1).
                required_wins = current_week_num - 1
                
                if "Total_Wins" in pick_df.columns:
                    # Convert Total_Wins to numeric for comparison, handling potential non-numeric data
                    wins_series = pd.to_numeric(pick_df["Total_Wins"], errors='coerce').fillna(0)
                    
                    # Entries are alive if their current total wins meets the requirement 
                    # to have survived all prior weeks (Week 1 to Week X-1).
                    alive_entries_mask = (wins_series >= required_wins).copy()
                    total_alive_entries = alive_entries_mask.sum()
                    
                else:
                    # FALLBACK: Use original logic if Total_Wins is missing (less reliable)
                    previous_week_col_name = f"Week_{current_week_num - 1}"
                    
                    if previous_week_col_name in pick_df.columns:
                        # Entries are "alive" if they were NOT ELIMINATED in the previous week
                        alive_entries_mask = (pick_df[previous_week_col_name] != ELIMINATED_MARKER).copy()
                        total_alive_entries = alive_entries_mask.sum()
                    else:
                        # Final Fallback to total pool size
                        total_alive_entries = get_total_entries(year) 
                        alive_entries_mask = pd.Series(True, index=pick_df.index)
    
            # NOTE: If total_alive_entries is 0, subsequent pick calculations will correctly be 0.
                
            main_df.loc[index, alive_entries_col_name] = total_alive_entries
    
            # Get the names of entries that are still alive for subsequent calculations
            alive_entry_names_current_week = pick_df.index[alive_entries_mask].tolist()
            # --- END CRITICAL FIX 1 ---
    
            cumulative_prior_team_picks = 0
            current_week_specific_picks = 0
    
            if not full_team_name:
                print(f"Warning: Row {index}: Team abbreviation '{team_abbr}' not in dictionary. Team pick count will be 0.")
            elif not alive_entry_names_current_week:
                # If no entries are alive, picks and usage are 0
                pass 
            else:
                relevant_picks_df = pd.DataFrame()
                try:
                    # Filter the full pick_df to only include rows for entries currently alive
                    # .copy() is used to avoid SettingWithCopyWarning, though .loc usually handles it.
                    relevant_picks_df = pick_df.loc[alive_entry_names_current_week].copy()
                except KeyError as e: 
                    print(f"Warning: Row {index} (Year {year}, Week {current_week_num}): KeyError when filtering for alive entries: {e}. Pick count may be affected.")
                
                if not relevant_picks_df.empty:
                    # 1. Calculate picks for the *current* week from the *alive* entries
                    current_week_col_name_in_picks = f"Week_{current_week_num}"
                    if current_week_col_name_in_picks in relevant_picks_df.columns:
                        current_week_specific_picks = (relevant_picks_df[current_week_col_name_in_picks] == full_team_name).sum()
                    
                    # 2. Calculate *cumulative* picks for the team from *prior* weeks by the *alive* entries
                    # Iterate from week 1 up to the current_week_num - 1 (the PRIOR week)
                    for w in range(1, current_week_num): # Range stops before current_week_num
                        week_w_col_name_in_picks = f"Week_{w}"
                        if week_w_col_name_in_picks in relevant_picks_df.columns:
                            cumulative_prior_team_picks += (relevant_picks_df[week_w_col_name_in_picks] == full_team_name).sum()
            
            main_df.loc[index, prior_picks_col_name] = cumulative_prior_team_picks
            main_df.loc[index, weeks_team_picks_col_name] = current_week_specific_picks
    
            if (index + 1) % 500 == 0:
                print(f"Processed {index + 1} rows...")
    
        print("\nAll rows processed for base calculations. Now calculating derived columns.")
    
        # --- Convert to numeric before calculations ---
        main_df['Year'] = pd.to_numeric(main_df['Year'], errors='coerce') 
        main_df['Week'] = pd.to_numeric(main_df['Week'], errors='coerce') 
        main_df[alive_entries_col_name] = pd.to_numeric(main_df[alive_entries_col_name], errors='coerce')
        main_df[prior_picks_col_name] = pd.to_numeric(main_df[prior_picks_col_name], errors='coerce') 
        main_df[weeks_team_picks_col_name] = pd.to_numeric(main_df[weeks_team_picks_col_name], errors='coerce')
    
    
        # --- Calculate "Availability" column ---
        alive_entries_for_avail = main_df[alive_entries_col_name]
        prior_team_picks_for_avail = main_df[prior_picks_col_name]
        
        # Availability is: (Alive Entries - Prior Picks Used) / (Alive Entries)
        valid_availability_mask = alive_entries_for_avail.gt(0) & alive_entries_for_avail.notna() & prior_team_picks_for_avail.notna()
        
        main_df.loc[valid_availability_mask, availability_col_name] = \
            (alive_entries_for_avail[valid_availability_mask] - prior_team_picks_for_avail[valid_availability_mask]) / alive_entries_for_avail[valid_availability_mask]
        
        main_df.loc[~valid_availability_mask, availability_col_name] = 0.0
        
        print(f"'Availability' column calculated. {valid_availability_mask.sum()} rows had valid data for calculation.")
    
        # --- Calculate "Pick %" column (picks this week / entries alive this week) ---
        current_week_picks_for_perc = main_df[weeks_team_picks_col_name]
        current_week_alive_for_perc = main_df[alive_entries_col_name]
    
        valid_pick_percentage_mask = current_week_alive_for_perc.gt(0) & \
                                     current_week_alive_for_perc.notna() & \
                                     current_week_picks_for_perc.notna()
        
        main_df.loc[valid_pick_percentage_mask, pick_percentage_col_name] = \
            current_week_picks_for_perc[valid_pick_percentage_mask] / current_week_alive_for_perc[valid_pick_percentage_mask]
        
        main_df.loc[~valid_pick_percentage_mask, pick_percentage_col_name] = 0.0
        
        print(f"'Pick %' column calculated. {valid_pick_percentage_mask.sum()} rows had valid data for calculation.")
    
        # --- Calculate "Entry Remaining Percent" column ---
        main_df['Total_Entries_For_Year'] = main_df['Year'].apply(get_total_entries)
    
        remaining_entries_for_calc = main_df[alive_entries_col_name]
        total_entries_for_year_for_calc = main_df['Total_Entries_For_Year']
    
        valid_entry_remaining_mask = remaining_entries_for_calc.notna() & \
                                     total_entries_for_year_for_calc.notna() & \
                                     total_entries_for_year_for_calc.gt(0)
    
        main_df.loc[valid_entry_remaining_mask, entry_remaining_percent_col_name] = \
            remaining_entries_for_calc[valid_entry_remaining_mask] / total_entries_for_year_for_calc[valid_entry_remaining_mask]
        
        main_df.loc[~valid_entry_remaining_mask, entry_remaining_percent_col_name] = 0.0
        
        del main_df['Total_Entries_For_Year'] # Clean up temporary column
        print(f"'{entry_remaining_percent_col_name}' column calculated. {valid_entry_remaining_mask.sum()} rows had valid data for calculation.")
    
    
        try:
            # Convert calculated columns to appropriate types
            main_df[alive_entries_col_name] = main_df[alive_entries_col_name].astype(pd.Int64Dtype())
            main_df[prior_picks_col_name] = main_df[prior_picks_col_name].astype(pd.Int64Dtype()) 
            main_df[weeks_team_picks_col_name] = main_df[weeks_team_picks_col_name].astype(pd.Int64Dtype())
            main_df[availability_col_name] = main_df[availability_col_name].astype(pd.Float64Dtype())
            main_df[pick_percentage_col_name] = main_df[pick_percentage_col_name].astype(pd.Float64Dtype()) 
            main_df[entry_remaining_percent_col_name] = main_df[entry_remaining_percent_col_name].astype(pd.Float64Dtype()) 
            
            main_df.to_csv(output_file, index=False)
            print(f"\nSuccessfully updated data including 'Availability', 'Current Week Picks', 'Pick %', '{entry_remaining_percent_col_name}' and saved to '{output_file}'")
        except Exception as e:
            print(f"\nError saving updated CSV file '{output_file}': {e}")
    
    if __name__ == '__main__':
        update_historical_data()
    
    df = pd.read_csv(f'contest-historical-data/Circa_historical_data_{current_year}.csv')
    # Define group keys for weekly calculations
    group_keys = ['Year', 'Week']
    
    # 1. Calculate Weekly Win % Stats
    # Using .transform() to broadcast the group-level stats to every row in that group
    print("  Calculating weekly Win % statistics (mean, max, min, std)...")
    df['Week_Mean_Availability'] = df.groupby(group_keys)['Availability'].transform('mean')
    df['Week_Max_Availability'] = df.groupby(group_keys)['Availability'].transform('max')
    df['Week_Min_Availability'] = df.groupby(group_keys)['Availability'].transform('min')
    df['Week_Std_Availability'] = df.groupby(group_keys)['Availability'].transform('std')
    
    # Fill NaN for Std on weeks with only one game (if any)
    df['Week_Std_Availability'] = df['Week_Std_Availability'].fillna(0)
    
    # 2. Calculate Team-Specific Relative Stats
    print("  Calculating team-relative Win % stats...")
    df['Team_Availability_RelativeToWeekMean'] = df['Availability'] - df['Week_Mean_Availability']
    
    # Handle potential division by zero if Max_WinPct is 0 (unlikely, but safe)
    df['Team_Availability_RelativeToTopTeam'] = df['Availability'] / df['Week_Max_Availability']
    df['Team_Availability_RelativeToTopTeam'] = df['Team_Availability_RelativeToTopTeam'].fillna(0).replace([np.inf, -np.inf], 0)                                                                                        
    
    # 3. Calculate Ranks (Win % and Star Rating)
    df['Availability_Rank'] = df.groupby(group_keys)['Availability'].rank(ascending=False, method='min')
    
    # This normalizes the rank based on the number of available teams that week
    df['Availability_Rank_Density'] = df['Availability_Rank'] / df['Num_Teams_This_Week']
    
    print("✅ Feature engineering complete.")
    
    df.to_csv(f'contest-historical-data/Circa_historical_data_{current_year}.csv', index = False)
    
    def process_game_data(input_path):
        """
        Loads historical game data, adds opponent pick percentages,
        and filters to one unique row per game.
        
        Args:
            input_path (str): The file path for the input CSV.
            
        Returns:
            pd.DataFrame: A DataFrame with 'Opponent Pick %' added
                          and filtered to one row per game.
        """
        
        # --- 1. Load Data ---
        print(f"Loading data from {input_path}...")
        try:
            # Suppress potential low_memory warning if columns are mixed types
            df = pd.read_csv(input_path, low_memory=False) 
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_path}", file=sys.stderr)
            print("Please make sure the file is in the correct path.", file=sys.stderr)
            return None # Return None instead of sys.exit(1) for cleaner Jupyter execution
        except Exception as e:
            print(f"An error occurred while loading the CSV: {e}", file=sys.stderr)
            return None
            
        print(f"Successfully loaded {len(df)} total rows.")
    
        # --- 2. Check for Required Columns ---
        # NOTE: 'Week' in your required_cols vs 'Week' in your main logic.
        # Assuming the input file uses 'Week'.
        required_cols = ['Year', 'Week', 'Team', 'Opponent', 'Pick %']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Input CSV is missing one or more required columns.", file=sys.stderr)
            print(f"It must contain: {', '.join(required_cols)}", file=sys.stderr)
            return None
            
        # --- 3. Create 'Opponent Pick %' Column ---
        print("Processing data to add 'Opponent Pick %'...")
        
        # Create a lightweight "mirror" DataFrame to match opponent rows
        opponent_picks = df[['Year', 'Week', 'Team', 'Opponent', 'Pick %']].rename(
            columns={
                'Team': 'Opponent',        # Swap Team
                'Opponent': 'Team',        # Swap Opponent
                'Pick %': 'Opponent Pick %' # This is the value we want
            }
        )
        
        # Merge the original DataFrame with the mirrored one.
        df_merged = pd.merge(
            df,
            opponent_picks,
            on=['Year', 'Week', 'Team', 'Opponent'],
            how='left' # Use 'left' to keep all original rows
        )
        
        # Check if any matches failed (should be 0 if data is clean)
        null_opp_picks = df_merged['Opponent Pick %'].isnull().sum()
        if null_opp_picks > 0:
            print(f"Warning: Found {null_opp_picks} rows where an opponent match was not found.")
    
        # --- 4. Filter to One Row Per Game ---
        print("Filtering to one unique row per game...")
        
        # Keep only rows where 'Team' name comes alphabetically before 'Opponent'.
        df_final = df_merged[df_merged['Team'] < df_merged['Opponent']].copy()
    
        temporary_historical_data_df = df_final
        
        print(f"Filtered from {len(df)} rows down to {len(temporary_historical_data_df)} unique game rows.")
    
        # --- 5. Return Processed DataFrame ---
        return temporary_historical_data_df
        
    
    
    def calculate_ev(df):
        """
        Calculates the EV for all teams, correctly grouped by both Year and Week.
        
        Args:
            df (pd.DataFrame): The historical game data, filtered to one row per game,
                               including 'Opponent Pick %'.
                               
        Returns:
            pd.DataFrame: The input DataFrame with 'Home Team EV' and 'Opponent EV' added/updated.
        """
    
        # --- Nested Helper Functions (Unchanged Logic, now self-contained) ---
    
        def calculate_all_scenarios(week_df):
            num_games = len(week_df)
            if num_games == 0:
                return pd.Series(dtype=float), np.array([]), np.array([])
                
            teams = week_df['Team'].tolist() + week_df['Opponent'].tolist()
            num_teams = len(teams)
            print("DEBUG: TEAMS:")
            print(teams)
    
            all_outcomes_matrix = np.array(list(itertools.product(['Team Win', 'Opponent Win'], repeat=num_games)))
            num_scenarios = all_outcomes_matrix.shape[0]
    
            ev_df = pd.DataFrame(index=range(num_scenarios), columns=teams)
            scenario_weights = np.zeros(num_scenarios)
    
            # Vectorized calculations within the scenario loop
            for i in range(num_scenarios):
                outcome = all_outcomes_matrix[i]
                winning_teams = np.where(outcome == 'Team Win', week_df['Team'].values, week_df['Opponent'].values)
                winning_team_indices = np.isin(teams, winning_teams)
    
                winning_probs = np.where(outcome == 'Team Win', week_df['Win %'].values, 1 - week_df['Win %'].values)
    
                scenario_weights[i] = np.prod(winning_probs)
    
                pick_percentages = np.where(outcome == 'Team Win', week_df['Pick %'].values, week_df['Opponent Pick %'].values)
                surviving_entries = np.sum(pick_percentages)
    
                ev_values = np.zeros(num_teams)
                ev_values[winning_team_indices] = 1 / surviving_entries if surviving_entries > 0 else 0
                ev_df.iloc[i] = ev_values
    
            if scenario_weights.sum() == 0:
                return pd.Series(0.0, index=teams), all_outcomes_matrix, scenario_weights
    
            weighted_avg_ev = (ev_df * scenario_weights[:, np.newaxis]).sum(axis=0) / scenario_weights.sum()
            return weighted_avg_ev, all_outcomes_matrix, scenario_weights
    
        def get_pick_percentage(week_df, team_name):
            # Check if the team is a home team in any game this week
            if team_name in week_df['Team'].values:
                return week_df[week_df['Team'] == team_name]['Pick %'].iloc[0]
            # Check if the team is an away team
            elif team_name in week_df['Opponent'].values:
                return week_df[week_df['Opponent'] == team_name]['Opponent Pick %'].iloc[0]
            # Return 0 if the team is not found
            return 0.0
    
        def calculate_all_scenarios_two_picks(week_df):
            num_games = len(week_df)
            if num_games == 0:
                return pd.Series(dtype=float), np.array([]), np.array([])
                
            teams = week_df['Team'].tolist() + week_df['Opponent'].tolist()
            
            all_outcomes_matrix = np.array(list(itertools.product(['Team Win', 'Opponent Win'], repeat=num_games)))
            num_scenarios = all_outcomes_matrix.shape[0]
            
            team_pairs = list(itertools.combinations(teams, 2))
            
            pair_ev_df = pd.DataFrame(0.0, index=range(num_scenarios), columns=team_pairs)
            scenario_weights = np.zeros(num_scenarios)
            
            for i in range(num_scenarios):
                outcome = all_outcomes_matrix[i]
                winning_teams = np.where(outcome == 'Team Win', week_df['Team'].values, week_df['Opponent'].values)
                
                # Leaving your original logic here for Win % vs Pick % on the opponent side
                winning_probs = np.where(outcome == 'Team Win', week_df['Win %'].values, 1 - week_df['Pick %'].values)
                scenario_weights[i] = np.prod(winning_probs)
                
                # Calculate EV for each pair
                for pair in team_pairs:
                    team1, team2 = pair
                    if team1 in winning_teams and team2 in winning_teams:
                        pick_perc1 = get_pick_percentage(week_df, team1)
                        pick_perc2 = get_pick_percentage(week_df, team2)
                        
                        surviving_entries = pick_perc1 * pick_perc2
                        
                        if surviving_entries > 0:
                            pair_ev_df.loc[i, pair] = 1 / surviving_entries
            
            if scenario_weights.sum() == 0:
                return pd.Series(0.0, index=teams), all_outcomes_matrix, scenario_weights
    
            # Now, calculate the weighted average EV for each pair
            weighted_avg_pair_ev = (pair_ev_df.mul(scenario_weights, axis=0)).sum(axis=0) / scenario_weights.sum()
            
            # Consolidate pair EVs into single-team EVs
            weighted_avg_ev = pd.Series(0.0, index=teams)
            for pair, ev in weighted_avg_pair_ev.items():
                team1, team2 = pair
                weighted_avg_ev[team1] += ev
                weighted_avg_ev[team2] += ev
            
            return weighted_avg_ev, all_outcomes_matrix, scenario_weights
    
        # --- Main Function Logic (Adjusted for Jupyter) ---
    
        all_weeks_ev = {} # Store the EV values for each (year, week)
        
        # 1. Check for required columns
        if 'Year' not in df.columns or 'Week' not in df.columns:
            print("Error: Dataframe must contain 'Year' and 'Week' columns.", file=sys.stderr)
            return df
    
        # 2. Get all unique combinations of 'Year' and 'Week'
        year_week_pairs = df[['Year', 'Week']].drop_duplicates()
        
        # 3. Sort them so we process chronologically
        year_week_pairs_sorted = year_week_pairs.sort_values(by=['Year', 'Week']).values
        
        # 4. Iterate through the (year, week) pairs using TQDM
        for year, week in tqdm(year_week_pairs_sorted, desc="Processing Year/Weeks"):
            
            # Filter the dataframe for the specific year AND week
            week_df = df[(df['Year'] == year) & (df['Week'] == week)].copy()
            print(f"Getting EV Calculations for Week {week} in Year {year}.")
            if week_df.empty:
                print(f"Warning: No data found for Year {year}, Week {week}. Skipping.")
                continue
    
            # Using the existing function for single-pick weeks (since 'week_requiring_two_selections' is undefined)
            weighted_avg_ev, all_outcomes, scenario_weights = calculate_all_scenarios(week_df)
    
            # Store the EV values using a (year, week) tuple as the key
            all_weeks_ev[(year, week)] = weighted_avg_ev
    
            # Update the main 'df' using the year, week, AND team
            if not weighted_avg_ev.empty:
                
                # Identify rows for the current year and week
                current_week_mask = (df['Year'] == year) & (df['Week'] == week)
    
                # Map EV values for the Home Team (Team column)
                home_team_ev_map = df.loc[current_week_mask, 'Team'].map(weighted_avg_ev)
                # Update 'Home Team EV' where the mapping was successful (not NaN)
                df.loc[current_week_mask & home_team_ev_map.notna(), 'Team EV'] = home_team_ev_map[home_team_ev_map.notna()]
    
                # Map EV values for the Opponent (Opponent column)
                opponent_ev_map = df.loc[current_week_mask, 'Opponent'].map(weighted_avg_ev)
                # Update 'Opponent EV' where the mapping was successful (not NaN)
                df.loc[current_week_mask & opponent_ev_map.notna(), 'Opponent EV'] = opponent_ev_map[opponent_ev_map.notna()]
    
        print("EV calculations complete for all seasons.")
    
        csv_df = pd.read_csv(f'contest-historical-data/Circa_historical_data_{current_year}.csv')
        csv_df = csv_df[csv_df['Year'] == current_year]
    ####    csv_df = pd.read_csv(f'contest-historical-data/Circa_historical_data.csv')
        merge_cols = ['Team', 'Opponent', 'Week', 'Year']
        merged_1 = pd.merge(
            csv_df,
            df[['Team', 'Opponent', 'Week', 'Year', 'Team EV']],
            on=merge_cols,
            how='left',
            suffixes=('_csv', '_df1')
        )
        merged_1.rename(columns={'Team EV': 'EV_from_df1'}, inplace=True)
    
        merged_2 = pd.merge(
            csv_df,
            df[['Team', 'Opponent', 'Week', 'Year', 'Opponent EV']],
            left_on=['Team', 'Opponent', 'Week', 'Year'],
            right_on=['Opponent', 'Team', 'Week', 'Year'], # Note the swap here!
            how='left',
            suffixes=('_csv', '_df2')
        )
        
        # Rename the relevant column for clarity after the merge
        merged_2.rename(columns={'Opponent EV': 'EV_from_df2'}, inplace=True)
    
        csv_df['EV'] = merged_1['EV_from_df1'].combine_first(csv_df['EV'])
        csv_df['EV'] = merged_2['EV_from_df2'].combine_first(csv_df['EV'])
    
        # Save to a CSV
        output_filename = f"contest-historical-data/Circa_historical_data_{current_year}.csv"
    ####    output_filename = f"contest-historical-data/Circa_historical_data.csv"
        csv_df.to_csv(output_filename, index=False)
        print(f"Updated data saved to {output_filename}")
        
        return df
    
    
    initial_input_filename = f"contest-historical-data/Circa_historical_data_{current_year}.csv"
    ####initial_input_filename = f"contest-historical-data/Circa_historical_data.csv"
    
    historical_df = process_game_data(initial_input_filename)
    
    # Ensure 'Home Team EV' and 'Opponent EV' columns exist
    if 'Team EV' not in historical_df.columns:
        historical_df['Team EV'] = np.nan
    if 'Opponent EV' not in historical_df.columns:
        historical_df['Opponent EV'] = np.nan
        
    print(f"\n--- Starting EV Calculation ---")
    
    # Run the calculation
    updated_df = calculate_ev(historical_df.copy())
    
    
    # Display sample output in Jupyter
    if updated_df is not None:
        print("\n--- Sample of Updated Data (Team EV) ---")
        print(updated_df[updated_df['Team EV'].notna()][['Year', 'Week', 'Team', 'Team EV']].head())
        
        print("\n--- Sample of Updated Data (Opponent EV) ---")
        print(updated_df[updated_df['Opponent EV'].notna()][['Year', 'Week', 'Opponent', 'Opponent EV']].head())
    
    df_historical = pd.read_csv("contest-historical-data/Circa_historical_data.csv")
    df_current_year = pd.read_csv(f"contest-historical-data/Circa_historical_data_{current_year}.csv")
    
    df_filtered = df_historical[df_historical['Year'] != current_year].copy()
    
    df_combined = pd.concat([df_filtered, df_current_year], ignore_index=True)
    
    # --- Step 4: Save the result (Optional, but recommended) ---
    df_combined.to_csv("contest-historical-data/Circa_historical_data.csv", index=False)
            
    print(df_combined)
        
    def calculate_team_availability(historical_data_path: str, picks_data_path: str) -> Optional[pd.DataFrame]:
        """
        Calculates the availability of each team for the next week in the Circa Survivor Contest.
    
        Availability is defined as the percentage of currently 'alive' entries 
        (Total_Wins >= last completed week) that have NOT yet used a given team.
    
        Args:
            historical_data_path: File path to the Circa historical data (used to find W_max).
            picks_data_path: File path to the survivor picks history (used to find usage).
    
        Returns:
            A pandas DataFrame with team, next week, availability percentage, and counts,
            or None if an error occurs.
        """
        
        print(f"Loading data from: {historical_data_path} and {picks_data_path}")
    
        # --- 1. Load Data ---
        try:
            df_hist = pd.read_csv(historical_data_path)
            df_picks = pd.read_csv(picks_data_path)
        except FileNotFoundError as e:
            print(f"ERROR: One of the files was not found. Please ensure both are correctly named and accessible. {e}")
            return None
        except Exception as e:
            print(f"An error occurred while reading the files: {e}")
            return None
    
        # --- 2. Determine the Last Completed Week (W_max) ---
        if 'Week' not in df_hist.columns:
            print("ERROR: 'contest-historical-data/Circa_historical_data.csv' must contain a 'Week' column.")
            return None
    
        # Find the maximum completed week (W_max) and the next week for calculation (W_next)
        try:
            # Ensure 'Week' is numeric and find the maximum completed week
            df_hist['Week_Numeric'] = pd.to_numeric(df_hist['Week'], errors='coerce')
            W_max = int(df_hist['Week_Numeric'].max(skipna=True))
        except (ValueError, TypeError):
            print("ERROR: Could not determine the last completed week from the 'Week' column in the historical data.")
            return None
            
        W_next = W_max
    
        print(f"\n--- Analysis Parameters ---")
        print(f"Last completed pick week (W_max): {W_max}")
        print(f"Calculating availability for Week: {W_next}")
    
        # Define the columns that represent the picks made up to the last completed week
        pick_cols = [f'Week_{i}' for i in range(1, W_max)]
        
        # Check for missing pick columns and adjust W_max if necessary
        available_pick_cols = [col for col in pick_cols if col in df_picks.columns]
    
        if len(available_pick_cols) < W_max:
            print(f"WARNING: Picks data is incomplete. Only found picks up to Week {len(available_pick_cols)}. Adjusting W_max.")
            W_max = len(available_pick_cols)
            W_next = W_max
            print(f"Adjusted analysis: Last pick available is Week {W_max}. Calculating availability for Week: {W_next}")
        
        if W_max == 0:
            print("ERROR: No 'Week_X' columns found in the picks data to determine usage.")
            return None
    
        # --- 3. Filter Alive Entries ---
        if 'Total_Wins' not in df_picks.columns:
            print("ERROR: '2025_survivor_picks.csv' must contain a 'Total_Wins' column.")
            return None
    
        # Entries are considered 'alive' if Total_Wins >= W_max (they survived W_max picks)
        try:
            # Convert Total_Wins to numeric for reliable comparison
            df_picks['Total_Wins_Numeric'] = pd.to_numeric(df_picks['Total_Wins'], errors='coerce').fillna(0)
            df_alive = df_picks[df_picks['Total_Wins_Numeric'] >= W_max-1].copy()
        except Exception as e:
            print(f"ERROR: Failed to filter entries by 'Total_Wins'. Check the data type and format. {e}")
            return None
    
        N_alive = len(df_alive)
        print(f"Total entries considered alive (Total_Wins > {W_max}): {N_alive}")
    
        if N_alive == 0:
            print("\nNo entries are currently alive (Total_Wins < W_max). Cannot calculate availability.")
            return pd.DataFrame({'Team': [], 'Availability_Percent': []})
    
    
        # --- 4 & 5. Calculate Availability for Each Team ---
        
        # Get the unique list of all teams from the historical data
        # Note: We use the historical data loaded inside the function (df_hist)
        all_teams_home = df_hist['Team'].dropna().unique()
        all_teams_away = df_hist['Opponent'].dropna().unique()
        all_teams = np.unique(np.concatenate((all_teams_home, all_teams_away)))
    
        availability_list = []
    
        for team in all_teams:
            # Build a single boolean mask for all relevant pick columns
            # The mask is True if the entry picked the 'team' in Week_1 OR Week_2 OR ... Week_W_max
            used_mask = pd.Series(False, index=df_alive.index)
            
            for col in pick_cols[:W_max]: # Use the adjusted W_max
                if col in df_alive.columns:
                    # Use str.strip() to handle potential whitespace in team names
                    used_mask = used_mask | (df_alive[col].astype(str).str.strip() == str(team).strip())
    
            # Count of ALIVE entries that have used this team
            N_used = used_mask.sum()
            
            # Count of ALIVE entries that have NOT used this team
            N_available = N_alive - N_used
    
            # Availability: Percentage of alive entries that have NOT used the team
            availability_percent = N_available / N_alive
            
            # Store result
            availability_list.append({
                'Team': team,
                'Availability_Week': W_next,
                'Entries_Used_Count': N_used,
                'Entries_Available_Count': N_available,
                'Total_Alive_Entries': N_alive,
                'Availability_Percent': f"{availability_percent:.4f}"
            })
    
        # Create the final results DataFrame, sorted by availability
        # Convert 'Availability_Percent' to float for sorting, then back to string for output
        df_availability = pd.DataFrame(availability_list)
        df_availability['Availability_Percent_Float'] = pd.to_numeric(df_availability['Availability_Percent'], errors='coerce')
        df_availability = df_availability.sort_values(by='Availability_Percent_Float', ascending=False).drop(columns=['Availability_Percent_Float'])
    
        print("\n--- Availability Calculation Complete ---")
        return df_availability
    
    # --- Main Execution Block ---
    
    # Define the file paths as provided by the user
    historical_file = "contest-historical-data/Circa_historical_data.csv"
    picks_file = f"circa-pick-history/{current_year}_survivor_picks.csv"
    output_file = "contest-historical-data/Circa_historical_data.csv" # New output file name
    
    # Run the calculation
    results_df = calculate_team_availability(historical_file, picks_file)
    
    if results_df is not None:
        # Display the top 10 most available teams
        print("\nTop 10 Team Availability for the Next Pick Week:")
        print(results_df.head(10).to_markdown(index=False))
    
        # Example: How to get the availability for a specific team like 'DEN'
        denver_row = results_df[results_df['Team'] == 'DEN']
        if not denver_row.empty:
            denver_availability = denver_row.iloc[0]['Availability_Percent']
            print(f"\nAvailability for Denver (DEN): {denver_availability}")
    
        # --- NEW LOGIC: Save Availability back to Historical Data ---
    
        # 1. Load the original historical data again for modification
        df_hist_original = pd.read_csv(historical_file)
        
        # 2. Recalculate W_max and W_next (must match logic in function)
        df_hist_original['Week_Numeric'] = pd.to_numeric(df_hist_original['Week'], errors='coerce')
        W_max = int(df_hist_original['Week_Numeric'].max(skipna=True))
        W_next = W_max + 1
    
        # 3. Prepare the mapping dictionary
        # Convert the Availability_Percent from string back to float for storing
        results_df['Calculated_Availability_Float'] = pd.to_numeric(
            results_df['Availability_Percent'], errors='coerce'
        )
        
        # Create a mapping dictionary: {'Team': Availability_Float}
        availability_map = results_df.set_index('Team')['Calculated_Availability_Float'].to_dict()
    
        # 4. Apply the calculated availability to the historical data
        
        # Define the mask for the rows that represent the next week's games
        week_mask = (df_hist_original['Week_Numeric'] == W_next)
        
        # Use .loc to ensure we only update the rows corresponding to W_next
        # Map the availability for the 'Team' (which is the column we're overwriting)
        # .astype(str).str.strip() is used to ensure clean merging keys
        df_hist_original.loc[week_mask, 'Availability'] = (
            df_hist_original.loc[week_mask, 'Team'].astype(str).str.strip().map(availability_map)
        )
    
        # 5. Save the modified DataFrame to the new CSV
        df_hist_original = df_hist_original.drop(columns=['Week_Numeric'], errors='ignore')
        df_hist_original.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully updated Availability for Week {W_next} games and saved to '{output_file}'.")

    def calculate_ev2(starting_week, target_year, config: dict = {}):
        """
        Calculates EV for all probability models (sportsbook, mp, gsf, sim, consensus)
        for the upcoming week only. Writes EV columns back to the main sim results CSV.

        Uses starting_week (= upcoming_week in this script's terminology) as the
        starting point — prior weeks already have EV from previous runs.

        Zero-availability teams (pick% == 0) are excluded from the expected survivors
        denominator so they don't dilute EV for eligible teams.
        """        

        sim_file = (
            f"nfl-power-ratings/final_sim_results_with_variance_week_"
            f"{starting_week}_{target_year}.csv"
        )
        if not os.path.exists(sim_file):
            print(f"⚠️  calculate_ev: sim file not found: {sim_file}")
            return
        
        df = pd.read_csv(sim_file)
        print(f"   📊 Loaded sim results: {len(df)} rows from {sim_file}")
        start_w = starting_week  # upcoming_week in this script

        # Enforce team abbreviation standardization
        replace_dict = {'JAC': 'JAX', 'LAR': 'LA'}
        df['Away Team'] = df['Away Team'].replace(replace_dict)
        df['Home Team'] = df['Home Team'].replace(replace_dict)

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
            expected survivors calculation.
            """
            home_probs = week_df[home_prob_col].values
            away_probs = week_df[away_prob_col].values
            home_picks = week_df['Home Pick %'].values
            away_picks = week_df['Away Pick %'].values

            home_eligible = home_picks > 0
            away_eligible = away_picks > 0

            expected_survivors = (
                np.sum(home_probs[home_eligible] * home_picks[home_eligible]) +
                np.sum(away_probs[away_eligible] * away_picks[away_eligible])
            )

            ev_results = {}
            for i, row in week_df.iterrows():
                if expected_survivors > 0:
                    ev_results[row['Home Team']] = (
                        row[home_prob_col] / expected_survivors
                        if row['Home Pick %'] > 0 else 0
                    )
                    ev_results[row['Away Team']] = (
                        row[away_prob_col] / expected_survivors
                        if row['Away Pick %'] > 0 else 0
                    )
                else:
                    ev_results[row['Home Team']] = 0
                    ev_results[row['Away Team']] = 0

            return ev_results

        for scenario_name, scenario_config in probability_scenarios.items():
            away_prob_col = scenario_config["away_col"]
            home_prob_col = scenario_config["home_col"]
            prefix = scenario_config["prefix"]

            home_ev_col = f"{prefix}_Home_EV"
            away_ev_col = f"{prefix}_Away_EV"

            # Initialize columns — prior weeks keep their existing values
            if home_ev_col not in df.columns:
                df[home_ev_col] = 0.0
            if away_ev_col not in df.columns:
                df[away_ev_col] = 0.0

            for week in tqdm(range(start_w, end_w), desc=f"Processing {prefix.upper()} EV", leave=False):
                week_df = df[df['Week_x'] == week].copy()

                if week_df.empty:
                    continue

                # Skip if required probability columns don't exist for this scenario
                if away_prob_col not in week_df.columns or home_prob_col not in week_df.columns:
                    print(f"   ⚠️  Skipping {prefix} EV for week {week} — columns not found")
                    continue

                ev_results = calculate_all_scenarios(
                    week_df,
                    away_prob_col=away_prob_col,
                    home_prob_col=home_prob_col
                )

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

        # Save updated EV columns back to the main sim results file
        main_file_path = (
            f"nfl-power-ratings/final_sim_results_with_variance_week_"
            f"{starting_week}_{target_year}.csv"
        )
        df.to_csv(main_file_path, index=False)
        print(f"\n✅ EV columns saved to: {main_file_path}")

    calculate_ev2(starting_week, target_year)


    def create_actual_historical_data(last_played_week, starting_week, target_year, current_year):
        """
        Builds a historical record for the just-completed week using:
          - Actual closing sportsbook odds and game scores from nflreadpy
          - Actual pick percentages from the Circa historical data CSV
          - Actual EV recalculated using the above real data

        Handles holiday weeks (Thanksgiving, Christmas) correctly by grouping
        EV calculation by Circa Week rather than numeric Week_x, ensuring
        holiday games are never pooled with regular week games that share
        the same Week_x number.

        Output files:
          nfl-power-ratings/Week_{last_played_week}_{target_year}_Final_Data.csv
              — this week's games only
          nfl-power-ratings/Season_{target_year}_Final_Data.csv
              — full season append (idempotent — safe to re-run)
        """


        # Full name lookup — nflreadpy abbreviation → sim file full team name
        ABBR_TO_FULL = {
            "ARI": "Arizona Cardinals",    "ATL": "Atlanta Falcons",
            "BAL": "Baltimore Ravens",     "BUF": "Buffalo Bills",
            "CAR": "Carolina Panthers",    "CHI": "Chicago Bears",
            "CIN": "Cincinnati Bengals",   "CLE": "Cleveland Browns",
            "DAL": "Dallas Cowboys",       "DEN": "Denver Broncos",
            "DET": "Detroit Lions",        "GB":  "Green Bay Packers",
            "HOU": "Houston Texans",       "IND": "Indianapolis Colts",
            "JAX": "Jacksonville Jaguars", "JAC": "Jacksonville Jaguars",
            "KC":  "Kansas City Chiefs",   "LA":  "Los Angeles Rams",
            "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
            "LV":  "Las Vegas Raiders",    "LVR": "Las Vegas Raiders",
            "MIA": "Miami Dolphins",       "MIN": "Minnesota Vikings",
            "NE":  "New England Patriots", "NO":  "New Orleans Saints",
            "NOR": "New Orleans Saints",   "NYG": "New York Giants",
            "NYJ": "New York Jets",        "PHI": "Philadelphia Eagles",
            "PIT": "Pittsburgh Steelers",  "SEA": "Seattle Seahawks",
            "SF":  "San Francisco 49ers",  "SFO": "San Francisco 49ers",
            "TB":  "Tampa Bay Buccaneers", "TAM": "Tampa Bay Buccaneers",
            "TEN": "Tennessee Titans",     "WAS": "Washington Commanders",
            "WSH": "Washington Commanders","GNB": "Green Bay Packers",
            "KAN": "Kansas City Chiefs",
        }
        
        # Reverse: full name → abbreviation (for pick% lookup)
        FULL_TO_ABBR = {v: k for k, v in ABBR_TO_FULL.items()
                        if k in {"ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE",
                                 "DAL","DEN","DET","GB","HOU","IND","JAX","KC",
                                 "LA","LAC","LV","MIA","MIN","NE","NO","NYG",
                                 "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"}}
        # ── 1. Load sim results and filter to completed week ──────────────────
        sim_file = (
            f"nfl-power-ratings/final_sim_results_with_variance_week_"
            f"{last_played_week}_{target_year}.csv"
        )

        if not os.path.exists(sim_file):
            print(f"⚠️  create_actual_historical_data: sim file not found: {sim_file}")
            return

        sim_df = pd.read_csv(sim_file)
        week_col = "Week_x" if "Week_x" in sim_df.columns else "Week"

        # Debug — remove after confirming
        print(f"   📋 Sim file columns: {list(sim_df.columns[:10])}")
        print(f"   📋 Unique weeks in file: {sorted(sim_df[week_col].unique().tolist())}")
        print(f"   📋 Looking for week: {last_played_week}")

        actual_data = sim_df[sim_df[week_col] == last_played_week].copy()

        if actual_data.empty:
            print(f"⚠️  No rows found for week {last_played_week} in {sim_file}")
            return

        print(f"\n📊 Building actual historical data for Week {last_played_week} {target_year}...")
        print(f"   Loaded {len(actual_data)} games")

        # ── 2. Pull actual closing odds and scores from nflreadpy ─────────────
        try:
            schedule = nfl.load_schedules([target_year])
            schedule = schedule.to_pandas() if hasattr(schedule, 'to_pandas') else schedule

            sched_week = schedule[schedule["week"] == last_played_week].copy()

            if sched_week.empty:
                print(f"   ⚠️  No nflreadpy data found for week {last_played_week}")
            else:
                def ml_to_implied(ml):
                    ml = float(ml)
                    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)

                def remove_vig(away_ml, home_ml):
                    imp_away = ml_to_implied(away_ml)
                    imp_home = ml_to_implied(home_ml)
                    total = imp_away + imp_home
                    if total == 0:
                        return 0.5, 0.5
                    return round(imp_away / total, 4), round(imp_home / total, 4)

                replaced_odds = 0
                for _, sched_row in sched_week.iterrows():
                    away_abbr = str(sched_row.get("away_team", "")).upper()
                    home_abbr = str(sched_row.get("home_team", "")).upper()

                    # Convert abbreviations to full team names
                    away_full = ABBR_TO_FULL.get(away_abbr, away_abbr)
                    home_full = ABBR_TO_FULL.get(home_abbr, home_abbr)

                    game_mask = (
                        (actual_data["Away Team"] == away_full) &
                        (actual_data["Home Team"] == home_full)
                    )

                    if not game_mask.any():
                        continue

                    # Actual closing moneylines → fair odds (vig removed)
                    away_ml = sched_row.get("away_moneyline")
                    home_ml = sched_row.get("home_moneyline")
                    spread   = sched_row.get("spread_line")  # home team spread

                    if pd.notna(away_ml) and pd.notna(home_ml):
                        fair_away, fair_home = remove_vig(away_ml, home_ml)
                        actual_data.loc[game_mask, "Away Team Sportsbook Fair Odds"] = fair_away
                        actual_data.loc[game_mask, "Home Team Sportsbook Fair Odds"] = fair_home
                        actual_data.loc[game_mask, "Away Team Sportsbook Moneyline"] = float(away_ml)
                        actual_data.loc[game_mask, "Home Team Sportsbook Moneyline"] = float(home_ml)

                    if pd.notna(spread):
                        actual_data.loc[game_mask, "Home Team Sportsbook Spread"] = float(spread)
                        actual_data.loc[game_mask, "Away Team Sportsbook Spread"] = -float(spread)

                    # Actual scores
                    home_score = sched_row.get("home_score")
                    away_score = sched_row.get("away_score")
                    if pd.notna(home_score):
                        actual_data.loc[game_mask, "Home Team Points"] = int(home_score)
                    if pd.notna(away_score):
                        actual_data.loc[game_mask, "Away Team Points"] = int(away_score)

                    # Actual win result (1 = won, 0 = lost, 0.5 = tie)
                    if pd.notna(home_score) and pd.notna(away_score):
                        if int(home_score) > int(away_score):
                            actual_data.loc[game_mask, "Actual Home Team Win %"] = 1.0
                            actual_data.loc[game_mask, "Actual Away Team Win %"] = 0.0
                        elif int(away_score) > int(home_score):
                            actual_data.loc[game_mask, "Actual Home Team Win %"] = 0.0
                            actual_data.loc[game_mask, "Actual Away Team Win %"] = 1.0
                        else:
                            actual_data.loc[game_mask, "Actual Home Team Win %"] = 0.5
                            actual_data.loc[game_mask, "Actual Away Team Win %"] = 0.5

                    replaced_odds += 1

                print(f"   ✅ Updated odds/scores for {replaced_odds} games from nflreadpy")
                # Debug — show unmatched games if any
                if replaced_odds < len(sched_week):
                    for _, r in sched_week.iterrows():
                        away = ABBR_TO_FULL.get(str(r.get('away_team', '')).upper(), '?')
                        home = ABBR_TO_FULL.get(str(r.get('home_team', '')).upper(), '?')
                        game_mask = (
                            (actual_data["Away Team"] == away) &
                            (actual_data["Home Team"] == home)
                        )
                        if not game_mask.any():
                            print(f"   ❌ No match: {away} @ {home} (raw: {r.get('away_team')} @ {r.get('home_team')})")

        except Exception as e:
            print(f"   ⚠️  nflreadpy pull failed: {e}")

        # ── 3. Replace predicted pick% with actual pick% from Circa data ──────
        hist_file = f"contest-historical-data/Circa_historical_data_{current_year}.csv"

        if os.path.exists(hist_file):
            hist_df = pd.read_csv(hist_file)
            hist_week = hist_df[
                (hist_df["Year"] == target_year) &
                (hist_df["Week"].astype(int) == last_played_week)
            ].copy()

            if not hist_week.empty:
                # Build lookup: team abbreviation → actual pick %
                pick_pct_map = {}
                for _, row in hist_week.iterrows():
                    team_abbr = str(row.get("Team", "")).strip()
                    pick_pct  = row.get("Pick %")
                    if team_abbr and pd.notna(pick_pct):
                        pick_pct_map[team_abbr] = float(pick_pct)

                if pick_pct_map:
                    def get_pick(team_full_name):
                        abbr = FULL_TO_ABBR.get(team_full_name, team_full_name)
                        return pick_pct_map.get(abbr, None)
                
                    away_pcts = actual_data["Away Team"].apply(get_pick)
                    home_pcts = actual_data["Home Team"].apply(get_pick)

                    actual_data.loc[away_pcts.notna(), "Away Actual Pick %"] = \
                        away_pcts[away_pcts.notna()]
                    actual_data.loc[home_pcts.notna(), "Home Actual Pick %"] = \
                        home_pcts[home_pcts.notna()]

                    print(
                        f"   ✅ Replaced pick%: "
                        f"{away_pcts.notna().sum()} away, "
                        f"{home_pcts.notna().sum()} home teams"
                    )
                else:
                    print(f"   ⚠️  No pick% data found for week {last_played_week}")
            else:
                print(f"   ⚠️  No historical rows for week {last_played_week} in {hist_file}")
        else:
            print(f"   ⚠️  Historical data file not found: {hist_file}")

        # ── 4. Calculate actual EV using real pick% and real sportsbook odds ───
        # Group by Circa Week to keep holiday games (Thanksgiving, Christmas)
        # separate from regular week games that share the same numeric Week_x.
        # This prevents Thanksgiving games from being pooled with Week 12 games, etc.
        try:
            circa_col = "Circa Week" if "Circa Week" in actual_data.columns else None

            if circa_col:
                circa_groups = actual_data[circa_col].unique()
            else:
                # Fall back to a single group if Circa Week column doesn't exist
                circa_groups = [last_played_week]
                circa_col = week_col

            # Initialize EV columns
            actual_data["Actual Away Team EV"] = 0.0
            actual_data["Actual Home Team EV"] = 0.0

            for circa_week in circa_groups:
                group_mask = actual_data[circa_col] == circa_week
                group = actual_data[group_mask].copy()

                if group.empty:
                    continue

                away_probs = pd.to_numeric(
                    group["Away Team Sportsbook Fair Odds"], errors="coerce"
                ).fillna(0)
                home_probs = pd.to_numeric(
                    group["Home Team Sportsbook Fair Odds"], errors="coerce"
                ).fillna(0)

                # Use actual pick% where available, fall back to predicted pick%
                away_picks = pd.to_numeric(
                    group["Away Actual Pick %"]
                    if "Away Actual Pick %" in group.columns
                    else group.get("Away Pick %", pd.Series(0, index=group.index)),
                    errors="coerce"
                ).fillna(0)
                home_picks = pd.to_numeric(
                    group["Home Actual Pick %"]
                    if "Home Actual Pick %" in group.columns
                    else group.get("Home Pick %", pd.Series(0, index=group.index)),
                    errors="coerce"
                ).fillna(0)

                # Only eligible teams (pick% > 0) contribute to expected survivors
                away_eligible = away_picks > 0
                home_eligible = home_picks > 0

                expected_survivors = (
                    (away_probs[away_eligible] * away_picks[away_eligible]).sum() +
                    (home_probs[home_eligible] * home_picks[home_eligible]).sum()
                )

                if expected_survivors > 0:
                    actual_data.loc[group_mask, "Actual Away Team EV"] = (
                        (away_probs / expected_survivors)
                        .where(away_eligible, 0)
                        .round(4)
                        .values
                    )
                    actual_data.loc[group_mask, "Actual Home Team EV"] = (
                        (home_probs / expected_survivors)
                        .where(home_eligible, 0)
                        .round(4)
                        .values
                    )
                    print(
                        f"   ✅ {circa_week}: EV calculated "
                        f"(expected survivors: {expected_survivors:.4f})"
                    )
                else:
                    print(f"   ⚠️  {circa_week}: expected survivors = 0 — EV set to 0")

        except Exception as e:
            print(f"   ⚠️  EV calculation failed: {e}")
            actual_data["Actual Away Team EV"] = None
            actual_data["Actual Home Team EV"] = None


        # ── 5b. Calculate Win/Loss and P/L for each bet type ─────────────────────
        try:
            DEFAULT_UNIT_RISK = 110.0
            DEFAULT_UNIT_WIN  = 100.0

            # ── Helper functions ───────────────────────────────────────────────

            def spread_result(bet_team, home_team, home_pts, away_pts,
                              home_spread, away_spread):
                """True=cover, False=no cover, None=push or no data."""
                if pd.isna(home_pts) or pd.isna(away_pts):
                    return None
                home_pts = float(home_pts)
                away_pts = float(away_pts)
                if bet_team == home_team:
                    margin = (home_pts - away_pts) + float(home_spread)
                else:
                    margin = (away_pts - home_pts) + float(away_spread)
                if margin > 0:  return True
                if margin < 0:  return False
                return None  # push

            def ml_result(bet_team, home_team, home_pts, away_pts):
                """True=win, False=loss, None=tie or no data."""
                if pd.isna(home_pts) or pd.isna(away_pts):
                    return None
                home_pts = float(home_pts)
                away_pts = float(away_pts)
                if home_pts == away_pts:
                    return None  # tie
                return (home_pts > away_pts) if bet_team == home_team \
                    else (away_pts > home_pts)

            def total_result(direction, total_pts, total_line):
                """True=cover, False=no cover, None=push or no data."""
                if total_pts is None or pd.isna(total_line):
                    return None
                d = str(direction).strip().lower()
                if d in ("no bet", "", "nan"):
                    return None
                total_pts  = float(total_pts)
                total_line = float(total_line)
                if total_pts == total_line:
                    return None  # push
                if d == "over":  return total_pts > total_line
                if d == "under": return total_pts < total_line
                return None

            def ml_profit(wager, moneyline, won):
                """Profit/loss from a moneyline bet."""
                if won is None:
                    return 0.0
                ml = float(moneyline)
                if won:
                    return wager * (100 / abs(ml)) if ml < 0 else wager * (ml / 100)
                return -abs(wager)

            def spread_total_profit(unit_risk, unit_win, won):
                """Profit/loss from a spread or total bet."""
                if won is None:
                    return 0.0
                return float(unit_win) if won else -abs(float(unit_risk))

            def result_label(won, has_scores):
                """Human-readable result string."""
                if not has_scores: return "Pending"
                if won is None:    return "Push"
                return "Win" if won else "Loss"

            def get_val(row, key, default=DEFAULT_UNIT_RISK):
                """
                If key is a column name, return that row's value.
                If key is already a number, return it directly.
                Falls back to default on missing/null.
                """
                if isinstance(key, str) and key in row.index:
                    v = row.get(key)
                    return float(v) if pd.notna(v) else default
                return key if isinstance(key, (int, float)) else default

            # ── Bet configurations ─────────────────────────────────────────────
            # All spread and total bets are evaluated against the SPORTSBOOK line.
            # Model projections (Monte Carlo Total, Consensus spreads, etc.) are
            # only used for edge calculation — never for win/loss evaluation.
            # All moneyline profit calculations use the closing sportsbook moneyline.

            bet_configs = [

                # ── Spread bets ──────────────────────────────────────────────
                {
                    "label":       "GSF Spread",
                    "type":        "spread",
                    "bet_col":     "GSF Spread Bet",
                    "home_spread": "Home Team Sportsbook Spread",
                    "away_spread": "Away Team Sportsbook Spread",
                    "unit_risk":   DEFAULT_UNIT_RISK,
                    "unit_win":    DEFAULT_UNIT_WIN,
                },
                {
                    "label":       "MP Spread",
                    "type":        "spread",
                    "bet_col":     "Massey-Peabody Spread Bet",
                    "home_spread": "Home Team Sportsbook Spread",
                    "away_spread": "Away Team Sportsbook Spread",
                    "unit_risk":   DEFAULT_UNIT_RISK,
                    "unit_win":    DEFAULT_UNIT_WIN,
                },
                {
                    "label":       "Sim Spread",
                    "type":        "spread",
                    "bet_col":     "Monte Carlo Spread Bet",
                    "home_spread": "Home Team Sportsbook Spread",
                    "away_spread": "Away Team Sportsbook Spread",
                    "unit_risk":   "MC Spread Unit Wager",
                    "unit_win":    "MC Spread Unit to Win",
                },
                {
                    "label":       "Sim Spread (Kelly)",
                    "type":        "spread",
                    "bet_col":     "Monte Carlo Spread Bet",
                    "home_spread": "Home Team Sportsbook Spread",
                    "away_spread": "Away Team Sportsbook Spread",
                    "unit_risk":   "MC Spread Kelly Wager",
                    "unit_win":    "MC Spread Kelly To Win",
                },
                {
                    "label":       "Consensus Spread",
                    "type":        "spread",
                    "bet_col":     "Consensus Spread Bet",
                    "home_spread": "Home Team Sportsbook Spread",
                    "away_spread": "Away Team Sportsbook Spread",
                    "unit_risk":   DEFAULT_UNIT_RISK,
                    "unit_win":    DEFAULT_UNIT_WIN,
                },

                # ── Moneyline bets ───────────────────────────────────────────
                # All use closing sportsbook moneyline for profit calculation.
                {
                    "label":     "GSF Moneyline",
                    "type":      "moneyline",
                    "bet_col":   "GSF Moneyline Bet",
                    "unit_risk": DEFAULT_UNIT_RISK,
                },
                {
                    "label":     "MP Moneyline",
                    "type":      "moneyline",
                    "bet_col":   "Massey-Peabody Moneyline Bet",
                    "unit_risk": DEFAULT_UNIT_RISK,
                },
                {
                    "label":     "Sim Moneyline",
                    "type":      "moneyline",
                    "bet_col":   "Monte Carlo Moneyline Bet",
                    "unit_risk": "MC ML Unit Wager",
                    "unit_win":  "MC ML Unit to Win",
                },
                {
                    "label":     "Sim Moneyline (Kelly)",
                    "type":      "moneyline",
                    "bet_col":   "Monte Carlo Moneyline Bet",
                    "unit_risk": "MC ML Kelly Wager",
                    "unit_win":  "MC ML Kelly To Win",
                },
                {
                    "label":     "Consensus Moneyline",
                    "type":      "moneyline",
                    "bet_col":   "Consensus Moneyline Bet",
                    "unit_risk": DEFAULT_UNIT_RISK,
                },

                # ── Total bets ───────────────────────────────────────────────
                # Evaluated against Total Line (sportsbook O/U line).
                # Direction comes from Monte Carlo Total Bet ("Over"/"Under"/"No Bet").
                {
                    "label":         "Sim Total",
                    "type":          "total",
                    "bet_col":       "Monte Carlo Total Bet",
                    "direction_col": "Monte Carlo Total Bet",
                    "total_line":    "Total Line",
                    "unit_risk":     "MC Total Unit Wager",
                    "unit_win":      "MC Total Unit to Win",
                },
                {
                    "label":         "Sim Total (Kelly)",
                    "type":          "total",
                    "bet_col":       "Monte Carlo Total Bet",
                    "direction_col": "Monte Carlo Total Bet",
                    "total_line":    "Total Line",
                    "unit_risk":     "MC Total Kelly Wager",
                    "unit_win":      "MC Total Kelly To Win",
                },
            ]

            # ── Per-row evaluation ─────────────────────────────────────────────
            for config in bet_configs:
                label    = config["label"]
                bet_col  = config["bet_col"]
                bet_type = config["type"]
                wl_col   = f"{label} Win/Loss"
                pnl_col  = f"{label} P/L"

                actual_data[wl_col]  = "Pending"
                actual_data[pnl_col] = 0.0

                if bet_col not in actual_data.columns:
                    print(f"   ⚠️  {label}: column '{bet_col}' not found — skipping")
                    continue

                for idx, row in actual_data.iterrows():
                    bet_team  = row.get(bet_col)
                    home_team = row.get("Home Team")
                    home_pts  = row.get("Home Team Points")
                    away_pts  = row.get("Away Team Points")
                    has_scores = pd.notna(home_pts) and pd.notna(away_pts)

                    # No bet recommended for this game
                    if pd.isna(bet_team) or str(bet_team).strip().lower() in ("", "nan", "no bet"):
                        actual_data.loc[idx, wl_col]  = "No Bet"
                        actual_data.loc[idx, pnl_col] = 0.0
                        continue

                    if bet_type == "spread":
                        home_s = row.get(config["home_spread"], 0)
                        away_s = row.get(config["away_spread"], 0)
                        won    = spread_result(
                            bet_team, home_team, home_pts, away_pts,
                            home_s, away_s
                        )
                        unit_risk = get_val(row, config["unit_risk"])
                        unit_win  = get_val(row, config.get("unit_win", DEFAULT_UNIT_WIN),
                                           DEFAULT_UNIT_WIN)
                        profit = spread_total_profit(unit_risk, unit_win, won)

                    elif bet_type == "moneyline":
                        won = ml_result(bet_team, home_team, home_pts, away_pts)
                        ml_col = ("Home Team Sportsbook Moneyline"
                                  if bet_team == home_team
                                  else "Away Team Sportsbook Moneyline")
                        ml        = row.get(ml_col)
                        unit_risk = get_val(row, config["unit_risk"])
                        profit    = ml_profit(unit_risk, ml, won) \
                            if pd.notna(ml) else 0.0

                    elif bet_type == "total":
                        direction = row.get(config["direction_col"], "")

                        # No bet for this game
                        if str(direction).strip().lower() in ("no bet", "", "nan"):
                            actual_data.loc[idx, wl_col]  = "No Bet"
                            actual_data.loc[idx, pnl_col] = 0.0
                            continue

                        total_line = row.get(config["total_line"])
                        total_pts  = (
                            float(home_pts) + float(away_pts)
                            if has_scores and pd.notna(home_pts) and pd.notna(away_pts)
                            else None
                        )
                        won       = total_result(direction, total_pts, total_line)
                        unit_risk = get_val(row, config["unit_risk"])
                        unit_win  = get_val(row, config.get("unit_win", DEFAULT_UNIT_WIN),
                                           DEFAULT_UNIT_WIN)
                        profit    = spread_total_profit(unit_risk, unit_win, won)

                    else:
                        won    = None
                        profit = 0.0

                    actual_data.loc[idx, wl_col]  = result_label(won, has_scores)
                    actual_data.loc[idx, pnl_col] = round(profit, 2)

            # ── Summary printout ───────────────────────────────────────────────
            print(f"\n   📊 Betting Results — Week {last_played_week} {target_year}")
            print(f"   {'Model':<28} {'Record':<28} P/L")
            print(f"   {'-'*28} {'-'*28} {'-'*12}")
            for config in bet_configs:
                label   = config["label"]
                wl_col  = f"{label} Win/Loss"
                pnl_col = f"{label} P/L"
                if wl_col not in actual_data.columns:
                    continue
                wins    = (actual_data[wl_col] == "Win").sum()
                losses  = (actual_data[wl_col] == "Loss").sum()
                pushes  = (actual_data[wl_col] == "Push").sum()
                pending = (actual_data[wl_col] == "Pending").sum()
                no_bet  = (actual_data[wl_col] == "No Bet").sum()
                total_pl = actual_data[pnl_col].sum()
                record  = f"{wins}W-{losses}L"
                if pushes:  record += f"-{pushes}P"
                if pending: record += f" ({pending} pending)"
                if no_bet:  record += f" ({no_bet} no bet)"
                print(f"   {label:<28} {record:<28} ${total_pl:+.2f}")

        except Exception as e:
            print(f"   ⚠️  Win/Loss calculation failed: {e}")
            import traceback
            traceback.print_exc()
        # ── 5. Add metadata columns ────────────────────────────────────────────
        actual_data["Data_Type"]   = "Final"
        actual_data["Week_Final"]  = last_played_week
        actual_data["Year_Final"]  = target_year

        # ── 6. Save this week's standalone file ────────────────────────────────
        weekly_output = (
            f"nfl-power-ratings/final_data/{target_year}_final_data/Week_{last_played_week}_{target_year}_Final_Data.csv"
        )
        actual_data.to_csv(weekly_output, index=False)
        print(f"   ✅ Saved weekly file → {weekly_output}")

        # ── 7. Append to the season-long Final Data file (idempotent) ──────────
        season_file = f"nfl-power-ratings/final_data/{target_year}_final_data/Season_{target_year}_Through_Week_{last_played_week}_Final_Data.csv"

        if last_played_week >= 1 and os.path.exists(season_file):
            existing = pd.read_csv(season_file)
            # Remove any existing rows for this week so re-runs are safe
            existing = existing[existing["Week_Final"] != last_played_week]
            season_df = pd.concat([existing, actual_data], ignore_index=True)
            sort_col = week_col if week_col in season_df.columns else "Week_Final"
            season_df = season_df.sort_values(sort_col).reset_index(drop=True)
            print(f"   📎 Appended to existing season file ({len(existing)} prior rows)")
        else:
            season_df = actual_data.copy()
            print(
                f"   📄 {'Week 1 — creating' if last_played_week == 1 else 'Creating'} "
                f"new season file"
            )

        season_df.to_csv(season_file, index=False)
        print(
            f"   ✅ Saved season file → {season_file} "
            f"({len(season_df)} total rows)"
        )

    create_actual_historical_data(last_played_week, starting_week, target_year, current_year)


if __name__ == "__main__":
    formatted_date = datetime.now().strftime("%m/%d/%Y")
    week_starting_dates = [
#        "09/03/2025", 
#        "09/10/2025"#, 
#        "09/17/2025",
#        "09/24/2025"#, 
#        "10/01/2025",
#        "10/08/2025", 
        "10/15/2025", 
        "10/22/2025", 
        "10/29/2025", 
        "11/05/2025", 
        "11/12/2025", 
        "11/19/2025", 
        "11/26/2025", 
#        "11/29/2025", 
#        "12/03/2025", 
#        "12/10/2025", 
#        "12/17/2025", 
#        "12/24/2025", 
#        "12/26/2025", 
#        "12/31/2025",
#        "01/06/2026",
        
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
#        "01/08/2025",
        
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
#        "01/10/2024",
        
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
#        "12/28/2022",
#        "01/04/2023",
#        "01/11/2023",
        
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
#        "12/29/2021", 
#        "01/05/2022",
#        "01/12/2022",
        
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
#        "01/05/2021",
        
#        formatted_date
    ]

    for date in week_starting_dates:
        loop_through_historical_final_data(date)
