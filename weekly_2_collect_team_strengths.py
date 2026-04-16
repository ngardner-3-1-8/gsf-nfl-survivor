import pandas as pd
import numpy as np
import nflreadpy as nfl
from scipy.stats import percentileofscore
from datetime import datetime, timedelta
import warnings
import calendar

def loop_through_strengths(date):
    # 1. Get current date
    today = pd.to_datetime(date)
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

    START_YEAR = target_year
    END_YEAR = target_year
    DECAY_RATE = 0.00475
    MIN_WEIGHTED_PLAYS = 30
    GARBAGE_MIN = 0.05
    GARBAGE_MAX = 0.95
    HFA_EPA_VAL = 0.04
    
    warnings.filterwarnings("ignore")
    
    # --- 1. DATA LOADING & DECAY CALCULATION ---
    def load_data(target_year, ref_date):
        """
        Loads data for target_year and the two prior years.
        Calculates decay weights relative to ref_date.
        """
        years = [target_year - 2, target_year - 1, target_year]
        print(f"\n--- Processing Season {target_year} (Window: {years}) ---")
        
        try:
            pbp = nfl.load_pbp(seasons=years).to_pandas()
        except Exception as e:
            print(f"CRITICAL ERROR loading PBP: {e}")
            return pd.DataFrame(), False
        
        # Standard Filters
        pbp = pbp[(pbp['wp'] >= GARBAGE_MIN) & (pbp['wp'] <= GARBAGE_MAX)]
        pbp = pbp[pbp['play_type'].isin(['run', 'pass'])]
        pbp = pbp.dropna(subset=['epa', 'posteam', 'defteam'])
        
        # --- TIME DECAY WEIGHTS (Relative to History) ---
        pbp['game_date'] = pd.to_datetime(pbp['game_date'])
        
        # Calculate days ago relative to the reference date (e.g., Feb after the season)
        pbp['days_ago'] = (ref_date - pbp['game_date']).dt.days
        
        # Filter out any games that happened AFTER the reference date (just in case)
        pbp = pbp[pbp['days_ago'] >= 0]
        
        # Formula: weight = e^(-rate * days_ago)
        pbp['time_weight'] = np.exp(-DECAY_RATE * pbp['days_ago'])
        
        # Load FTN Data (Only available for recent years)
        try:
            ftn = nfl.load_ftn_charting(seasons=years).to_pandas()
            if 'nflverse_game_id' in ftn.columns:
                ftn = ftn.rename(columns={'nflverse_game_id': 'game_id', 'nflverse_play_id': 'play_id'})
            
            cols = ['game_id', 'play_id', 'is_blitz', 'is_pressure', 'is_play_action', 
                    'is_rpo', 'is_slot', 'is_motion']
            ftn_subset = ftn[[c for c in cols if c in ftn.columns]]
            pbp = pd.merge(pbp, ftn_subset, on=['game_id', 'play_id'], how='left')
            has_ftn = True
        except:
            # FTN data likely won't exist for 2008-2021, this is expected
            has_ftn = False

        # Load Participation Data (For Man/Zone and Pressure)
        try:
            participation = nfl.load_participation(seasons=years).to_pandas()
            part_cols = ['nflverse_game_id', 'play_id', 'defense_man_zone_type', 'was_pressure']
            
            # Merge it into your main PBP dataframe
            pbp = pd.merge(pbp, participation[part_cols], 
                           left_on=['game_id', 'play_id'], 
                           right_on=['nflverse_game_id', 'play_id'], 
                           how='left')
            has_participation = True
        except Exception as e:
            print(f"Failed to load participation data: {e}")
            has_participation = False
    
        # Load Rosters
        try:
            rosters = nfl.load_rosters(seasons=years).to_pandas()
            if 'gsis_id' in rosters.columns:
                roster_map = rosters.set_index('gsis_id')['position'].to_dict()
                pbp['receiver_pos'] = pbp['receiver_player_id'].map(roster_map)
            else:
                pbp['receiver_pos'] = np.nan
        except:
            pbp['receiver_pos'] = np.nan
        
        return pbp, has_ftn, has_participation
    
    # --- 2. CATEGORIES (Unchanged) ---
    def get_categories(df, has_ftn, has_participation):
        cats = {}
        cats['Overall'] = (df['play_type'].isin(['run', 'pass']))
        cats['Run'] = (df['play_type'] == 'run')
        cats['Pass'] = (df['play_type'] == 'pass')
        cats['Pass_Deep'] = (df['play_type'] == 'pass') & (df['air_yards'] >= 20)
        cats['Pass_Intermediate'] = (df['play_type'] == 'pass') & (df['air_yards'].between(10, 19))
        cats['Pass_Short'] = (df['play_type'] == 'pass') & (df['air_yards'].between(0, 9))
        cats['Pass_BehindLoS'] = (df['play_type'] == 'pass') & (df['air_yards'] < 0)
        cats['Redzone'] = (df['yardline_100'] <= 20)
        cats['3rd_Down'] = (df['down'] == 3)
        cats['1st_Down'] = (df['down'] == 1)
        cats['Scramble'] = (df['play_type'] == 'run') & (df['qb_scramble'] == 1)
        
        cats['Target_WR'] = (df['play_type'] == 'pass') & (df['receiver_pos'] == 'WR')
        cats['Target_TE'] = (df['play_type'] == 'pass') & (df['receiver_pos'] == 'TE')
        cats['Target_RB'] = (df['play_type'] == 'pass') & (df['receiver_pos'] == 'RB')
    
        cats['Quick_Game_Proxy'] = (df['play_type'] == 'pass') & (df['air_yards'] < 5) & \
                                   (df['qb_scramble'] == 0) & (df['sack'] == 0)
    
        if has_ftn:
            if 'is_blitz' in df.columns:
                cats['Versus_Blitz'] = (df['play_type'] == 'pass') & (df['is_blitz'] == 1)
                cats['No_Blitz']      = (df['play_type'] == 'pass') & (df['is_blitz'] == 0)
            if 'is_pressure' in df.columns:
                cats['Under_Pressure'] = (df['play_type'] == 'pass') & (df['is_pressure'] == 1)
                cats['Clean_Pocket']   = (df['play_type'] == 'pass') & (df['is_pressure'] == 0)
            if 'is_play_action' in df.columns:
                cats['Play_Action']    = (df['play_type'] == 'pass') & (df['is_play_action'] == 1)
            if 'is_rpo' in df.columns:
                cats['RPO'] = (df['is_rpo'] == 1)
            if 'is_slot' in df.columns:
                cats['Target_Slot'] = (df['play_type'] == 'pass') & (df['is_slot'] == 1)

        if has_participation:
            if 'defense_man_zone_type' in df.columns:
                cats['Vs_Man'] = (df['play_type'] == 'pass') & (df['defense_man_zone_type'] == 'MAN_COVERAGE')
                cats['Vs_Zone'] = (df['play_type'] == 'pass') & (df['defense_man_zone_type'] == 'ZONE_COVERAGE')
            if 'was_pressure' in df.columns:
                # was_pressure is usually a boolean or 1/0
                cats['Pressured'] = (df['play_type'] == 'pass') & (df['was_pressure'] == True)
                cats['Clean_Pocket_Part'] = (df['play_type'] == 'pass') & (df['was_pressure'] == False)
    
        return cats
    
    # --- 3. WEIGHTED CALCULATOR (Unchanged) ---
    def calculate_sos_adjusted_stats(df, has_ftn, has_participation):
        categories = get_categories(df, has_ftn, has_participation)
        results = []
        
        # Global League Averages for this window
        league_avg_epa = np.average(df['epa'], weights=df['time_weight'])
        league_avg_sr = np.average(df['success'], weights=df['time_weight'])
        
        for cat_name, mask in categories.items():
            subset = df[mask].copy()
            if subset.empty: continue
            
            subset['epa_w'] = subset['epa'] * subset['time_weight']
            subset['sr_w'] = subset['success'] * subset['time_weight']
            
            # --- DEFENSE ADJUSTMENTS ---
            def_scouting = subset.groupby('defteam')[['epa_w', 'sr_w', 'time_weight']].sum()
            def_scouting['def_avg_epa'] = def_scouting['epa_w'] / def_scouting['time_weight']
            def_scouting['def_avg_sr'] = def_scouting['sr_w'] / def_scouting['time_weight']
            
            # Vectorized Opponent Adjustment
            valid_defs = def_scouting[def_scouting['time_weight'] >= MIN_WEIGHTED_PLAYS]
            
            # Create dictionary mappings for instant lookup
            def_epa_map = (valid_defs['def_avg_epa'] - league_avg_epa).to_dict()
            def_sr_map = (valid_defs['def_avg_sr'] - league_avg_sr).to_dict()
    
            # Map directly to the series (100x faster than .apply)
            subset['opp_epa_adj'] = subset['defteam'].map(def_epa_map).fillna(0)
            subset['opp_sr_adj'] = subset['defteam'].map(def_sr_map).fillna(0)
            
            # HFA
            subset['is_home'] = np.where(subset['posteam'] == subset['home_team'], 1, 0)
            subset['hfa_adj'] = np.where(subset['is_home'] == 1, -HFA_EPA_VAL, HFA_EPA_VAL)
            
            # Apply Adj
            subset['epa_adjusted'] = subset['epa'] - subset['opp_epa_adj'] + subset['hfa_adj']
            subset['sr_adjusted'] = subset['success'] - subset['opp_sr_adj']
            
            subset['adj_epa_w'] = subset['epa_adjusted'] * subset['time_weight']
            subset['adj_sr_w'] = subset['sr_adjusted'] * subset['time_weight']
            
            # --- OFFENSE AGGREGATION ---
            off_stats = subset.groupby('posteam')[['adj_epa_w', 'adj_sr_w', 'time_weight', 'play_id']].agg(
                adj_epa_sum=('adj_epa_w', 'sum'),
                adj_sr_sum=('adj_sr_w', 'sum'),
                total_weight=('time_weight', 'sum'),
                raw_plays=('play_id', 'count')
            )
            
            off_stats['final_epa'] = off_stats['adj_epa_sum'] / off_stats['total_weight']
            off_stats['final_sr'] = off_stats['adj_sr_sum'] / off_stats['total_weight']
            off_stats = off_stats[off_stats['total_weight'] >= MIN_WEIGHTED_PLAYS]
            
            if not off_stats.empty:
                epas = off_stats['final_epa'].values
                srs = off_stats['final_sr'].values
                off_stats['epa_pct'] = off_stats['final_epa'].apply(lambda x: percentileofscore(epas, x, kind='weak'))
                off_stats['sr_pct'] = off_stats['final_sr'].apply(lambda x: percentileofscore(srs, x, kind='weak'))
                
                for team, row in off_stats.iterrows():
                    results.append({
                        'Team': team, 'Category': cat_name, 'Side': 'Offense',
                        'Plays': row['raw_plays'], 'Weight_Score': row['total_weight'],
                        'EPA_Adj': row['final_epa'], 'EPA_Pct': row['epa_pct'],
                        'SR_Adj': row['final_sr'],   'SR_Pct': row['sr_pct']
                    })
                    
            # --- OFFENSE ADJUSTMENTS (FOR DEFENSE) ---
            off_scouting = subset.groupby('posteam')[['epa_w', 'sr_w', 'time_weight']].sum()
            off_scouting['off_avg_epa'] = off_scouting['epa_w'] / off_scouting['time_weight']
            off_scouting['off_avg_sr'] = off_scouting['sr_w'] / off_scouting['time_weight']


            # Vectorized Opponent Adjustment
            valid_offs = off_scouting[off_scouting['time_weight'] >= MIN_WEIGHTED_PLAYS]
            
            # Create dictionary mappings for instant lookup
            off_epa_map = (valid_offs['off_avg_epa'] - league_avg_epa).to_dict()
            off_sr_map = (valid_offs['off_avg_sr'] - league_avg_sr).to_dict()
    
            # Map directly to the series (100x faster than .apply)
            subset['off_epa_adj'] = subset['posteam'].map(off_epa_map).fillna(0)
            subset['off_sr_adj'] = subset['posteam'].map(off_sr_map).fillna(0)
    
            subset['def_epa_adjusted'] = subset['epa'] - subset['off_epa_adj'] + subset['hfa_adj']
            subset['def_sr_adjusted'] = subset['success'] - subset['off_sr_adj']
            
            subset['def_epa_w'] = subset['def_epa_adjusted'] * subset['time_weight']
            subset['def_sr_w'] = subset['def_sr_adjusted'] * subset['time_weight']
            
            # --- DEFENSE AGGREGATION ---
            def_stats = subset.groupby('defteam')[['def_epa_w', 'def_sr_w', 'time_weight', 'play_id']].agg(
                adj_epa_sum=('def_epa_w', 'sum'),
                adj_sr_sum=('def_sr_w', 'sum'),
                total_weight=('time_weight', 'sum'),
                raw_plays=('play_id', 'count')
            )
            
            def_stats['final_epa'] = def_stats['adj_epa_sum'] / def_stats['total_weight']
            def_stats['final_sr'] = def_stats['adj_sr_sum'] / def_stats['total_weight']
            def_stats = def_stats[def_stats['total_weight'] >= MIN_WEIGHTED_PLAYS]
            
            if not def_stats.empty:
                # Invert for Defense (Lower EPA is better, so we rank inverted)
                def_stats['inv_epa'] = def_stats['final_epa'] * -1
                def_stats['epa_pct'] = def_stats['inv_epa'].apply(lambda x: percentileofscore(def_stats['inv_epa'].values, x, kind='weak'))
                
                def_stats['inv_sr'] = def_stats['final_sr'] * -1
                def_stats['sr_pct'] = def_stats['inv_sr'].apply(lambda x: percentileofscore(def_stats['inv_sr'].values, x, kind='weak'))
                
                for team, row in def_stats.iterrows():
                    results.append({
                        'Team': team, 'Category': cat_name, 'Side': 'Defense',
                        'Plays': row['raw_plays'], 'Weight_Score': row['total_weight'],
                        'EPA_Adj': row['final_epa'], 'EPA_Pct': row['epa_pct'],
                        'SR_Adj': row['final_sr'],   'SR_Pct': row['sr_pct']
                    })
    
        return pd.DataFrame(results)
    
    # --- MAIN LOOP ---
    if __name__ == "__main__":
        all_history = []
        
        # Loop from 2008 to 2024
        for year in range(START_YEAR, END_YEAR + 1):
            # We set the "Anchor Date" to mid-February of the following year
            # This ensures the decay treats the Super Bowl as "recent" and the season opener as "older"
            anchor_date = datetime(year + 1, 2, 20)
            
            df, has_ftn, has_participation = load_data(year, anchor_date)
            
            if not df.empty:
                print(f"Calculating Weighted Percentiles for {year}...")
                year_results = calculate_sos_adjusted_stats(df, has_ftn, has_participation)
                year_results['Season'] = year  # Tag the data with the season
                all_history.append(year_results)
            else:
                print(f"Skipping {year} due to data load failure.")
    
        # Combine all years
        if all_history:
            print("\n--- Combining Historical Data ---")
            full_df = pd.concat(all_history, ignore_index=True)
            
            # Save Long Format
            filename_long = f"nfl-pbp-data/nfl_history_{START_YEAR}_{END_YEAR}_weighted.csv"
            full_df.to_csv(filename_long, index=False)
            print(f"Saved: {filename_long}")
            
            # Example: Create a pivot for a specific view (e.g. 2024 only, or just overall ranks)
            # This creates a pivot of the *most recent* year just to check
            recent_year = full_df[full_df['Season'] == END_YEAR]
            if not recent_year.empty:
                pivot_df = recent_year.pivot_table(index='Team', columns=['Side', 'Category'], values='EPA_Pct')
                pivot_df.to_csv(f"nfl-power-ratings/nfl_ranks_{END_YEAR}_weighted.csv")
                print(f"Saved snapshot for {END_YEAR}")
    
            print("\nSUCCESS: Historical Database Created.")
        else:
            print("No data generated.")
    
    
    INPUT_FILE = f"nfl-pbp-data/nfl_history_{START_YEAR}_{END_YEAR}_weighted.csv"
    OUTPUT_FILE = f"nfl-pbp-data/nfl_games_with_schematic_data_{START_YEAR}_{END_YEAR}.csv"
    
    # File Paths for your ratings
    POWER_RATINGS_FILE = f"nfl-power-ratings/nfl_power_ratings_blended_week_{CURRENT_UPCOMING_WEEK}_{END_YEAR}.csv"
    HFA_RATINGS_FILE = "nfl-power-ratings/nfl_hfa_ratings.csv"
    
    # --- HELPER: TEAM MAPPING ---
    def get_rating_team_map():
        # Maps nflreadpy abbreviations to your Rating File abbreviations
        return {
            'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU',
            'LAR': 'LA',  'STL': 'LA',  'SD': 'LAC',  'OAK': 'LV'
        }
    
    def add_rest_and_ratings(sched):
        print("Calculating Rest Advantages and Imputing Spreads...")
        
        # 1. Load User Ratings
        try:
            pr_df = pd.read_csv(POWER_RATINGS_FILE)
            # Create dictionary: Team -> Rating
            power_map = pr_df.set_index('Team')['Power Rating'].to_dict()
            
            hfa_df = pd.read_csv(HFA_RATINGS_FILE)
            hfa_map = hfa_df.set_index('Team')['HFA (Points)'].to_dict()
            
            has_ratings = True
        except FileNotFoundError:
            print("WARNING: Rating files not found. Spread imputation will be skipped.")
            has_ratings = False
            power_map, hfa_map = {}, {}
    
        # 2. Prepare for Rest Calculation
        # Ensure dates are datetime
        sched['gameday'] = pd.to_datetime(sched['gameday'])
        sched = sched.sort_values(['season', 'gameday', 'game_id'])
        
        # Initialize Columns
        sched['home_rest'] = 7
        sched['away_rest'] = 7
        sched['home_rest_adv'] = 0.0
        sched['away_rest_adv'] = 0.0
        sched['home_cum_rest_adv'] = 0.0
        sched['away_cum_rest_adv'] = 0.0
        sched['my_projected_spread'] = np.nan
        
        # Track state per team: {Team: Last_Game_Date} and {Team: Cumulative_Score}
        # We reset this every season to avoid carry-over across years
        team_map = get_rating_team_map()
        
        # Iterate by Season to keep boundaries clean
        for season in sched['season'].unique():
            season_mask = sched['season'] == season
            season_games = sched.loc[season_mask].copy()
            
            last_game_date = {} # Stores date of last game
            cum_rest_adv = {}   # Stores running total of rest advantage
            
            # Iterate through games in chronological order
            for idx, row in season_games.iterrows():
                home = row['home_team']
                away = row['away_team']
                gameday = row['gameday']
                
                # --- HOME REST ---
                if home in last_game_date:
                    days_off = (gameday - last_game_date[home]).days
                    # Cap at 14 to avoid skewing data with byes too heavily in raw days? 
                    # User asked for raw days, so we keep it. 
                    # Note: Week 1 vs Week 1 will usually be handled by 'else' (7 days)
                    h_rest = days_off
                else:
                    h_rest = 7 # Default standard rest
                
                # --- AWAY REST ---
                if away in last_game_date:
                    days_off = (gameday - last_game_date[away]).days
                    a_rest = days_off
                else:
                    a_rest = 7
                    
                # --- CALCULATE ADVANTAGES ---
                # "add .125 for every extra day of rest they have over the opponent"
                h_adv = h_rest - a_rest
                a_adv = a_rest - h_rest # Inverse
                
                # --- CUMULATIVE TRACKING ---
                # Initialize if not present
                if home not in cum_rest_adv: cum_rest_adv[home] = 0
                if away not in cum_rest_adv: cum_rest_adv[away] = 0
                
                # "add up the team's rest advantage... including this week"
                cum_rest_adv[home] += h_adv
                cum_rest_adv[away] += a_adv
                
                # Store values
                sched.at[idx, 'home_rest'] = h_rest
                sched.at[idx, 'away_rest'] = a_rest
                sched.at[idx, 'home_rest_adv'] = h_adv
                sched.at[idx, 'away_rest_adv'] = a_adv
                sched.at[idx, 'home_cum_rest_adv'] = cum_rest_adv[home]
                sched.at[idx, 'away_cum_rest_adv'] = cum_rest_adv[away]
                
                # Update Last Game Date
                last_game_date[home] = gameday
                last_game_date[away] = gameday
                
                # --- IMPUTE SPREAD (If Ratings Exist) ---
                if has_ratings:
                    # Map teams to rating file names
                    h_name = team_map.get(home, home)
                    a_name = team_map.get(away, away)
                    
                    h_pow = power_map.get(h_name, np.nan)
                    a_pow = power_map.get(a_name, np.nan)
                    h_hfa = hfa_map.get(h_name, 0) # Default 0 if missing
                    
                    if pd.notna(h_pow) and pd.notna(a_pow):
                        # Logic: 
                        # Home Strength = Rating + HFA + (RestAdv * 0.125) + (CumRest * 0.0625)
                        # Away Strength = Rating + (RestAdv * 0.125) + (CumRest * 0.0625)
                        
                        h_adj = (h_adv * 0.125) + (cum_rest_adv[home] * 0.0625)
                        a_adj = (a_adv * 0.125) + (cum_rest_adv[away] * 0.0625)
                        
                        home_total = h_pow + h_hfa + h_adj
                        away_total = a_pow + a_adj
                        
                        # Spread: Points Home is Favored By
                        # If Home=24, Away=20, Spread should be roughly 4.
                        projected_spread = home_total - away_total
                        sched.at[idx, 'my_projected_spread'] = projected_spread
    
        # Fill missing spreads
        # "For games where the spread is not available..."
        mask_missing = sched['spread_line'].isna()
        sched.loc[mask_missing, 'spread_line'] = sched.loc[mask_missing, 'my_projected_spread']
        
        return sched
    
    
    def process_schematic_data(filepath):
        """
        Reads the long-format schematic history and pivots it to wide format
        so there is exactly one row per Team + Season.
        """
        print(f"Loading schematic history from {filepath}...")
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"ERROR: Could not find {filepath}. Please run the previous script first.")
            return pd.DataFrame()
    
        # Create a clean column name for the pivot
        # Format: Side_Category_Metric (e.g., Off_Run_EPA)
        df['Side_Short'] = df['Side'].map({'Offense': 'Off', 'Defense': 'Def'})
        
        # We want both the Percentile and the Adjusted Value
        # Pivot on EPA_Pct, SR_Pct, EPA_Adj, SR_Adj
        pivot_df = df.pivot_table(
            index=['Season', 'Team'],
            columns=['Side_Short', 'Category'],
            values=['EPA_Pct', 'SR_Pct', 'EPA_Adj', 'SR_Adj']
        )
    
        # Flatten the hierarchical columns
        # Example outcome: Off_Run_EPA_Pct
        pivot_df.columns = [f"{col[1]}_{col[2]}_{col[0]}" for col in pivot_df.columns]
        
        return pivot_df.reset_index()
    
    def load_and_clean_schedule(years):
        """
        Loads game schedules, fills missing moneylines, converts to decimal odds,
        calculates results, and cleans column names.
        """
        print("Loading NFL Schedule Data...")
        sched = nfl.load_schedules(years)
        
        # --- FIX: Convert to Pandas if it isn't already ---
        if hasattr(sched, "to_pandas"):
            sched = sched.to_pandas()
        
        # Filter for regular season and playoffs only
        sched = sched[sched['game_type'] != 'PRE'].copy()
    
        # --- 1. FILL MISSING MONEYLINES (From previous step) ---
        print("Imputing missing moneylines and converting to decimal...")
        
        odds_map = {
            0: [-110, -110], 0.5: [-116, -104], 1: [-122, 101], 1.5: [-128, 105], 2: [-131, 108],
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
    
        def get_imputed_odds(row):
            if pd.notna(row['home_moneyline']) and pd.notna(row['away_moneyline']):
                return row['home_moneyline'], row['away_moneyline']
            
            spread = row['spread_line']
            if pd.isna(spread):
                return row['home_moneyline'], row['away_moneyline']
    
            lookup_key = round(abs(spread) * 2) / 2
            lookup_key = min(lookup_key, 30.0)
            fav_ml, dog_ml = odds_map.get(lookup_key, (np.nan, np.nan))
            
            if spread > 0:
                return fav_ml, dog_ml
            elif spread < 0:
                return dog_ml, fav_ml
            else:
                return -110, -110
    
        filled_odds = sched.apply(get_imputed_odds, axis=1, result_type='expand')
        sched['home_moneyline'] = filled_odds[0]
        sched['away_moneyline'] = filled_odds[1]
    
        # --- 2. CONVERT TO DECIMAL ODDS ---
        def to_decimal(odds):
            if pd.isna(odds): return np.nan
            if odds > 0:
                return (odds / 100) + 1
            else:
                return (100 / abs(odds)) + 1
    
        sched['home_moneyline_decimal'] = sched['home_moneyline'].apply(to_decimal).round(2)
        sched['away_moneyline_decimal'] = sched['away_moneyline'].apply(to_decimal).round(2)
    
        # --- 3. DETERMINE RESULTS ---
        sched['winner'] = np.where(sched['result'] > 0, sched['home_team'],
                                   np.where(sched['result'] < 0, sched['away_team'], 'TIE'))
        
        sched['spread_margin'] = sched['result'] - sched['spread_line']
        sched['spread_winner'] = np.where(sched['spread_margin'] > 0, sched['home_team'],
                                          np.where(sched['spread_margin'] < 0, sched['away_team'], 'PUSH'))
        
        # --- 4. SELECT COLUMNS ---
        cols_to_keep = [
            'game_id', 'season', 'week', 'game_type', 'home_team', 'away_team', 
            'home_score', 'away_score', 'result', 'spread_line', 
            'home_moneyline', 'away_moneyline', 
            'home_moneyline_decimal', 'away_moneyline_decimal',  # <-- NEW COLUMNS
            'winner', 'spread_winner'
        ]
        
        return sched[cols_to_keep]
    
    def merge_data(sched, team_stats):
        """
        Merges the schedule with team stats for both Home and Away teams.
        """
        print("Merging Game Data with Team Schematic Data...")
        
        # Merge Home Team Stats
        # We look for matches on Season and Team
        merged = pd.merge(
            sched, 
            team_stats, 
            left_on=['season', 'home_team'], 
            right_on=['Season', 'Team'], 
            how='left'
        )
        
        # Rename the new columns to have a 'home_' prefix
        # Exclude the joining keys and original schedule columns from renaming
        sched_cols = list(sched.columns)
        new_cols = {c: f"home_{c}" for c in merged.columns if c not in sched_cols and c not in ['Season', 'Team']}
        merged = merged.rename(columns=new_cols)
        
        # Drop redundant join columns
        merged = merged.drop(columns=['Season', 'Team'], errors='ignore')
    
        # Merge Away Team Stats
        merged = pd.merge(
            merged, 
            team_stats, 
            left_on=['season', 'away_team'], 
            right_on=['Season', 'Team'], 
            how='left'
        )
        
        # Rename the new columns to have a 'away_' prefix
        new_cols_away = {c: f"away_{c}" for c in team_stats.columns if c not in ['Season', 'Team']}
        merged = merged.rename(columns=new_cols_away)
        
        # Drop redundant join columns
        merged = merged.drop(columns=['Season', 'Team'], errors='ignore')
        
        return merged
    
    # --- MAIN EXECUTION ---
    if __name__ == "__main__":
        # 1. Load Wide Schematic Data (Same as before)
        team_stats_wide = process_schematic_data(INPUT_FILE)
        
        if not team_stats_wide.empty:
            # 2. Load the Games (Raw)
            print("Loading Schedule...")
            schedule_df = nfl.load_schedules(list(range(START_YEAR, END_YEAR + 1)))
            if hasattr(schedule_df, "to_pandas"): schedule_df = schedule_df.to_pandas()
            schedule_df = schedule_df[schedule_df['game_type'] != 'PRE'].copy()
            
            # 3. Add Rest & Impute Spreads (NEW STEP)
            schedule_df = add_rest_and_ratings(schedule_df)
            
            # 4. Fill Moneylines & Convert (Same as before, using updated schedule_df)
            # Note: We use the function logic inline or call a modified version.
            # For brevity, here is the imputation block reused:
            
            print("Imputing missing moneylines...")
            odds_map = {
                0: [-110, -110], 0.5: [-116, -104], 1: [-122, 101], 1.5: [-128, 105], 2: [-131, 108],
                2.5: [-142, 117], 3: [-164, 135], 3.5: [-191, 156], 4: [-211, 171], 4.5: [-224, 181],
                5: [-234, 188], 5.5: [-244, 195], 6: [-261, 208], 6.5: [-282, 224], 7: [-319, 249],
                7.5: [-346, 268], 8: [-366, 282], 8.5: [-397, 302], 9: [-416, 314], 9.5: [-436, 327],
                10: [-483, 356], 10.5: [-538, 389], 11: [-567, 406], 11.5: [-646, 450], 12: [-660, 458],
                12.5: [-675, 466], 13: [-729, 494], 13.5: [-819, 539], 14: [-890, 573], 14.5: [-984, 615],
                15: [-1134, 677], 15.5: [-1197, 702], 16: [-1266, 728], 16.5: [-1267, 728], 17: [-1381, 769],
                17.5: [-1832, 906], 18: [-2149, 986], 18.5: [-2590, 1079], 19: [-3245, 1190], 19.5: [-4323, 1324],
                20: [-4679, 1359], 20.5: [-5098, 1396], 21: [-5597, 1434], 21.5: [-6000, 1500], 22: [-6500, 1600],
                22.5: [-7000, 1650], 23: [-7500, 1700], 23.5: [-8000, 1750], 24: [-8500, 1800], 24.5: [-9000, 1850],
                25: [-9500, 1900], 25.5: [-10000, 2000], 30: [-10000, 2000]
            }
            
            def get_imputed_odds(row):
                if pd.notna(row['home_moneyline']) and pd.notna(row['away_moneyline']):
                    return row['home_moneyline'], row['away_moneyline']
                spread = row['spread_line']
                if pd.isna(spread): return row['home_moneyline'], row['away_moneyline']
                lookup_key = min(round(abs(spread) * 2) / 2, 30.0)
                fav, dog = odds_map.get(lookup_key, (-110, -110))
                if spread > 0: return fav, dog
                elif spread < 0: return dog, fav
                return -110, -110
    
            filled_odds = schedule_df.apply(get_imputed_odds, axis=1, result_type='expand')
            schedule_df['home_moneyline'] = filled_odds[0]
            schedule_df['away_moneyline'] = filled_odds[1]
            
            # Decimal Conversion
            def to_decimal(odds):
                if pd.isna(odds): return np.nan
                if odds > 0: return (odds / 100) + 1
                return (100 / abs(odds)) + 1
                
            schedule_df['home_moneyline_decimal'] = schedule_df['home_moneyline'].apply(to_decimal).round(2)
            schedule_df['away_moneyline_decimal'] = schedule_df['away_moneyline'].apply(to_decimal).round(2)
            
            # Results
            schedule_df['winner'] = np.where(schedule_df['result'] > 0, schedule_df['home_team'],
                                       np.where(schedule_df['result'] < 0, schedule_df['away_team'], 'TIE'))
            schedule_df['spread_margin'] = schedule_df['result'] - schedule_df['spread_line']
            schedule_df['spread_winner'] = np.where(schedule_df['spread_margin'] > 0, schedule_df['home_team'],
                                              np.where(schedule_df['spread_margin'] < 0, schedule_df['away_team'], 'PUSH'))
    
    
            is_home_fav = schedule_df['spread_line'] > 0
            is_away_fav = schedule_df['spread_line'] < 0
            
            upset_condition = (is_home_fav & (schedule_df['winner'] == schedule_df['away_team'])) | \
                              (is_away_fav & (schedule_df['winner'] == schedule_df['home_team']))
            
            schedule_df['Upset'] = upset_condition
    
            final_df = merge_data(schedule_df, team_stats_wide)
            
            # --- NEW: REMOVE SPECIFIC COLUMNS ---
            # Add any other columns you want to remove here
            cols_to_remove = [
                'old_game_id', 'nfl_detail_id', 'pfr', 'pff', 'espn', 
                'away_qb_id', 'home_qb_id', 'away_qb_name', 'home_qb_name',
                'stadium', 'weather', 'wind', 'roof', 'surface', 'temp',
                'div_game', 'location', 'gameday', 'weekday', 'gametime',
                'referee', 'stadium_id'
            ]
            
            # Only drop columns that actually exist in the dataframe
            existing_cols_to_drop = [c for c in cols_to_remove if c in final_df.columns]
            final_df = final_df.drop(columns=existing_cols_to_drop)
            mask = final_df['week'] >= CURRENT_UPCOMING_WEEK
            final_df = final_df[mask]
    
            
            final_df.to_csv(OUTPUT_FILE, index=False)
            print(f"\nSUCCESS! File created: {OUTPUT_FILE}")

if __name__ == "__main__":
    formatted_date = datetime.now().strftime("%m/%d/%Y")
    week_starting_dates = [
#        "09/03/2025", "09/10/2025", "09/17/2025", "09/24/2025", 
#        "10/01/2025", "10/08/2025", "10/15/2025", "10/22/2025", 
#        "10/29/2025", "11/05/2025", "11/12/2025", "11/19/2025", 
#        "11/26/2025", "11/29/2025", "12/03/2025", "12/10/2025", 
#        "12/17/2025", "12/24/2025", "12/26/2025", 
        "12/31/2025"
#        formatted_date
    ]

    for date in week_starting_dates:
        loop_through_strengths(date)
