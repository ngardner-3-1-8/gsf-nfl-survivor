###WITHOUT DIVISIONAL, AWAY, AND THURSDAY NIGHT GAMES

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import nflreadpy as nfl
from datetime import datetime

# 1. Get current date
today = datetime.now()

current_cal_year = today.year 

# 2. Initial Year Logic based on Month (User Rule)
# If Jan-May (< 6), assume we are finishing the previous season.
if today.month < 6:
    target_year = current_cal_year - 1

    # 3. Pre-Season Check (User Rule)
    # We need to see if the season has actually started yet.
    try:
        # Load the schedule for the target year
        schedule = nfl.load_schedules([target_year])
        
        schedule = schedule.to_pandas() # Convert here!
        
        # Now all the standard Pandas filtering works:
        reg_season_games = schedule[schedule['game_type'] == 'REG']
        
        if not reg_season_games.empty:
            # Find the very first game date of the season
            first_game_date = pd.to_datetime(reg_season_games['gameday'].min())
            
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
        print(games_played)
        
        if not games_played.empty:
            # Ensure gameday is datetime for accurate comparisons
            schedule['gameday'] = pd.to_datetime(schedule['gameday'])
            
            # Create a Series: index = week, value = the latest game date for that week
            week_end_dates = schedule[schedule['game_type'] == 'REG'].groupby('week')['gameday'].max()
            
            # Find weeks where the FINAL game has already occurred
            completed_weeks = week_end_dates[week_end_dates <= pd.to_datetime(today)]
            
            if not completed_weeks.empty:
                # The latest completed week
                last_played_week = int(completed_weeks.index.max())
                starting_week = last_played_week + 1
            else:
                # No weeks have fully completed yet
                starting_week = 1
                
            # Bound check
            if starting_week > 20: 
                starting_week = 20
        else:
            # If we fell back a year but that season is fully over, or if no games played yet
            starting_week = 1 
    
    except Exception as e:
        print(f"⚠️ Error in dynamic detection: {e}. Falling back to defaults.")
        # Fallback defaults to prevent crash
        target_year = 2025
        starting_week = 20
else:
    target_year = current_cal_year

    # 3. Pre-Season Check (User Rule)
    # We need to see if the season has actually started yet.
    try:
        # Load the schedule for the target year
        schedule = nfl.load_schedules([target_year])
        
        schedule = schedule.to_pandas() # Convert here!
        
        # Now all the standard Pandas filtering works:
        reg_season_games = schedule[schedule['game_type'] == 'REG']
        
        if not reg_season_games.empty:
            # Find the very first game date of the season
            first_game_date = pd.to_datetime(reg_season_games['gameday'].min())
            
            # Check if today is BEFORE the first game
            if pd.to_datetime(today) < first_game_date:
                print(f"Today ({today.date()}) is before the first game ({first_game_date.date()}). dropping year by 1.")
                target_year -= 0
                # Reload schedule for the adjusted year so we can calculate the week correctly below
                schedule = nfl.load_schedules([target_year])
                schedule = schedule.to_pandas()
                starting_week = 1
        
        # 4. Calculate the Current Week
        # We find the latest game that has happened to determine "current" week
        games_played = schedule[
            pd.to_datetime(schedule['gameday']) <= pd.to_datetime(today)
        ]
        print(games_played)
        
        if not games_played.empty:
            # Ensure gameday is datetime for accurate comparisons
            schedule['gameday'] = pd.to_datetime(schedule['gameday'])
            
            # Create a Series: index = week, value = the latest game date for that week
            week_end_dates = schedule[schedule['game_type'] == 'REG'].groupby('week')['gameday'].max()
            
            # Find weeks where the FINAL game has already occurred
            completed_weeks = week_end_dates[week_end_dates <= pd.to_datetime(today)]
            
            if not completed_weeks.empty:
                # The latest completed week
                last_played_week = int(completed_weeks.index.max())
                starting_week = last_played_week + 1
            else:
                # No weeks have fully completed yet
                starting_week = 1
                
            # Bound check
            if starting_week > 20: 
                starting_week = 20
        else:
            # If we fell back a year but that season is fully over, or if no games played yet
            starting_week = 1 
    
    except Exception as e:
        print(f"⚠️ Error in dynamic detection: {e}. Falling back to defaults.")
        # Fallback defaults to prevent crash
        target_year = 2025
        starting_week = 19

# --- CONFIGURATION ---
TRAIN_FILE = f"nfl-pbp-data/nfl_games_with_schematic_data_2008_{target_year - 1}.csv"
TEST_FILE = f"nfl-pbp-data/nfl_games_with_schematic_data_{target_year}_{target_year}.csv"
OUTPUT_FILE = f"nfl-power-ratings/nfl_{target_year}_matchup_upset_predictions.csv"

# --- UPDATED CATEGORIES ---
MATCHUP_CATEGORIES = [
    'Overall', 'Run', 'Pass', 'Pass_Deep', 'Pass_Short', 
    'Redzone', '3rd_Down', '1st_Down', 'Play_Action', 
    'Quick_Game_Proxy', 'Under_Pressure'
]

def calculate_mismatches(df):
    """
    Creates new columns representing the difference between Offense and Defense.
    Positive Value = Offense has the advantage.
    Negative Value = Defense has the advantage.
    """
    df = df.copy()
    
    print("Engineering Matchup Features...")
    
    for cat in MATCHUP_CATEGORIES:
        # Construct column names based on your file structure
        # Format in file: home_Off_Run_EPA_Pct, away_Def_Run_EPA_Pct
        
        h_off_col = f"home_Off_{cat}_EPA_Pct"
        a_def_col = f"away_Def_{cat}_EPA_Pct"
        
        a_off_col = f"away_Off_{cat}_EPA_Pct"
        h_def_col = f"home_Def_{cat}_EPA_Pct"
        
        # Check if columns exist before calculating (Crucial for 'Under_Pressure')
        if h_off_col in df.columns and a_def_col in df.columns:
            # 1. Home Offense vs Away Defense Matchup
            df[f'Matchup_HomeOff_{cat}'] = df[h_off_col] - df[a_def_col]
            
        if a_off_col in df.columns and h_def_col in df.columns:
            # 2. Away Offense vs Home Defense Matchup
            df[f'Matchup_AwayOff_{cat}'] = df[a_off_col] - df[h_def_col]

    return df

def run_matchup_analysis():
    # 1. Load Data
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    if test_df.empty:
        print("WARNING: Test file is empty. Cannot predict.")
        return

    # 2. Engineer Features (Calculate Mismatches)
    train_df = calculate_mismatches(train_df)
    test_df = calculate_mismatches(test_df)

    # 3. Define Features
    # We explicitly include our new 'Matchup_' columns
    mismatch_cols = [c for c in train_df.columns if c.startswith('Matchup_')]
    
    base_cols = ['spread_line', 'total_line', 'home_moneyline_decimal', 'away_moneyline_decimal', 
                 'home_rest_adv', 'away_rest_adv']
    
    feature_cols = base_cols + mismatch_cols
    
    # Filter for columns that actually exist in BOTH datasets
    feature_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
    
    print(f"Training on {len(feature_cols)} features...")

    # 4. Prepare Data
    X_train = train_df[feature_cols].copy()
    y_train = train_df['Upset'].astype(int)
    X_test = test_df[feature_cols].copy()

    # Impute missing data
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 5. Train Model
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, 
                                class_weight='balanced', random_state=42)
    rf.fit(X_train_imputed, y_train)

    # 6. Predict
    probs = rf.predict_proba(X_test_imputed)[:, 1]
    
    # 7. Format Results
    results = test_df.copy()
    results['Upset_Probability'] = probs
    
    # --- INTERPRETATION: IDENTIFY THE "BAD MATCHUP" ---
    def find_key_mismatch(row):
        spread = row['spread_line']
        if pd.isna(spread): return "Unknown"
        
        # If Home is Favorite (Spread > 0), we look for Away Advantages
        if spread > 0: 
            cols = [c for c in mismatch_cols if 'AwayOff' in c]
            if not cols: return "None"
            # Get values for this row
            vals = row[cols]
            best_cat = vals.idxmax()
            score = vals.max()
            if score > 30: 
                clean_cat = best_cat.replace('Matchup_AwayOff_', '')
                return f"Underdog Edge: {clean_cat} (+{score:.1f})"
            return "No Glaring Mismatch"
            
        # If Away is Favorite (Spread < 0), we look for Home Advantages
        elif spread < 0:
            cols = [c for c in mismatch_cols if 'HomeOff' in c]
            if not cols: return "None"
            vals = row[cols]
            best_cat = vals.idxmax()
            score = vals.max()
            if score > 30:
                clean_cat = best_cat.replace('Matchup_HomeOff_', '')
                return f"Underdog Edge: {clean_cat} (+{score:.1f})"
            return "No Glaring Mismatch"
        return "Pick'em"

    results['Key_Schematic_Edge'] = results.apply(find_key_mismatch, axis=1)

    # Output columns
    out_cols = ['week', 'away_team', 'home_team', 'spread_line', 'Upset_Probability', 'Key_Schematic_Edge']
    results = results.sort_values('Upset_Probability', ascending=False)
    
    results[out_cols].to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSUCCESS: Predictions saved to {OUTPUT_FILE}")
    print("\n--- Top Potential Upsets & The Matchup to Watch ---")
    print(results[out_cols].head(5))
    
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n--- Top Factors Driving the Model ---")
    for f in range(min(10, len(feature_cols))):
        print(f"{f+1}. {feature_cols[indices[f]]} ({importances[indices[f]]:.4f})")

if __name__ == "__main__":
    run_matchup_analysis()
