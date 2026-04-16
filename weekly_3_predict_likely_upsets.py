###WITHOUT DIVISIONAL, AWAY, AND THURSDAY NIGHT GAMES

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import nflreadpy as nfl
from datetime import datetime

def loop_through_predictions(date):
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

    # --- CONFIGURATION ---
    TRAIN_FILE = f"nfl-pbp-data/nfl_games_with_schematic_data_2008_{target_year - 1}.csv"
    TEST_FILE = f"nfl-pbp-data/nfl_games_with_schematic_data_{target_year}_{target_year}.csv"
    OUTPUT_FILE = f"nfl-power-ratings/nfl_{target_year}_week_{upcoming_week}_matchup_upset_predictions.csv"
    
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
        loop_through_predictions(date)
