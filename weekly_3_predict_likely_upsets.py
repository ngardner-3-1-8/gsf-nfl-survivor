###WITHOUT DIVISIONAL, AWAY, AND THURSDAY NIGHT GAMES

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import nflreadpy as nfl
from datetime import datetime, timedelta
import calendar

def loop_through_predictions(date):
    # 1. Get current date
    today = pd.to_datetime(date)
    current_cal_year = today.year 
    
    # 2. Initial Year Logic based on Month
    target_year = current_cal_year - 1 if today.month < 6 else current_cal_year
    
    schedule_df = pd.read_csv(f"nfl-schedules/schedule_{target_year}.csv")
    schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])
    first_game_date = schedule_df['Date'].min()
    
    # 3. Calculate Important Dates Automatically
    def get_thanksgiving(year):
        c = calendar.monthcalendar(year, 11)
        thursdays = [row[calendar.THURSDAY] for row in c if row[calendar.THURSDAY] != 0]
        return datetime(year, 11, thursdays[3])
    
    thanksgiving_date = get_thanksgiving(target_year)
    black_friday = thanksgiving_date + timedelta(days=1)
    christmas_day = datetime(target_year, 12, 25)
    boxing_day = datetime(target_year, 12, 26)
    
    thanksgiving_week = int((thanksgiving_date - first_game_date).days/7) + 1 
    christmas_week = int((christmas_day - first_game_date).days/7) + 2 
    
    if today <= first_game_date:
        starting_week = 1
        upcoming_week = starting_week
    else:
        week_end_dates = schedule_df.groupby('Week')['Date'].max()
        completed_weeks = week_end_dates[week_end_dates <= today]
        if not completed_weeks.empty:
            standard_nfl_week = int(completed_weeks.index.max())           
            starting_week = standard_nfl_week + 1
            upcoming_week = starting_week
            
            # --- ADJUST FOR CIRCA SPECIAL WEEKS ---
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
            starting_week = 1
    
    # 5. Final Assignment to your variables
    current_year = target_year
    starting_year = target_year
    current_year_plus_1 = current_year + 1
    season_start_date = first_game_date - timedelta(days=1)
    
    thanksgiving_reset_date = black_friday + timedelta(days=1) 
    christmas_reset_date = boxing_day

    # --- CONFIGURATION ---
    TRAIN_FILE = f"nfl-pbp-data/nfl_games_with_schematic_data_2008_{target_year - 1}.csv"
    TEST_FILE = f"nfl-pbp-data/nfl_games_with_schematic_data_{target_year}_{target_year}.csv"
    OUTPUT_FILE = f"nfl-power-ratings/nfl_{target_year}_week_{upcoming_week}_matchup_upset_predictions.csv"
    
    # --- UPDATED CATEGORIES ---
    MATCHUP_CATEGORIES = [
        'Overall', 'Run', 'Pass', 'Pass_Deep', 'Pass_Short', 
        'Redzone', '3rd_Down', '1st_Down', 'Play_Action', 
        'Quick_Game_Proxy', 'Under_Pressure', 'Vs_Man', 'Vs_Zone', 'Pressured'
    ]
    
    def calculate_mismatches(df):
        df = df.copy()
        for cat in MATCHUP_CATEGORIES:
            # SWITCHED to EPA_Adj for mathematical magnitude
            h_off_col = f"home_Off_{cat}_EPA_Adj"
            a_def_col = f"away_Def_{cat}_EPA_Adj"
            
            a_off_col = f"away_Off_{cat}_EPA_Adj"
            h_def_col = f"home_Def_{cat}_EPA_Adj"
            
            if h_off_col in df.columns and a_def_col in df.columns:
                df[f'Matchup_HomeOff_{cat}'] = df[h_off_col] - df[a_def_col]
                
            if a_off_col in df.columns and h_def_col in df.columns:
                df[f'Matchup_AwayOff_{cat}'] = df[a_off_col] - df[h_def_col]
        return df
    
    def run_matchup_analysis():
        try:
            train_df = pd.read_csv(TRAIN_FILE)
            test_df = pd.read_csv(TEST_FILE)
        except Exception as e:
            print(f"Error loading files: {e}")
            return
    
        if test_df.empty:
            print("WARNING: Test file is empty. Cannot predict.")
            return
    
        print(f"\n--- Processing Predictions for {target_year} Week {upcoming_week} ---")
        
        train_df = calculate_mismatches(train_df)
        test_df = calculate_mismatches(test_df)
    
        mismatch_cols = [c for c in train_df.columns if c.startswith('Matchup_')]
        base_cols = ['spread_line', 'total_line', 'home_moneyline_decimal', 'away_moneyline_decimal', 
                     'home_rest_adv', 'away_rest_adv']
        
        feature_cols = base_cols + mismatch_cols
        feature_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
    
        # Prepare Data
        X_train = train_df[feature_cols].copy()
        y_train = train_df['Upset'].astype(int)
        
        # ISOLATE test set to only the week we want to predict
        predict_df = test_df[test_df['week'] == upcoming_week].copy()
        if predict_df.empty:
            print(f"No games found for Week {upcoming_week} in the test data.")
            return
            
        X_predict = predict_df[feature_cols].copy()
    
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_predict_imputed = imputer.transform(X_predict)
    
        # Train Model
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, 
                                    class_weight='balanced', random_state=42)
        rf.fit(X_train_imputed, y_train)
    
        # Predict
        probs = rf.predict_proba(X_predict_imputed)[:, 1]
        
        results = predict_df.copy()
        results['Model_Upset_Prob'] = probs
        
        # --- TRUE EDGE CALCULATION ---
        results['Underdog_ML_Decimal'] = np.where(results['spread_line'] > 0, 
                                                  results['away_moneyline_decimal'], 
                                                  results['home_moneyline_decimal'])
        
        results['Implied_Upset_Prob'] = 1 / results['Underdog_ML_Decimal']
        results['Model_Edge'] = results['Model_Upset_Prob'] - results['Implied_Upset_Prob']
        
        # --- INTERPRETATION ---
        def find_key_mismatch(row):
            spread = row['spread_line']
            if pd.isna(spread): return "Unknown"
            
            # Adjusted threshold for Raw EPA (0.15 is roughly a massive mismatch)
            THRESHOLD = 0.15 
            
            if spread > 0: 
                cols = [c for c in mismatch_cols if 'AwayOff' in c]
                if not cols: return "None"
                vals = row[cols]
                best_cat = vals.idxmax()
                score = vals.max()
                if score > THRESHOLD: 
                    clean_cat = best_cat.replace('Matchup_AwayOff_', '')
                    return f"Underdog Edge: {clean_cat} (+{score:.2f} EPA)"
                return "No Glaring Mismatch"
                
            elif spread < 0:
                cols = [c for c in mismatch_cols if 'HomeOff' in c]
                if not cols: return "None"
                vals = row[cols]
                best_cat = vals.idxmax()
                score = vals.max()
                if score > THRESHOLD:
                    clean_cat = best_cat.replace('Matchup_HomeOff_', '')
                    return f"Underdog Edge: {clean_cat} (+{score:.2f} EPA)"
                return "No Glaring Mismatch"
            return "Pick'em"
    
        results['Key_Schematic_Edge'] = results.apply(find_key_mismatch, axis=1)
    
        # Sort by EDGE instead of raw probability
        out_cols = ['week', 'away_team', 'home_team', 'spread_line', 
                    'Model_Upset_Prob', 'Implied_Upset_Prob', 'Model_Edge', 'Key_Schematic_Edge']
        results = results.sort_values('Model_Edge', ascending=False)
        
        results[out_cols].to_csv(OUTPUT_FILE, index=False)
        print(f"SUCCESS: Predictions saved to {OUTPUT_FILE}")
        
        print("\n--- Top Expected Value (EV) Upset Spots ---")
        for _, row in results.head(5).iterrows():
            print(f"{row['away_team']} @ {row['home_team']} (Spread: {row['spread_line']})")
            print(f"  Model Prob: {row['Model_Upset_Prob']:.1%} | Implied: {row['Implied_Upset_Prob']:.1%}")
            print(f"  Edge: {row['Model_Edge']:+.1%} | {row['Key_Schematic_Edge']}\n")
    
    # Execute the analysis for this loop iteration
    run_matchup_analysis()

if __name__ == "__main__":
    week_starting_dates = [
        "12/31/2025"
    ]

    for date in week_starting_dates:
        loop_through_predictions(date)
