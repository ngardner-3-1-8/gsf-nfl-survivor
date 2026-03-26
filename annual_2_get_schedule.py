import pandas as pd
import nflreadpy as nfl
import os
from datetime import datetime

def update_annual_schedule():
    # 1. Determine the Year
    today = datetime.now()
    current_cal_year = today.year 
    
    if today.month < 5:
        target_year = current_cal_year - 1
    else:
        target_year = current_cal_year
    
    print(f"Fetching official NFL schedule for {target_year}...")

    try:
        # 2. Load schedule and team metadata
        schedule_raw = nfl.load_schedules([target_year])
        schedule_df = schedule_raw.to_pandas()
        
        # Load team info to get full names
        teams_df = nfl.load_teams().to_pandas()
        
        if schedule_df.empty:
            print(f"⚠️ Warning: No schedule found for {target_year} yet.")
            return

        # 3. Create Mapping Dictionary {Abbreviation: Full Name}
        # We use 'team_abbr' as the key and 'team_name' as the value
        team_map = dict(zip(teams_df['team_abbr'], teams_df['team_name']))

        # 4. Clean the Schedule Data
        schedule_df = schedule_df[schedule_df['game_type'] == 'REG'].copy()

        # Replace abbreviations with full names in away_team and home_team columns
        schedule_df['away_team'] = schedule_df['away_team'].map(team_map).fillna(schedule_df['away_team'])
        schedule_df['home_team'] = schedule_df['home_team'].map(team_map).fillna(schedule_df['home_team'])

        schedule_df = schedule_df.rename(columns={
            "away_team": "Away Team",
            "home_team": "Home Team",
            "week": "Week",
            "season": "Season",
            "gameday": "Date",
            "gametime": "Time",
            "weekday": "Day of Week",
            "away_score": "Away Score",
            "home_score": "Home Score",
            "location": "Location",
            "away_qb_name": "Away QB",
            "home_qb_name": "Home QB",
#            "referee": "Referee",
            "stadium": "Stadium",
            "roof": "Roof Type",
            "surface": "Surface Type",
            "temp": "Temperature",
            "wind": "Wind Speed"
        })

        # Define the list of columns to remove
        cols_to_remove = [
            'game_type', 'result', 'overtime', 'total', 'nfl_detail_id', 
            'total_line', 'under_odds', 'over_odds', 'div_game', 'home_coach', 
            'away_coach', 'referee', 'stadium_id', 'away_rest', 'home_rest', 
            'away_moneyline', 'home_moneyline', 'spread_line', 'away_spread_odds', 
            'home_spread_odds'
        ]
        
        # Drop the columns
        schedule_df = schedule_df.drop(columns=cols_to_remove)



        # 5. Save to CSV
        output_dir = "nfl-schedules"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = f"{output_dir}/schedule_{target_year}.csv"
        schedule_df.to_csv(file_path, index=False)
        
        print(f"✅ Success! {target_year} schedule saved with full team names to {file_path}")
        print(f"Total Games Found: {len(schedule_df)}")

    except Exception as e:
        print(f"❌ Error fetching schedule: {e}")

if __name__ == "__main__":
    update_annual_schedule()
