#GET THE BASIC RANKINGS

import pandas as pd
import numpy as np
import nflreadpy as nfl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
from datetime import datetime, timedelta


def loop_through_rankings(date):
    output_folders = ["nfl-power-ratings"] 
    
    for folder in output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")
    
    
    # 1. Get current date
    today = pd.to_datetime(date)
    
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
            
            if not games_played.empty:
                # If games have been played, the "starting_week" for your script 
                # (which usually scrapes the *upcoming* week) should be the last played week + 1.
                last_played_week = int(games_played['week'].max())
                starting_week = last_played_week + 1
                
                # Bound check: If season is over (e.g. Week 22), cap it or handle as needed
                if starting_week > 19: 
                    starting_week = 19 
            else:
                # If we fell back a year but that season is fully over, or if no games played yet
                starting_week = 1 
        
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
            
            schedule = schedule.to_pandas() # Convert here!
            
            # Now all the standard Pandas filtering works:
            reg_season_games = schedule[schedule['game_type'] == 'REG']
            
            if not reg_season_games.empty:
                # Find the very first game date of the season
                first_game_date = pd.to_datetime(reg_season_games['gameday'].min())
                
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
                if starting_week > 19: 
                    starting_week = 19 
            else:
                # If we fell back a year but that season is fully over, or if no games played yet
                starting_week = 1 
        
        except Exception as e:
            print(f"⚠️ Error in dynamic detection: {e}. Falling back to defaults.")
            # Fallback defaults to prevent crash
            target_year = 2025
            starting_week = 19
    
    CURRENT_UPCOMING_WEEK = starting_week
    
    # --- Configuration ---
    DAYS_WINDOW = 840 #840
    DECAY_RATE = 0.00475
    ITERATIONS = 100 
    
    # BLENDING WEIGHTS
    WEIGHT_EPA = 0.42
    WEIGHT_SR = 0.58
    
    # --- NEW OPTIMIZATION PARAMETERS ---
    SOS_MULTIPLIER = 0.86       # Boosts value of Strength of Schedule (1.0 = Neutral, 1.15 = Heavy Weight)
    REGRESSION_RATE = 0.01      # Pulls ratings toward 0.0 (League Avg) to fix "outliers"
    BLOWOUT_EPA_CAP = 21.25      # Soft cap for Game-Level Net EPA (Diminishing returns after this #)
    
    garbage_time_min = .05
    garbage_time_max = .95
    
    luck_adj_min = -4.875
    luck_adj_max = 4.875
    
    # CONVERSION FACTOR (+1% SR = 0.70 Points)
    SR_TO_POINTS_COEFF = 0.765 
    
    # MANUAL OVERRIDE: 
    MANUAL_CURRENT_STARTERS = {
    #    'KC': 'G.Minshew',
    }
    
    def load_pbp_data(years):
        print(f"Loading PBP data for {years}...")
        try:
            df = nfl.load_pbp(seasons=years).to_pandas()
            return df
        except Exception as e:
            print(f"Error loading PBP data: {e}")
            return pd.DataFrame()
    
    def get_qb_ratings_fast(years, target_year, current_upcoming_week):
        print(f"Loading Player Stats for {years}...")
        try:
            stats = nfl.load_player_stats(seasons=years).to_pandas()
            qbs = stats[stats['position'] == 'QB'].copy()
            
            # --- NEW FIX: Filter out future stats ---
            qbs = qbs[
                (qbs['season'] < target_year) | 
                ((qbs['season'] == target_year) & (qbs['week'] < current_upcoming_week))
            ].copy()
            
            # --- FIX 1: COLUMN NAMES ---
            team_col = 'recent_team' if 'recent_team' in qbs.columns else 'team'
            
            if 'sacks_suffered' in qbs.columns:
                qbs['sacks_val'] = qbs['sacks_suffered']
            elif 'sacks' in qbs.columns:
                qbs['sacks_val'] = qbs['sacks']
            else:
                qbs['sacks_val'] = 0 
                
            cols_to_fix = ['passing_epa', 'rushing_epa', 'attempts', 'carries']
            for col in cols_to_fix:
                if col in qbs.columns:
                    qbs[col] = qbs[col].fillna(0)
                else:
                    qbs[col] = 0
            qbs['sacks_val'] = qbs['sacks_val'].fillna(0)
                
            qbs['total_epa'] = qbs['passing_epa'] + qbs['rushing_epa']
            qbs['total_involvement'] = qbs['attempts'] + qbs['sacks_val'] + qbs['carries']
            
            # --- EFFICIENCY RATING ---
            qb_career = qbs.groupby('player_name').agg(
                career_epa=('total_epa', 'sum'),
                career_plays=('total_involvement', 'sum')
            ).reset_index()
            
            qb_career['epa_per_play'] = qb_career['career_epa'] / qb_career['career_plays']
            qb_career = qb_career[qb_career['career_plays'] > 50]
            qb_rating_map = pd.Series(qb_career.epa_per_play.values, index=qb_career.player_name).to_dict()
            
            # --- TEAM VOLUME ---
            team_game_stats = qbs.groupby([team_col, 'season', 'week'])['total_involvement'].sum().reset_index()
            team_volume = team_game_stats.groupby(team_col)['total_involvement'].mean()
            team_volume_map = team_volume.to_dict()
            
            return qb_rating_map, team_volume_map
            
        except Exception as e:
            print(f"Error loading player stats: {e}")
            return {}, {}
    
    def weighted_avg(values, weights):
        if len(values) == 0: return 0
        return np.average(values, weights=weights)
    
    # --- NEW HELPER: Diminishing Returns for Blowouts ---
    def apply_game_cap(net_val, limit):
        """
        If a team wins by a massive amount (e.g. +30 EPA), 
        logarithmically dampen the excess so it doesn't skew the model.
        """
        sign = np.sign(net_val)
        abs_val = abs(net_val)
        if abs_val > limit:
            excess = abs_val - limit
            # Add log of excess to the limit
            return sign * (limit + np.log(excess + 1) * 2.0)
        return net_val
    
    def build_power_ratings():
        # 1. Setup
        current_date = today
        start_date = current_date - timedelta(days=DAYS_WINDOW)
        years_to_load = [target_year, target_year - 1, target_year - 2, target_year - 3]
        
        # 2. Load
        pbp = load_pbp_data(years_to_load)
        if pbp.empty: return pd.DataFrame()
        
        qb_rating_map, team_qb_vol_map = get_qb_ratings_fast(years_to_load)
    
        # 3. Process PBP
        print("Processing PBP data...")
        pbp['game_date'] = pd.to_datetime(pbp['game_date'])
        pbp = pbp[(pbp['game_date'] >= start_date) & (pbp['game_date'] <= current_date)].copy()
        
        valid_types = ['run', 'pass', 'punt', 'field_goal', 'kickoff', 'extra_point']
        pbp = pbp[pbp['play_type'].isin(valid_types)]
        
        pbp = pbp.dropna(subset=['wp', 'epa', 'posteam', 'defteam', 'success'])
        pbp = pbp[(pbp['wp'] >= garbage_time_min) & (pbp['wp'] <= garbage_time_max)]
        pbp['capped_epa'] = pbp['epa'].clip(lower = luck_adj_min, upper = luck_adj_max)
        
        pbp['days_ago'] = (current_date - pbp['game_date']).dt.days
        pbp['weight'] = np.exp(-DECAY_RATE * pbp['days_ago'])
    
        # 4. Map Games to QBs
        pass_plays = pbp[pbp['play_type'] == 'pass']
        game_qb_map = pass_plays.groupby(['game_id', 'posteam', 'passer_player_name']).size().reset_index(name='counts')
        game_qb_map = game_qb_map.sort_values('counts', ascending=False).drop_duplicates(subset=['game_id', 'posteam'])
        game_qb_dict = game_qb_map.set_index(['game_id', 'posteam'])['passer_player_name'].to_dict()
    
        # 5. Build Schedule & Metrics
        game_stats = pbp.groupby(['game_id', 'posteam', 'defteam', 'game_date']).agg(
            total_epa_sum=('capped_epa', 'sum'),
            success_sum=('success', 'sum'),
            play_count=('play_id', 'count'),
            weight=('weight', 'mean')
        ).reset_index()
    
        df_games = []
        for g_id in game_stats['game_id'].unique():
            g_data = game_stats[game_stats['game_id'] == g_id]
            if len(g_data) != 2: continue 
            
            team_a = g_data.iloc[0] 
            team_b = g_data.iloc[1] 
            
            # Raw Net EPA
            raw_net_epa = team_a['total_epa_sum'] - team_b['total_epa_sum']
            
            # --- OPTIMIZATION: Apply Diminishing Returns (Blowout Cap) ---
            net_epa_a = apply_game_cap(raw_net_epa, BLOWOUT_EPA_CAP)
            
            sr_a = team_a['success_sum'] / team_a['play_count'] if team_a['play_count'] > 0 else 0
            sr_b = team_b['success_sum'] / team_b['play_count'] if team_b['play_count'] > 0 else 0
            net_sr_a = sr_a - sr_b
            
            qb_a = game_qb_dict.get((g_id, team_a['posteam']), None)
            qb_b = game_qb_dict.get((g_id, team_b['posteam']), None)
            
            df_games.append({
                'team': team_a['posteam'], 'opponent': team_b['posteam'], 
                'net_epa': net_epa_a, 'net_sr': net_sr_a,
                'weight': team_a['weight'], 'game_qb': qb_a
            })
            df_games.append({
                'team': team_b['posteam'], 'opponent': team_a['posteam'], 
                'net_epa': -net_epa_a, 'net_sr': -net_sr_a,
                'weight': team_b['weight'], 'game_qb': qb_b
            })
            
        df_sched = pd.DataFrame(df_games)
        
        # 6. Parallel SRS Solver
        print(f"Calculating Schedule Adjustments ({ITERATIONS} iterations)...")
        teams = df_sched['team'].unique()
        
        curr_epa_ratings = {t: 0.0 for t in teams}
        curr_sr_ratings = {t: 0.0 for t in teams}
        final_sos = {}
        
        for i in range(ITERATIONS):
            temp_epa_ratings = {}
            temp_sr_ratings = {}
            for team in teams:
                t_games = df_sched[df_sched['team'] == team]
                if t_games.empty: 
                    temp_epa_ratings[team] = 0
                    temp_sr_ratings[team] = 0
                    continue
                
                # EPA
                avg_net_epa = weighted_avg(t_games['net_epa'], t_games['weight'])
                opp_epa_ratings = [curr_epa_ratings.get(opp, 0) for opp in t_games['opponent']]
                sos_epa = weighted_avg(opp_epa_ratings, t_games['weight'])
                
                # --- OPTIMIZATION: SOS Multiplier ---
                # We give extra credit (or penalty) for the strength of schedule
                temp_epa_ratings[team] = avg_net_epa + (sos_epa * SOS_MULTIPLIER)
                
                # SR
                avg_net_sr = weighted_avg(t_games['net_sr'], t_games['weight'])
                opp_sr_ratings = [curr_sr_ratings.get(opp, 0) for opp in t_games['opponent']]
                sos_sr = weighted_avg(opp_sr_ratings, t_games['weight'])
                
                # Same SOS Multiplier for SR
                temp_sr_ratings[team] = avg_net_sr + (sos_sr * SOS_MULTIPLIER)
                
                if i == ITERATIONS - 1: final_sos[team] = sos_epa 
                
            curr_epa_ratings = temp_epa_ratings
            curr_sr_ratings = temp_sr_ratings
            
        # 7. Final Aggregation
        print("Blending Ratings and Finalizing...")
        final_stats = []
        
        def get_epa_per_game_sum(slice_df, epa_col='capped_epa'):
            if slice_df.empty: return 0.0
            g_sums = slice_df.groupby('game_id')[epa_col].sum()
            g_weights = slice_df.groupby('game_id')['weight'].mean()
            return weighted_avg(g_sums, g_weights)
        
        st_types = ['punt', 'field_goal', 'kickoff', 'extra_point']
    
        for team in teams:
            off_slice = pbp[(pbp['posteam'] == team) & (pbp['play_type'].isin(['run', 'pass']))]
            def_slice = pbp[(pbp['defteam'] == team) & (pbp['play_type'].isin(['run', 'pass']))]
            st_kick = pbp[(pbp['posteam'] == team) & (pbp['play_type'].isin(st_types))].copy()
            st_kick['net_epa_for_team'] = st_kick['capped_epa']
            st_ret = pbp[(pbp['defteam'] == team) & (pbp['play_type'].isin(st_types))].copy()
            st_ret['net_epa_for_team'] = -st_ret['capped_epa']
            st_all = pd.concat([st_kick, st_ret])
    
            off_epa_game = get_epa_per_game_sum(off_slice)
            def_epa_game = get_epa_per_game_sum(def_slice)
            st_epa_game = get_epa_per_game_sum(st_all, epa_col='net_epa_for_team')
            
            # QB Logic
            curr_starter = MANUAL_CURRENT_STARTERS.get(team, None)
            if curr_starter is None:
                team_games = game_qb_map[game_qb_map['posteam'] == team]
                if not team_games.empty:
                    last_game_slice = pass_plays[pass_plays['posteam'] == team].sort_values('game_date').tail(1)
                    if not last_game_slice.empty:
                        last_game_id = last_game_slice['game_id'].values[0]
                        curr_starter = game_qb_dict.get((last_game_id, team), 'Unknown')
            
            curr_qb_rating = qb_rating_map.get(curr_starter, 0.0)
            t_games_sched = df_sched[df_sched['team'] == team]
            
            total_weight = 0
            starter_weight = 0
            hist_qb_ratings = []
            hist_weights = []
            for _, row in t_games_sched.iterrows():
                g_qb = row['game_qb']
                w = row['weight']
                rating = qb_rating_map.get(g_qb, 0.0)
                hist_qb_ratings.append(rating)
                hist_weights.append(w)
                total_weight += w
                if g_qb == curr_starter: starter_weight += w
                    
            avg_hist_qb_rating = weighted_avg(hist_qb_ratings, hist_weights)
            starter_weight_pct = starter_weight / total_weight if total_weight > 0 else 0
            
            # QB Lift (Efficiency * Volume)
            team_qb_vol = team_qb_vol_map.get(team, 40.0)
            qb_adj_total = (curr_qb_rating - avg_hist_qb_rating) * (team_qb_vol / 2.1)
            
            # Blending
            epa_rating_final = curr_epa_ratings[team] + qb_adj_total
            sr_rating_pct = curr_sr_ratings[team]
            sr_rating_points = (sr_rating_pct * 100) * SR_TO_POINTS_COEFF
            
            raw_power_rating = (epa_rating_final * WEIGHT_EPA) + (sr_rating_points * WEIGHT_SR)
            
            # --- OPTIMIZATION: Market Regression ---
            # Pull the final rating slightly towards 0.0 to account for variance
            final_power_rating = raw_power_rating * (1.0 - REGRESSION_RATE)
    
            # Your manual scaler from before (kept for consistency)
            final_power_rating = final_power_rating * 1.04 
    
            adj_off_epa_game = off_epa_game + qb_adj_total
    
            final_stats.append({
                'Team': team,
                'Power Rating': final_power_rating,
                'Projected QB': curr_starter,
                'Starter Weight %': starter_weight_pct,
                'QB Adj (Pts)': qb_adj_total,
                'EPA Rating': epa_rating_final,
                'SR Rating (Pts)': sr_rating_points,
                'Strength of Schedule': final_sos[team],
                'Offensive EPA/Game': adj_off_epa_game,
                'Defensive EPA/Game': def_epa_game,
                'Special Teams EPA/Game': st_epa_game
            })
            
        df = pd.DataFrame(final_stats)
        
        cols = ['Team', 'Power Rating', 'Projected QB', 'Starter Weight %', 'QB Adj (Pts)',
                'EPA Rating', 'SR Rating (Pts)', 'Strength of Schedule', 
                'Offensive EPA/Game', 'Defensive EPA/Game', 'Special Teams EPA/Game']
        
        df = df[cols].sort_values(by='Power Rating', ascending=False).reset_index(drop=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(3)
        
        return df
    
    def compare_models(my_df, mp_file_path):
        mp_df = pd.read_csv(mp_file_path)
        mp_df.rename(columns={'Rating': 'MP_Rating'}, inplace=True)
        mp_df['Team'] = mp_df['Team'].replace({'JAC': 'JAX', 'LAR': 'LA'})
        comparison = pd.merge(my_df, mp_df, on='Team', suffixes=('_My', '_MP'))
        
        pearson_corr, _ = pearsonr(comparison['Power Rating'], comparison['MP_Rating'])
        spearman_corr, _ = spearmanr(comparison['Power Rating'], comparison['MP_Rating'])
        
        comparison['Diff'] = comparison['Power Rating'] - comparison['MP_Rating']
        mae = comparison['Diff'].abs().mean()
        rmse = np.sqrt((comparison['Diff'] ** 2).mean())
        
        print("--- MODEL COMPARISON REPORT ---")
        print(f"Pearson Correlation (Values):  {pearson_corr:.4f}")
        print(f"Spearman Correlation (Ranks):  {spearman_corr:.4f}")
        print(f"Mean Absolute Error (Points):  {mae:.4f}")
        print(f"RMSE (Outlier Sensitivity):    {rmse:.4f}")
        print("-" * 30)
        
        comparison['Abs_Diff'] = comparison['Diff'].abs()
        top_disagreements = comparison.sort_values('Abs_Diff', ascending=False).head(5)
        
        print("\nTop 5 Disagreements (Where do we differ?):")
        print(top_disagreements[['Team', 'Power Rating', 'MP_Rating', 'Diff']].to_string(index=False))
        
        # --- VOLATILITY CHECK ---
        print("\n--- VOLATILITY CHECK ---")
        my_std = comparison['Power Rating'].std()
        mp_std = comparison['MP_Rating'].std()
        print(f"Your Spread (StdDev): {my_std:.4f}")
        print(f"MP Spread (StdDev):   {mp_std:.4f}")
        if mp_std > 0 and my_std > 0:
            ratio = mp_std / my_std
            print(f"Scalar Needed to match volatility: {ratio:.4f}")
    
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.regplot(x='MP_Rating', y='Power Rating', data=comparison, 
                    scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        
        for i, row in top_disagreements.iterrows():
            plt.text(row['MP_Rating']+0.2, row['Power Rating'], row['Team'], fontsize=9)
            
        plt.title(f"Model Correlation: r = {pearson_corr:.3f}")
        plt.xlabel("Massey-Peabody Rating")
        plt.ylabel("Your Power Rating")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        return comparison
    
    if __name__ == "__main__":
        # 1. Build
        df = build_power_ratings()
        
        if not df.empty:
            filename = f"nfl-power-ratings/nfl_power_ratings_blended_week_{CURRENT_UPCOMING_WEEK}_{target_year}.csv"
            
            # 2. Compare if file exists
            if os.path.exists('nfl-power-ratings/mp_ratings.csv'):
                print("Found MP Ratings file. Merging and analyzing...")
                try:
                    final_df = compare_models(df, 'nfl-power-ratings/mp_ratings.csv')
                except Exception as e:
                    print(f"Error during merge: {e}. Saving internal ratings only.")
                    final_df = df
            else:
                print("mp_ratings.csv not found. Skipping comparison.")
                final_df = df
    
            # 3. Save
            final_df.to_csv(filename, index=False)
            print(f"\nSuccessfully created {filename}")
    
    #GET THE HOME FIELD ADVANTAGE
    
    # --- Configuration ---
    DAYS_WINDOW = 840 
    DECAY_RATE = 0.00475
    ITERATIONS = 50  # Lower iterations needed for HFA once global is set
    
    # BLENDING WEIGHTS
    WEIGHT_EPA = 0.42
    WEIGHT_SR = 0.58
    
    # OPTIMIZATION PARAMETERS
    SOS_MULTIPLIER = 1.0        # Keep neutral for HFA calculation to avoid double counting
    BLOWOUT_EPA_CAP = 21.25     
    
    garbage_time_min = .05
    garbage_time_max = .95
    
    luck_adj_min = -4.875
    luck_adj_max = 4.875
    
    # CONVERSION FACTOR
    SR_TO_POINTS_COEFF = 0.765 
    
    def load_pbp_data(years):
        print(f"Loading PBP data for {years}...")
        try:
            df = nfl.load_pbp(seasons=years).to_pandas()
            return df
        except Exception as e:
            print(f"Error loading PBP data: {e}")
            return pd.DataFrame()
    
    def weighted_avg(values, weights):
        if len(values) == 0: return 0
        return np.average(values, weights=weights)
    
    def apply_game_cap(net_val, limit):
        sign = np.sign(net_val)
        abs_val = abs(net_val)
        if abs_val > limit:
            excess = abs_val - limit
            return sign * (limit + np.log(excess + 1) * 2.0)
        return net_val
    
    def calculate_hfa():
        # 1. Setup
        current_date = today
        start_date = current_date - timedelta(days=DAYS_WINDOW)
        years_to_load = [current_date.year, current_date.year - 1, current_date.year - 2]
        
        # 2. Load
        pbp = load_pbp_data(years_to_load)
        if pbp.empty: return pd.DataFrame()
        
        # 3. Process PBP
        print("Processing PBP data...")
        pbp['game_date'] = pd.to_datetime(pbp['game_date'])
        pbp = pbp[pbp['game_date'] >= start_date].copy()
        
        valid_types = ['run', 'pass', 'punt', 'field_goal', 'kickoff', 'extra_point']
        pbp = pbp[pbp['play_type'].isin(valid_types)]
        
        pbp = pbp.dropna(subset=['wp', 'epa', 'posteam', 'defteam', 'success', 'home_team'])
        pbp = pbp[(pbp['wp'] >= garbage_time_min) & (pbp['wp'] <= garbage_time_max)]
        pbp['capped_epa'] = pbp['epa'].clip(lower = luck_adj_min, upper = luck_adj_max)
        
        pbp['days_ago'] = (current_date - pbp['game_date']).dt.days
        pbp['weight'] = np.exp(-DECAY_RATE * pbp['days_ago'])
    
        # 4. Build Schedule & Metrics
        # We aggregate by game first
        game_stats = pbp.groupby(['game_id', 'posteam', 'defteam', 'home_team', 'game_date']).agg(
            total_epa_sum=('capped_epa', 'sum'),
            success_sum=('success', 'sum'),
            play_count=('play_id', 'count'),
            weight=('weight', 'mean')
        ).reset_index()
    
        df_games = []
        unique_games = game_stats['game_id'].unique()
        
        for g_id in unique_games:
            g_data = game_stats[game_stats['game_id'] == g_id]
            if len(g_data) != 2: continue 
            
            # Identify Team A and Team B
            team_a = g_data.iloc[0] 
            team_b = g_data.iloc[1] 
            
            # Raw Net Calculations
            raw_net_epa = team_a['total_epa_sum'] - team_b['total_epa_sum']
            net_epa_a = apply_game_cap(raw_net_epa, BLOWOUT_EPA_CAP)
            
            sr_a = team_a['success_sum'] / team_a['play_count'] if team_a['play_count'] > 0 else 0
            sr_b = team_b['success_sum'] / team_b['play_count'] if team_b['play_count'] > 0 else 0
            net_sr_a = sr_a - sr_b
            
            # Determine Home/Away status
            # Note: posteam is the team currently on offense in the row, but we grouped by posteam.
            # So team_a['posteam'] is the team name.
            is_a_home = (team_a['posteam'] == team_a['home_team'])
            
            df_games.append({
                'team': team_a['posteam'], 
                'opponent': team_b['posteam'], 
                'net_epa': net_epa_a, 
                'net_sr': net_sr_a,
                'weight': team_a['weight'],
                'is_home': is_a_home
            })
            df_games.append({
                'team': team_b['posteam'], 
                'opponent': team_a['posteam'], 
                'net_epa': -net_epa_a, 
                'net_sr': -net_sr_a,
                'weight': team_b['weight'],
                'is_home': not is_a_home
            })
            
        df_sched = pd.DataFrame(df_games)
        
        # ---------------------------------------------------------
        # 5. PHASE 1: Global SRS Solver
        # We need to know the TRUE global strength of every team 
        # to properly adjust for who they played at home vs away.
        # ---------------------------------------------------------
        print(f"Calculating Baseline Global Ratings ({ITERATIONS} iterations)...")
        teams = df_sched['team'].unique()
        
        global_epa = {t: 0.0 for t in teams}
        global_sr = {t: 0.0 for t in teams}
        
        # Standard SRS Loop
        for i in range(ITERATIONS):
            temp_epa = {}
            temp_sr = {}
            for team in teams:
                t_games = df_sched[df_sched['team'] == team]
                if t_games.empty: 
                    temp_epa[team] = 0; temp_sr[team] = 0
                    continue
                
                # EPA Logic
                avg_net_epa = weighted_avg(t_games['net_epa'], t_games['weight'])
                opp_epa_vals = [global_epa.get(opp, 0) for opp in t_games['opponent']]
                sos_epa = weighted_avg(opp_epa_vals, t_games['weight'])
                temp_epa[team] = avg_net_epa + sos_epa # Simple SRS
                
                # SR Logic
                avg_net_sr = weighted_avg(t_games['net_sr'], t_games['weight'])
                opp_sr_vals = [global_sr.get(opp, 0) for opp in t_games['opponent']]
                sos_sr = weighted_avg(opp_sr_vals, t_games['weight'])
                temp_sr[team] = avg_net_sr + sos_sr
                
            global_epa = temp_epa
            global_sr = temp_sr
    
        # ---------------------------------------------------------
        # 6. PHASE 2: Calculate HFA (Home vs Away Splits)
        # HFA = (Adjusted Home Rating - Adjusted Away Rating) / 2
        # ---------------------------------------------------------
        print("Calculating Home Field Advantage Splits...")
        hfa_stats = []
        
        for team in teams:
            t_games = df_sched[df_sched['team'] == team]
            
            home_games = t_games[t_games['is_home'] == True]
            away_games = t_games[t_games['is_home'] == False]
            
            # --- HOME RATING ---
            if not home_games.empty:
                # How did we perform at home?
                raw_home_epa = weighted_avg(home_games['net_epa'], home_games['weight'])
                raw_home_sr  = weighted_avg(home_games['net_sr'], home_games['weight'])
                
                # Who did we play at home? (Use Global Ratings for Opponents)
                home_opp_epa = weighted_avg([global_epa.get(o,0) for o in home_games['opponent']], home_games['weight'])
                home_opp_sr  = weighted_avg([global_sr.get(o,0) for o in home_games['opponent']], home_games['weight'])
                
                # Adjusted Home Rating = Raw Performance + Opponent Strength
                adj_home_epa = raw_home_epa + home_opp_epa
                adj_home_sr  = raw_home_sr + home_opp_sr
            else:
                adj_home_epa = global_epa.get(team, 0)
                adj_home_sr = global_sr.get(team, 0)
    
            # --- AWAY RATING ---
            if not away_games.empty:
                raw_away_epa = weighted_avg(away_games['net_epa'], away_games['weight'])
                raw_away_sr  = weighted_avg(away_games['net_sr'], away_games['weight'])
                
                away_opp_epa = weighted_avg([global_epa.get(o,0) for o in away_games['opponent']], away_games['weight'])
                away_opp_sr  = weighted_avg([global_sr.get(o,0) for o in away_games['opponent']], away_games['weight'])
                
                adj_away_epa = raw_away_epa + away_opp_epa
                adj_away_sr  = raw_away_sr + away_opp_sr
            else:
                adj_away_epa = global_epa.get(team, 0)
                adj_away_sr = global_sr.get(team, 0)
                
            # --- HFA CALCULATION ---
            # HFA EPA
            hfa_val_epa = (adj_home_epa - adj_away_epa) / 2
            
            # HFA SR (Convert to Points)
            hfa_val_sr_raw = (adj_home_sr - adj_away_sr) / 2
            hfa_val_sr_pts = (hfa_val_sr_raw * 100) * SR_TO_POINTS_COEFF
            
            # Blended HFA (Points)
            # Note: EPA is already in points, SR is converted to points above
            blended_hfa = (hfa_val_epa * WEIGHT_EPA) + (hfa_val_sr_pts * WEIGHT_SR)
            
            # Optional: Regress extreme outliers slightly?
            # HFA usually falls between -3 and +6. We can leave it raw for pure data analysis.
            
            hfa_stats.append({
                'Team': team,
                'HFA (Points)': blended_hfa,
                'Home Rating (Adj)': (adj_home_epa*WEIGHT_EPA) + (adj_home_sr*100*SR_TO_POINTS_COEFF*WEIGHT_SR),
                'Away Rating (Adj)': (adj_away_epa*WEIGHT_EPA) + (adj_away_sr*100*SR_TO_POINTS_COEFF*WEIGHT_SR),
                'Global Rating': (global_epa[team]*WEIGHT_EPA) + (global_sr[team]*100*SR_TO_POINTS_COEFF*WEIGHT_SR),
                'HFA EPA': hfa_val_epa,
                'HFA SR': hfa_val_sr_raw
            })
    
        df = pd.DataFrame(hfa_stats)
        
        # 7. Final Formatting
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(3)
        
        df = df.sort_values(by='HFA (Points)', ascending=False).reset_index(drop=True)
        return df
    
    if __name__ == "__main__":
        df_hfa = calculate_hfa()
        
        if not df_hfa.empty:
            print("\n--- HOME FIELD ADVANTAGE RANKINGS ---")
            print(df_hfa[['Team', 'HFA (Points)', 'Home Rating (Adj)', 'Away Rating (Adj)']].head(10).to_string(index=False))
            
            filename = "nfl-power-ratings/nfl_hfa_ratings.csv"
            df_hfa.to_csv(filename, index=False)
            print(f"\nSaved to {filename}")
    
    # --- Configuration ---
    DAYS_WINDOW = 3650  # 5 Years of data
    DECAY_RATE = 0.0025   
    ITERATIONS = 50
    
    # WEIGHTS
    WEIGHT_EPA = 0.42
    WEIGHT_SR = 0.58
    
    # OPTIMIZATION
    BLOWOUT_EPA_CAP = 22.0
    SR_TO_POINTS_COEFF = 0.765 
    
    # MAP: Host Team -> Time Zone
    TZ_MAP = {
        'ARI': 'MST', 'ATL': 'EST', 'BAL': 'EST', 'BUF': 'EST', 'CAR': 'EST', 
        'CHI': 'CST', 'CIN': 'EST', 'CLE': 'EST', 'DAL': 'CST', 'DEN': 'MST', 
        'DET': 'EST', 'GB':  'CST', 'HOU': 'CST', 'IND': 'EST', 'JAX': 'EST', 
        'KC':  'CST', 'LA':  'PST', 'LAC': 'PST', 'LV':  'PST', 'MIA': 'EST', 
        'MIN': 'CST', 'NE':  'EST', 'NO':  'CST', 'NYG': 'EST', 'NYJ': 'EST', 
        'PHI': 'EST', 'PIT': 'EST', 'SEA': 'PST', 'SF':  'PST', 'TB':  'EST', 
        'TEN': 'CST', 'WAS': 'EST'
    }
    
    def load_pbp_data(years):
        print(f"Loading PBP data for {years}...")
        try:
            df = nfl.load_pbp(seasons=years).to_pandas()
            return df
        except Exception as e:
            print(f"Error loading PBP data: {e}")
            return pd.DataFrame()
    
    def weighted_avg(values, weights):
        if len(values) == 0: return 0
        return np.average(values, weights=weights)
    
    def apply_game_cap(net_val, limit):
        sign = np.sign(net_val)
        abs_val = abs(net_val)
        if abs_val > limit:
            excess = abs_val - limit
            return sign * (limit + np.log(excess + 1) * 2.0)
        return net_val
    
    def calculate_tz_splits():
        # 1. Setup
        current_date = today
        start_date = current_date - timedelta(days=DAYS_WINDOW)
        years_to_load = range(start_date.year, current_date.year + 1)
        
        # 2. Load
        pbp = load_pbp_data(list(years_to_load))
        if pbp.empty: return pd.DataFrame()
        
        # 3. Process
        print("Processing PBP data...")
        pbp['game_date'] = pd.to_datetime(pbp['game_date'])
        pbp = pbp[pbp['game_date'] >= start_date].copy()
        
        valid_types = ['run', 'pass', 'punt', 'field_goal', 'kickoff', 'extra_point']
        pbp = pbp[pbp['play_type'].isin(valid_types)]
        pbp = pbp.dropna(subset=['epa', 'posteam', 'defteam', 'home_team'])
        
        # Weights
        pbp['days_ago'] = (current_date - pbp['game_date']).dt.days
        pbp['weight'] = np.exp(-DECAY_RATE * pbp['days_ago'])
    
        # 4. Game Aggregation
        game_stats = pbp.groupby(['game_id', 'posteam', 'defteam', 'home_team', 'game_date']).agg(
            total_epa_sum=('epa', 'sum'),
            success_sum=('success', 'sum'),
            play_count=('play_id', 'count'),
            weight=('weight', 'mean')
        ).reset_index()
    
        df_games = []
        unique_games = game_stats['game_id'].unique()
        
        for g_id in unique_games:
            g_data = game_stats[game_stats['game_id'] == g_id]
            if len(g_data) != 2: continue 
            
            team_a = g_data.iloc[0]
            team_b = g_data.iloc[1]
            
            # Determine Home/Away
            if team_a['posteam'] == team_a['home_team']:
                home, away = team_a, team_b
            else:
                home, away = team_b, team_a
                
            # Calc Stats from AWAY perspective
            raw_net_epa = away['total_epa_sum'] - home['total_epa_sum']
            net_epa_away = apply_game_cap(raw_net_epa, BLOWOUT_EPA_CAP)
            
            sr_away = away['success_sum'] / away['play_count'] if away['play_count'] > 0 else 0
            sr_home = home['success_sum'] / home['play_count'] if home['play_count'] > 0 else 0
            net_sr_away = sr_away - sr_home
            
            # Add to schedule 
            df_games.append({
                'team': away['posteam'], 'opponent': home['posteam'], 
                'net_epa': net_epa_away, 'net_sr': net_sr_away, 
                'weight': away['weight'], 'is_away': True, 
                'host_tz': TZ_MAP.get(home['posteam'], 'EST')
            })
            df_games.append({
                'team': home['posteam'], 'opponent': away['posteam'], 
                'net_epa': -net_epa_away, 'net_sr': -net_sr_away, 
                'weight': home['weight'], 'is_away': False,
                'host_tz': TZ_MAP.get(home['posteam'], 'EST')
            })
    
        df_sched = pd.DataFrame(df_games)
    
        # 5. Global Solver
        print("Calculating Global Opponent Adjustments...")
        teams = df_sched['team'].unique()
        global_epa = {t: 0.0 for t in teams}
        global_sr = {t: 0.0 for t in teams}
        
        for i in range(ITERATIONS):
            temp_epa = {}
            temp_sr = {}
            for team in teams:
                t_games = df_sched[df_sched['team'] == team]
                if t_games.empty: 
                    temp_epa[team] = 0; temp_sr[team] = 0; continue
                
                avg_net_epa = weighted_avg(t_games['net_epa'], t_games['weight'])
                sos_epa = weighted_avg([global_epa.get(o,0) for o in t_games['opponent']], t_games['weight'])
                temp_epa[team] = avg_net_epa + sos_epa
                
                avg_net_sr = weighted_avg(t_games['net_sr'], t_games['weight'])
                sos_sr = weighted_avg([global_sr.get(o,0) for o in t_games['opponent']], t_games['weight'])
                temp_sr[team] = avg_net_sr + sos_sr
                
            global_epa = temp_epa
            global_sr = temp_sr
    
        # 6. Time Zone Analysis (Away Games Only)
        print("Analyzing Time Zone Splits...")
        df_away = df_sched[df_sched['is_away'] == True].copy()
        
        final_stats = []
        
        for team in teams:
            team_games = df_away[df_away['team'] == team]
            if team_games.empty: continue
            
            # A. General Away Baseline
            adj_perfs = []
            weights = []
            
            for _, row in team_games.iterrows():
                opp = row['opponent']
                opp_val = (global_epa[opp] * WEIGHT_EPA) + (global_sr[opp] * 100 * SR_TO_POINTS_COEFF * WEIGHT_SR)
                game_val = (row['net_epa'] * WEIGHT_EPA) + (row['net_sr'] * 100 * SR_TO_POINTS_COEFF * WEIGHT_SR)
                adj_perfs.append(game_val + opp_val)
                weights.append(row['weight'])
                
            baseline_rating = weighted_avg(adj_perfs, weights)
            
            # B. Specific TZ Baselines
            zones = ['EST', 'CST', 'MST', 'PST']
            zone_diffs = {}
            
            for zone in zones:
                z_games = team_games[team_games['host_tz'] == zone]
                
                if len(z_games) < 2: 
                    zone_diffs[zone] = np.nan # Use NaN so it doesn't skew averages with 0.0
                    continue
                    
                z_perfs = []
                z_weights = []
                
                for _, row in z_games.iterrows():
                    opp = row['opponent']
                    opp_val = (global_epa[opp] * WEIGHT_EPA) + (global_sr[opp] * 100 * SR_TO_POINTS_COEFF * WEIGHT_SR)
                    game_val = (row['net_epa'] * WEIGHT_EPA) + (row['net_sr'] * 100 * SR_TO_POINTS_COEFF * WEIGHT_SR)
                    z_perfs.append(game_val + opp_val)
                    z_weights.append(row['weight'])
                
                zone_rating = weighted_avg(z_perfs, z_weights)
                zone_diffs[zone] = zone_rating - baseline_rating
                
            final_stats.append({
                'Team': team,
                'Origin_TZ': TZ_MAP.get(team, 'Unknown'),
                'General Away Rating': baseline_rating,
                'EST Diff': zone_diffs.get('EST', 0),
                'CST Diff': zone_diffs.get('CST', 0),
                'MST Diff': zone_diffs.get('MST', 0),
                'PST Diff': zone_diffs.get('PST', 0)
            })
    
        df = pd.DataFrame(final_stats)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(2)
        df = df.sort_values('General Away Rating', ascending=False).reset_index(drop=True)
        return df
    
    def print_regional_averages(df):
        """Calculates and prints averages by Origin Region."""
        # Define Regions
        # EST = East Coast, CST = Central, MST = Mountain, PST = West Coast
        
        print("\n" + "="*80)
        print("REGIONAL TRAVEL REPORT: Average Advantage/Disadvantage (Points)")
        print("="*80)
        print("Reading the matrix: 'How do [ROW] teams perform when traveling to [COLUMN]?'")
        print("Positive = They play BETTER than their normal away baseline.")
        print("Negative = They play WORSE than their normal away baseline.")
        print("-" * 80)
    
        # Group by Origin Time Zone
        regions = ['EST', 'CST', 'MST', 'PST']
        summary_data = []
    
        for region in regions:
            region_df = df[df['Origin_TZ'] == region]
            if region_df.empty: continue
            
            # Calculate mean of the diffs (ignoring NaNs automatically)
            avg_est = region_df['EST Diff'].mean()
            avg_cst = region_df['CST Diff'].mean()
            avg_mst = region_df['MST Diff'].mean()
            avg_pst = region_df['PST Diff'].mean()
            
            summary_data.append({
                'Origin Region': region,
                'Teams': len(region_df),
                'at EST': avg_est,
                'at CST': avg_cst,
                'at MST': avg_mst,
                'at PST': avg_pst
            })
    
        summary_df = pd.DataFrame(summary_data)
        
        # Formatting for print
        numeric_cols = ['at EST', 'at CST', 'at MST', 'at PST']
        summary_df[numeric_cols] = summary_df[numeric_cols].round(2)
        
        # Rename for clarity
        display_map = {'EST': 'East Coast (EST)', 'CST': 'Central (CST)', 'MST': 'Mountain (MST)', 'PST': 'West Coast (PST)'}
        summary_df['Origin Region'] = summary_df['Origin Region'].map(display_map)
        
        print(summary_df.to_string(index=False))
        print("-" * 80)
        print("* Note: 'at MST' sample sizes are often small (only DEN/ARI host games).")
    
    if __name__ == "__main__":
        df_splits = calculate_tz_splits()
        
        if not df_splits.empty:
            # 1. Print Regional Summary
            print_regional_averages(df_splits)
    
            # 2. Print Detailed Team Table
            print("\n--- DETAILED TEAM SPLITS ---")
            print(df_splits.to_string(index=False))
            
            # 3. Save
            fn = "nfl-power-ratings/nfl_timezone_splits.csv"
            df_splits.to_csv(fn, index=False)
            print(f"\nSaved detailed data to {fn}")
if __name__ == "__main__":
    week_starting_dates = [
        "09/03/2025", "09/10/2025", "09/17/2025", "09/24/2025", 
        "10/01/2025", "10/08/2025", "10/15/2025", "10/22/2025", 
        "10/29/2025", "11/05/2025", "11/12/2025", "11/19/2025", 
        "11/26/2025", "11/29/2025", "12/03/2025", "12/10/2025", 
        "12/17/2025", "12/24/2025", "12/26/2025", "12/31/2025"
    ]

    for date in week_starting_dates:
        loop_through_rankings(date)
