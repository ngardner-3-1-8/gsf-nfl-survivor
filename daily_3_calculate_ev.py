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
from season_dates import resolve_week_context
    
def loop_through_ev(date_str):   
    # 1. Get current date
    # --- Season / week context (shared with daily_2 via season_dates.py) ---
    # Replaces the old inline target_year + hardcoded per-year Christmas
    # if/elif; resolve_week_context reproduces daily_2's numbers exactly.
    ctx = resolve_week_context(date_str)
    today = ctx.today
    current_cal_year = ctx.current_cal_year
    target_year = ctx.target_year
    schedule_df = ctx.schedule_df
    first_game_date = ctx.first_game_date
    thanksgiving_date = ctx.thanksgiving_date
    black_friday = ctx.black_friday
    christmas_day = ctx.christmas_day
    boxing_day = ctx.boxing_day
    thanksgiving_week = ctx.thanksgiving_week
    christmas_week = ctx.christmas_week
    starting_week = ctx.starting_week
    upcoming_week = ctx.upcoming_week
    
    
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

        # NOTE: team names in this pipeline are full names ("Los Angeles
        # Chargers"), not abbreviations, so a {'JAC': 'JAX', 'LAR': 'LA'}
        # replace on 'Away Team'/'Home Team' is a silent no-op here — it
        # never matches anything and was removed. If an upstream data
        # source ever goes back to abbreviated codes, re-add a mapping
        # that matches whatever abbreviations that source actually uses
        # (and watch for historically relocated/renamed franchises —
        # Raiders OAK/LV, Chargers SD/LAC, Rams STL/LA, Commanders'
        # several names — if any historical training data merges on team
        # name/abbreviation across seasons).

        # Find ending week bounds based on the full file
        end_w = int(df['Week_x'].max()) + 1

        # Expected total pick-% mass across a full slate in a given week.
        # Every remaining entry picks exactly one team, so Home Pick % +
        # Away Pick % across ALL of that week's games should sum to ~1.0
        # (bump this to 2.0/3.0 if daily_2's target_pick_sum is ever set
        # higher for a multi-pick week). If the actual mass comes in well
        # below this, the pick% data for that week is incomplete — e.g.
        # the Week 2 Splash bug, where public pick % had only posted for
        # a handful of games and the rest were still NaN. Dividing by an
        # incomplete mass is exactly what inflated EV past 3.0, so treat
        # low coverage as missing data rather than compute from it.
        EXPECTED_PICK_MASS = 1.0
        MIN_PICK_MASS_COVERAGE = 0.85  # fraction of EXPECTED_PICK_MASS required to trust the week

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

        # Max games we'll exact-enumerate (2**G scenarios). An NFL week is at
        # most 16 games, so 16 -> 65,536 scenarios is the real ceiling; the
        # cap is just a guardrail so a malformed week can't blow up memory.
        MAX_ENUM_GAMES = 20

        def calculate_all_scenarios(week_df, week_num, away_prob_col, home_prob_col,
                                    home_pick_col='Home Pick %',
                                    away_pick_col='Away Pick %'):
            """
            TRUE EV via exact enumeration of every possible weekly outcome.

            EV(team T) = E[ 1{T wins} / S ]
                       = sum over all 2**G win/loss scenarios of
                         P(scenario) * 1{T wins in that scenario} / S(scenario)

            where G is the number of games that week (<=16, so <=65,536
            scenarios), P(scenario) is the product of each game's per-team
            win probability (games treated as independent, same assumption
            daily_2's Monte Carlo makes), and S(scenario) is the surviving
            pick-mass in that scenario -- the sum of pick% over the teams
            that won. An entry survives iff its picked team won, and its
            equity that week is inversely proportional to how much of the
            field survives alongside it, hence 1/S.

            This REPLACES the old ratio-of-expectations approximation
            EV = P(win) / E[survivors]. That form is a first-order estimate;
            it ignores that 1/S is convex and, more importantly, that when a
            heavily-picked favorite wins, S is LARGER (its own mass survives),
            so 1{T wins} and 1/S are negatively correlated. Enumerating every
            scenario prices both effects in exactly, which is why a full
            65k-scenario pass is more accurate than the 5k-sample Monte Carlo.

            Same coverage guard as before: if pick% data doesn't cover
            (essentially) the whole slate, bail to all-zero EV rather than
            compute from a partial distribution. home_pick_col/away_pick_col
            let the same routine run on Circa pick% or Splash pick%.
            """
            home_probs = week_df[home_prob_col].to_numpy(dtype=float)
            away_probs = week_df[away_prob_col].to_numpy(dtype=float)
            home_picks = week_df[home_pick_col].fillna(0).to_numpy(dtype=float)
            away_picks = week_df[away_pick_col].fillna(0).to_numpy(dtype=float)
            n = len(week_df)

            total_pick_mass = home_picks.sum() + away_picks.sum()
            if total_pick_mass < EXPECTED_PICK_MASS * MIN_PICK_MASS_COVERAGE:
                print(f"⚠️ Week {week_num}: {home_pick_col}/{away_pick_col} pick% "
                      f"coverage is only {total_pick_mass:.1%} of expected — "
                      f"treating as missing data and zeroing EV for this week "
                      f"(check upstream pick% generation).")
                zero = np.zeros(n)
                return (dict(zip(week_df['Home Team'], zero)),
                        dict(zip(week_df['Away Team'], zero)))

            # Clean per-game home-win probability. Fair-odds columns are meant
            # to be de-vigged (home + away ~ 1), so fill a missing side from
            # its complement; fall back to 0.5 only if both sides are missing.
            hp = home_probs.copy()
            ap = away_probs.copy()
            hp = np.where(np.isnan(hp) & ~np.isnan(ap), 1.0 - ap, hp)
            hp = np.where(np.isnan(hp), 0.5, hp)
            hp = np.clip(hp, 0.0, 1.0)

            home_ev = np.zeros(n)
            away_ev = np.zeros(n)

            # A game only affects survival if at least one side is pickable
            # (pick% > 0). Games with no pick mass on either side can't change
            # S no matter who wins, so drop them from the enumeration -- that
            # keeps the scenario count at 2**(relevant games) instead of 2**16
            # when byes/eliminations have thinned the pickable slate.
            relevant = (home_picks > 0) | (away_picks > 0)
            idx = np.nonzero(relevant)[0]
            G = int(idx.size)
            if G == 0:
                return (dict(zip(week_df['Home Team'], home_ev)),
                        dict(zip(week_df['Away Team'], away_ev)))
            if G > MAX_ENUM_GAMES:
                # Should never happen for real NFL weeks; guardrail only.
                print(f"⚠️ Week {week_num}: {G} pickable games exceeds the "
                      f"{MAX_ENUM_GAMES}-game enumeration cap — skipping true-EV "
                      f"for this week/column.")
                return (dict(zip(week_df['Home Team'], home_ev)),
                        dict(zip(week_df['Away Team'], away_ev)))

            p = hp[idx]              # home-win prob per relevant game  (G,)
            hm = home_picks[idx]     # home pick mass                   (G,)
            am = away_picks[idx]     # away pick mass                   (G,)

            # bits[s, g] == 1  <=>  the HOME team wins game g in scenario s.
            n_scen = 1 << G
            scen = np.arange(n_scen, dtype=np.int64)[:, None]
            gbit = np.arange(G, dtype=np.int64)[None, :]
            bits = ((scen >> gbit) & 1).astype(np.float64)          # (n_scen, G)

            # P(scenario) = product over games of (p if home wins else 1-p).
            prob_per_game = bits * p + (1.0 - bits) * (1.0 - p)     # (n_scen, G)
            p_scen = prob_per_game.prod(axis=1)                     # (n_scen,)

            # S(scenario) = surviving pick mass = winners' pick% summed.
            s_mass = bits @ hm + (1.0 - bits) @ am                  # (n_scen,)
            with np.errstate(divide='ignore', invalid='ignore'):
                weight = np.where(s_mass > 0, p_scen / s_mass, 0.0)  # (n_scen,)

            # EV per side = sum of scenario weights over scenarios where that
            # side won.  home wins => bit==1, away wins => bit==0.
            home_ev_g = weight @ bits                               # (G,)
            away_ev_g = weight @ (1.0 - bits)                       # (G,)

            # A team you can't pick (pick% == 0) has no EV even though its game
            # was enumerated (its outcome still moved S for the other side).
            home_ev[idx] = np.where(hm > 0, home_ev_g, 0.0)
            away_ev[idx] = np.where(am > 0, away_ev_g, 0.0)

            return (dict(zip(week_df['Home Team'], home_ev)),
                    dict(zip(week_df['Away Team'], away_ev)))

        def compute_and_write_ev(home_prob_col, away_prob_col, home_ev_col, away_ev_col,
                                  home_pick_col, away_pick_col, desc):
            df[home_ev_col] = 0.0
            df[away_ev_col] = 0.0

            for week in tqdm(range(start_w, end_w), desc=desc, leave=False):
                week_mask = df['Week_x'] == week
                week_df = df.loc[week_mask]

                if week_df.empty:
                    continue

                home_ev_map, away_ev_map = calculate_all_scenarios(
                    week_df, week,
                    away_prob_col=away_prob_col,
                    home_prob_col=home_prob_col,
                    home_pick_col=home_pick_col,
                    away_pick_col=away_pick_col,
                )

                # Vectorized write-back: one map() per side instead of a
                # per-team boolean-mask scan over the whole df (this was
                # O(teams) redundant full-column scans per week per
                # scenario before).
                df.loc[week_mask, home_ev_col] = df.loc[week_mask, 'Home Team'].map(home_ev_map)
                df.loc[week_mask, away_ev_col] = df.loc[week_mask, 'Away Team'].map(away_ev_map)

        for scenario_name, scenario_config in probability_scenarios.items():
            away_prob_col = scenario_config["away_col"]
            home_prob_col = scenario_config["home_col"]
            prefix = scenario_config["prefix"]

            compute_and_write_ev(
                home_prob_col, away_prob_col,
                home_ev_col=f"{prefix}_Home_EV",
                away_ev_col=f"{prefix}_Away_EV",
                home_pick_col='Home Pick %',
                away_pick_col='Away Pick %',
                desc=f"Processing {prefix.upper()} EV",
            )

        # ── 🌊 SPLASH EV — same 5 scenarios, using the Splash pick% columns ──
        # ── 🌊 SHARED SPLASH EV — same 5 scenarios, using the shared Splash
        #    pick% columns (the public-feed proxy daily_2 writes for the
        #    upcoming week). This still serves any Splash sub-contest that does
        #    NOT have its own dedicated projection (e.g. 4-for-4, High Roller,
        #    …). The two first-class contests (Big Splash, World Championship)
        #    additionally get their own per-contest EV columns below.
        if 'Home Splash Pick %' in df.columns and 'Away Splash Pick %' in df.columns:
            print("Computing shared Splash EV from Splash pick% columns...")
            for scenario_name, scenario_config in probability_scenarios.items():
                away_prob_col = scenario_config["away_col"]
                home_prob_col = scenario_config["home_col"]
                prefix = scenario_config["prefix"]

                compute_and_write_ev(
                    home_prob_col, away_prob_col,
                    home_ev_col=f"Splash_{prefix}_Home_EV",
                    away_ev_col=f"Splash_{prefix}_Away_EV",
                    home_pick_col='Home Splash Pick %',
                    away_pick_col='Away Splash Pick %',
                    desc=f"Processing SPLASH {prefix.upper()} EV",
                )
        else:
            print("(No shared Splash pick% columns — skipping shared Splash EV)")

        # ── 🌊 PER-CONTEST SPLASH EV — Big Splash + Survivor World Championship
        #    each have their OWN projected pick% columns (daily_2 merges
        #    'Home/Away {Contest} Pick %' from that contest's model into this
        #    file). Run the exact same true-EV enumeration on each contest's
        #    own pick% so its EV reflects its own crowd, and write clearly
        #    namespaced EV columns ({Tag}_{prefix}_Home/Away_EV, e.g.
        #    BigSplash_sportsbook_Home_EV) that never collide with Circa's or
        #    the shared Splash EV. The API points each contest's optimizer at
        #    its own columns; sub-contests without a projection keep using the
        #    shared Splash EV above.
        from contest_config import CONTESTS as _ALL_CONTESTS
        for _ck in ('big_splash', 'world_championship'):
            _prefix_name = _ALL_CONTESTS[_ck]['proj_prefix']      # e.g. 'Big Splash'
            _tag = _prefix_name.replace(' ', '')                  # e.g. 'BigSplash'
            _home_pick = f'Home {_prefix_name} Pick %'
            _away_pick = f'Away {_prefix_name} Pick %'
            if _home_pick in df.columns and _away_pick in df.columns:
                print(f"Computing {_prefix_name} EV from its projected pick% columns...")
                for scenario_name, scenario_config in probability_scenarios.items():
                    away_prob_col = scenario_config["away_col"]
                    home_prob_col = scenario_config["home_col"]
                    prefix = scenario_config["prefix"]

                    compute_and_write_ev(
                        home_prob_col, away_prob_col,
                        home_ev_col=f"{_tag}_{prefix}_Home_EV",
                        away_ev_col=f"{_tag}_{prefix}_Away_EV",
                        home_pick_col=_home_pick,
                        away_pick_col=_away_pick,
                        desc=f"Processing {_tag.upper()} {prefix.upper()} EV",
                    )
            else:
                print(f"(No {_prefix_name} pick% columns — skipping {_prefix_name} EV)")

        # Save the updated main dataframe overwriting the original input file
        df.to_csv(main_file_path, index=False)
        print(f"\nSuccessfully appended all EV columns and saved to: {main_file_path}")

    calculate_ev(df, config={})

if __name__ == "__main__":
    formatted_date = datetime.now().strftime("%m/%d/%Y")
    week_starting_dates = [
#        "09/08/2026", #Leading up to Week 1
        "09/15/2026",
        
#        "09/03/2025", #Leading up to Week 1
#        "09/10/2025", #Leading up to Week 2
#        "09/17/2025", #Leading up to Week 3
#        "09/24/2025", #Leading up to Week 4
#        "10/01/2025", #Leading up to Week 5
#        "10/08/2025", #Leading up to Week 6
#        "10/15/2025", #Leading up to Week 7
#        "10/22/2025", #Leading up to Week 8
#        "10/29/2025", #Leading up to Week 9
#        "11/05/2025", #Leading up to Week 10
#        "11/12/2025", #Leading up to Week 11
#        "11/19/2025", #Leading up to Week 12
#        "11/26/2025", #Leading up to Week 13
#        "11/29/2025", #Leading up to Week 14
#        "12/03/2025", #Leading up to Week 15
#        "12/10/2025", #Leading up to Week 16
#        "12/17/2025", #Leading up to Week 17
#        "12/24/2025", #Leading up to Week 18
#        "12/26/2025", #Leading up to Week 19
#        "12/31/2025", #Leading up to Week 20
        
#        "09/04/2024", #Leading up to Week 1
#        "09/11/2024", #Leading up to Week 2
#        "09/18/2024", #Leading up to Week 3
#        "09/25/2024", #Leading up to Week 4
#        "10/02/2024", #Leading up to Week 5
#        "10/09/2024", #Leading up to Week 6
#        "10/16/2024", #Leading up to Week 7
#        "10/23/2024", #Leading up to Week 8
#        "10/30/2024", #Leading up to Week 9
#        "11/06/2024", #Leading up to Week 10
#        "11/13/2024", #Leading up to Week 11
#        "11/20/2024", #Leading up to Week 12
#        "11/27/2024", #Leading up to Week 13
#        "11/30/2024", #Leading up to Week 14
#        "12/11/2024", #Leading up to Week 16
#        "12/18/2024", #Leading up to Week 17
#        "12/24/2024", #Leading up to Week 18
#        "12/27/2024", #Leading up to Week 19
#        "01/01/2025", #Leading up to Week 20
        
#        "09/06/2023", #Leading up to Week 1
#        "09/13/2023", #Leading up to Week 2
#        "09/20/2023", #Leading up to Week 3
#        "09/27/2023", #Leading up to Week 4
#        "10/04/2023", #Leading up to Week 5
#        "10/11/2023", #Leading up to Week 6
#        "10/18/2023", #Leading up to Week 7
#        "10/25/2023", #Leading up to Week 8
#        "11/01/2023", #Leading up to Week 9
#        "11/08/2023", #Leading up to Week 10
#        "11/15/2023", #Leading up to Week 11
#        "11/22/2023", #Leading up to Week 12
#        "11/25/2023", #Leading up to Week 13
#        "11/29/2023", #Leading up to Week 14
#        "12/06/2023", #Leading up to Week 15
#        "12/13/2023", #Leading up to Week 16
#        "12/20/2023", #Leading up to Week 17
#        "12/25/2023",  #Leading up to Week 18
#        "12/27/2023", #Leading up to Week 19
#        "01/03/2024", #Leading up to Week 20
        
#        "09/07/2022", #Leading up to Week 1
#        "09/14/2022", #Leading up to Week 2
#        "09/21/2022", #Leading up to Week 3
#        "09/28/2022", #Leading up to Week 4
#        "10/05/2022", #Leading up to Week 5
#        "10/12/2022", #Leading up to Week 6
#        "10/19/2022", #Leading up to Week 7
#        "10/26/2022", #Leading up to Week 8
#        "11/02/2022", #Leading up to Week 9
#        "11/09/2022", #Leading up to Week 10
#        "11/16/2022", #Leading up to Week 11
#        "11/23/2022", #Leading up to Week 12
#        "11/26/2022", #Leading up to Week 13
#        "11/30/2022", #Leading up to Week 14
#        "12/07/2022", #Leading up to Week 15
#        "12/14/2022", #Leading up to Week 16
#        "12/21/2022", #Leading up to Week 17
#        "12/25/2022", #Leading up to Week 18
#        "12/28/2022", #Leading up to Week 19
#        "01/04/2023", #Leading up to Week 20
        
#        "09/08/2021", #Leading up to Week 1
#        "09/15/2021", #Leading up to Week 2
#        "09/22/2021", #Leading up to Week 3
#        "09/29/2021", #Leading up to Week 4
#        "10/06/2021", #Leading up to Week 5
#        "10/13/2021", #Leading up to Week 6
#        "10/20/2021", #Leading up to Week 7
#        "10/27/2021", #Leading up to Week 8
#        "11/03/2021", #Leading up to Week 9
#        "11/10/2021", #Leading up to Week 10
#        "11/17/2021", #Leading up to Week 11
#        "11/24/2021", #Leading up to Week 12
#        "11/27/2021", #Leading up to Week 13
#        "12/01/2021", #Leading up to Week 14
#        "12/08/2021", #Leading up to Week 15
#        "12/15/2021", #Leading up to Week 16
#        "12/22/2021", #Leading up to Week 17
#        "12/26/2021", #Leading up to Week 18
#        "12/29/2021", #Leading up to Week 19
#        "01/05/2022", #Leading up to Week 20
        
#        "09/09/2020", #Leading up to Week 1
#        "09/16/2020", #Leading up to Week 2
#        "09/23/2020", #Leading up to Week 3
#        "09/30/2020", #Leading up to Week 4
#        "10/07/2020", #Leading up to Week 5
#        "10/14/2020", #Leading up to Week 6
#        "10/21/2020", #Leading up to Week 7
#        "10/28/2020", #Leading up to Week 8
#        "11/04/2020", #Leading up to Week 9
#        "11/11/2020", #Leading up to Week 10
#        "11/18/2020", #Leading up to Week 11
#        "11/25/2020", #Leading up to Week 12
#        "11/28/2020", #Leading up to Week 13
#        "12/02/2020", #Leading up to Week 14
#        "12/09/2020", #Leading up to Week 15
#        "12/16/2020", #Leading up to Week 16
#        "12/23/2020", #Leading up to Week 17
#        "12/30/2020", #Leading up to Week 18
        
#        formatted_date
    ]

    for date in week_starting_dates:
        loop_through_ev(date)
