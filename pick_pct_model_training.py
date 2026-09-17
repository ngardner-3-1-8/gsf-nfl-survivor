"""
pick_pct_model_training.py

Rewritten training logic for the Circa / Splash pick-percentage models used
by daily_2_consolidate_predict_weekly_data.py (function get_predicted_pick_percentages).

WHAT CHANGED AND WHY
=====================================================================
1. Holiday signal, computed ONE way, used in both training and prediction.
   The original script computed a team's win% on its own upcoming
   Thanksgiving/Christmas game only inside the live prediction block
   (lines ~5502-5518 of the original file), where it OVERWROTE the
   'Pre Thanksgiving' / 'Pre Christmas' columns in place -- turning them
   from the plain 0/1 flags the model was trained on into a continuous
   win%-weighted value at prediction time only. That mismatch (model
   trained on {0,1}, fed something like 0.03 or 0.41 at inference) is very
   likely the direct reason "I've had a hard time getting that picked up by
   the model": the feature the model learned during training does not
   exist in that form at prediction time.

   compute_holiday_lookahead_features() below fixes this by computing the
   signal into NEW, separate columns (never overwriting the original 0/1
   flags), and it must be called on both the training frame and the live
   prediction frame so the feature means the same thing in both places.

2. The temporal leakage guard was disabled (`df_historical = df` in the
   original, instead of `df[valid_history_mask]`). Restored below.

3. Target switched from absolute Pick % to Pick % relative to that week's
   mean -- matching your point that a pick is a relative choice among that
   week's options, not an absolute quantity. This is still non-negative,
   so it plugs into the existing downstream proportional-renormalization /
   availability water-filling code with zero changes needed there.

4. Feature selection switched from RandomForest impurity importance (fit
   once on all the training data, no holdout) to permutation importance
   measured on a held-out, chronologically-later slice of weeks. Impurity
   importance splits credit across correlated feature families (this
   dataset has ~6 repeated mean/max/min/std blocks) and is biased toward
   dense continuous columns -- it will reliably bury a feature like "this
   team's holiday win%" that's nonzero for maybe 2 weeks a season, even
   when it has a real effect on those weeks. The new holiday features are
   also forced into `mandatory_features` so they're never dropped by
   ranking at all.

5. A real validation split + MAE + rank-correlation, logged to CSV.
   `train_test_split` and `mean_absolute_error` were imported in the
   original script but never called anywhere -- there was no way to know
   whether any change helped versus hurt except watching live results.
   Now, every training run appends a row (timestamp, model, MAE in
   percentage points, and mean weekly Spearman rank-correlation -- i.e.
   "did we get the ordering of who's more/less popular right", which is
   what actually feeds EV) to a CSV so you can track this over time.

HOW TO WIRE THIS INTO daily_2_consolidate_predict_weekly_data.py
=====================================================================
A. Save this file next to your scripts and add near the top of
   daily_2_consolidate_predict_weekly_data.py:

       from pick_pct_model_training import (
           compute_holiday_lookahead_features,
           train_pick_pct_models,
       )

   (Or paste these functions directly above `get_predicted_pick_percentages`
   if you'd rather not add a new module -- either works.)

B. Inside get_predicted_pick_percentages(), REPLACE the block that currently
   runs from:
       df = pd.read_csv('contest-historical-data/Circa_historical_data.csv')
       df.rename(columns={"Week": "Date"}, inplace=True)
       ...
   down through the end of the splash-model try/except (originally ending
   around the "End of Training Block" comment) with:

       df = pd.read_csv('contest-historical-data/Circa_historical_data.csv')
       df.rename(columns={"Week": "Date"}, inplace=True)
       df['Pick %'] = df['Pick %'].fillna(0.0)

       base_feature_candidates = [
           'Win %', 'Future Value (Stars)', 'Date', 'Away Team', 'Availability',
           'Divisional Matchup?', 'Week_Mean_WinPct', 'Week_Mean_FV', 'Week_Max_WinPct',
           'Week_Max_FV', 'Week_Min_WinPct', 'Week_Min_FV', 'Week_Std_WinPct', 'Week_Std_FV',
           'Team_WinPct_RelativeToWeekMean', 'Team_FV_RelativeToWeekMean',
           'Team_WinPct_RelativeToTopTeam', 'Team_FV_RelativeToTopTeam', 'Win % Rank',
           'Star Rating Rank', 'Num_Teams_This_Week', 'Rank_Density', 'FV_Rank_Density',
           'Future_Weeks_Top_Team', 'Future_Weeks_Over_80', 'Future_Weeks_70_80',
           'Future_Weeks_60_70', 'Thanksgiving Underdog', 'Christmas Favorite',
           'Thanksgiving Favorite', 'thanksgiving_week', 'christmas_week', 'Thursday_Home',
           'Thursday_Away', 'Thursday_Underdog', 'Thursday_Favorite', 'Week_Mean_80',
           'Week_Max_80', 'Week_Min_80', 'Week_Std_80', 'Team_80_RelativeToWeekMean',
           'Team_80_RelativeToTopTeam', '80_Rank', '80_Rank_Density', 'Week_Mean_70_80',
           'Week_Max_70_80', 'Week_Min_70_80', 'Week_Std_70_80',
           'Team_70_80_RelativeToWeekMean', 'Team_70_80_RelativeToTopTeam', '70_80_Rank',
           '70_80_Rank_Density', 'Week_Mean_60_70', 'Week_Max_60_70', 'Week_Min_60_70',
           'Week_Std_60_70', 'Team_60_70_RelativeToWeekMean', 'Team_60_70_RelativeToTopTeam',
           '60_70_Rank', '60_70_Rank_Density', 'Week_Mean_Top_Team', 'Week_Max_Top_Team',
           'Week_Min_Top_Team', 'Week_Std_Top_Team', 'Team_Top_Team_RelativeToWeekMean',
           'Team_Top_Team_RelativeToTopTeam', 'Top_Team_Rank', 'Top_Team_Rank_Density',
           'Week_Mean_Availability', 'Week_Max_Availability', 'Week_Min_Availability',
           'Week_Std_Availability', 'Team_Availability_RelativeToWeekMean',
           'Team_Availability_RelativeToTopTeam', 'Availability_Rank',
           'Availability_Rank_Density',
           # NOTE: 'Holiday Strength' intentionally dropped from this list --
           # it's superseded by the Holiday_Lookahead_* columns train_pick_pct_models
           # computes itself. Leaving the stale column in df is harmless either way.
       ]
       base_feature_candidates.extend([c for c in holiday_cols if c in df.columns])
       base_feature_candidates = [f for f in base_feature_candidates
                                   if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

       trained_models, splash_model, splash_features, run_metrics = train_pick_pct_models(
           df_historical=df,
           target_year=target_year,
           upcoming_week=upcoming_week,
           base_feature_candidates=base_feature_candidates,
       )

   Everything after this point in the original function (the "Starting
   week-by-week pick percentage predictions..." section and onward) reads
   `trained_models`, `splash_model`, and `splash_features` exactly as
   before -- nothing downstream needs to change because of this swap.

C. Inside the SAME function, further down, REPLACE the ad hoc block that
   currently reads (originally around lines 5501-5518):

       xmas_map = pick_predictions_df[pick_predictions_df['christmas_week'] == 1]...
       tgiving_map = pick_predictions_df[pick_predictions_df['thanksgiving_week'] == 1]...
       pick_predictions_df['christmas_win_pct'] = ...
       pick_predictions_df['thanksgiving_win_pct'] = ...
       pick_predictions_df['Pre Christmas'] = pick_predictions_df['Pre Christmas'] * ...
       pick_predictions_df['Pre Thanksgiving'] = pick_predictions_df['Pre Thanksgiving'] * ...
       pick_predictions_df['Holiday Strength'] = ...

   with:

       pick_predictions_df = compute_holiday_lookahead_features(pick_predictions_df)

   This is the fix for the train/inference mismatch described above -- it
   computes the exact same columns, the exact same way, that training saw.
   'Pre Thanksgiving' / 'Pre Christmas' are left untouched (still 0/1) in
   both places now.

D. Add these imports near the top of daily_2_consolidate_predict_weekly_data.py
   if they aren't already there:

       from sklearn.model_selection import train_test_split
       from sklearn.metrics import mean_absolute_error
       from sklearn.inspection import permutation_importance
       from scipy.stats import spearmanr
       from datetime import datetime
       import os

THINGS TO VERIFY / DECIDE (not guessed at here on purpose)
=====================================================================
- The old `1 / Date` decay in the original holiday-strength formula makes
  the signal WEAKEST the closer you get to the holiday (since 'Date' is
  the week number counting up through the season) and strongest early --
  which may be backwards from the real effect you described ("people save
  a strong team for the holiday slate"). Rather than guess a replacement
  decay curve, this rewrite hands the model two separate, honest inputs
  (Christmas/Thanksgiving_WinPct_Lookahead, and Weeks_To_Christmas /
  Weeks_To_Thanksgiving) and lets the RandomForest learn whatever shape
  that decay actually has from the data -- check feature importance /
  partial dependence on these once you have a few runs logged.
- metrics_log_path defaults to 'logs/pick_pct_model_metrics.csv' -- point
  it wherever you want this tracked; the directory is created if missing.
- This validates by holding out the most recent ~15% of *weeks*
  (chronological, not random) rather than the most recent season, so even
  early-season runs get a real (if small) holdout. If you'd rather always
  validate against the most recently completed full season specifically,
  say so and the split logic is a small change.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# =====================================================================
# 1. Holiday lookahead features -- computed identically for training and
#    live prediction. See module docstring, point 1.
# =====================================================================
def compute_holiday_lookahead_features(df, year_col='Year', team_col='Team',
                                        winpct_col='Win %', week_col=None):
    """
    Attaches, to every pre-holiday row, THAT SAME TEAM's Win % on its own
    upcoming Thanksgiving / Christmas game within that season.

    Rationale: a team can look mediocre on paper in an early week and still
    be unpopular, because sharp entries are deliberately saving it for a
    holiday slate where it projects much stronger. The raw weekly Win %
    can't see that -- it only shows up if the model can look ahead to the
    team's own holiday-week matchup. This function makes that lookahead an
    explicit, leak-safe feature (it only ever uses each team's OWN game
    result on its OWN holiday week within the SAME season -- no future
    weeks from other teams or other seasons leak in).

    Must be called on the training frame and the prediction frame the same
    way -- that's the whole point of pulling it out into one function
    instead of duplicating slightly different logic in two places.

    Requires these columns to already exist: Year, Team, Win %,
    christmas_week, thanksgiving_week, Pre Christmas, Pre Thanksgiving
    (all already produced earlier in the existing pipeline).
    """
    df = df.copy()
    week_col = week_col or ('Date' if 'Date' in df.columns else 'Week')

    required = [year_col, team_col, winpct_col, 'christmas_week', 'thanksgiving_week',
                'Pre Christmas', 'Pre Thanksgiving', week_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"compute_holiday_lookahead_features is missing required "
                        f"column(s): {missing}")

    def _lookahead_lookup(holiday_flag_col):
        # One row per (season, team): that team's Win % on its own holiday game.
        holiday_rows = df.loc[df[holiday_flag_col] == 1]
        return holiday_rows.groupby([year_col, team_col])[winpct_col].mean().to_dict()

    xmas_lookup = _lookahead_lookup('christmas_week')
    tgiving_lookup = _lookahead_lookup('thanksgiving_week')

    keys = list(zip(df[year_col], df[team_col]))
    df['Christmas_WinPct_Lookahead'] = [xmas_lookup.get(k, 0.0) for k in keys]
    df['Thanksgiving_WinPct_Lookahead'] = [tgiving_lookup.get(k, 0.0) for k in keys]

    # Only meaningful on the declared pre-holiday window -- zero elsewhere,
    # so this doesn't accidentally leak a "this team is generally strong on
    # holidays" signal into unrelated mid-season weeks.
    pre_xmas = df['Pre Christmas'].fillna(0).astype(bool)
    pre_tgiving = df['Pre Thanksgiving'].fillna(0).astype(bool)
    df['Christmas_WinPct_Lookahead'] = np.where(pre_xmas, df['Christmas_WinPct_Lookahead'], 0.0)
    df['Thanksgiving_WinPct_Lookahead'] = np.where(pre_tgiving, df['Thanksgiving_WinPct_Lookahead'], 0.0)

    # How many weeks out from that holiday game -- a separate, honest input
    # instead of hand-baking a 1/week decay curve (see module docstring).
    xmas_week_num = df.loc[df['christmas_week'] == 1].groupby(year_col)[week_col].first()
    tgiving_week_num = df.loc[df['thanksgiving_week'] == 1].groupby(year_col)[week_col].first()

    df['Weeks_To_Christmas'] = (df[year_col].map(xmas_week_num) - df[week_col]).clip(lower=0)
    df['Weeks_To_Thanksgiving'] = (df[year_col].map(tgiving_week_num) - df[week_col]).clip(lower=0)
    df['Weeks_To_Christmas'] = np.where(pre_xmas, df['Weeks_To_Christmas'].fillna(0), 0.0)
    df['Weeks_To_Thanksgiving'] = np.where(pre_tgiving, df['Weeks_To_Thanksgiving'].fillna(0), 0.0)

    # Convenience combined column (successor to the old 'Holiday Strength').
    df['Holiday_Lookahead_Strength'] = df[['Christmas_WinPct_Lookahead',
                                            'Thanksgiving_WinPct_Lookahead']].max(axis=1)
    return df


HOLIDAY_LOOKAHEAD_COLS = [
    'Christmas_WinPct_Lookahead', 'Thanksgiving_WinPct_Lookahead',
    'Weeks_To_Christmas', 'Weeks_To_Thanksgiving', 'Holiday_Lookahead_Strength',
]


# =====================================================================
# 2. Chronological (not random) validation split, by whole weeks so a
#    held-out week's teams all move together, matching how the model is
#    actually used in production (predict one full week at a time).
# =====================================================================
def _time_based_split(df, year_col='Year', week_col='Date', val_frac=0.15):
    week_keys = (df[[year_col, week_col]]
                 .drop_duplicates()
                 .sort_values([year_col, week_col])
                 .reset_index(drop=True))
    n_weeks = len(week_keys)
    if n_weeks < 4:
        # Too little history for a meaningful holdout yet -- caller falls
        # back to training on everything with no logged validation metric
        # for this run, rather than crashing on an early week-1 run.
        return df.index, pd.Index([])

    n_val_weeks = max(1, int(round(n_weeks * val_frac)))
    train_weeks, val_weeks = train_test_split(week_keys, test_size=n_val_weeks, shuffle=False)

    train_keys = set(map(tuple, train_weeks[[year_col, week_col]].to_numpy()))
    val_keys = set(map(tuple, val_weeks[[year_col, week_col]].to_numpy()))
    row_keys = list(zip(df[year_col], df[week_col]))
    train_mask = [k in train_keys for k in row_keys]
    val_mask = [k in val_keys for k in row_keys]
    return df.index[train_mask], df.index[val_mask]


# =====================================================================
# 3. Permutation-importance feature ranking (measured on the held-out
#    weeks), replacing impurity importance measured on the training data.
# =====================================================================
def _select_features_by_permutation_importance(X_train, y_train, X_val, y_val,
                                                 top_n, random_state=42, n_repeats=8):
    if len(X_val) == 0:
        # No holdout yet (very first run) -- fall back to impurity
        # importance rather than failing outright.
        probe = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state)
        probe.fit(X_train, y_train)
        ranks = pd.Series(probe.feature_importances_, index=X_train.columns)
    else:
        probe = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state,
                                       min_samples_leaf=5)
        probe.fit(X_train, y_train)
        result = permutation_importance(
            probe, X_val, y_val, n_repeats=n_repeats,
            random_state=random_state, scoring='neg_mean_absolute_error', n_jobs=-1,
        )
        ranks = pd.Series(result.importances_mean, index=X_train.columns)

    return ranks.sort_values(ascending=False).head(top_n).index.tolist()


def _log_metrics(metrics, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    row = pd.DataFrame([metrics])
    header = not os.path.exists(path)
    row.to_csv(path, mode='a', header=header, index=False)


# =====================================================================
# 4. Train ONE model against ONE target (reused for the 9-feature and
#    7-feature Circa models, and for the Splash/public-pick model -- the
#    original script had near-duplicate copies of this block for each).
# =====================================================================
def train_relative_pick_pct_model(df_historical, target_col, feature_candidates,
                                   mandatory_features, top_n, model_label,
                                   year_col='Year', week_col='Date',
                                   metrics_log_path='logs/pick_pct_model_metrics.csv',
                                   random_state=42):
    """
    Trains a RandomForest to predict `target_col` RELATIVE TO that week's
    mean (a ratio, always >= 0), which matches "picks are a relative choice
    among that week's slate" and stays compatible with the existing
    proportional renormalization downstream (it just rescales whatever
    positive values it's given, so predicting a ratio instead of a raw
    percentage requires no changes to that code).

    Returns None if there's no data for this target at all (e.g. Public
    Pick % not populated yet), otherwise a dict:
        {'model': fitted RandomForestRegressor,
         'features': [feature names, in the order the model expects],
         'metrics': {} or a dict of this run's validation metrics}
    """
    df_use = df_historical.dropna(subset=[target_col]).copy()
    if df_use.empty:
        print(f"⚠️ No rows with {target_col} available -- skipping {model_label}.")
        return None

    week_mean = df_use.groupby([year_col, week_col])[target_col].transform('mean')
    df_use['_relative_target'] = df_use[target_col] / week_mean.clip(lower=1e-6)

    feat_list = [f for f in feature_candidates
                 if f in df_use.columns and pd.api.types.is_numeric_dtype(df_use[f])]

    train_idx, val_idx = _time_based_split(df_use, year_col, week_col)
    X_all = df_use[feat_list].fillna(0)
    y_all = df_use['_relative_target']
    X_train, y_train = X_all.loc[train_idx], y_all.loc[train_idx]
    X_val, y_val = X_all.loc[val_idx], y_all.loc[val_idx]

    print(f"⚙️  Ranking features for {model_label} "
          f"({len(feat_list)} candidates, {len(X_train)} train rows / {len(X_val)} val rows)...")
    ranked = _select_features_by_permutation_importance(
        X_train, y_train, X_val, y_val, top_n=top_n, random_state=random_state)

    final_features = list(dict.fromkeys(ranked + [f for f in mandatory_features if f in feat_list]))

    metrics = {}
    if len(X_val) > 0:
        eval_model = RandomForestRegressor(n_estimators=100, random_state=random_state,
                                            n_jobs=-1, min_samples_leaf=5)
        eval_model.fit(X_train[final_features], y_train)
        val_pred_ratio = eval_model.predict(X_val[final_features])

        mae_ratio = mean_absolute_error(y_val, val_pred_ratio)

        val_df = df_use.loc[val_idx, [year_col, week_col, target_col]].copy()
        val_df['_pred_ratio'] = val_pred_ratio
        val_week_mean = df_use.loc[val_idx].groupby([year_col, week_col])[target_col].transform('mean')
        val_df['_pred_abs'] = val_df['_pred_ratio'] * val_week_mean.to_numpy()
        mae_abs = mean_absolute_error(val_df[target_col], val_df['_pred_abs'])

        # Mean weekly rank-correlation: did we get the ORDERING of who's
        # more/less popular right? That's what actually drives EV -- a
        # model can miss the absolute percentage by a bit and still be
        # very useful if it ranks teams correctly within each week.
        weekly_corrs = []
        for _, wk in val_df.groupby([year_col, week_col]):
            if wk[target_col].nunique() > 1 and wk['_pred_abs'].nunique() > 1:
                corr, _ = spearmanr(wk[target_col], wk['_pred_abs'])
                if pd.notna(corr):
                    weekly_corrs.append(corr)
        mean_spearman = float(np.mean(weekly_corrs)) if weekly_corrs else float('nan')

        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_label': model_label,
            'target_col': target_col,
            'n_features': len(final_features),
            'features': ';'.join(final_features),
            'n_train_rows': int(len(X_train)),
            'n_val_rows': int(len(X_val)),
            'mae_ratio_scale': round(float(mae_ratio), 5),
            'mae_pct_points': round(float(mae_abs), 5),
            'mean_weekly_spearman': round(mean_spearman, 4) if pd.notna(mean_spearman) else None,
        }
        _log_metrics(metrics, metrics_log_path)
        spearman_str = f"{mean_spearman:.3f}" if pd.notna(mean_spearman) else "n/a"
        print(f"✅ {model_label}: val MAE={mae_abs:.4f} pts | "
              f"mean weekly rank-corr={spearman_str} ({len(weekly_corrs)} weeks) | "
              f"features={final_features}")
    else:
        print(f"⚠️ {model_label}: not enough distinct weeks yet for a validation split -- "
              f"training on everything, no MAE logged this run.")

    # Deploy model: refit on ALL available rows (train + held-out) using the
    # features/hyperparameters already locked in above via validation, so
    # the model that actually makes predictions uses every bit of history
    # rather than permanently holding 15% back.
    final_model = RandomForestRegressor(n_estimators=100, random_state=random_state,
                                         n_jobs=-1, min_samples_leaf=5)
    final_model.fit(X_all[final_features], y_all)

    return {'model': final_model, 'features': final_features, 'metrics': metrics}


# =====================================================================
# 5. Orchestrator -- drop-in replacement for the original "DUAL MODEL
#    TRAINING" + "SPLASH MODEL" sections.
# =====================================================================
def train_pick_pct_models(df_historical, target_year, upcoming_week,
                           base_feature_candidates, mandatory_features=None,
                           metrics_log_path='logs/pick_pct_model_metrics.csv',
                           random_state=42):
    """
    Returns (trained_models, splash_model, splash_features, run_metrics):
      - trained_models: {9: {'model', 'features'}, 7: {'model', 'features'}}
        -- same shape the rest of the original pipeline already expects,
        so nothing after this call needs to change.
      - splash_model, splash_features: same as the original variables.
      - run_metrics: list of metric dicts logged this run (for printing /
        inspecting immediately without re-reading the CSV).

    `df_historical` should be the raw df straight off
    pd.read_csv('contest-historical-data/Circa_historical_data.csv')
    with the Week->Date rename and Pick % NaN-fill already applied, exactly
    as the original code did before this block.
    """
    mandatory_features = mandatory_features or ['Pre Thanksgiving', 'Pre Christmas',
                                                  'christmas_week', 'thanksgiving_week']

    # --- Restore the leakage guard (this was disabled in the original
    #     script: `df_historical = df` unconditionally overwrote the
    #     masked version, so training saw the full dataset regardless of
    #     upcoming_week). ---
    past_years_mask = df_historical['Year'] < target_year
    current_year_past_weeks_mask = ((df_historical['Year'] == target_year) &
                                     (df_historical['Date'] < upcoming_week))
    df_historical = df_historical.loc[past_years_mask | current_year_past_weeks_mask].copy()

    if df_historical.empty:
        raise ValueError(f"No historical training data available prior to "
                          f"{target_year} Week {upcoming_week}. If this is the very "
                          f"first week of your very first historical year, you'll "
                          f"need a fallback (e.g. skip training and use a flat/prior "
                          f"guess) rather than calling this function.")

    # --- Holiday lookahead features (see compute_holiday_lookahead_features
    #     docstring). Computed here on the training frame; the SAME call
    #     must also be made on the live prediction frame -- see this
    #     module's top-of-file integration notes, step C. ---
    df_historical = compute_holiday_lookahead_features(df_historical)
    mandatory_features = list(dict.fromkeys(mandatory_features + HOLIDAY_LOOKAHEAD_COLS))

    assumed_public_pick_col = 'Public Pick %'
    clean_base = [f for f in base_feature_candidates if f != assumed_public_pick_col]
    candidate_pool = list(dict.fromkeys(clean_base + HOLIDAY_LOOKAHEAD_COLS))

    model_configs = {
        9: {'features': candidate_pool + [assumed_public_pick_col], 'target_n': 9},
        7: {'features': candidate_pool, 'target_n': 7},
    }

    trained_models = {}
    run_metrics = []
    for n_key, cfg in model_configs.items():
        result = train_relative_pick_pct_model(
            df_historical, target_col='Pick %', feature_candidates=cfg['features'],
            mandatory_features=mandatory_features, top_n=cfg['target_n'],
            model_label=f'circa_pick_pct_model_{n_key}',
            metrics_log_path=metrics_log_path, random_state=random_state,
        )
        if result is not None:
            trained_models[n_key] = {'model': result['model'], 'features': result['features']}
            if result['metrics']:
                run_metrics.append(result['metrics'])
        else:
            print(f"⚠️ Model {n_key} could not be trained this run (see warning above).")

    # --- Splash (public/square pick %) model -- same treatment, same
    #     holiday features, own validation + logging. ---
    splash_model, splash_features = None, []
    try:
        splash_result = train_relative_pick_pct_model(
            df_historical, target_col=assumed_public_pick_col, feature_candidates=clean_base,
            mandatory_features=mandatory_features, top_n=9, model_label='splash_public_pick_pct',
            metrics_log_path=metrics_log_path, random_state=random_state,
        )
        if splash_result is not None:
            splash_model = splash_result['model']
            splash_features = splash_result['features']
            if splash_result['metrics']:
                run_metrics.append(splash_result['metrics'])
        else:
            print("⚠️ No Public Pick % history -- Splash pick% will be blank.")
    except Exception as _e:
        print(f"⚠️ Splash model training failed ({_e}); Splash pick% blank.")

    return trained_models, splash_model, splash_features, run_metrics
