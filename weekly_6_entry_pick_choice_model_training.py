"""
entry_pick_choice_model_training.py

Trains the actual entry-choice model: given a survivor ENTRY's as-of
archetype state and the pre-game features of every team it still has
available in a given week, score how likely that entry is to pick each
candidate team. At inference time (a separate, later script -- not this
one) those per-team scores get softmax-normalized within one entry's
choice set to produce a probability split, e.g. "30% SF, 20% LAC, 15%
PHI...".

INPUT
=====
training_data/entry_pick_choice_training_data.parquet (or .csv), produced
by build_entry_pick_training_data.py. One row per (Year, Week, EntryName,
candidate Team), with a binary `Picked` label -- exactly one positive row
per (Year, Week, EntryName) group.

MODEL
=====
A single LightGBM binary classifier (objective='binary') trained on the
expanded rows -- NOT a per-group softmax/multiclass model. This matches
the "choice-set expansion" design from build_entry_pick_training_data.py:
train a plain binary scorer on (candidate, label) rows, then normalize
its raw scores within a group at inference time. Training directly on
softmax-normalized probabilities would require a fixed-size choice set,
which survivor pools don't have (available pool shrinks every week).

WHY A CHRONOLOGICAL SPLIT
===========================
Same reasoning as pick_pct_model_training.py: a random row split would
leak information across time (the model could see a later week's outcome
patterns while "predicting" an earlier one), and would leak information
across an ENTRY's own picks (rows from the same EntryName's Week 5 and
Week 6 picks would end up on both sides of the split). Splitting whole
SEASONS into train/val/test keeps every group's rows together and keeps
the evaluation honest about the real task: predicting a future season's
picks from patterns learned in past seasons.

WHY TOP-1 / TOP-3 ACCURACY, NOT JUST LOGLOSS/AUC
====================================================
Binary logloss and AUC measure row-level classification quality across
ALL entries/weeks pooled together, but the real question is per-group:
"out of THIS entry's available teams THIS week, did the model rank the
team they actually picked at or near the top?" That's what top-1/top-3
accuracy (computed per (Year, Week, EntryName) group) actually measures,
and it's the metric that matters once scores get softmax-normalized into
a probability split at inference time.
"""

import datetime
import json
import os

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError(
        "This script needs lightgbm (`pip install lightgbm`) -- see the "
        "matching train_entry_pick_choice_model.yml workflow."
    ) from e

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
from contest_config import get_contest, tagged

_CFG = get_contest()

# Contest-tagged so each contest keeps its own training data / model
# (Circa's filenames are preserved unchanged; Splash contests get a suffix).
DATA_PATH_PARQUET = tagged("training_data/entry_pick_choice_training_data", ".parquet")
DATA_PATH_CSV = tagged("training_data/entry_pick_choice_training_data", ".csv")  # fallback

MODEL_DIR = "models"
MODEL_PATH = tagged(os.path.join(MODEL_DIR, "entry_pick_choice_model"), ".pkl")
FEATURE_META_PATH = tagged(os.path.join(MODEL_DIR, "entry_pick_choice_model_features"), ".json")

# Train/Val/Test years are NOT hardcoded -- see split_by_year() below.
# They're derived from whichever seasons are actually present in the
# loaded data: newest season = Test (the live, still-accumulating
# holdout for the in-progress season), second-newest = Val, everything
# older = Train. Since pool behavior has been evolving season to season,
# this also means a season doesn't sit in Val/Test forever -- the
# moment a newer season shows up, it rolls back into Train automatically.
# This needs zero edits at each season transition (e.g. 2026 -> 2027).

GROUP_COLS = ['Year', 'Week', 'EntryName']
LABEL_COL = 'Picked'

# Columns that must stay as text/category no matter what -- never run
# through the numeric-recovery heuristic in load_training_data().
FORCE_CATEGORICAL = ['Team', 'Primary_Archetype_AsOf', 'Circa Week']

NUMERIC_FEATURES = [
    'Win_Pct', 'Sportsbook_EV', 'Future_Value', 'Public_Pick_Pct',
    'Predicted_Pick_Pct',  # known ~100% NaN as of this writing -- kept
                           # in case it gets populated later; LightGBM
                           # just won't split on an all-NaN column.
    'Win_Pct_Pctile', 'Sportsbook_EV_Pctile', 'Future_Value_Pctile',
    'Teams_Remaining_In_Pool', 'Total Remaining Entries at Start of Week',
    'Week',
    'Planner', 'Contrarian', 'EV Hunter', 'Sprinter', 'Hoarder', 'Tourist',
    'Picks_Used_So_Far',
    # Boolean-ish flags -- recovered to 0/1 floats by load_training_data,
    # so they belong in the numeric list, not FORCE_CATEGORICAL.
    'Expected_Availability', 'Pre_Thanksgiving', 'Pre_Christmas',
    'Divisional Matchup Boolean',
]
CATEGORICAL_FEATURES = list(FORCE_CATEGORICAL)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------
# 1. Load + robust dtype recovery
# --------------------------------------------------------------------
def load_training_data(parquet_path=DATA_PATH_PARQUET, csv_path=DATA_PATH_CSV):
    """build_entry_pick_training_data.py normalizes every mixed-type
    ("object") column to a plain string dtype before saving, so Parquet
    can write it (see that script's comments -- e.g. "Circa Week" holds
    both week numbers and the text "Thanksgiving" in the same column).
    That's the right move for a lossless write, but it means everything
    that got stringified needs its real type recovered here before it's
    useful as a model feature: a numeric column that picked up string
    dtype only because of one rare non-numeric value elsewhere in the
    6-year table should come back as numeric, not stay text.
    """
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df):,} rows from {parquet_path}")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded {len(df):,} rows from {csv_path}")
    else:
        raise FileNotFoundError(
            f"Training data not found at {parquet_path} or {csv_path}. "
            "Run build_entry_pick_training_data.py (or download its "
            "workflow artifact) first."
        )

    bool_map = {'True': 1.0, 'False': 0.0, 'true': 1.0, 'false': 0.0,
                'TRUE': 1.0, 'FALSE': 0.0}

    for col in df.columns:
        if col in FORCE_CATEGORICAL:
            df[col] = df[col].astype(str).astype('category')
            continue
        if df[col].dtype.name in ('object', 'string'):
            non_null = df[col].notna()
            if non_null.sum() == 0:
                continue  # all-NaN column -- leave as-is, harmless
            uniques = set(df.loc[non_null, col].unique().tolist())
            if uniques <= set(bool_map.keys()):
                df[col] = df[col].map(bool_map)
                continue
            numeric = pd.to_numeric(df[col], errors='coerce')
            # If nearly everything that had a real value converted
            # cleanly, trust the numeric recovery. Anything genuinely
            # textual (that we didn't already know to force-categorize
            # above) falls back to category so training doesn't crash
            # on it, with a printed heads-up since it means a column
            # showed up that this script doesn't already know about.
            if numeric.notna()[non_null].mean() > 0.9:
                df[col] = numeric
            else:
                print(f"   Note: '{col}' didn't recover as numeric or "
                      f"boolean (sample values: "
                      f"{list(uniques)[:5]}) -- keeping as category.")
                df[col] = df[col].astype('category')

    return df


# --------------------------------------------------------------------
# 2. Chronological split -- purely a function of which seasons are
#    actually present in the data. No year list to maintain here.
# --------------------------------------------------------------------
def compute_dynamic_year_splits(available_years):
    """Newest season present = Test (the live, still-accumulating
    holdout -- e.g. the currently in-progress season, retrained weekly
    as more of its weeks complete). Second-newest = Val (the most
    recently fully-completed season). Everything older = Train. Because
    this is recomputed from whatever years actually show up in the
    loaded table, a season transition (2026 -> 2027, 2027 -> 2028, ...)
    requires no code change: the day a new season's picks start showing
    up in the training data, it becomes Test, the old Test season slides
    into Val, and the old Val season slides into Train."""
    years = sorted(available_years)
    if len(years) >= 3:
        return years[:-2], years[-2:-1], years[-1:]
    if len(years) == 2:
        return years[:1], [], years[-1:]
    return years, [], []  # 0 or 1 season present -- can't split meaningfully


def split_by_year(df):
    available_years = df['Year'].unique().tolist()
    train_years, val_years, test_years = compute_dynamic_year_splits(available_years)

    def _subset(years, label):
        sub = df[df['Year'].isin(years)]
        n_groups = sub.groupby(GROUP_COLS).ngroups if len(sub) else 0
        print(f"   {label}: {len(sub):,} rows, {n_groups:,} entry-week "
              f"groups, years {sorted(sub['Year'].unique().tolist())}")
        return sub

    print("Splitting by season (Test = current/newest season present, "
          "Val = season before that, Train = everything older):")
    train = _subset(train_years, 'Train')
    val = _subset(val_years, 'Val')
    test = _subset(test_years, 'Test')
    return train, val, test, train_years, val_years, test_years


# --------------------------------------------------------------------
# 3. Per-group evaluation -- the metric that actually matters here
# --------------------------------------------------------------------
def topk_accuracy(frame, scores, ks=(1, 3)):
    """frame must include GROUP_COLS and LABEL_COL, aligned index-for-
    index with `scores` (the model's predicted probability per row).
    For each (Year, Week, EntryName) group, ranks candidate teams by
    predicted score and checks whether the team the entry ACTUALLY
    picked landed in the top k."""
    ranked = frame[GROUP_COLS + [LABEL_COL]].copy()
    ranked['_score'] = np.asarray(scores)
    ranked['_rank'] = ranked.groupby(GROUP_COLS)['_score'] \
        .rank(method='first', ascending=False)

    actual = ranked.loc[ranked[LABEL_COL] == 1, '_rank']
    results = {'n_groups': len(actual)}
    for k in ks:
        results[f'top_{k}_accuracy'] = float((actual <= k).mean()) if len(actual) else float('nan')
    results['mean_rank_of_actual_pick'] = float(actual.mean()) if len(actual) else float('nan')
    return results


# --------------------------------------------------------------------
# 4. Train
# --------------------------------------------------------------------
def train_model(train_df, val_df):
    X_train, y_train = train_df[ALL_FEATURES], train_df[LABEL_COL]
    X_val, y_val = val_df[ALL_FEATURES], val_df[LABEL_COL]

    model = LGBMClassifier(
        objective='binary',
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        # Deliberately NOT using is_unbalance/scale_pos_weight: the ~3%
        # positive rate here is a structural property of the choice-set
        # expansion (one pick out of an N-team pool), not class noise to
        # correct for. Reweighting would distort the raw scores that
        # get softmax-normalized at inference time.
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['binary_logloss', 'auc'],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[
            lgb.early_stopping(stopping_rounds=75),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


# --------------------------------------------------------------------
# 5. Permutation importance (mirrors pick_pct_model_training.py's use
#    of this on the validation set -- shows what the model actually
#    leans on, separate from LightGBM's own split-count importances,
#    which can overweight high-cardinality columns like Team).
# --------------------------------------------------------------------
def permutation_importance(model, val_df, n_repeats=5, random_state=42):
    from sklearn.metrics import roc_auc_score

    X_val, y_val = val_df[ALL_FEATURES].copy(), val_df[LABEL_COL]
    rng = np.random.default_rng(random_state)

    baseline_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    importances = []
    for col in ALL_FEATURES:
        drops = []
        for _ in range(n_repeats):
            shuffled = X_val.copy()
            shuffled[col] = shuffled[col].sample(frac=1.0, random_state=int(rng.integers(1e9))).values
            auc = roc_auc_score(y_val, model.predict_proba(shuffled)[:, 1])
            drops.append(baseline_auc - auc)
        importances.append({
            'feature': col,
            'mean_auc_drop': float(np.mean(drops)),
            'std_auc_drop': float(np.std(drops)),
        })

    imp_df = pd.DataFrame(importances).sort_values('mean_auc_drop', ascending=False)
    return baseline_auc, imp_df


# --------------------------------------------------------------------
# 6. Orchestration
# --------------------------------------------------------------------
def main():
    df = load_training_data()

    missing_feature_cols = [c for c in ALL_FEATURES + GROUP_COLS + [LABEL_COL] if c not in df.columns]
    if missing_feature_cols:
        raise ValueError(
            f"Training data is missing expected column(s): {missing_feature_cols}. "
            "Did build_entry_pick_training_data.py change its output schema?"
        )

    train_df, val_df, test_df, train_years, val_years, test_years = split_by_year(df)
    if train_df.empty or val_df.empty:
        seasons = sorted(df['Year'].unique().tolist())
        print(
            "\n⏭️  Skipping choice-model training: the data only spans "
            f"{seasons} season(s), which isn't enough for a 3-way "
            "train/val/test split (need at least 2 seasons). This is expected "
            "for the earliest replay year(s); the model will train once more "
            "seasons have accumulated. No model written -- daily_4 will fall "
            "back to the base projection."
        )
        return

    print("\nTraining LightGBM binary classifier...")
    model = train_model(train_df, val_df)

    print("\nEvaluating on each split:")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if split_df.empty:
            print(f"   {name}: (empty, skipped)")
            continue
        preds = model.predict_proba(split_df[ALL_FEATURES])[:, 1]
        metrics = topk_accuracy(split_df, preds)
        print(f"   {name}: {metrics['n_groups']:,} groups -- "
              f"top-1 accuracy {metrics['top_1_accuracy']:.4f}, "
              f"top-3 accuracy {metrics['top_3_accuracy']:.4f}, "
              f"mean rank of actual pick {metrics['mean_rank_of_actual_pick']:.2f}")

    print("\nLightGBM split-count feature importances (top 15):")
    imp = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    print(imp.head(15).to_string())

    print("\nPermutation importance on Val (AUC drop when a feature is shuffled, top 15):")
    baseline_auc, perm_imp = permutation_importance(model, val_df)
    print(f"   Val baseline AUC: {baseline_auc:.4f}")
    print(perm_imp.head(15).to_string(index=False))

    os.makedirs(MODEL_DIR, exist_ok=True)
    import joblib
    joblib.dump(model, MODEL_PATH)
    with open(FEATURE_META_PATH, 'w') as f:
        json.dump({
            'features': ALL_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'group_cols': GROUP_COLS,
            'label_col': LABEL_COL,
            'train_years': train_years,
            'val_years': val_years,
            'test_years': test_years,
            'trained_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, f, indent=2)

    print(f"\n✅ Saved model to {MODEL_PATH}")
    print(f"✅ Saved feature metadata to {FEATURE_META_PATH}")


if __name__ == '__main__':
    main()
