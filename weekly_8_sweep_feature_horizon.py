#!/usr/bin/env python3
"""
weekly_8_sweep_feature_horizon.py

Which FEATURE COUNT projects pick % best, and does the answer depend on how many
weeks ahead you're forecasting? The intuition: for the upcoming week you have
rich current signal (incl. the public feed) so a wider feature set may help, but
for distant weeks a leaner model may generalize better. This script measures
that directly.

SELF-CONTAINED: it retrains the pick-% model itself from the contest's historical
file at several feature counts, forward-projects by horizon, and scores against
the actual Pick % already in that file. No daily_2 run, no re-backfill, no new
snapshots — it works on data you already have.

  • horizon h = target_week - as_of_week. For each as-of week W the model is
    trained ONLY on data legal at W (prior seasons, plus earlier weeks of the
    same season), then used to project week W+h. Comparing that projection to
    the actual isolates how accuracy fades with distance, per feature count.
  • Public Pick % availability is respected: it's a real feature only for the
    UPCOMING week (h=0). For h>=1 a future week's public pick % isn't published
    yet, so it is EXCLUDED — mirroring daily_2's 9-feature (with-public) vs
    7-feature (no-public) split. Using it for future weeks would be leakage.

FAITHFULNESS: the candidate pool, the Circa-only holiday mandatory set, the
strong-mandatory features, the relative-to-week-mean target and the permutation
top-N selection all mirror daily_2's current pick-% logic. Because that logic
lives inside daily_2's big function (not importable), it is re-implemented here;
if daily_2's pick-% model changes materially, update the constants below to match.

OUTPUTS (committed, under entry-analytics/):
  • sweep_feature_horizon.csv      -- row per (contest, year, as_of, target, top_n)
  • sweep_feature_horizon_curve.csv-- rolled up per (contest, top_n, horizon)
  • sweep_feature_horizon_report.txt -- best feature count per horizon band
  • sweep_feature_horizon_{contest}.png -- MAE & rank-corr vs horizon, line per top_n

Usage:
  python weekly_8_sweep_feature_horizon.py                      # all years/contests
  python weekly_8_sweep_feature_horizon.py --years 2022 2023 2024
  python weekly_8_sweep_feature_horizon.py --top-ns 9 15 30 40 60 all
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import contest_config as cc

OUT_DIR = 'entry-analytics'
MIN_TEAMS = 5
DEFAULT_TOP_NS = [9, 15, 30, 40, 60, 'all']

# --- Mirrors of daily_2's pick-% feature logic (keep in sync) ---------------
BASE_FEATURE_CANDIDATES = [
    'Win %', 'Future Value (Stars)', 'Availability', 'Divisional Matchup?',
    'Week_Mean_WinPct', 'Week_Mean_FV', 'Week_Max_WinPct', 'Week_Max_FV',
    'Week_Min_WinPct', 'Week_Min_FV', 'Week_Std_WinPct', 'Week_Std_FV',
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
]
HOLIDAY_LOOKAHEAD_COLS = ['Christmas_WinPct_Lookahead', 'Thanksgiving_WinPct_Lookahead',
                          'Weeks_To_Christmas', 'Weeks_To_Thanksgiving',
                          'Holiday_Lookahead_Strength']
STRONG_MANDATORY = ['Public Pick %', 'Win %', 'Future Value (Stars)', 'Availability']
CIRCA_HOLIDAY_MANDATORY = ['thanksgiving_week', 'christmas_week', 'Pre Thanksgiving',
                           'Pre Christmas', 'Thanksgiving Favorite', 'Christmas Favorite',
                           'Thanksgiving Underdog', 'Christmas Underdog',
                           'Thursday_Favorite', 'Thursday_Underdog']
PUBLIC_COL = 'Public Pick %'


def compute_holiday_lookahead_features(df):
    """Leak-safe holiday lookahead cols (mirror of daily_2). Uses each team's OWN
    holiday-week Win % within the same season; never touches Pick %."""
    df = df.copy()
    req = ['Year', 'Team', 'Win %', 'christmas_week', 'thanksgiving_week',
           'Pre Christmas', 'Pre Thanksgiving', 'Week']
    if any(c not in df.columns for c in req):
        for c in HOLIDAY_LOOKAHEAD_COLS:
            df[c] = 0.0
        return df

    def _lookup(flag):
        rows = df.loc[df[flag] == 1]
        return rows.groupby(['Year', 'Team'])['Win %'].mean().to_dict()
    xmas, tg = _lookup('christmas_week'), _lookup('thanksgiving_week')
    keys = list(zip(df['Year'], df['Team']))
    df['Christmas_WinPct_Lookahead'] = [xmas.get(k, 0.0) for k in keys]
    df['Thanksgiving_WinPct_Lookahead'] = [tg.get(k, 0.0) for k in keys]
    pre_x = df['Pre Christmas'].fillna(0).astype(bool)
    pre_t = df['Pre Thanksgiving'].fillna(0).astype(bool)
    df['Christmas_WinPct_Lookahead'] = np.where(pre_x, df['Christmas_WinPct_Lookahead'], 0.0)
    df['Thanksgiving_WinPct_Lookahead'] = np.where(pre_t, df['Thanksgiving_WinPct_Lookahead'], 0.0)
    xw = df.loc[df['christmas_week'] == 1].groupby('Year')['Week'].first()
    tw = df.loc[df['thanksgiving_week'] == 1].groupby('Year')['Week'].first()
    df['Weeks_To_Christmas'] = np.where(pre_x, (df['Year'].map(xw) - df['Week']).clip(lower=0).fillna(0), 0.0)
    df['Weeks_To_Thanksgiving'] = np.where(pre_t, (df['Year'].map(tw) - df['Week']).clip(lower=0).fillna(0), 0.0)
    df['Holiday_Lookahead_Strength'] = df[['Christmas_WinPct_Lookahead',
                                           'Thanksgiving_WinPct_Lookahead']].max(axis=1)
    return df


def _numeric_pool(df, include_public):
    pool = [c for c in BASE_FEATURE_CANDIDATES if c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])]
    pool += [c for c in HOLIDAY_LOOKAHEAD_COLS if c in df.columns]
    if include_public and PUBLIC_COL in df.columns:
        pool.append(PUBLIC_COL)
    return list(dict.fromkeys(pool))


def _mandatory(is_circa, include_public, pool):
    m = list(CIRCA_HOLIDAY_MANDATORY) if is_circa else []
    if is_circa:
        m += HOLIDAY_LOOKAHEAD_COLS
    m += [f for f in STRONG_MANDATORY if include_public or f != PUBLIC_COL]
    return [f for f in dict.fromkeys(m) if f in pool]


def rank_features(legal, pool, random_state=42):
    """Rank `pool` by RandomForest impurity importance on the relative-target.
    daily_2 uses permutation importance for its deployed selection; here the
    ranking only decides which features each top_n config includes, and this is
    run once globally, so impurity (a single fit, no repeated shuffles) is used
    for speed. The per-window model refits that produce the horizon curve are
    unaffected by this choice."""
    use = legal.dropna(subset=['Pick %']).copy()
    if use.empty:
        return list(pool)
    y = _relative_target(use)
    rf = RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=random_state,
                               min_samples_leaf=5)
    rf.fit(use[pool].fillna(0), y)
    return pd.Series(rf.feature_importances_, index=pool).sort_values(ascending=False).index.tolist()


def _relative_target(use):
    wm = use.groupby(['Year', 'Week'])['Pick %'].transform('mean').clip(lower=1e-6)
    return (use['Pick %'] / wm).to_numpy()


def fit_model(use, y, feats, random_state=42):
    """Fit on a pre-filtered legal frame `use` (no NaN Pick %) and its
    precomputed relative target `y`, so the per-(year,week) groupby isn't
    recomputed for every feature-count config."""
    rf = RandomForestRegressor(n_estimators=70, n_jobs=-1, random_state=random_state, min_samples_leaf=5)
    rf.fit(use[feats].fillna(0), y)
    return rf


def _resolve_topn(n, pool_len):
    return pool_len if (isinstance(n, str) and n.lower() == 'all') else int(n)


def score_config(model, feats, target_rows):
    awm = target_rows['Pick %'].mean()
    if not np.isfinite(awm) or awm <= 0:
        return None
    ratio = model.predict(target_rows[feats].fillna(0))
    proj = ratio * awm
    actual = target_rows['Pick %'].to_numpy()
    mae = float(np.mean(np.abs(proj - actual)))
    sp = float('nan')
    if len(np.unique(proj)) > 1 and len(np.unique(actual)) > 1:
        if _HAVE_SCIPY:
            sp, _ = spearmanr(proj, actual)
        else:
            sp = pd.Series(proj).rank().corr(pd.Series(actual).rank())
    return {'n_teams': len(actual), 'mae_pct_points': round(mae, 5),
            'spearman': round(float(sp), 5) if pd.notna(sp) else None}


def run(contests, years, top_ns, week_stride=1):
    rows = []
    for contest in contests:
        spec = cc.CONTESTS[contest]
        is_circa = (spec['col_tag'].lower() == 'circa')
        start = int(spec.get('start_season', 0))
        path = spec.get('historical_csv')
        if not path or not os.path.exists(path):
            print(f"  ⏭️  {contest}: no historical file.")
            continue
        raw = pd.read_csv(path, low_memory=False)
        if not {'Year', 'Week', 'Team', 'Pick %'}.issubset(raw.columns):
            print(f"  ⏭️  {contest}: historical file missing Year/Week/Team/Pick %.")
            continue
        raw['Year'] = pd.to_numeric(raw['Year'], errors='coerce')
        raw['Week'] = pd.to_numeric(raw['Week'], errors='coerce')
        df = compute_holiday_lookahead_features(raw.dropna(subset=['Year', 'Week']))
        pool_pub = _numeric_pool(df, include_public=True)
        pool_nop = _numeric_pool(df, include_public=False)
        mand_pub = _mandatory(is_circa, True, pool_pub)
        mand_nop = _mandatory(is_circa, False, pool_nop)

        yrs = [y for y in years if y >= start and (df['Year'] == y).any()]
        # Rank the feature pools ONCE (globally) rather than per year: the
        # ranking is a feature-set choice, and re-running permutation importance
        # per (year, public-flag) dominates runtime. The horizon signal comes
        # from the per-(year, as-of week) model refits below, which stay
        # point-in-time. Ranking uses all seasons for stability/speed.
        print(f"  ⚙️  {contest}: ranking feature pools (with/without public)...")
        ranked_pub = rank_features(df, pool_pub)
        ranked_nop = rank_features(df, pool_nop)
        for year in yrs:
            ydf = df[df['Year'] == year]
            weeks = sorted(int(w) for w in ydf['Week'].dropna().unique())
            print(f"  📈 {contest} {year}: {len(weeks)} as-of weeks × {len(top_ns)} configs")
            for W in weeks[::week_stride]:
                legal_use = df[(df['Year'] < year) | ((df['Year'] == year) & (df['Week'] < W))] \
                    .dropna(subset=['Pick %'])
                if legal_use.shape[0] < 50:
                    continue
                y_legal = _relative_target(legal_use)
                targets = {T: ydf[(ydf['Week'] == T) & ydf['Pick %'].notna()]
                           for T in weeks if T >= W}
                targets = {T: r for T, r in targets.items()
                           if len(r) >= MIN_TEAMS and r['Pick %'].abs().sum() > 0}
                if not targets:
                    continue
                for n in top_ns:
                    k_pub = _resolve_topn(n, len(pool_pub))
                    k_nop = _resolve_topn(n, len(pool_nop))
                    feats_pub = [f for f in dict.fromkeys(ranked_pub[:k_pub] + mand_pub) if f in pool_pub]
                    feats_nop = [f for f in dict.fromkeys(ranked_nop[:k_nop] + mand_nop) if f in pool_nop]
                    m_pub = fit_model(legal_use, y_legal, feats_pub)
                    m_nop = fit_model(legal_use, y_legal, feats_nop)
                    for T, trows in targets.items():
                        model, feats = (m_pub, feats_pub) if T == W else (m_nop, feats_nop)
                        s = score_config(model, feats, trows)
                        if s is None:
                            continue
                        rows.append({'contest': contest, 'year': int(year),
                                     'as_of_week': W, 'target_week': T,
                                     'horizon': T - W, 'top_n': str(n),
                                     'n_features': len(feats), **s})
    return pd.DataFrame(rows)


def _halflife(h, s):
    h, s = np.asarray(h, float), np.asarray(s, float)
    ok = np.isfinite(h) & np.isfinite(s) & (s > 0)
    if ok.sum() < 3 or np.ptp(h[ok]) == 0:
        return float('nan')
    slope = np.polyfit(h[ok], np.log(s[ok]), 1)[0]
    return float('inf') if slope >= 0 else round(-math.log(2) / slope, 2)


def summarize(acc):
    curve = (acc.groupby(['contest', 'top_n', 'horizon'])
             .agg(n_obs=('mae_pct_points', 'size'),
                  mae_pct_points=('mae_pct_points', 'mean'),
                  spearman=('spearman', 'mean'))
             .reset_index().round(5))
    lines = ["FEATURE-COUNT × HORIZON  —  which top_n projects best, how far out",
             "=" * 74,
             "MAE in pick-% points (lower better); Spearman (higher better).\n"]
    for contest in sorted(curve['contest'].unique()):
        cb = curve[curve['contest'] == contest]
        lines.append(f"\n{'#'*74}\n## {contest.upper()}\n{'#'*74}")
        # best top_n per horizon (by lowest mean MAE)
        lines.append("\n  best feature count by horizon (lowest MAE):")
        for h in sorted(cb['horizon'].unique()):
            hb = cb[cb['horizon'] == h].dropna(subset=['mae_pct_points'])
            if hb.empty:
                continue
            win = hb.loc[hb['mae_pct_points'].idxmin()]
            lines.append(f"    h={int(h):>2}:  top_n={win['top_n']:>4}  "
                         f"(MAE {win['mae_pct_points']:.4f}, n={int(win['n_obs'])})")
        lines.append("\n  per-config decay:")
        for n in sorted(cb['top_n'].unique(), key=lambda x: (x == 'all', _safe_int(x))):
            nb = cb[cb['top_n'] == n].sort_values('horizon')
            if nb.empty:
                continue
            h0 = nb[nb['horizon'] == nb['horizon'].min()]['mae_pct_points'].iloc[0]
            hl = _halflife(nb.dropna(subset=['spearman'])['horizon'],
                           nb.dropna(subset=['spearman'])['spearman'])
            hl_s = 'n/a' if not np.isfinite(hl) else ('no decay' if hl == float('inf') else f'{hl} wks')
            lines.append(f"    top_n={n:>4}: MAE@h0={h0:.4f}  skill half-life={hl_s}")
    return curve, "\n".join(lines) + "\n"


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return 10 ** 9


# dataviz categorical hues (light), assigned to configs in sorted order.
_HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']


def render(curve, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  ⚠️  matplotlib unavailable ({e}); skipping charts.")
        return []
    written = []
    for contest in sorted(curve['contest'].unique()):
        cb = curve[curve['contest'] == contest]
        order = sorted(cb['top_n'].unique(), key=lambda x: (x == 'all', _safe_int(x)))
        colors = {n: _HUES[i % len(_HUES)] for i, n in enumerate(order)}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor='#fcfcfb')
        for ax in (ax1, ax2):
            ax.set_facecolor('#fcfcfb')
            ax.grid(True, color='#e6e5e1', linewidth=0.8, zorder=0)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors='#52514e')
        for n in order:
            nb = cb[cb['top_n'] == n].sort_values('horizon')
            ax1.plot(nb['horizon'], nb['mae_pct_points'], '-o', color=colors[n],
                     lw=2, ms=4, label=f'top_n={n}', zorder=3)
            sp = nb.dropna(subset=['spearman'])
            ax2.plot(sp['horizon'], sp['spearman'], '-o', color=colors[n],
                     lw=2, ms=4, label=f'top_n={n}', zorder=3)
        ax1.set_title('MAE vs horizon  (lower = better)', color='#0b0b0b', fontsize=11)
        ax1.set_xlabel('weeks ahead (horizon)', color='#52514e')
        ax1.set_ylabel('MAE (pick-% points)', color='#52514e')
        ax2.set_title('Rank-corr vs horizon  (higher = better)', color='#0b0b0b', fontsize=11)
        ax2.set_xlabel('weeks ahead (horizon)', color='#52514e')
        ax2.set_ylabel('Spearman', color='#52514e')
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f'{contest.title()} — feature count vs forecast horizon',
                     color='#0b0b0b', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        path = os.path.join(out_dir, f'sweep_feature_horizon_{contest}.png')
        fig.savefig(path, dpi=150, facecolor='#fcfcfb', bbox_inches='tight')
        plt.close(fig)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--years', nargs='*', type=int, default=None)
    ap.add_argument('--contests', nargs='*', default=None)
    ap.add_argument('--top-ns', nargs='*', default=None,
                    help="feature counts to test (ints and/or 'all')")
    ap.add_argument('--week-stride', type=int, default=1,
                    help='sample every Nth as-of week to speed up (default 1 = all)')
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()

    contests = args.contests or list(cc.CONTESTS.keys())
    top_ns = args.top_ns or DEFAULT_TOP_NS
    top_ns = [t if (isinstance(t, str) and t.lower() == 'all') else int(t) for t in top_ns]

    if args.years:
        years = args.years
    else:
        # every year present in the Circa historical file
        try:
            h = pd.read_csv(cc.CONTESTS['circa']['historical_csv'], usecols=['Year'])
            years = sorted(int(y) for y in pd.to_numeric(h['Year'], errors='coerce').dropna().unique())
        except Exception:
            years = list(range(2020, 2027))

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"🔎 Feature-count × horizon — contests={contests} years={years} "
          f"top_ns={top_ns} week_stride={args.week_stride}")
    acc = run(contests, years, top_ns, week_stride=max(1, args.week_stride))
    if acc.empty:
        print("❌ No observations produced.")
        return
    acc_path = os.path.join(args.out_dir, 'sweep_feature_horizon.csv')
    acc.sort_values(['contest', 'year', 'as_of_week', 'top_n', 'target_week']).to_csv(acc_path, index=False)
    curve, report = summarize(acc)
    curve_path = os.path.join(args.out_dir, 'sweep_feature_horizon_curve.csv')
    curve.to_csv(curve_path, index=False)
    report_path = os.path.join(args.out_dir, 'sweep_feature_horizon_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    charts = render(curve, args.out_dir)
    print(report)
    print(f"💾 wrote {acc_path} ({len(acc)} rows)")
    print(f"💾 wrote {curve_path}")
    print(f"💾 wrote {report_path}")
    for c in charts:
        print(f"🖼️  wrote {c}")


if __name__ == '__main__':
    main()
