#!/usr/bin/env python3
"""
weekly_9_flavor_accuracy.py

Which pick-% projection column is most accurate? This compares every projection
FLAVOR — Predicted (the deployed blend), Top Down (the pure daily_2 projection),
and Archetype (the behavioral/daily_4 estimate) — against the ACTUAL contest
pick %, with a full battery of accuracy measures.

DATA SOURCE (one file per week, no cross-joins):
  nfl-power-ratings/final_data/{year}_final_data/Week_{W}_{year}_Final_Data.csv
Each carries, per game, every flavor's projection for that week AND that week's
Actual pick %, all on the same (full-name) team key. The flavor projections in
these files are the horizon-0 (live, as-of-that-week) values — verified to match
the committed sim snapshots — so this is an apples-to-apples "what we projected
for the week vs what actually happened."

MEASURES (per flavor, per contest):
  Error / magnitude      MAE, RMSE, Median AE, Mean Error (signed bias)
  Correlation / ranking  Pearson r, Spearman rho (pooled), mean weekly Spearman
  Regression / calibration  R² and the actual~projected OLS slope & intercept
                         (ideal: slope 1, intercept 0 — slope<1 = over-confident)
  Decision quality       Top-1 hit rate (called the single most-picked team),
                         Top-3 overlap (share of the week's 3 most-picked teams
                         it identified)
  Head-to-head           % of weeks each flavor wins on MAE and on Spearman
  Coverage               team-week observations, number of weeks

OUTPUTS (committed, under entry-analytics/):
  • flavor_accuracy.csv          -- per (contest, year, week, flavor)
  • flavor_accuracy_summary.csv  -- pooled per (contest, flavor), all measures
  • flavor_accuracy_report.txt   -- ranked verdict per contest
  • flavor_calibration_{contest}.png -- predicted-vs-actual scatter per flavor
  • flavor_metrics_{contest}.png     -- error & skill bars per flavor

Usage:
  python weekly_9_flavor_accuracy.py                 # all years, all contests
  python weekly_9_flavor_accuracy.py --years 2021 2022 2023 2024 2025
  python weekly_9_flavor_accuracy.py --contests circa
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, pearsonr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

import contest_config as cc

FINAL_DIR = 'nfl-power-ratings/final_data/{year}_final_data'
WEEK_RE = re.compile(r'Week_(\d+)_(\d+)_Final_Data\.csv$')
OUT_DIR = 'entry-analytics'
FLAVORS = ['Predicted', 'Top Down', 'Archetype', 'Blend']
MIN_TEAMS = 5

FLAVOR_COLOR = {'Predicted': '#2a78d6', 'Top Down': '#eb6834',
                'Archetype': '#1baf7a', 'Blend': '#eda100'}

_TEAM_CANON = {'LA': 'LAR', 'JAC': 'JAX', 'WSH': 'WAS', 'WFT': 'WAS'}


def _canon(t):
    t = str(t).strip()
    return _TEAM_CANON.get(t, t)


def _spearman(a, b):
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float('nan')
    if _HAVE_SCIPY:
        return float(spearmanr(a, b).correlation)
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def _pearson(a, b):
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float('nan')
    if _HAVE_SCIPY:
        return float(pearsonr(a, b)[0])
    return float(np.corrcoef(a, b)[0, 1])


def _week_col(df):
    return 'Week_x' if 'Week_x' in df.columns else ('Week' if 'Week' in df.columns else None)


def _long(df, home_col, away_col, name):
    h = df[['Home Team', home_col]].rename(columns={'Home Team': 'team', home_col: name})
    a = df[['Away Team', away_col]].rename(columns={'Away Team': 'team', away_col: name})
    out = pd.concat([h, a], ignore_index=True)
    out['team'] = out['team'].map(_canon)
    out[name] = pd.to_numeric(out[name], errors='coerce')
    return out


def collect_pairs(contest, years):
    """Team-level frame: one row per (year, week, team) with each flavor's
    projection and the actual, for one contest."""
    flav_cols = {fl: cc.flavor_cols(fl, contest) for fl in FLAVORS}
    act_cols = cc.flavor_cols('Actual', contest)
    frames = []
    for year in years:
        for path in glob.glob(os.path.join(FINAL_DIR.format(year=year),
                                           f'Week_*_{year}_Final_Data.csv')):
            m = WEEK_RE.search(path)
            if not m:
                continue
            week = int(m.group(1))
            df = pd.read_csv(path, low_memory=False)
            if 'Home Team' not in df.columns or act_cols[0] not in df.columns:
                continue
            base = _long(df, act_cols[0], act_cols[1], 'actual').dropna(subset=['actual'])
            if base.empty:
                continue
            merged = base.groupby('team', as_index=False)['actual'].mean()
            for fl, (hc, ac) in flav_cols.items():
                if hc in df.columns and ac in df.columns:
                    fl_long = _long(df, hc, ac, fl).groupby('team', as_index=False)[fl].mean()
                    merged = merged.merge(fl_long, on='team', how='left')
            merged['year'] = year
            merged['week'] = week
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out['contest'] = contest
    return out


def per_week_metrics(pairs):
    rows = []
    for (contest, year, week), g in pairs.groupby(['contest', 'year', 'week']):
        actual = g['actual'].to_numpy()
        for fl in FLAVORS:
            if fl not in g.columns:
                continue
            sub = g[[fl, 'actual']].dropna()
            if len(sub) < MIN_TEAMS or sub['actual'].abs().sum() == 0:
                continue
            p = sub[fl].to_numpy()
            a = sub['actual'].to_numpy()
            err = p - a
            top1 = int(np.argmax(p) == np.argmax(a))
            k = min(3, len(a))
            t_p = set(np.argsort(-p)[:k])
            t_a = set(np.argsort(-a)[:k])
            top3 = len(t_p & t_a) / k
            rows.append({
                'contest': contest, 'year': int(year), 'week': int(week), 'flavor': fl,
                'n_teams': len(a),
                'mae': round(float(np.mean(np.abs(err))), 6),
                'rmse': round(float(np.sqrt(np.mean(err ** 2))), 6),
                'me_bias': round(float(np.mean(err)), 6),
                'spearman': round(_spearman(p, a), 4),
                'top1_hit': top1, 'top3_overlap': round(top3, 4),
            })
    return pd.DataFrame(rows)


def summarize(pairs, perweek):
    srows = []
    for contest in sorted(pairs['contest'].unique()):
        cp = pairs[pairs['contest'] == contest]
        cw = perweek[perweek['contest'] == contest]
        for fl in FLAVORS:
            if fl not in cp.columns:
                continue
            sub = cp[[fl, 'actual']].dropna()
            if sub.empty:
                continue
            p = sub[fl].to_numpy()
            a = sub['actual'].to_numpy()
            err = p - a
            r = _pearson(p, a)
            # calibration: OLS actual ~ projected
            if len(p) >= 2 and len(np.unique(p)) > 1:
                slope, intercept = np.polyfit(p, a, 1)
            else:
                slope = intercept = float('nan')
            fw = cw[cw['flavor'] == fl]
            srows.append({
                'contest': contest, 'flavor': fl,
                'n_obs': len(a), 'n_weeks': int(fw.shape[0]),
                'MAE': round(float(np.mean(np.abs(err))), 6),
                'RMSE': round(float(np.sqrt(np.mean(err ** 2))), 6),
                'MdAE': round(float(np.median(np.abs(err))), 6),
                'ME_bias': round(float(np.mean(err)), 6),
                'Pearson_r': round(r, 4) if pd.notna(r) else None,
                'Spearman_pooled': round(_spearman(p, a), 4),
                'Spearman_weekly': round(float(fw['spearman'].mean()), 4) if not fw.empty else None,
                'R2': round(r ** 2, 4) if pd.notna(r) else None,
                'calib_slope': round(float(slope), 4) if pd.notna(slope) else None,
                'calib_intercept': round(float(intercept), 6) if pd.notna(intercept) else None,
                'Brier': round(float(np.mean(err ** 2)), 6),
                'Top1_hit_rate': round(float(fw['top1_hit'].mean()), 4) if not fw.empty else None,
                'Top3_overlap': round(float(fw['top3_overlap'].mean()), 4) if not fw.empty else None,
            })
    summ = pd.DataFrame(srows)

    # head-to-head weekly win rates (lowest MAE / highest Spearman per week)
    for metric, better, col in [('mae', 'min', 'winrate_MAE'),
                                ('spearman', 'max', 'winrate_Spearman')]:
        wins = {}
        for (contest, year, week), g in perweek.groupby(['contest', 'year', 'week']):
            gg = g.dropna(subset=[metric])
            if gg.empty:
                continue
            idx = gg[metric].idxmin() if better == 'min' else gg[metric].idxmax()
            win_fl = gg.loc[idx, 'flavor']
            wins.setdefault(contest, {}).setdefault('tot', 0)
            wins[contest]['tot'] += 1
            wins[contest][win_fl] = wins[contest].get(win_fl, 0) + 1
        vals = []
        for _, row in summ.iterrows():
            w = wins.get(row['contest'], {})
            tot = w.get('tot', 0)
            vals.append(round(w.get(row['flavor'], 0) / tot, 4) if tot else None)
        summ[col] = vals
    return summ


def build_report(summ):
    lines = ["PICK-% FLAVOR ACCURACY  —  which projection column is most accurate",
             "=" * 74,
             "MAE/RMSE/Brier: lower better. Spearman/Top-k/winrate: higher better.",
             "calib_slope: 1.0 = well-calibrated; < 1 = projections too extreme.\n"]
    for contest in sorted(summ['contest'].unique()):
        cb = summ[summ['contest'] == contest].copy()
        lines.append(f"\n{'#'*74}\n## {contest.upper()}\n{'#'*74}")
        # rank by the two headline metrics
        by_mae = cb.sort_values('MAE')['flavor'].tolist()
        by_sp = cb.sort_values('Spearman_weekly', ascending=False)['flavor'].tolist()
        lines.append(f"  most accurate by MAE:            {' > '.join(by_mae)}")
        lines.append(f"  best ranking by weekly Spearman: {' > '.join(by_sp)}\n")
        for _, r in cb.sort_values('MAE').iterrows():
            lines.append(f"  {r['flavor']:<10} "
                         f"MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
                         f"bias={r['ME_bias']:+.4f}  "
                         f"Spearman(wk)={_fmt(r['Spearman_weekly'])}  "
                         f"R²={_fmt(r['R2'])}  calib={_fmt(r['calib_slope'])}")
            lines.append(f"  {'':<10} Top1={_fmt(r['Top1_hit_rate'])}  "
                         f"Top3={_fmt(r['Top3_overlap'])}  "
                         f"win%(MAE)={_fmt(r['winrate_MAE'])}  "
                         f"win%(Spear)={_fmt(r['winrate_Spearman'])}  "
                         f"n={int(r['n_obs'])} over {int(r['n_weeks'])} wks")
        # verdict
        if not cb.empty:
            best = cb.sort_values(['MAE']).iloc[0]['flavor']
            best_sp = cb.sort_values('Spearman_weekly', ascending=False).iloc[0]['flavor']
            if best == best_sp:
                lines.append(f"\n  → {best} is most accurate on both error and ranking.")
            else:
                lines.append(f"\n  → {best} has the lowest error; {best_sp} ranks teams best. "
                             f"Pick by what you need (absolute % vs who's popular).")
    return "\n".join(lines) + "\n"


def _fmt(v):
    return 'n/a' if v is None or (isinstance(v, float) and pd.isna(v)) else f'{v:.3f}'


def render(pairs, summ, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  ⚠️  matplotlib unavailable ({e}); skipping charts.")
        return []
    written = []
    for contest in sorted(pairs['contest'].unique()):
        cp = pairs[pairs['contest'] == contest]
        flavs = [f for f in FLAVORS if f in cp.columns and cp[f].notna().any()]
        if not flavs:
            continue
        # calibration scatters
        fig, axes = plt.subplots(1, len(flavs), figsize=(4.2 * len(flavs), 4.2),
                                 facecolor='#fcfcfb', squeeze=False)
        for ax, fl in zip(axes[0], flavs):
            sub = cp[[fl, 'actual']].dropna()
            p, a = sub[fl].to_numpy(), sub['actual'].to_numpy()
            lim = max(p.max(), a.max()) * 1.05 if len(p) else 1
            ax.set_facecolor('#fcfcfb')
            ax.scatter(p, a, s=8, alpha=0.25, color=FLAVOR_COLOR[fl], edgecolors='none', zorder=3)
            ax.plot([0, lim], [0, lim], '--', color='#b0afab', lw=1, zorder=2)  # y = x
            if len(p) >= 2 and len(np.unique(p)) > 1:
                s, b = np.polyfit(p, a, 1)
                ax.plot([0, lim], [b, b + s * lim], '-', color=FLAVOR_COLOR[fl], lw=2, zorder=4)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.tick_params(colors='#52514e')
            ax.set_title(fl, color='#0b0b0b', fontsize=11)
            ax.set_xlabel('projected pick %', color='#52514e')
            ax.set_ylabel('actual pick %', color='#52514e')
        fig.suptitle(f'{contest.title()} — calibration (projected vs actual; dashed = perfect)',
                     color='#0b0b0b', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p1 = os.path.join(out_dir, f'flavor_calibration_{contest}.png')
        fig.savefig(p1, dpi=150, facecolor='#fcfcfb', bbox_inches='tight')
        plt.close(fig)
        written.append(p1)

        # metrics bars
        cb = summ[summ['contest'] == contest].set_index('flavor').reindex(flavs)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor='#fcfcfb')
        x = np.arange(len(flavs))
        cols = [FLAVOR_COLOR[f] for f in flavs]
        axL.bar(x - 0.2, cb['MAE'], 0.38, label='MAE', color=cols, zorder=3)
        axL.bar(x + 0.2, cb['RMSE'], 0.38, label='RMSE', color=cols, alpha=0.5, zorder=3)
        axL.set_title('Error (lower = better)', color='#0b0b0b', fontsize=11)
        axL.set_xticks(x)
        axL.set_xticklabels(flavs)
        axL.legend(frameon=False)
        axR.bar(x - 0.2, cb['Spearman_weekly'], 0.38, label='Spearman (wk)', color=cols, zorder=3)
        axR.bar(x + 0.2, cb['Top1_hit_rate'], 0.38, label='Top-1 hit', color=cols, alpha=0.5, zorder=3)
        axR.set_title('Skill (higher = better)', color='#0b0b0b', fontsize=11)
        axR.set_xticks(x)
        axR.set_xticklabels(flavs)
        axR.legend(frameon=False)
        for ax in (axL, axR):
            ax.set_facecolor('#fcfcfb')
            ax.grid(True, axis='y', color='#e6e5e1', lw=0.8, zorder=0)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors='#52514e')
        fig.suptitle(f'{contest.title()} — flavor accuracy', color='#0b0b0b',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p2 = os.path.join(out_dir, f'flavor_metrics_{contest}.png')
        fig.savefig(p2, dpi=150, facecolor='#fcfcfb', bbox_inches='tight')
        plt.close(fig)
        written.append(p2)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--years', nargs='*', type=int, default=None)
    ap.add_argument('--contests', nargs='*', default=None)
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()

    contests = args.contests or list(cc.CONTESTS.keys())
    if args.years:
        years = args.years
    else:
        years = sorted({int(WEEK_RE.search(p).group(2))
                        for p in glob.glob('nfl-power-ratings/final_data/*/Week_*_Final_Data.csv')
                        if WEEK_RE.search(p)})
    if not years:
        print("❌ No per-week Final_Data files found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"🔎 Flavor accuracy — contests={contests} years={years}")
    all_pairs = []
    for contest in contests:
        p = collect_pairs(contest, years)
        if not p.empty:
            all_pairs.append(p)
            print(f"  📊 {contest}: {len(p)} team-week rows across "
                  f"{p.groupby(['year','week']).ngroups} weeks")
        else:
            print(f"  ⏭️  {contest}: no usable Week_*_Final_Data files.")
    if not all_pairs:
        print("❌ No data.")
        return
    pairs = pd.concat(all_pairs, ignore_index=True)
    perweek = per_week_metrics(pairs)
    summ = summarize(pairs, perweek)

    perweek.sort_values(['contest', 'year', 'week', 'flavor']).to_csv(
        os.path.join(args.out_dir, 'flavor_accuracy.csv'), index=False)
    summ.to_csv(os.path.join(args.out_dir, 'flavor_accuracy_summary.csv'), index=False)
    report = build_report(summ)
    with open(os.path.join(args.out_dir, 'flavor_accuracy_report.txt'), 'w') as f:
        f.write(report)
    charts = render(pairs, summ, args.out_dir)

    print(report)
    print(f"💾 wrote {args.out_dir}/flavor_accuracy.csv ({len(perweek)} rows)")
    print(f"💾 wrote {args.out_dir}/flavor_accuracy_summary.csv")
    print(f"💾 wrote {args.out_dir}/flavor_accuracy_report.txt")
    for c in charts:
        print(f"🖼️  wrote {c}")


if __name__ == '__main__':
    main()
