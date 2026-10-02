#!/usr/bin/env python3
"""
weekly_7_analyze_projection_horizon.py

How does a pick-% projection decay as the forecast horizon grows? When you're in
Week 2, the model already projects Weeks 2..18 -- but a Week-18 projection made
in Week 2 has far less information than one made in Week 17. This script
quantifies that decay so you can say, concretely, "1 week out the projection is
sharp; 4+ weeks out it's murky."

DATA SOURCES (both already committed -- no new logging needed):
  • Projection snapshots: nfl-power-ratings/final_sim_results_with_variance_week_{W}_{year}.csv
      Each file is the pipeline's view AS OF week W, carrying projected pick %
      for every week T >= W. The projection made in week W for week T is a
      forecast at horizon h = T - W.
  • Ground truth: nfl-power-ratings/final_data/{year}_final_data/
      Season_{year}_Through_Week_{N}_Final_Data.csv  (the highest N available --
      the most complete actuals for that season), carrying the real
      'Actual Pick %' for every week.

For each (contest, year, as-of week W, target week T) it pairs the projection
from the W-snapshot with the actual for T and, across that week's teams,
computes MAE (points), rank-correlation (Spearman) and mean signed bias. Then it
rolls those up by horizon and fits the decay:

  • decay slope       -- points of MAE added per extra week of horizon
  • skill horizon     -- the largest horizon where Spearman still clears a
                         threshold (default 0.5); beyond it, the ordering is
                         unreliable ("murky")
  • skill half-life    -- fit Spearman(h) = s0 * exp(-h/tau); half-life = tau*ln2,
                         a single "how fast does it rot" number per flavor

FLAVORS: Predicted (deployed blend), Top Down (pure projection), Archetype
(behavioral). Old snapshots (pre-flavor-migration) only carry the deployed
'Home/Away Pick %' -> mapped to Predicted; the other flavors light up once you
re-run backfills with the flavor-writing pipeline. Each flavor is resolved with
a legacy-name fallback and simply skipped where its column is absent.

OUTPUTS (all under entry-analytics/, committed):
  • flavor_horizon_accuracy.csv -- one row per (contest, year, as_of, target, flavor)
  • horizon_curve.csv           -- rolled up per (contest, flavor, horizon)
  • horizon_report.txt          -- the fitted trend metrics, human-readable
  • horizon_{contest}.png       -- MAE and Spearman vs horizon, one line per flavor

Usage:
  python weekly_7_analyze_projection_horizon.py                 # all years, all contests
  python weekly_7_analyze_projection_horizon.py --years 2020 2021
  python weekly_7_analyze_projection_horizon.py --skill-threshold 0.5
"""

import argparse
import glob
import math
import os
import re

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, linregress
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

import contest_config as cc

SIM_DIR = 'nfl-power-ratings'
SIM_GLOB = 'final_sim_results_with_variance_week_*_{year}.csv'
SIM_RE = re.compile(r'final_sim_results_with_variance_week_(\d+)_(\d+)\.csv$')
FINAL_DIR = 'nfl-power-ratings/final_data/{year}_final_data'
OUT_DIR = 'entry-analytics'

MIN_TEAMS = 5          # need at least this many matched teams in a week to score it
SKILL_THRESHOLD = 0.5  # Spearman floor that defines the "skill horizon"

# Flavor -> ordered (home_col, away_col) candidates. First present pair wins;
# {tag} is the contest's col_tag (Circa / Big Splash / World Championship). The
# second entry in each list is the legacy (pre-flavor-migration) column name.
def _flavor_proj_candidates(tag):
    return {
        'Predicted': [(f'Predicted {tag} Home Pick %', f'Predicted {tag} Away Pick %'),
                      ('Home Pick %', 'Away Pick %')],
        'Top Down':  [(f'Top Down {tag} Home Pick %', f'Top Down {tag} Away Pick %'),
                      ('Topdown Home Pick %', 'Topdown Away Pick %')],
        'Archetype': [(f'Archetype {tag} Home Pick %', f'Archetype {tag} Away Pick %')],
    }


def _actual_candidates(tag):
    return [(f'Actual {tag} Home Pick %', f'Actual {tag} Away Pick %'),
            ('Home Actual Pick %', 'Away Actual Pick %')]


# Fixed flavor colors (dataviz categorical slots 1/2/3, light surface).
FLAVOR_COLOR = {'Predicted': '#2a78d6', 'Top Down': '#eb6834', 'Archetype': '#1baf7a'}
FLAVOR_ORDER = ['Predicted', 'Top Down', 'Archetype']

_TEAM_CANON = {'LA': 'LAR', 'JAC': 'JAX', 'WSH': 'WAS', 'WFT': 'WAS'}


def _canon(team):
    t = str(team).strip()
    return _TEAM_CANON.get(t, t)


def _week_col(df):
    for c in ('Week_x', 'Week'):
        if c in df.columns:
            return c
    return None


def _first_present(df, pairs):
    """First (home, away) column pair that's actually in df, else None."""
    for home, away in pairs:
        if home in df.columns and away in df.columns:
            return home, away
    return None


def _to_team_level(df, wkcol, home_col, away_col):
    """Long frame: one row per (week, team) with a 'value' from the home/away col."""
    h = df[[wkcol, 'Home Team', home_col]].rename(
        columns={'Home Team': 'Team', home_col: 'value'})
    a = df[[wkcol, 'Away Team', away_col]].rename(
        columns={'Away Team': 'Team', away_col: 'value'})
    out = pd.concat([h, a], ignore_index=True)
    out['Team'] = out['Team'].map(_canon)
    out['value'] = pd.to_numeric(out['value'], errors='coerce')
    out = out.rename(columns={wkcol: 'week'})
    out['week'] = pd.to_numeric(out['week'], errors='coerce')
    return out.dropna(subset=['week', 'Team'])


def _reduce_actual(long):
    """Collapse a team-level long frame to one actual per (week, team)."""
    long = long.dropna(subset=['value']).rename(columns={'value': 'actual'})
    return long.groupby(['week', 'Team'], as_index=False)['actual'].mean()


def _truth_from_final_data(year, contest_tag):
    """Actuals from the final_data files (the source you asked for). Prefers the
    highest-week Season-Through file that actually carries the actual-pick
    columns; else stitches together the per-week Week_{T}_{year} files that do.
    None when no final_data file for this year carries actuals."""
    d = FINAL_DIR.format(year=year)
    cands = _actual_candidates(contest_tag)

    season = glob.glob(os.path.join(d, f'Season_{year}_Through_Week_*_Final_Data.csv'))
    have = []
    for p in season:
        m = re.search(r'Through_Week_(\d+)_Final_Data', p)
        cols = pd.read_csv(p, nrows=0).columns
        if m and _first_present_cols(cols, cands):
            have.append((int(m.group(1)), p))
    if have:
        _, path = max(have, key=lambda t: t[0])
        df = pd.read_csv(path, low_memory=False)
        wk = _week_col(df)
        pair = _first_present(df, cands)
        if wk and pair:
            return _reduce_actual(_to_team_level(df, wk, pair[0], pair[1])), os.path.basename(path)

    weekly = glob.glob(os.path.join(d, f'Week_*_{year}_Final_Data.csv'))
    frames = []
    for p in weekly:
        cols = pd.read_csv(p, nrows=0).columns
        pair = _first_present_cols(cols, cands)
        if not pair:
            continue
        df = pd.read_csv(p, low_memory=False)
        wk = _week_col(df)
        if wk:
            frames.append(_to_team_level(df, wk, pair[0], pair[1]))
    if frames:
        return _reduce_actual(pd.concat(frames, ignore_index=True)), \
            f'{year} per-week Final_Data files'
    return None


def _truth_from_historical(year, contest):
    """Fallback: the contest's historical training file, whose 'Pick %' column IS
    the observed official pick % (same 0–1 scale as the projections). Full
    coverage across every season, so this is what fills years the final_data
    files don't carry actuals for."""
    path = cc.CONTESTS[contest].get('historical_csv')
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    if not {'Year', 'Team', 'Pick %'}.issubset(df.columns):
        return None
    wk = 'Week' if 'Week' in df.columns else ('Date' if 'Date' in df.columns else None)
    if wk is None:
        return None
    sub = df[pd.to_numeric(df['Year'], errors='coerce') == year].copy()
    if sub.empty:
        return None
    long = pd.DataFrame({
        'week': pd.to_numeric(sub[wk], errors='coerce'),
        'Team': sub['Team'].map(_canon),
        'value': pd.to_numeric(sub['Pick %'], errors='coerce'),
    }).dropna(subset=['week', 'Team'])
    return _reduce_actual(long), os.path.basename(path)


def _first_present_cols(cols, pairs):
    cols = set(cols)
    for home, away in pairs:
        if home in cols and away in cols:
            return home, away
    return None


def load_truth(year, contest):
    """(target_week, team) -> actual pick %. Tries the final_data files first
    (per the requested source), then the historical file for full coverage."""
    tag = cc.CONTESTS[contest]['col_tag']
    return _truth_from_final_data(year, tag) or _truth_from_historical(year, contest)


def score_week(proj_long, truth_long, target_week):
    """MAE / Spearman / bias / n for one target week, across matched teams."""
    p = proj_long[proj_long['week'] == target_week][['Team', 'value']].dropna()
    merged = p.merge(truth_long[truth_long['week'] == target_week][['Team', 'actual']],
                     on='Team', how='inner').dropna()
    n = len(merged)
    if n < MIN_TEAMS:
        return None
    err = merged['value'] - merged['actual']
    mae = float(err.abs().mean())
    bias = float(err.mean())
    sp = float('nan')
    if merged['value'].nunique() > 1 and merged['actual'].nunique() > 1:
        if _HAVE_SCIPY:
            sp, _ = spearmanr(merged['value'], merged['actual'])
        else:
            sp = merged['value'].rank().corr(merged['actual'].rank())
    return {'n_teams': n, 'mae_pct_points': round(mae, 5),
            'mean_bias': round(bias, 5),
            'spearman': round(float(sp), 5) if pd.notna(sp) else None}


def build_accuracy_rows(years, contests):
    rows = []
    for contest in contests:
        tag = cc.CONTESTS[contest]['col_tag']
        start = int(cc.CONTESTS[contest].get('start_season', 0))
        proj_map = _flavor_proj_candidates(tag)
        for year in years:
            if year < start:
                continue
            truth = load_truth(year, contest)
            if truth is None:
                print(f"  ⏭️  {contest} {year}: no usable actuals (final_data or historical).")
                continue
            truth_long, truth_name = truth
            sims = sorted(glob.glob(os.path.join(SIM_DIR, SIM_GLOB.format(year=year))),
                          key=lambda p: int(SIM_RE.search(p).group(1)) if SIM_RE.search(p) else 0)
            if not sims:
                print(f"  ⏭️  {contest} {year}: no sim snapshot files.")
                continue
            print(f"  📈 {contest} {year}: {len(sims)} snapshot(s) vs {truth_name}")
            for sp_path in sims:
                m = SIM_RE.search(sp_path)
                as_of = int(m.group(1))
                snap = pd.read_csv(sp_path, low_memory=False)
                wk = _week_col(snap)
                if wk is None or 'Home Team' not in snap.columns:
                    continue
                for flavor in FLAVOR_ORDER:
                    pair = _first_present(snap, proj_map[flavor])
                    if pair is None:
                        continue
                    proj_long = _to_team_level(snap, wk, pair[0], pair[1])
                    for target in sorted(proj_long['week'].dropna().unique()):
                        target = int(target)
                        if target < as_of:
                            continue  # only forward (and same-week) forecasts
                        s = score_week(proj_long, truth_long, target)
                        if s is None:
                            continue
                        rows.append({'contest': contest, 'year': int(year),
                                     'as_of_week': as_of, 'target_week': target,
                                     'horizon': target - as_of, 'flavor': flavor,
                                     **s})
    return pd.DataFrame(rows)


def _fit_halflife(horizons, skills):
    """Half-life (weeks) from Spearman(h) = s0*exp(-h/tau). NaN if unfittable."""
    h = np.asarray(horizons, float)
    s = np.asarray(skills, float)
    ok = np.isfinite(h) & np.isfinite(s) & (s > 0)
    if ok.sum() < 3:
        return float('nan')
    h, s = h[ok], s[ok]
    if np.ptp(h) == 0:
        return float('nan')
    try:
        # log-linear fit: ln(s) = ln(s0) - h/tau
        b = np.polyfit(h, np.log(s), 1)  # slope = -1/tau
        slope = b[0]
        if slope >= 0:      # skill not decaying -> no finite half-life
            return float('inf')
        tau = -1.0 / slope
        return round(tau * math.log(2), 2)
    except Exception:
        return float('nan')


def build_curve_and_report(acc, skill_threshold):
    if acc.empty:
        return pd.DataFrame(), "No horizon observations were produced.\n"
    curve = (acc.groupby(['contest', 'flavor', 'horizon'])
             .agg(n_obs=('mae_pct_points', 'size'),
                  mae_pct_points=('mae_pct_points', 'mean'),
                  spearman=('spearman', 'mean'),
                  mean_bias=('mean_bias', 'mean'))
             .reset_index().round(5))

    lines = []
    lines.append("PROJECTION HORIZON DECAY  —  how pick-% accuracy fades with weeks-ahead")
    lines.append("=" * 74)
    lines.append(f"skill horizon = largest horizon with mean Spearman >= {skill_threshold}")
    lines.append("MAE in pick-% points (lower better); Spearman in [-1,1] (higher better).\n")

    for contest in sorted(curve['contest'].unique()):
        cblock = curve[curve['contest'] == contest]
        lines.append(f"\n{'#' * 74}\n## {contest.upper()}\n{'#' * 74}")
        for flavor in FLAVOR_ORDER:
            fb = cblock[cblock['flavor'] == flavor].sort_values('horizon')
            if fb.empty:
                continue
            hz = fb['horizon'].to_numpy()
            mae = fb['mae_pct_points'].to_numpy()
            spv = fb['spearman'].to_numpy()
            # decay slope of MAE vs horizon
            if _HAVE_SCIPY and len(hz) >= 2 and np.ptp(hz) > 0:
                lr = linregress(hz, mae)
                slope, intercept, r = lr.slope, lr.intercept, lr.rvalue
            elif len(hz) >= 2 and np.ptp(hz) > 0:
                slope, intercept = np.polyfit(hz, mae, 1)
                r = float('nan')
            else:
                slope = intercept = r = float('nan')
            valid_sp = fb.dropna(subset=['spearman'])
            skill_h = (int(valid_sp[valid_sp['spearman'] >= skill_threshold]['horizon'].max())
                       if (valid_sp['spearman'] >= skill_threshold).any() else None)
            half = _fit_halflife(valid_sp['horizon'], valid_sp['spearman'])

            lines.append(f"\n  {flavor}:")
            lines.append(f"    horizons {int(hz.min())}–{int(hz.max())} "
                         f"({int(fb['n_obs'].sum())} week-observations)")
            lines.append(f"    MAE @ horizon 0 : "
                         f"{mae[hz == hz.min()][0]:.3f} pts" if len(mae) else "    MAE: n/a")
            lines.append(f"    decay slope     : {slope:+.4f} pts per week of horizon"
                         + (f"  (R={r:.2f})" if pd.notna(r) else ""))
            lines.append(f"    skill horizon   : "
                         + (f"{skill_h} weeks (Spearman>= {skill_threshold})"
                            if skill_h is not None else f"never reaches {skill_threshold}"))
            lines.append(f"    skill half-life : "
                         + ("n/a" if not np.isfinite(half) else
                            ("no decay" if half == float('inf') else f"{half} weeks")))
        # simple crossover note: best flavor at h=0 vs at max horizon
        piv = cblock.pivot_table(index='horizon', columns='flavor',
                                 values='mae_pct_points')
        if piv.shape[1] >= 2 and len(piv) >= 2:
            best_near = piv.iloc[0].idxmin()
            best_far = piv.iloc[-1].idxmin()
            if best_near != best_far:
                lines.append(f"\n  ↳ lowest MAE flips from '{best_near}' (near) to "
                             f"'{best_far}' (far) as horizon grows.")
    return curve, "\n".join(lines) + "\n"


def render_charts(curve, out_dir):
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor='#fcfcfb')
        for ax in (ax1, ax2):
            ax.set_facecolor('#fcfcfb')
            ax.grid(True, color='#e6e5e1', linewidth=0.8, zorder=0)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors='#52514e')
        for flavor in FLAVOR_ORDER:
            fb = cb[cb['flavor'] == flavor].sort_values('horizon')
            if fb.empty:
                continue
            col = FLAVOR_COLOR[flavor]
            ax1.plot(fb['horizon'], fb['mae_pct_points'], '-o', color=col,
                     linewidth=2, markersize=5, label=flavor, zorder=3)
            sp = fb.dropna(subset=['spearman'])
            ax2.plot(sp['horizon'], sp['spearman'], '-o', color=col,
                     linewidth=2, markersize=5, label=flavor, zorder=3)
        ax1.set_title('MAE vs horizon  (lower = better)', color='#0b0b0b', fontsize=11)
        ax1.set_xlabel('weeks ahead (horizon)', color='#52514e')
        ax1.set_ylabel('MAE (pick-% points)', color='#52514e')
        ax2.set_title('Rank-corr vs horizon  (higher = better)', color='#0b0b0b', fontsize=11)
        ax2.set_xlabel('weeks ahead (horizon)', color='#52514e')
        ax2.set_ylabel('Spearman', color='#52514e')
        ax2.axhline(SKILL_THRESHOLD, color='#b0afab', linestyle='--', linewidth=1, zorder=1)
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=len(handles),
                       frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f'{contest.title()} — pick-% projection decay by horizon',
                     color='#0b0b0b', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        path = os.path.join(out_dir, f'horizon_{contest}.png')
        fig.savefig(path, dpi=150, facecolor='#fcfcfb', bbox_inches='tight')
        plt.close(fig)
        written.append(path)
    return written


def main():
    global SKILL_THRESHOLD
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--years', nargs='*', type=int, default=None,
                    help='years to analyze (default: every year with sim snapshots)')
    ap.add_argument('--contests', nargs='*', default=None,
                    help='contest keys (default: all in contest_config)')
    ap.add_argument('--skill-threshold', type=float, default=SKILL_THRESHOLD)
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()

    SKILL_THRESHOLD = args.skill_threshold

    contests = args.contests or list(cc.CONTESTS.keys())
    if args.years:
        years = args.years
    else:
        years = sorted({int(SIM_RE.search(p).group(2))
                        for p in glob.glob(os.path.join(SIM_DIR, 'final_sim_results_with_variance_week_*_*.csv'))
                        if SIM_RE.search(p)})
    if not years:
        print("❌ No sim snapshot files found; nothing to analyze.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"🔎 Horizon analysis — contests={contests} years={years}")
    acc = build_accuracy_rows(years, contests)
    if acc.empty:
        print("❌ No matched projection/actual observations. "
              "(Actuals may not be populated yet for these weeks.)")
        return

    acc_path = os.path.join(args.out_dir, 'flavor_horizon_accuracy.csv')
    acc.sort_values(['contest', 'year', 'as_of_week', 'flavor', 'target_week']).to_csv(
        acc_path, index=False)
    curve, report = build_curve_and_report(acc, SKILL_THRESHOLD)
    curve_path = os.path.join(args.out_dir, 'horizon_curve.csv')
    curve.to_csv(curve_path, index=False)
    report_path = os.path.join(args.out_dir, 'horizon_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    charts = render_charts(curve, args.out_dir)

    print(report)
    print(f"💾 wrote {acc_path} ({len(acc)} rows)")
    print(f"💾 wrote {curve_path} ({len(curve)} rows)")
    print(f"💾 wrote {report_path}")
    for c in charts:
        print(f"🖼️  wrote {c}")


if __name__ == '__main__':
    main()
