#!/usr/bin/env python3
"""
analyze_pick_pct_sweep.py

Reads logs/pick_pct_model_metrics.csv and ranks the pick-% feature-count
configs against each other, per contest, so you can pick the best `top_n`
empirically instead of by intuition.

It focuses on the rows written by the daily_2 sweep (model_label like
`<contest>_pick_pct_sweep_top<N>`, produced when SWEEP_PICK_PCT_MODELS=1), and
also summarises the deployed models (`<contest>_pick_pct_model_primary` /
`..._nopublic`) if they're present.

For each (contest, config) it averages, across every logged run/week:
  • mean_weekly_spearman  – did it rank the right teams as popular? (higher is
                            better; this is what actually drives EV/optimizer)
  • mae_pct_points        – how far off the absolute pick % was, in points
                            (lower is better)

The winner per contest is the config with the best average spearman, ties
broken by lower average MAE. Run it after one or more sweep runs:

    python analyze_pick_pct_sweep.py
    python analyze_pick_pct_sweep.py --metrics logs/pick_pct_model_metrics.csv
    python analyze_pick_pct_sweep.py --min-runs 2   # ignore configs with <2 weeks

Nothing here re-trains anything; it only reads the metrics log.
"""

import argparse
import os
import sys

import pandas as pd

SWEEP_MARKER = '_pick_pct_sweep_top'
MODEL_MARKER = '_pick_pct_model_'


def _parse_label(label):
    """(contest, config_key, is_sweep) from a model_label, or (None, None, None).

    Sweep labels:  '<contest>_pick_pct_sweep_top<N>'  -> (contest, 'top<N>', True)
    Deployed:      '<contest>_pick_pct_model_primary' -> (contest, 'primary', False)
    Contest names may contain underscores (big_splash, world_championship), so
    we split on the marker rather than on '_'.
    """
    label = str(label)
    if SWEEP_MARKER in label:
        contest, n = label.split(SWEEP_MARKER, 1)
        return contest, f'top{n}', True
    if MODEL_MARKER in label:
        contest, key = label.split(MODEL_MARKER, 1)
        return contest, key, False
    return None, None, None


def _sort_key(config_key):
    """Order configs sensibly: numeric top_n ascending, 'topall' last,
    deployed models after the sweep."""
    if config_key.startswith('top'):
        tail = config_key[3:]
        if tail == 'all':
            return (0, 10 ** 9)
        try:
            return (0, int(tail))
        except ValueError:
            return (0, 10 ** 8)
    return (1, config_key)  # primary / nopublic after the sweep block


def summarise(df, min_runs=1):
    rows = []
    for label, grp in df.groupby('model_label'):
        contest, config_key, is_sweep = _parse_label(label)
        if contest is None:
            continue
        g = grp.copy()
        for col in ('mean_weekly_spearman', 'mae_pct_points', 'n_features'):
            g[col] = pd.to_numeric(g.get(col), errors='coerce')
        sp = g['mean_weekly_spearman'].dropna()
        mae = g['mae_pct_points'].dropna()
        n_runs = int(max(len(sp), len(mae)))
        if n_runs < min_runs:
            continue
        rows.append({
            'contest': contest,
            'config': config_key,
            'is_sweep': is_sweep,
            'n_runs': n_runs,
            'avg_spearman': round(sp.mean(), 4) if len(sp) else float('nan'),
            'avg_mae_pts': round(mae.mean(), 4) if len(mae) else float('nan'),
            'avg_n_features': round(g['n_features'].dropna().mean(), 1)
                              if g['n_features'].notna().any() else float('nan'),
        })
    return pd.DataFrame(rows)


def _fmt(v):
    return 'n/a' if pd.isna(v) else f'{v:.4f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--metrics', default='logs/pick_pct_model_metrics.csv',
                    help='path to the metrics CSV (default: %(default)s)')
    ap.add_argument('--min-runs', type=int, default=1,
                    help='ignore configs logged fewer than this many times')
    args = ap.parse_args()

    if not os.path.exists(args.metrics):
        print(f"❌ No metrics file at {args.metrics}. Run daily_2 with "
              f"SWEEP_PICK_PCT_MODELS=1 first.")
        sys.exit(1)

    df = pd.read_csv(args.metrics)
    if 'model_label' not in df.columns:
        print(f"❌ {args.metrics} has no 'model_label' column — is it the right file?")
        sys.exit(1)

    summary = summarise(df, min_runs=args.min_runs)
    if summary.empty:
        print(f"❌ No usable rows in {args.metrics} (min-runs={args.min_runs}). "
              f"Have you run a sweep yet?")
        sys.exit(1)

    for contest in sorted(summary['contest'].unique()):
        block = summary[summary['contest'] == contest].copy()
        block['_ord'] = block['config'].map(_sort_key)
        block = block.sort_values('_ord')

        print(f"\n{'=' * 68}")
        print(f"  {contest.upper()}  —  pick-% feature-count comparison")
        print(f"{'=' * 68}")
        print(f"  {'config':<12}{'runs':>6}{'avg spearman':>15}"
              f"{'avg MAE pts':>14}{'~features':>12}")
        print(f"  {'-' * 62}")
        for _, r in block.iterrows():
            tag = '' if r['is_sweep'] else '  (deployed)'
            print(f"  {r['config']:<12}{r['n_runs']:>6}"
                  f"{_fmt(r['avg_spearman']):>15}{_fmt(r['avg_mae_pts']):>14}"
                  f"{_fmt(r['avg_n_features']):>12}{tag}")

        # Winner = best avg spearman among SWEEP configs, ties -> lower MAE.
        sweep = block[block['is_sweep'] & block['avg_spearman'].notna()].copy()
        if not sweep.empty:
            sweep = sweep.sort_values(['avg_spearman', 'avg_mae_pts'],
                                      ascending=[False, True])
            win = sweep.iloc[0]
            print(f"  {'-' * 62}")
            print(f"  🏆 best config: {win['config']} "
                  f"(spearman {_fmt(win['avg_spearman'])}, "
                  f"MAE {_fmt(win['avg_mae_pts'])} pts over {win['n_runs']} run(s))")
            if len(sweep) > 1:
                nxt = sweep.iloc[1]
                d_sp = win['avg_spearman'] - nxt['avg_spearman']
                if d_sp < 0.005:
                    print(f"     ↳ within 0.005 spearman of {nxt['config']} — "
                          f"prefer the smaller/simpler config.")

    print(f"\nRead: higher spearman = better ranking of who's popular (drives EV); "
          f"lower MAE = closer absolute pick %.")
    print(f"If top15 ties or beats the larger configs, 15 is the sweet spot. "
          f"If a larger config clearly wins, raise PRIMARY_TOP_N in daily_2.\n")


if __name__ == '__main__':
    main()
