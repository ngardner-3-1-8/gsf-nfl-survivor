"""
ablate_pick_model.py

Runs the 2025 backtest under a series of configurations, toggling each new
model layer off in turn, to isolate which layer is responsible for the
aggregate MAE regression. Prints a comparison table at the end.

Usage:
    python ablate_pick_model.py            # 2025
    python ablate_pick_model.py 2024
"""

import sys
import numpy as np
import pandas as pd

import entry_analytics as ea
import backtest_pick_predictions as bt


CONFIGS = [
    # name,                    n_iter, diversify, fe_scale, priors
    ("FULL (current)",              3,     2.0,      1.0,   True),
    ("no self-consistency",         1,     2.0,      1.0,   True),
    ("no multi-entry coord",        3,     0.0,      1.0,   True),
    ("no team FE",                  3,     2.0,      0.0,   True),
    ("no contestant priors",        3,     2.0,      1.0,   False),
    ("baseline (all new OFF)",      1,     0.0,      0.0,   False),
]


def run(year):
    rows = []
    for name, n_iter, div, fes, priors in CONFIGS:
        print(f"\n{'='*60}\nCONFIG: {name}\n{'='*60}")
        ea.SELF_CONSISTENCY_ITERS = n_iter
        ea.DIVERSIFY_COEFF = div
        ea.FE_SCALE = fes
        ea.USE_CONTESTANT_PRIORS = priors

        bt.run_backtest(year)

        df = pd.read_csv(f"entry-analytics/backtest_{year}_results.csv")
        mae = np.abs(df["behavioral"] - df["actual"]).mean()
        rmse = np.sqrt(((df["behavioral"] - df["actual"]) ** 2).mean())
        late = df[df["week"] >= 8]
        late_mae = np.abs(late["behavioral"] - late["actual"]).mean()
        ll = df.dropna(subset=["entry_logloss"]).groupby("week").first()
        rows.append({
            "config": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "late_MAE(w8+)": round(late_mae, 4),
            "entry_LL": round(ll["entry_logloss"].mean(), 3) if len(ll) else None,
            "top1%": round(ll["entry_top1"].mean() * 100, 1) if len(ll) else None,
        })

    out = pd.DataFrame(rows)
    print(f"\n{'='*60}\nABLATION SUMMARY — {year}\n{'='*60}")
    print(out.to_string(index=False))
    topdown_mae = np.abs(df["topdown"] - df["actual"]).mean()
    print(f"\n  (top-down reference MAE: {topdown_mae:.4f})")
    out.to_csv(f"entry-analytics/ablation_{year}.csv", index=False)


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    run(year)
