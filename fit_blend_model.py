"""
fit_blend_model.py  (replaces fit_blend_alpha.py)

Fits a stacked ridge blend:  actual ≈ c0 + c1·behavioral + c2·topdown
                                     + c3·week + c4·log(alive)
from backtest results CSVs. Strictly generalizes the scalar alpha blend.

Also prints the per-band alphas for reference/fallback.

Reads:  entry-analytics/backtest_*_results.csv  (must include 'alive' column
        — regenerate backtests with the v3 script first)
Writes: entry-analytics/blend_model.json
"""

import os
import glob
import json
import numpy as np
import pandas as pd

RIDGE_LAMBDA = 1e-4


def fit():
    files = glob.glob("entry-analytics/backtest_*_results.csv")
    if not files:
        print("No backtest results found — run backtest_pick_predictions.py first")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "alive" not in df.columns:
        print("Backtest CSVs missing 'alive' column — regenerate with v3 backtest")
        return
    print(f"Fitting stacked blend on {len(df)} team-weeks "
          f"from {len(files)} file(s)")

    X = np.stack([df["behavioral"], df["topdown"],
                  df["week"].astype(float), np.log(df["alive"])], axis=1)
    y = df["actual"].values

    Xb = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    A = Xb.T @ Xb + RIDGE_LAMBDA * np.eye(Xb.shape[1])
    A[0, 0] -= RIDGE_LAMBDA          # don't penalize intercept
    coef = np.linalg.solve(A, Xb.T @ y)

    pred = np.clip(Xb @ coef, 0, None)
    mae_stack = np.abs(pred - y).mean()
    mae_b = np.abs(df["behavioral"] - y).mean()
    mae_t = np.abs(df["topdown"] - y).mean()
    print(f"  MAE — stacked: {mae_stack:.4f}   "
          f"behavioral: {mae_b:.4f}   topdown: {mae_t:.4f}")

    # Reference: best scalar alpha per band
    for lo, hi in [(2, 4), (5, 8), (9, 99)]:
        band = df[(df.week >= lo) & (df.week <= hi)]
        if band.empty:
            continue
        best_a, best_m = 0, np.inf
        for a in np.arange(0, 1.01, 0.05):
            m = np.abs(a * band.behavioral + (1 - a) * band.topdown
                       - band.actual).mean()
            if m < best_m:
                best_m, best_a = m, a
        print(f"  (reference α weeks {lo}-{hi}: {best_a:.2f}, MAE {best_m:.4f})")

    os.makedirs("entry-analytics", exist_ok=True)
    with open("entry-analytics/blend_model.json", "w") as f:
        json.dump({"intercept": float(coef[0]),
                   "coef": coef[1:].tolist(),
                   "features": ["behavioral", "topdown", "week", "log_alive"],
                   "lambda": RIDGE_LAMBDA,
                   "n_rows": len(df)}, f, indent=2)
    print("✅ Saved → entry-analytics/blend_model.json")


if __name__ == "__main__":
    fit()
