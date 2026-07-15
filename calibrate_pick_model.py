"""
calibrate_pick_model.py  (v3)

Conditional-logit fit of the pick-utility model on historical pick decisions.

v3 feature set (9 + 32 team fixed effects):
  base 6 (with profile interactions):  win, ev, pop, home, fv, hfv
  global 3 (no interactions):          win×stage, pop×stage, thursday
  team FE 32 (L2-regularized):         brand bias — marquee teams get
                                       over-picked beyond their numbers

  stage = log(alive / total entries) — the field's behavioral mix shifts
  as survivors self-select.

Output: entry-analytics/calibrated_weights.json
Usage:  python calibrate_pick_model.py 2022 2023 2024 2025
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from entry_analytics import (
    build_week_team_data, build_future_value, build_feature_tensors,
    build_entry_profiles, norm_abbr, TEAM_IDX, ALL_ABBRS,
    detect_holiday_weeks, build_holiday_future_value,
)

MAX_ENTRIES_PER_WEEK = 5000
FE_L2 = 0.01          # L2 penalty on team fixed effects
N_BASE = 6            # features with profile interactions
N_FEAT = 9            # total linear features
RNG = np.random.default_rng(7)


def load_season_final(year):
    files = glob.glob(f"nfl-power-ratings/final_data/{year}_final_data/"
                      f"Season_{year}_Through_Week_*_Final_Data.csv")
    if not files:
        return None
    def wk(p):
        try:
            return int(os.path.basename(p).split("_Week_")[1].split("_Final")[0])
        except (IndexError, ValueError):
            return 0
    return pd.read_csv(max(files, key=wk))


def collect_choice_data(years):
    X_list, chosen_list, dev_list, team_list = [], [], [], []

    for year in years:
        final_df = load_season_final(year)
        picks_path = f"circa-pick-history/{year}_survivor_picks.csv"
        if final_df is None or not os.path.exists(picks_path):
            print(f"   {year}: missing data — skipping")
            continue

        picks_df = pd.read_csv(picks_path)
        week_cols = [c for c in picks_df.columns if c.startswith("Week_")]
        max_week = len(week_cols)
        total_entries = len(picks_df)

        wtd = build_week_team_data(final_df, max_week + 1)
        fv = build_future_value(wtd, 1, max_week)
        holidays = detect_holiday_weeks(final_df)
        hfv = build_holiday_future_value(wtd, holidays, 1, max_week)
        feats = build_feature_tensors(wtd, fv, sorted(wtd.keys()),
                                      holiday_fv=hfv)

        n_year = 0
        for W in sorted(wtd.keys()):
            if W < 2:
                continue
            wcol = f"Week_{W}"
            if wcol not in picks_df.columns:
                continue

            trunc = picks_df.copy()
            for col in week_cols:
                if int(col.replace("Week_", "")) >= W:
                    trunc[col] = ""
            profiles = build_entry_profiles(trunc, wtd, list(range(1, W)))

            made_pick = picks_df[
                picks_df[wcol].notna()
                & (picks_df[wcol].astype(str).str.strip() != "")
                & (picks_df[wcol].astype(str).str.upper() != "ELIMINATED")
            ]
            n_alive = len(made_pick)
            stage = float(np.log(max(n_alive, 1) / max(total_entries, 1)))

            if n_alive > MAX_ENTRIES_PER_WEEK:
                made_pick = made_pick.sample(MAX_ENTRIES_PER_WEEK, random_state=7)

            fw = feats[W]
            playing = fw["plays"].astype(bool)
            base = np.stack([fw["win"], fw["ev"], fw["pop"],
                             fw["home"] - 0.5, -fw["fv"], -fw["hfv"]], axis=1)
            glob_f = np.stack([fw["win"] * stage, fw["pop"] * stage,
                               fw["thu"]], axis=1)
            feat_mat = np.concatenate([base, glob_f], axis=1)   # (32, 9)

            for _, row in made_pick.iterrows():
                pick = norm_abbr(row[wcol])
                if pick not in TEAM_IDX:
                    continue
                pick_idx = TEAM_IDX[pick]

                used = np.zeros(32, dtype=bool)
                for col in week_cols:
                    if int(col.replace("Week_", "")) < W:
                        t = norm_abbr(row.get(col, ""))
                        if t in TEAM_IDX:
                            used[TEAM_IDX[t]] = True

                avail = (~used) & playing
                if not avail[pick_idx]:
                    continue
                cand = np.where(avail)[0]
                if len(cand) < 2:
                    continue

                p = profiles.get(row["EntryName"])
                if p is None:
                    continue
                dev = np.array([
                    p["win_pref"] - 0.65,
                    p["ev_align"] - 0.5,
                    p["chalk"]    - 0.5,
                    p["home_rate"]- 0.55,
                    p["ev_align"] - 0.5,
                    p["ev_align"] - 0.5,
                ])

                X_list.append(feat_mat[cand])
                chosen_list.append(int(np.where(cand == pick_idx)[0][0]))
                dev_list.append(dev)
                team_list.append(cand)          # global team indices
                n_year += 1

        print(f"   {year}: {n_year} pick decisions collected")

    return X_list, np.array(chosen_list), np.array(dev_list), team_list


def fit(X_list, chosen, devs, teams):
    n = len(X_list)
    print(f"\n   Fitting conditional logit on {n} decisions "
          f"({N_FEAT} features + 32 team FE, L2={FE_L2})...")

    def unpack(params):
        return params[:N_FEAT], params[N_FEAT:N_FEAT + N_BASE], params[N_FEAT + N_BASE:]

    def neg_ll(params):
        beta, gamma, fe = unpack(params)
        total = 0.0
        grad = np.zeros(len(params))
        for i in range(n):
            X = X_list[i]                              # (n_cand, 9)
            w = beta.copy()
            w[:N_BASE] = w[:N_BASE] + gamma * devs[i]
            u = X @ w + fe[teams[i]]
            u -= u.max()
            ex = np.exp(u)
            p = ex / ex.sum()
            total -= np.log(max(p[chosen[i]], 1e-12))
            xbar = p @ X
            dW = X[chosen[i]] - xbar                   # (9,)
            grad[:N_FEAT] -= dW
            grad[N_FEAT:N_FEAT + N_BASE] -= dW[:N_BASE] * devs[i]
            # team FE gradient
            grad_fe = np.zeros(32)
            grad_fe[teams[i][chosen[i]]] -= 1
            np.add.at(grad_fe, teams[i], p)
            grad[N_FEAT + N_BASE:] += grad_fe
        # L2 on fe
        total = total / n + FE_L2 * np.sum(unpack(params)[2] ** 2)
        grad = grad / n
        grad[N_FEAT + N_BASE:] += 2 * FE_L2 * unpack(params)[2]
        return total, grad

    x0 = np.concatenate([
        [1.4, 0.6, 0.5, 0.75, 0.35, 0.10, 0.0, 0.0, 0.0],   # beta (warm from v2 fit)
        [5.5, -2.5, 0.4, 0.4, -0.6, 0.5],                    # gamma
        np.zeros(32),                                        # team FE
    ])
    res = minimize(neg_ll, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 300})
    beta, gamma, fe = unpack(res.x)

    print(f"   Converged: {res.success} (nll/decision = {res.fun:.4f})")
    names = ["win", "ev", "pop", "home", "fv", "hfv",
             "win×stage", "pop×stage", "thursday"]
    print(f"   {'feature':<10} {'beta':>8} {'gamma':>8}")
    for k in range(N_FEAT):
        g = f"{gamma[k]:>8.3f}" if k < N_BASE else "       —"
        print(f"   {names[k]:<10} {beta[k]:>8.3f} {g}")
    fe_rank = np.argsort(-fe)
    print(f"   Most over-picked brands:  " +
          ", ".join(f"{ALL_ABBRS[i]} {fe[i]:+.2f}" for i in fe_rank[:5]))
    print(f"   Most under-picked brands: " +
          ", ".join(f"{ALL_ABBRS[i]} {fe[i]:+.2f}" for i in fe_rank[-5:]))
    return beta.tolist(), gamma.tolist(), fe.tolist()


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or [2022, 2023, 2024, 2025]
    print(f"Calibrating on years: {years}")

    X_list, chosen, devs, teams = collect_choice_data(years)
    if len(X_list) < 1000:
        print("Not enough choice data — check paths.")
        sys.exit(1)

    beta, gamma, fe = fit(X_list, chosen, devs, teams)

    os.makedirs("entry-analytics", exist_ok=True)
    out = "entry-analytics/calibrated_weights.json"
    with open(out, "w") as f:
        json.dump({"beta": beta, "gamma": gamma, "team_fe": fe,
                   "fitted_on": years, "n_decisions": len(X_list),
                   "version": 3}, f, indent=2)
    print(f"\n   ✅ Saved calibrated weights → {out}")
