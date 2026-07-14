"""
calibrate_pick_model.py

Fits the pick-utility weights by maximum likelihood (conditional logit) on
actual historical pick decisions, replacing the hand-tuned heuristics in
profile_to_weights.

Model:
    u(entry i, team j, week w) = sum_k  (beta_k + gamma_k * dev_i,k) * x_j,k  [k=1..6]
    P(i picks j) = softmax over available teams playing week w

  x_j = [z_win, z_ev, z_pop, home-0.5, -z_fv, -z_hfv]   (hfv = holiday future value)
  dev_i = profile deviations:
      [win_pref-0.65, ev_align-0.5, chalk-0.5, home_rate-0.55, ev_align-0.5]

  beta  = league-average feature weights
  gamma = how much each behavioral trait amplifies its matching feature

Output: entry-analytics/calibrated_weights.json
        (entry_analytics.profile_to_weights auto-loads this if present)

Usage:
    python calibrate_pick_model.py 2022 2023 2024 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from entry_analytics import (
    build_week_team_data, build_future_value, build_feature_tensors,
    build_entry_profiles, norm_abbr, TEAM_IDX,
    detect_holiday_weeks, build_holiday_future_value,
)

MAX_ENTRIES_PER_WEEK = 3000   # sample cap for speed; raise for final fit
RNG = np.random.default_rng(7)


def load_season_final(year):
    import glob
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
    """
    Builds the choice dataset:
      X:      (n_choices, n_alternatives, 5) feature matrix
      chosen: (n_choices,) index of the picked alternative
      devs:   (n_choices, 5) entry profile deviations
      mask:   (n_choices, n_alternatives) availability mask
    Alternatives are padded to the max candidate count.
    """
    X_list, chosen_list, dev_list = [], [], []

    for year in years:
        final_df = load_season_final(year)
        picks_path = f"circa-pick-history/{year}_survivor_picks.csv"
        if final_df is None or not os.path.exists(picks_path):
            print(f"   {year}: missing data — skipping")
            continue

        picks_df = pd.read_csv(picks_path)
        week_cols = [c for c in picks_df.columns if c.startswith("Week_")]
        max_week = len(week_cols)

        wtd = build_week_team_data(final_df, max_week + 1)
        fv = build_future_value(wtd, 1, max_week)
        holidays = detect_holiday_weeks(final_df)
        hfv = build_holiday_future_value(wtd, holidays, 1, max_week)
        feats = build_feature_tensors(wtd, fv, sorted(wtd.keys()),
                                      holiday_fv=hfv)

        n_year = 0
        for W in sorted(wtd.keys()):
            if W < 2:      # week 1 has no history → profiles are all priors
                continue
            wcol = f"Week_{W}"
            if wcol not in picks_df.columns:
                continue

            # In-time profiles from weeks < W
            trunc = picks_df.copy()
            for col in week_cols:
                if int(col.replace("Week_", "")) >= W:
                    trunc[col] = ""
            profiles = build_entry_profiles(trunc, wtd, list(range(1, W)))

            # Entries that made a real pick in week W
            made_pick = picks_df[
                picks_df[wcol].notna()
                & (picks_df[wcol].astype(str).str.strip() != "")
                & (picks_df[wcol].astype(str).str.upper() != "ELIMINATED")
            ]
            if len(made_pick) > MAX_ENTRIES_PER_WEEK:
                made_pick = made_pick.sample(MAX_ENTRIES_PER_WEEK, random_state=7)

            fw = feats[W]
            playing = fw["plays"].astype(bool)
            feat_mat = np.stack([fw["win"], fw["ev"], fw["pop"],
                                 fw["home"] - 0.5, -fw["fv"],
                                 -fw["hfv"]], axis=1)  # (32, 6)

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
                    continue  # data inconsistency — skip

                cand = np.where(avail)[0]
                if len(cand) < 2:
                    continue
                chosen_pos = int(np.where(cand == pick_idx)[0][0])

                p = profiles.get(row["EntryName"])
                if p is None:
                    continue
                dev = np.array([
                    p["win_pref"] - 0.65,
                    p["ev_align"] - 0.5,
                    p["chalk"]    - 0.5,
                    p["home_rate"]- 0.55,
                    p["ev_align"] - 0.5,
                    p["ev_align"] - 0.5,   # hfv: planners hoard holiday teams
                ])

                X_list.append(feat_mat[cand])       # (n_cand, 5)
                chosen_list.append(chosen_pos)
                dev_list.append(dev)
                n_year += 1

        print(f"   {year}: {n_year} pick decisions collected")

    return X_list, np.array(chosen_list), np.array(dev_list)


def fit_conditional_logit(X_list, chosen, devs):
    """
    Maximizes sum log softmax(u)[chosen] over beta (5) and gamma (5).
    Weight for entry i on feature k: beta_k + gamma_k * dev_i,k
    """
    n = len(X_list)
    print(f"\n   Fitting conditional logit on {n} decisions...")

    def neg_ll(params):
        beta, gamma = params[:6], params[6:]
        total = 0.0
        grad = np.zeros(12)
        for i in range(n):
            X = X_list[i]                          # (n_cand, 5)
            w = beta + gamma * devs[i]             # (5,)
            u = X @ w                              # (n_cand,)
            u -= u.max()
            ex = np.exp(u)
            p = ex / ex.sum()
            total -= np.log(max(p[chosen[i]], 1e-12))
            # gradient
            xbar = p @ X                           # (5,)
            dW = X[chosen[i]] - xbar               # (5,) dLL/dw
            grad[:6] -= dW
            grad[6:] -= dW * devs[i]
        return total / n, grad / n

    # Warm start at the current hand-tuned league-average weights
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.8,    # beta
                   4.0, 3.0, 6.0, 3.0, 1.5, 2.0])   # gamma
    res = minimize(neg_ll, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 200})
    beta, gamma = res.x[:6], res.x[6:]

    print(f"   Converged: {res.success} (nll/decision = {res.fun:.4f})")
    names = ["win", "ev", "pop", "home", "fv", "hfv"]
    print(f"   {'feature':<8} {'beta':>8} {'gamma':>8}")
    for k in range(6):
        print(f"   {names[k]:<8} {beta[k]:>8.3f} {gamma[k]:>8.3f}")
    return beta.tolist(), gamma.tolist()


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or [2022, 2023, 2024, 2025]
    print(f"Calibrating on years: {years}")

    X_list, chosen, devs = collect_choice_data(years)
    if len(X_list) < 1000:
        print("Not enough choice data collected — check paths.")
        sys.exit(1)

    beta, gamma = fit_conditional_logit(X_list, chosen, devs)

    os.makedirs("entry-analytics", exist_ok=True)
    out = "entry-analytics/calibrated_weights.json"
    with open(out, "w") as f:
        json.dump({"beta": beta, "gamma": gamma,
                   "fitted_on": years,
                   "n_decisions": len(X_list)}, f, indent=2)
    print(f"\n   ✅ Saved calibrated weights → {out}")
    print("   entry_analytics.profile_to_weights will now use these automatically.")
