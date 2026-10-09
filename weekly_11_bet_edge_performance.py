"""
weekly_11_bet_edge_performance.py

"Which edges are most successful" — the repo version of the local edge-bucketing
analysis, broken out BY YEAR and with unit profit, plus model-agreement signals.

Two sections, written to one JSON the Bets tab reads on demand (NOT on page load):

  1. EDGE BUCKETS — for every measurement (MC/GSF/MP spread, MC/GSF/MP moneyline,
     MC total + over/under), bucket active bets by edge magnitude and report
     record, win %, and unit profit ($ P/L) for each bucket, per season and
     lifetime.

  2. AGREEMENT SIGNALS — results when models agree on the same pick, for both
     spread and moneyline: MC+GSF, MC+MP, MP+GSF, MC+MP+GSF. Anything involving
     Massey-Peabody is limited to 2026+ (MP wasn't reliably tracked before then).

Source:  nfl-power-ratings/final_data/{year}_final_data/
             Season_{year}_Through_Week_{max}_Final_Data.csv
Output:  entry-analytics/bet_edge_performance.json
"""

import os
import re
import glob
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Massey-Peabody wasn't reliably tracked before this season, so every
# MP-involving measurement (edge buckets AND agreement combos) starts here.
MP_MIN_YEAR = 2026

# Kelly sizing (mirrors the local analysis): fractional (quarter) Kelly sized
# off a flat bankroll, using the Monte Carlo simulation's probability for the
# PICKED side. Spread/total priced at -110; moneyline at the game's book line.
KELLY_FRACTION = 0.25
KELLY_BANKROLL = 10000.0

DATA_DIR = os.environ.get("GSF_DATA_DIR", ".")
OUT_DIR = os.path.join(DATA_DIR, "entry-analytics")
OUT_PATH = os.path.join(OUT_DIR, "bet_edge_performance.json")

INF = float("inf")
SPREAD_BINS = [0, 1, 2, 3, 4, INF]
SPREAD_LABELS = ["0.0–1.0", "1.0–2.0", "2.0–3.0", "3.0–4.0", "4.0+"]
TOTAL_BINS = [0, 1, 2, 3, 5, INF]
TOTAL_LABELS = ["0.0–1.0", "1.0–2.0", "2.0–3.0", "3.0–5.0", "5.0+"]
ML_BINS = [0, 0.05, 0.10, 0.15, 0.20, 1.0]
ML_LABELS = ["0–5%", "5–10%", "10–15%", "15–20%", "20%+"]

# label, category, edge_col, result_col, pnl_col, bet_col, bins, labels,
# mp_involved, bet_filter (None, or a specific bet value like "Over")
MEASUREMENTS = [
    ("MC Spread", "Spread", "Monte Carlo Spread Edge", "Sim Spread Win/Loss",
     "Sim Spread P/L", "Monte Carlo Spread Bet", SPREAD_BINS, SPREAD_LABELS, False, None),
    ("GSF Spread", "Spread", "GSF Spread Edge", "GSF Spread Win/Loss",
     "GSF Spread P/L", "GSF Spread Bet", SPREAD_BINS, SPREAD_LABELS, False, None),
    ("MP Spread", "Spread", "Massey-Peabody Spread Edge", "MP Spread Win/Loss",
     "MP Spread P/L", "Massey-Peabody Spread Bet", SPREAD_BINS, SPREAD_LABELS, True, None),
    ("MC Moneyline", "Moneyline", "Monte Carlo Moneyline Edge", "Sim Moneyline Win/Loss",
     "Sim Moneyline P/L", "Monte Carlo Moneyline Bet", ML_BINS, ML_LABELS, False, None),
    ("GSF Moneyline", "Moneyline", "GSF Moneyline Edge", "GSF Moneyline Win/Loss",
     "GSF Moneyline P/L", "GSF Moneyline Bet", ML_BINS, ML_LABELS, False, None),
    ("MP Moneyline", "Moneyline", "Massey-Peabody Moneyline Edge", "MP Moneyline Win/Loss",
     "MP Moneyline P/L", "Massey-Peabody Moneyline Bet", ML_BINS, ML_LABELS, True, None),
    ("MC Total", "Total", "Monte Carlo Total Edge", "Sim Total Win/Loss",
     "Sim Total P/L", "Monte Carlo Total Bet", TOTAL_BINS, TOTAL_LABELS, False, None),
    ("MC Total — Over", "Total", "Monte Carlo Total Edge", "Sim Total Win/Loss",
     "Sim Total P/L", "Monte Carlo Total Bet", TOTAL_BINS, TOTAL_LABELS, False, "Over"),
    ("MC Total — Under", "Total", "Monte Carlo Total Edge", "Sim Total Win/Loss",
     "Sim Total P/L", "Monte Carlo Total Bet", TOTAL_BINS, TOTAL_LABELS, False, "Under"),
]

# Per-model columns used to detect agreement and read its result.
MODEL_COLS = {
    "MC":  {"Spread": ("Monte Carlo Spread Bet", "Sim Spread Win/Loss", "Sim Spread P/L"),
            "Moneyline": ("Monte Carlo Moneyline Bet", "Sim Moneyline Win/Loss", "Sim Moneyline P/L")},
    "MP":  {"Spread": ("Massey-Peabody Spread Bet", "MP Spread Win/Loss", "MP Spread P/L"),
            "Moneyline": ("Massey-Peabody Moneyline Bet", "MP Moneyline Win/Loss", "MP Moneyline P/L")},
    "GSF": {"Spread": ("GSF Spread Bet", "GSF Spread Win/Loss", "GSF Spread P/L"),
            "Moneyline": ("GSF Moneyline Bet", "GSF Moneyline Win/Loss", "GSF Moneyline P/L")},
}
COMBOS = [
    ("MC + GSF", ["MC", "GSF"], False),
    ("MC + MP", ["MC", "MP"], True),
    ("MP + GSF", ["MP", "GSF"], True),
    ("MC + MP + GSF", ["MC", "MP", "GSF"], True),
]


# ── Loading ────────────────────────────────────────────────────────────────
def _week_of(path):
    try:
        return int(os.path.basename(path).split("_Through_Week_")[1].split("_Final")[0])
    except Exception:
        return 0


def load_all_seasons(years=None):
    base = os.path.join(DATA_DIR, "nfl-power-ratings", "final_data")
    frames, seasons = [], []
    for ydir in sorted(glob.glob(os.path.join(base, "*_final_data"))):
        m = re.match(r"(\d{4})", os.path.basename(ydir))
        if not m:
            continue
        y = int(m.group(1))
        if years and y not in years:
            continue
        files = glob.glob(os.path.join(ydir, "Season_*_Through_Week_*_Final_Data.csv"))
        if not files:
            continue
        df = pd.read_csv(max(files, key=_week_of))
        df["_season"] = y
        frames.append(df)
        seasons.append(y)
    if not frames:
        return pd.DataFrame(), []
    return pd.concat(frames, ignore_index=True), sorted(seasons)


# ── Helpers ────────────────────────────────────────────────────────────────
def _norm(series):
    return series.astype(str).str.strip()


def _valid_pick(series):
    s = _norm(series)
    return ~s.isin(["No Bet", "nan", "NaN", ""])


def _payout_mult(odds):
    """Profit multiple on a 1-unit win at American `odds`."""
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return np.nan
    if o == 0 or np.isnan(o):
        return np.nan
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def _kelly_stake_pct(prob, odds):
    """Fractional-Kelly stake as a share of bankroll (0 if no edge / bad input)."""
    b = _payout_mult(odds)
    if prob is None or pd.isna(prob) or np.isnan(b) or prob <= 0 or prob >= 1:
        return 0.0
    k = (b * prob - (1 - prob)) / b
    return max(0.0, k * KELLY_FRACTION)


def _kelly_cols(df, category, bet_col):
    """Per-row (stake_pct, profit) for Kelly staking, using the MC sim's prob for
    the picked side. profit needs a result; it's filled in by the caller's rows
    being pre-filtered to active bets, but we read the result here too."""
    n = len(df)
    nan = pd.Series(np.nan, index=df.index)
    if bet_col not in df.columns or "Home Team" not in df.columns:
        return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
    bet = _norm(df[bet_col])
    is_home = bet == _norm(df["Home Team"])
    def col(c):
        return pd.to_numeric(df[c], errors="coerce") if c in df.columns else nan
    if category == "Spread":
        prob = pd.Series(np.where(is_home, col("Sim_Home_Cover_Prob"), col("Sim_Away_Cover_Prob")), index=df.index)
        odds = pd.Series(-110.0, index=df.index)
    elif category == "Moneyline":
        prob = pd.Series(np.where(is_home, col("Sim_Home_Win_Pct"), col("Sim_Away_Win_Pct")), index=df.index)
        odds = pd.Series(np.where(is_home, col("Home Team Sportsbook Moneyline"),
                                  col("Away Team Sportsbook Moneyline")), index=df.index)
    else:  # Total
        is_over = bet == "Over"
        prob = pd.Series(np.where(is_over, col("Sim_Prob_Over"), col("Sim_Prob_Under")), index=df.index)
        odds = pd.Series(-110.0, index=df.index)
    stake_pct = pd.Series([_kelly_stake_pct(p, o) for p, o in zip(prob, odds)], index=df.index)
    return stake_pct, odds


def _record(res, pnl, df=None, category=None, bet_col=None):
    """record dict from a result Series + pnl Series (already active rows). When
    `df`/`category`/`bet_col` are given, also computes Kelly profit + avg stake
    for those same rows."""
    r = _norm(res)
    wins = int((r == "Win").sum())
    losses = int((r == "Loss").sum())
    pushes = int((r == "Push").sum())
    settled = wins + losses
    units = float(pd.to_numeric(pnl, errors="coerce").fillna(0).sum())
    out = {
        "wins": wins, "losses": losses, "pushes": pushes,
        "n": wins + losses + pushes,
        "win_pct": round(wins / settled * 100, 1) if settled else None,
        "units": round(units, 2),
    }
    if df is not None and category and bet_col:
        stake_pct, odds = _kelly_cols(df, category, bet_col)
        stake = stake_pct * KELLY_BANKROLL
        mult = odds.map(_payout_mult)
        profit = pd.Series(0.0, index=df.index)
        profit[r.values == "Win"] = (stake * mult)[r.values == "Win"]
        profit[r.values == "Loss"] = -stake[r.values == "Loss"]
        out["kelly"] = round(float(profit.fillna(0).sum()), 2)
        out["avg_kelly_pct"] = round(float(stake_pct.mean()) * 100, 2) if len(stake_pct) else None
    return out


def _bucketize(df, m):
    """Return {bucket_label: record} for one measurement over the given df."""
    (_, category, edge_col, res_col, pnl_col, bet_col, bins, labels, _, bet_filter) = m
    if res_col not in df.columns or edge_col not in df.columns:
        return {}
    sub = df
    if bet_filter is not None and bet_col in df.columns:
        sub = sub[_norm(sub[bet_col]) == bet_filter]
    active = sub[_norm(sub[res_col]).isin(["Win", "Loss", "Push"])].copy()
    if active.empty:
        return {}
    active["_bucket"] = pd.cut(pd.to_numeric(active[edge_col], errors="coerce"),
                               bins=bins, labels=labels, right=False)
    out = {}
    for lab in labels:
        b = active[active["_bucket"] == lab]
        if b.empty:
            continue
        out[lab] = _record(b[res_col], b[pnl_col], df=b, category=category, bet_col=bet_col)
    return out


def _combo_mask(df, models, category):
    """Rows where every model in `models` made the SAME valid pick for `category`."""
    first_bet = MODEL_COLS[models[0]][category][0]
    if first_bet not in df.columns:
        return pd.Series(False, index=df.index)
    base = _norm(df[first_bet])
    mask = _valid_pick(df[first_bet])
    for mdl in models[1:]:
        bc = MODEL_COLS[mdl][category][0]
        if bc not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= _valid_pick(df[bc]) & (_norm(df[bc]) == base)
    return mask


# ── Build ──────────────────────────────────────────────────────────────────
def build(years=None):
    df, seasons = load_all_seasons(years)
    if df.empty:
        raise SystemExit("No historical final_data files found to analyze.")
    df["_season"] = pd.to_numeric(df["_season"], errors="coerce").astype("Int64")

    # 1. Edge buckets per measurement, lifetime + by year.
    edge_buckets = {}
    for m in MEASUREMENTS:
        label, category, edge_col, res_col, pnl_col, bet_col, bins, labels, mp, _ = m
        if res_col not in df.columns:
            continue
        scope = df[df["_season"] >= MP_MIN_YEAR] if mp else df
        if scope.empty:
            continue
        by_year = {}
        for y in sorted(scope["_season"].dropna().unique().tolist()):
            yb = _bucketize(scope[scope["_season"] == y], m)
            if yb:
                by_year[str(int(y))] = yb
        edge_buckets[label] = {
            "category": category,
            "bucket_labels": labels,
            "mp_involved": mp,
            "min_year": MP_MIN_YEAR if mp else None,
            "lifetime": _bucketize(scope, m),
            "by_year": by_year,
        }

    # 2. Agreement signals, both categories, lifetime + by year.
    agreement = {}
    for combo_label, models, mp in COMBOS:
        for category in ("Spread", "Moneyline"):
            bet_col, res_col, pnl_col = MODEL_COLS[models[0]][category]
            if res_col not in df.columns:
                continue
            scope = df[df["_season"] >= MP_MIN_YEAR] if mp else df
            mask = _combo_mask(scope, models, category)
            agreed = scope[mask]
            active = agreed[_norm(agreed[res_col]).isin(["Win", "Loss", "Push"])]
            by_year = {}
            for y in sorted(active["_season"].dropna().unique().tolist()):
                ya = active[active["_season"] == y]
                by_year[str(int(y))] = _record(ya[res_col], ya[pnl_col],
                                               df=ya, category=category, bet_col=bet_col)
            key = f"{combo_label} ({category})"
            agreement[key] = {
                "combo": combo_label,
                "category": category,
                "mp_involved": mp,
                "min_year": MP_MIN_YEAR if mp else None,
                "lifetime": _record(active[res_col], active[pnl_col],
                                    df=active, category=category, bet_col=bet_col),
                "by_year": by_year,
            }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons_used": seasons,
        "mp_min_year": MP_MIN_YEAR,
        "edge_buckets": edge_buckets,
        "agreement": agreement,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def _fmt(rec):
    wp = f"{rec['win_pct']}%" if rec["win_pct"] is not None else "   -"
    k = f"   K:${rec.get('kelly', 0):>+10,.0f}" if "kelly" in rec else ""
    ak = f" ({rec.get('avg_kelly_pct')}%)" if rec.get("avg_kelly_pct") is not None else ""
    return f"{rec['wins']:>3}W-{rec['losses']:>3}L  {wp:>6}  {rec['units']:>+10,.0f}u{k}{ak}"


def _print_summary(payload):
    print(f"Seasons: {payload['seasons_used']}  |  MP measurements start {payload['mp_min_year']}")
    print("\n===== EDGE BUCKETS (lifetime) =====")
    for label, d in payload["edge_buckets"].items():
        print(f"\n--- {label}{'  [2026+]' if d['mp_involved'] else ''} ---")
        for lab in d["bucket_labels"]:
            r = d["lifetime"].get(lab)
            if r:
                print(f"   {lab:<10} {_fmt(r)}")
    print("\n===== AGREEMENT SIGNALS (lifetime) =====")
    for key, d in payload["agreement"].items():
        r = d["lifetime"]
        tag = "  [2026+]" if d["mp_involved"] else ""
        print(f"   {key:<26}{tag:<9} {_fmt(r)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    payload = build(args.years)
    print(f"Wrote {OUT_PATH}")
    if not args.quiet:
        _print_summary(payload)
