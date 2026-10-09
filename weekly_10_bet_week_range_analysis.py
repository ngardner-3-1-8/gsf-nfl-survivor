"""
weekly_10_bet_week_range_analysis.py

Identifies which parts of the season each bet type performs best and worst.

For every tracked bet type it pools realized Win/Loss across all historical
seasons by NFL week, computes a per-week win rate, then AUTO-DETECTS contiguous
STRONG / AVERAGE / WEAK week ranges relative to that bet's own baseline win rate
(e.g. "Weeks 1-4 weak, 5-16 strong, 17-18 average"). The result is written to a
JSON file that the Bets tab reads on demand — this script is NOT run on page
load; dispatch it weekly (or manually) so the analysis refreshes as results come
in.

Source:  nfl-power-ratings/final_data/{year}_final_data/
             Season_{year}_Through_Week_{max}_Final_Data.csv   (one per season)
Output:  entry-analytics/bet_week_range_analysis.json
"""

import os
import re
import glob
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────
# Regular-season weeks only; playoff weeks (19-20) are too sparse per week to
# classify cleanly and aren't what "best weeks of the season" is asking about.
MAX_WEEK = 18
# A week counts as STRONG/WEAK only if its (smoothed) win rate differs from the
# bet's baseline by at least this many percentage points...
MARGIN_PTS = 5.0
# ...and only when it has at least this many settled (Win/Loss) bets pooled
# across seasons; thinner weeks are treated as AVERAGE so they don't create
# spurious ranges.
MIN_WEEK_N = 25
# Per-week win rates are noisy (~90 games/week pooled), so classify on a
# centered, sample-weighted rolling mean rather than the raw weekly number.
SMOOTH_WINDOW = 3
# A STRONG/WEAK stretch must span at least this many weeks to be called out;
# shorter blips are folded back into AVERAGE so ranges stay broad and readable.
MIN_RUN_WEEKS = 2

# The 12 tracked bet types (label -> category), mirroring the betting-history
# endpoint so the Bets tab lines up.
BET_TYPES = [
    ("GSF Spread", "Spread"),
    ("MP Spread", "Spread"),
    ("Sim Spread", "Spread"),
    ("Sim Spread (Kelly)", "Spread"),
    ("Consensus Spread", "Spread"),
    ("GSF Moneyline", "Moneyline"),
    ("MP Moneyline", "Moneyline"),
    ("Sim Moneyline", "Moneyline"),
    ("Sim Moneyline (Kelly)", "Moneyline"),
    ("Consensus Moneyline", "Moneyline"),
    ("Sim Total", "Total"),
    ("Sim Total (Kelly)", "Total"),
]

DATA_DIR = os.environ.get("GSF_DATA_DIR", ".")
OUT_DIR = os.path.join(DATA_DIR, "entry-analytics")
OUT_PATH = os.path.join(OUT_DIR, "bet_week_range_analysis.json")


# ── Data loading ──────────────────────────────────────────────────────────
def _week_of(path):
    try:
        return int(os.path.basename(path).split("_Through_Week_")[1].split("_Final")[0])
    except Exception:
        return 0


def load_all_seasons(years=None):
    """Return a single long DataFrame of every season's most-complete cumulative
    file, with a numeric Week column, restricted to regular-season weeks."""
    base = os.path.join(DATA_DIR, "nfl-power-ratings", "final_data")
    frames = []
    seasons = []
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
        wk_col = "Week" if "Week" in df.columns else ("Week_x" if "Week_x" in df.columns else None)
        if wk_col is None:
            continue
        df["_week"] = pd.to_numeric(df[wk_col], errors="coerce")
        df = df[(df["_week"] >= 1) & (df["_week"] <= MAX_WEEK)]
        if df.empty:
            continue
        df["_season"] = y
        frames.append(df)
        seasons.append(y)
    if not frames:
        return pd.DataFrame(), []
    return pd.concat(frames, ignore_index=True), sorted(seasons)


# ── Analysis ──────────────────────────────────────────────────────────────
def _smoothed_win_pct(weeks):
    """Centered, sample-weighted rolling mean of weekly win % over SMOOTH_WINDOW,
    to damp week-to-week noise before classifying. Returns a list aligned to
    `weeks` (None where a week has no settled bets)."""
    half = SMOOTH_WINDOW // 2
    out = []
    for i, w in enumerate(weeks):
        if w["win_pct"] is None:
            out.append(None)
            continue
        num = den = 0.0
        for j in range(i - half, i + half + 1):
            if 0 <= j < len(weeks) and weeks[j]["win_pct"] is not None:
                num += weeks[j]["_wins"]
                den += weeks[j]["_settled"]
        out.append(round(num / den * 100, 1) if den else None)
    return out


def _classify_weeks(weeks, baseline):
    """Assign strong/weak/average to each week from its SMOOTHED win rate, then
    demote any strong/weak run shorter than MIN_RUN_WEEKS back to average."""
    smooth = _smoothed_win_pct(weeks)
    cls = []
    for w, sv in zip(weeks, smooth):
        if sv is None or w["n"] < MIN_WEEK_N:
            cls.append("average")
        elif sv >= baseline + MARGIN_PTS:
            cls.append("strong")
        elif sv <= baseline - MARGIN_PTS:
            cls.append("weak")
        else:
            cls.append("average")
    # Fold short strong/weak runs into average.
    i = 0
    while i < len(cls):
        j = i
        while j + 1 < len(cls) and cls[j + 1] == cls[i]:
            j += 1
        if cls[i] in ("strong", "weak") and (j - i + 1) < MIN_RUN_WEEKS:
            for k in range(i, j + 1):
                cls[k] = "average"
        i = j + 1
    return cls


def _merge_ranges(weeks):
    """Collapse consecutive weeks of the same class into labeled ranges with a
    pooled win rate."""
    ranges = []
    for wk in weeks:
        if ranges and ranges[-1]["class"] == wk["class"] and wk["week"] == ranges[-1]["end"] + 1:
            r = ranges[-1]
            r["end"] = wk["week"]
            r["_wins"] += wk["_wins"]
            r["_settled"] += wk["_settled"]
        else:
            ranges.append({
                "start": wk["week"], "end": wk["week"], "class": wk["class"],
                "_wins": wk["_wins"], "_settled": wk["_settled"],
            })
    out = []
    for r in ranges:
        wp = round(r["_wins"] / r["_settled"] * 100, 1) if r["_settled"] else None
        lo, hi = r["start"], r["end"]
        label = f"Week {lo}" if lo == hi else f"Weeks {lo}–{hi}"
        out.append({"start": lo, "end": hi, "class": r["class"],
                    "win_pct": wp, "n": int(r["_settled"]), "label": label})
    return out


def analyze_bet_type(df, label):
    wl_col = f"{label} Win/Loss"
    if wl_col not in df.columns:
        return None
    sub = df[["_week", wl_col]].copy()
    sub["res"] = sub[wl_col].astype(str).str.strip()
    settled = sub[sub["res"].isin(["Win", "Loss"])]
    if settled.empty:
        return None

    total_wins = int((settled["res"] == "Win").sum())
    total_settled = int(len(settled))
    baseline = round(total_wins / total_settled * 100, 1)

    weeks = []
    for wk in range(1, MAX_WEEK + 1):
        wsub = settled[settled["_week"] == wk]
        n = int(len(wsub))
        wins = int((wsub["res"] == "Win").sum())
        wp = round(wins / n * 100, 1) if n else None
        weeks.append({"week": wk, "win_pct": wp, "n": n,
                      "_wins": wins, "_settled": n})

    for w, cls in zip(weeks, _classify_weeks(weeks, baseline)):
        w["class"] = cls

    ranges = _merge_ranges(weeks)
    # strip private fields from the per-week list
    weeks_out = [{k: w[k] for k in ("week", "win_pct", "n", "class")} for w in weeks]

    rated = [r for r in ranges if r["win_pct"] is not None and r["n"] >= MIN_WEEK_N]
    best = max(rated, key=lambda r: r["win_pct"]) if rated else None
    worst = min(rated, key=lambda r: r["win_pct"]) if rated else None

    return {
        "baseline_win_pct": baseline,
        "n": total_settled,
        "weeks": weeks_out,
        "ranges": ranges,
        "best_range": best,
        "worst_range": worst,
    }


def build(years=None):
    df, seasons = load_all_seasons(years)
    if df.empty:
        raise SystemExit("No historical final_data files found to analyze.")
    by_bet_type = {}
    for label, category in BET_TYPES:
        res = analyze_bet_type(df, label)
        if res is None:
            continue
        res["category"] = category
        by_bet_type[label] = res

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons_used": seasons,
        "max_week": MAX_WEEK,
        "margin_pts": MARGIN_PTS,
        "min_week_n": MIN_WEEK_N,
        "by_bet_type": by_bet_type,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def _print_summary(payload):
    print(f"Seasons: {payload['seasons_used']}  |  weeks 1-{payload['max_week']}  "
          f"|  band ±{payload['margin_pts']}pts  |  min n/week {payload['min_week_n']}")
    for label, d in payload["by_bet_type"].items():
        print(f"\n{label}  (baseline {d['baseline_win_pct']}%, {d['n']} bets)")
        for r in d["ranges"]:
            tag = {"strong": "STRONG", "weak": "WEAK  ", "average": "avg   "}[r["class"]]
            wp = f"{r['win_pct']}%" if r["win_pct"] is not None else "n/a"
            print(f"   {tag}  {r['label']:<12}  {wp:>6}  (n={r['n']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=None,
                    help="Limit to specific seasons (default: all available).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    payload = build(args.years)
    print(f"Wrote {OUT_PATH}")
    if not args.quiet:
        _print_summary(payload)
