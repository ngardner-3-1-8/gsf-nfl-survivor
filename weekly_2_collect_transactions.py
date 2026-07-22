"""
weekly_2_collect_transactions.py  (nflreadpy version — no scraping)

Player transactions come from nflreadpy (API-based, no browser, no bot
detection, no Chrome version drift). Coaching moves come from a
hand-maintained CSV you control. Both are valued in power-rating points
and aggregated into a per-team inbound/outbound/net leaderboard.

Sources:
  - load_rosters(prior_year) vs load_rosters(year)  → signings / departures
  - load_trades(year)                                → trades (players + picks)
  - load_draft_picks(year)                           → draft additions
  - nfl-transactions/manual_coaching_{year}.csv      → coaching (you maintain)

Value model:
  - Skill players (QB/RB/WR/TE): prior-season total EPA × EPA_TO_POINTS
    (QBs fall back to the QB baseline when EPA is ~0, i.e. an unmatched backup)
  - Defense / OL / unknown: position baseline
  - Coaches: value column if provided, else role-based COACH_VALUES
  - Draft picks: expected-value-by-slot curve
  - Departures/releases/retirements: outbound value for the losing team

Outputs:
  nfl-transactions/{year}_transactions.csv    (one row per transaction)
  nfl-transactions/{year}_team_deltas.csv     (aggregated per team)
"""

import os
import numpy as np
import pandas as pd
import nflreadpy as nfl
from datetime import datetime

OUTPUT_DIR = "nfl-transactions"

SKILL_OFFENSE = {"QB", "RB", "WR", "TE", "FB"}
OL = {"T", "G", "C", "OL", "OT", "OG", "LT", "RT"}
DEFENSE = {"DE", "DT", "NT", "EDGE", "LB", "ILB", "OLB", "MLB",
           "CB", "S", "SS", "FS", "DB", "DL"}

# ── VALUE UNIT: net points per game of margin a player adds vs a replacement ──
# STARTER_PPG = value of a FULL-TIME STARTER at this position. Backups scale
# down by snap share; elite players scale UP via prior-season EPA (skill) or a
# manual override (any position). Numbers are per-GAME, so they're small and
# interpretable — a starting CB adds ~1.4 pts/game of margin vs a replacement.
STARTER_PPG = {
    "QB": 3.5, "RB": 1.0, "WR": 1.5, "TE": 0.9, "FB": 0.2,
    "T": 1.2, "G": 0.9, "C": 0.9, "OL": 1.0, "OT": 1.2, "OG": 0.9, "LT": 1.3, "RT": 1.1,
    "DE": 1.6, "EDGE": 1.8, "DT": 1.2, "NT": 0.8, "DL": 1.3,
    "LB": 1.1, "ILB": 1.0, "OLB": 1.3, "MLB": 1.1,
    "CB": 1.4, "S": 1.1, "SS": 1.1, "FS": 1.1, "DB": 1.2,
    "K": 0.7, "P": 0.3, "LS": 0.1,
}

# Coaching values are net points-per-game of MARGIN. A defensive-minded coach
# whose scheme yields 4 fewer points scored but 6 fewer allowed = +2 net.
COACH_VALUES = {
    "head coach": 2.0,
    "offensive coordinator": 1.2,
    "defensive coordinator": 1.2,
    "special teams coordinator": 0.4,
    "coordinator": 1.0,
    "coach": 0.4,
    "general manager": 0.8,
}

# Backup compatibility: some call sites still reference POSITION_BASELINE.
POSITION_BASELINE = STARTER_PPG

ABBR_MAP = {"JAC": "JAX", "LAR": "LA", "GNB": "GB", "KAN": "KC",
            "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV", "WSH": "WAS"}


def normalize_team(abbr):
    if abbr is None or (isinstance(abbr, float) and pd.isna(abbr)):
        return None
    a = str(abbr).strip().upper()
    if not a:
        return None
    return ABBR_MAP.get(a, a)


def get_target_year(today=None):
    today = today or datetime.now()
    return today.year - 1 if today.month < 6 else today.year


def draft_pick_value(overall_pick):
    if pd.isna(overall_pick) or overall_pick <= 0:
        return 0.0
    return round(7.0 * np.exp(-float(overall_pick) / 60.0), 2)


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


# ── Player valuation from prior-season EPA ──────────────────────────────────
def _load_snap_shares(prior_year):
    """pfr player name(lower) -> {off_pct, def_pct, games} averaged over the
    prior season. Used to scale position baselines by playing time so a
    full-time starter is worth more than a rotational backup."""
    try:
        snaps = _to_pandas(nfl.load_snap_counts([prior_year]))
    except Exception as e:
        print(f"   ⚠️  Snap counts unavailable ({e}); baselines won't be snap-scaled")
        return {}
    agg = {}
    for _, r in snaps.iterrows():
        name = str(r.get("player", "")).strip().lower()
        if not name:
            continue
        def pct(v):
            v = float(v or 0)
            return v / 100.0 if v > 1.5 else v   # handle 0-100 vs 0-1 encodings
        d = agg.setdefault(name, {"off": 0.0, "def": 0.0, "n": 0})
        d["off"] += pct(r.get("offense_pct"))
        d["def"] += pct(r.get("defense_pct"))
        d["n"] += 1
    out = {}
    for name, d in agg.items():
        if d["n"] > 0:
            out[name] = {"off_pct": d["off"] / d["n"],
                         "def_pct": d["def"] / d["n"],
                         "games": d["n"]}
    return out


def _load_manual_player_values(target_year, output_dir=OUTPUT_DIR):
    """Optional hand-set per-game values for specific players (the all-pros you
    care about). File: nfl-transactions/manual_player_values_{year}.csv
    Columns: player,value,note  — value is net points-per-game, overrides auto."""
    path = os.path.join(output_dir, f"manual_player_values_{target_year}.csv")
    if not os.path.exists(path):
        return {}
    try:
        m = pd.read_csv(path)
    except Exception:
        return {}
    out = {}
    for _, r in m.iterrows():
        name = str(r.get("player", "")).strip().lower()
        val = r.get("value")
        if name and pd.notna(val):
            out[name] = float(val)
    if out:
        print(f"   ✅ Loaded {len(out)} manual player-value overrides")
    return out


def build_player_values(prior_year, target_year=None):
    """
    name(lower) -> {value, position, epa}  where value is NET POINTS PER GAME.

    Skill players (QB/RB/WR/TE): prior-season total EPA / games played — this
      is already a per-game points figure and differentiates quality naturally
      (an all-pro WR has far more EPA than a mediocre one).
    Defense / OL / ST: STARTER_PPG[pos] scaled by snap share, so a full-time
      starter is worth the baseline and a rotational backup proportionally less.
    Manual overrides (if provided) win outright — use them for elite players
      whose value should exceed a full-time starter's baseline.
    """
    print(f"   Loading {prior_year} player stats for valuation...")
    stats = _to_pandas(nfl.load_player_stats([prior_year]))
    snap_shares = _load_snap_shares(prior_year)
    manual = _load_manual_player_values(target_year) if target_year else {}

    # Aggregate EPA + games from (possibly weekly) player stats
    weekly = "week" in stats.columns
    agg = {}
    for _, row in stats.iterrows():
        name = row.get("player_display_name") or row.get("player_name")
        if not name:
            continue
        key = str(name).strip().lower()
        pos = str(row.get("position", "")).upper()
        epa = (float(row.get("passing_epa", 0) or 0)
               + float(row.get("rushing_epa", 0) or 0)
               + float(row.get("receiving_epa", 0) or 0))
        d = agg.setdefault(key, {"position": pos, "total_epa": 0.0, "rows": 0,
                                 "games_col": None})
        d["total_epa"] += epa
        d["rows"] += 1
        if "games" in row and pd.notna(row.get("games")):
            d["games_col"] = float(row.get("games"))

    def games_for(name, d):
        if d["games_col"]:
            return max(1.0, d["games_col"])
        if weekly and d["rows"] > 0:
            return float(d["rows"])
        ss = snap_shares.get(name)
        if ss and ss["games"] > 0:
            return float(ss["games"])
        return 17.0

    values = {}
    for key, d in agg.items():
        pos = d["position"]

        if key in manual:                       # manual override wins
            values[key] = {"value": round(manual[key], 2), "position": pos,
                           "epa": round(d["total_epa"], 2)}
            continue

        if pos in SKILL_OFFENSE:
            games = games_for(key, d)
            ppg = d["total_epa"] / games        # points per game of margin
            if pos == "QB" and abs(ppg) < 0.5:  # unmatched backup QB
                ss = snap_shares.get(key, {})
                ppg = STARTER_PPG["QB"] * ss.get("off_pct", 0.5)
            values[key] = {"value": round(ppg, 2), "position": pos,
                           "epa": round(d["total_epa"], 2)}
        else:
            base = STARTER_PPG.get(pos, 0.5)
            ss = snap_shares.get(key)
            if ss:
                # Defensive positions use def snaps, OL/ST use offense/either
                share = ss["def_pct"] if pos in DEFENSE else max(ss["off_pct"], ss["def_pct"])
                share = share if share > 0 else 0.5
            else:
                share = 0.85   # no snap data → assume near-starter
            values[key] = {"value": round(base * share, 2), "position": pos,
                           "epa": None}

    # Players with snaps but no stats row (most defenders) — value them too
    for name, ss in snap_shares.items():
        if name in values or name in manual:
            continue
        # position not known here; leave for player_value() fallback at lookup
    for name, val in manual.items():
        if name not in values:
            values[name] = {"value": round(val, 2), "position": "", "epa": None}

    return values


def player_value(name, position, player_values):
    key = str(name).strip().lower()
    if key in player_values:
        d = player_values[key]
        return d["value"], d["position"] or position, d["epa"]
    # Unmatched → per-game starter baseline, mild backup discount
    base = STARTER_PPG.get(str(position).upper(), 0.5) * 0.85
    return round(base, 2), position, None


# ── Signings & departures from roster comparison ────────────────────────────
def _season_end_teams(roster_df):
    """Collapse weekly roster rows to one row per player: their LAST team that
    season (highest week). Returns {gsis_id: {'team','name','pos'}}."""
    df = roster_df
    if "week" in df.columns:
        df = df.sort_values("week")
    result = {}
    for _, row in df.iterrows():
        pid = row.get("gsis_id")
        if not pid or str(pid).strip() == "":
            continue
        team = normalize_team(row.get("team"))
        if not team:
            continue
        # Overwrite so the LAST (highest-week) team wins
        result[pid] = {
            "team": team,
            "name": row.get("full_name") or "",
            "pos": str(row.get("position", "")).upper(),
        }
    return result


def collect_roster_changes(target_year, prior_year, player_values):
    print("   Loading rosters (free agency / departures)...")
    try:
        prior = _to_pandas(nfl.load_rosters([prior_year]))
        curr = _to_pandas(nfl.load_rosters([target_year]))
    except Exception as e:
        print(f"   ⚠️  Could not load rosters: {e}")
        return []

    prior_end = _season_end_teams(prior)
    curr_end = _season_end_teams(curr)

    transactions = []

    # Players on a current roster whose team differs from prior season
    for pid, cur in curr_end.items():
        new_team = cur["team"]
        old = prior_end.get(pid)
        if old and old["team"] and old["team"] != new_team:
            val, rpos, epa = player_value(cur["name"], cur["pos"], player_values)
            transactions.append({
                "type": "Free Agent Signing", "player": cur["name"], "position": rpos,
                "from_team": old["team"], "to_team": new_team,
                "value": round(val, 2), "epa": epa, "season": target_year,
                "date": "", "description": f"{cur['name']} {old['team']}→{new_team}",
            })

    # Players on prior roster, absent from every current roster → departure
    for pid, old in prior_end.items():
        if pid in curr_end:
            continue
        val, rpos, epa = player_value(old["name"], old["pos"], player_values)
        if val and val > 0.3:  # per-game threshold now (values are smaller)
            transactions.append({
                "type": "Released / Retired", "player": old["name"], "position": rpos,
                "from_team": old["team"], "to_team": None,
                "value": round(val, 2), "epa": epa, "season": target_year,
                "date": "", "description": f"{old['name']} left {old['team']}",
            })

    n_moves = sum(1 for t in transactions if t["type"] == "Free Agent Signing")
    n_dep = sum(1 for t in transactions if t["type"] == "Released / Retired")
    print(f"   → {n_moves} team changes, {n_dep} departures")
    return transactions


def collect_trades(target_year, player_values):
    print("   Loading trades...")
    try:
        trades = _to_pandas(nfl.load_trades(seasons=[target_year]))
    except Exception as e:
        print(f"   ⚠️  Could not load trades: {e}")
        return []

    txs = []
    for _, row in trades.iterrows():
        gave = normalize_team(row.get("gave"))
        recv = normalize_team(row.get("received"))
        if pd.notna(row.get("pick_season")):
            val = draft_pick_value(row.get("pick_number"))
            txs.append({
                "type": "Trade (pick)",
                "player": f"Pick R{row.get('pick_round','?')} #{row.get('pick_number','?')}",
                "position": "PICK", "from_team": gave, "to_team": recv,
                "value": val, "epa": None, "season": target_year,
                "date": "", "description": "traded pick",
            })
        else:
            name = row.get("pfr_name") or "Unknown"
            val, pos, epa = player_value(name, "", player_values)
            txs.append({
                "type": "Trade (player)", "player": name, "position": pos,
                "from_team": gave, "to_team": recv,
                "value": round(val, 2), "epa": epa, "season": target_year,
                "date": "", "description": f"trade {gave}→{recv}",
            })
    print(f"   → {len(txs)} trade rows")
    return txs


def collect_draft(target_year):
    print("   Loading draft picks...")
    try:
        draft = _to_pandas(nfl.load_draft_picks([target_year]))
    except Exception as e:
        print(f"   ⚠️  Could not load draft: {e}")
        return []

    txs = []
    for _, row in draft.iterrows():
        team = normalize_team(row.get("team"))
        overall = row.get("pick")
        name = row.get("pfr_player_name") or row.get("player_name") or "Drafted Player"
        pos = str(row.get("position", "")).upper()
        txs.append({
            "type": "Draft Pick",
            "player": f"{name} (R{row.get('round','?')} #{overall})",
            "position": pos, "from_team": None, "to_team": team,
            "value": draft_pick_value(overall), "epa": None,
            "season": target_year, "date": "", "description": "drafted",
        })
    print(f"   → {len(txs)} draft picks")
    return txs


# ── Manual coaching file (you maintain) ─────────────────────────────────────
def load_manual_coaching(target_year, output_dir=OUTPUT_DIR):
    """
    Reads nfl-transactions/manual_coaching_{year}.csv.
    Columns: type,coach,role,from_team,to_team,value,date,note
      - type: 'Coaching Hire' or 'Coaching Departure'
      - role: used to derive value when the value column is blank
      - value: optional; falls back to role-based COACH_VALUES
      - from_team / to_team: blank where not applicable
    """
    path = os.path.join(output_dir, f"manual_coaching_{target_year}.csv")
    if not os.path.exists(path):
        print(f"   (no manual coaching file at {path} — skipping coaches)")
        return []
    manual = pd.read_csv(path)
    rows = []
    for _, r in manual.iterrows():
        role = str(r.get("role", "")).strip().lower()
        val = r.get("value")
        if pd.isna(val) or str(val).strip() == "":
            val = COACH_VALUES.get(role, COACH_VALUES["coach"])
        ft = r.get("from_team")
        tt = r.get("to_team")
        rows.append({
            "type": str(r.get("type", "Coaching Hire")),
            "player": str(r.get("coach", "")),
            "position": "COACH",
            "from_team": normalize_team(ft) if pd.notna(ft) and str(ft).strip() else None,
            "to_team": normalize_team(tt) if pd.notna(tt) and str(tt).strip() else None,
            "value": round(float(val), 2),
            "epa": None,
            "season": target_year,
            "date": str(r.get("date", "")),
            "description": str(r.get("note", "")),
        })
    print(f"   ✅ Loaded {len(rows)} manual coaching transactions")
    return rows


# ── Aggregation: inbound / outbound tracked independently ───────────────────
def aggregate_team_deltas(transactions):
    teams = {}

    def ensure(team):
        if team and team not in teams:
            teams[team] = {
                "team": team, "net_delta": 0.0,
                "inbound_value": 0.0, "outbound_value": 0.0,
                "offense_delta": 0.0, "defense_delta": 0.0, "coaching_delta": 0.0,
                "inbound_moves": 0, "outbound_moves": 0, "num_moves": 0,
            }

    OFFENSE_POS = SKILL_OFFENSE | OL

    def unit_of(pos):
        if pos == "COACH":
            return "coaching_delta"
        if pos in OFFENSE_POS or pos == "PICK":
            return "offense_delta"
        return "defense_delta"

    for t in transactions:
        val = abs(float(t.get("value") or 0))
        pos = t.get("position", "")
        unit = unit_of(pos)
        to_team, from_team = t.get("to_team"), t.get("from_team")

        if to_team:
            ensure(to_team)
            teams[to_team]["inbound_value"] += val
            teams[to_team]["net_delta"] += val
            teams[to_team][unit] += val
            teams[to_team]["inbound_moves"] += 1
            teams[to_team]["num_moves"] += 1

        if from_team:
            ensure(from_team)
            teams[from_team]["outbound_value"] += val
            teams[from_team]["net_delta"] -= val
            teams[from_team][unit] -= val
            teams[from_team]["outbound_moves"] += 1
            teams[from_team]["num_moves"] += 1

    result = []
    for team, d in teams.items():
        for k in ("net_delta", "inbound_value", "outbound_value",
                  "offense_delta", "defense_delta", "coaching_delta"):
            d[k] = round(d[k], 2)
        result.append(d)
    result.sort(key=lambda x: -x["net_delta"])
    return result


def main():
    target_year = get_target_year()
    prior_year = target_year - 1

    print(f"\n🏈 Collecting {target_year} transactions via nflreadpy "
          f"(valuing against {prior_year})...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    player_values = build_player_values(prior_year, target_year)
    print(f"   Built value map for {len(player_values)} players")

    transactions = []
    transactions += collect_roster_changes(target_year, prior_year, player_values)
    transactions += collect_trades(target_year, player_values)
    transactions += collect_draft(target_year)
    transactions += load_manual_coaching(target_year)

    if not transactions:
        print("   ⚠️  No transactions collected from any source")
        return

    tx_df = pd.DataFrame(transactions)
    tx_file = os.path.join(OUTPUT_DIR, f"{target_year}_transactions.csv")
    tx_df.to_csv(tx_file, index=False)
    print(f"   ✅ Saved transaction log → {tx_file} ({len(tx_df)} rows)")

    deltas = aggregate_team_deltas(transactions)
    deltas_df = pd.DataFrame(deltas)
    deltas_file = os.path.join(OUTPUT_DIR, f"{target_year}_team_deltas.csv")
    deltas_df.to_csv(deltas_file, index=False)
    print(f"   ✅ Saved team deltas → {deltas_file}")

    # Sanity: total inbound should equal total outbound (every move has 2 sides,
    # except draft picks & departures which are one-sided by design)
    ti = deltas_df["inbound_value"].sum()
    to = deltas_df["outbound_value"].sum()
    print(f"   Σ inbound={ti:.1f}  Σ outbound={to:.1f}")

    print(f"\n   📊 Top 5 net gainers:")
    for d in deltas[:5]:
        print(f"      {d['team']:<4} {d['net_delta']:>+7.2f}  "
              f"(in {d['inbound_value']:.0f} / out {d['outbound_value']:.0f})")
    print(f"   📉 Bottom 5:")
    for d in deltas[-5:]:
        print(f"      {d['team']:<4} {d['net_delta']:>+7.2f}  "
              f"(in {d['inbound_value']:.0f} / out {d['outbound_value']:.0f})")


if __name__ == "__main__":
    main()
