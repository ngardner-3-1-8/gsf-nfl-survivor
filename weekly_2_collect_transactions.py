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

POSITION_BASELINE = {
    "QB": 6.0, "RB": 8.0, "WR": 12.0, "TE": 8.0, "FB": 2.0,
    "T": 10.0, "G": 8.0, "C": 8.0, "OL": 9.0, "OT": 10.0, "OG": 8.0, "LT": 10.0, "RT": 10.0,
    "DE": 14.0, "EDGE": 14.0, "DT": 11.0, "NT": 8.0, "DL": 11.0,
    "LB": 9.0, "ILB": 8.0, "OLB": 10.0, "MLB": 9.0,
    "CB": 12.0, "S": 9.0, "SS": 9.0, "FS": 9.0, "DB": 10.0,
    "K": 4.0, "P": 2.0, "LS": 1.0,
}

COACH_VALUES = {
    "head coach": 12.0,
    "offensive coordinator": 6.0,
    "defensive coordinator": 6.0,
    "special teams coordinator": 2.0,
    "coordinator": 5.0,
    "coach": 1.5,
    "general manager": 4.0,
}

EPA_TO_POINTS = 0.15

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
def build_player_values(prior_year):
    """name(lower) -> {value, position, epa}"""
    print(f"   Loading {prior_year} player stats for valuation...")
    stats = _to_pandas(nfl.load_player_stats([prior_year]))

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
        if key not in agg:
            agg[key] = {"position": pos, "total_epa": 0.0}
        agg[key]["total_epa"] += epa

    values = {}
    for key, d in agg.items():
        pos = d["position"]
        if pos in SKILL_OFFENSE:
            epa_pts = d["total_epa"] * EPA_TO_POINTS
            if pos == "QB" and abs(epa_pts) < 1.0:
                pts = POSITION_BASELINE.get("QB", 6.0)
            else:
                pts = round(epa_pts, 2)
        else:
            pts = POSITION_BASELINE.get(pos, 2.0)
        values[key] = {"value": pts, "position": pos, "epa": round(d["total_epa"], 2)}
    return values


def player_value(name, position, player_values):
    key = str(name).strip().lower()
    if key in player_values:
        d = player_values[key]
        return d["value"], d["position"] or position, d["epa"]
    return POSITION_BASELINE.get(str(position).upper(), 2.0), position, None


# ── Signings & departures from roster comparison ────────────────────────────
def collect_roster_changes(target_year, prior_year, player_values):
    print("   Loading rosters (free agency / departures)...")
    try:
        prior = _to_pandas(nfl.load_rosters([prior_year]))
        curr = _to_pandas(nfl.load_rosters([target_year]))
    except Exception as e:
        print(f"   ⚠️  Could not load rosters: {e}")
        return []

    def pid_of(row):
        return row.get("gsis_id") or row.get("player_id")

    prior_team, prior_meta = {}, {}
    for _, row in prior.iterrows():
        pid = pid_of(row)
        if pid:
            prior_team[pid] = normalize_team(row.get("team"))
            prior_meta[pid] = {
                "name": row.get("full_name") or row.get("player_name") or "",
                "pos": str(row.get("position", "")).upper(),
            }

    transactions, seen = [], set()
    for _, row in curr.iterrows():
        pid = pid_of(row)
        if not pid or pid in seen:
            continue
        seen.add(pid)

        new_team = normalize_team(row.get("team"))
        old_team = prior_team.get(pid)
        name = row.get("full_name") or row.get("player_name") or ""
        pos = str(row.get("position", "")).upper()

        # Changed teams → both sides of the move
        if old_team and new_team and old_team != new_team:
            val, rpos, epa = player_value(name, pos, player_values)
            transactions.append({
                "type": "Free Agent Signing", "player": name, "position": rpos,
                "from_team": old_team, "to_team": new_team,
                "value": round(val, 2), "epa": epa, "season": target_year,
                "date": "", "description": f"{name} {old_team}→{new_team}",
            })

    # Departures — on prior roster, absent from every current roster
    curr_ids = set(seen)
    for pid, old_team in prior_team.items():
        if pid in curr_ids or not old_team:
            continue
        meta = prior_meta.get(pid, {})
        name, pos = meta.get("name", ""), meta.get("pos", "")
        val, rpos, epa = player_value(name, pos, player_values)
        if val and val > 1.0:  # only meaningful departures
            transactions.append({
                "type": "Released / Retired", "player": name, "position": rpos,
                "from_team": old_team, "to_team": None,
                "value": round(val, 2), "epa": epa, "season": target_year,
                "date": "", "description": f"{name} left {old_team}",
            })
    print(f"   → {len(transactions)} roster changes")
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

    player_values = build_player_values(prior_year)
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
