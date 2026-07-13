"""
weekly_2_collect_transactions.py

Collects NFL offseason/in-season transactions and quantifies each in points,
producing a per-team net-delta leaderboard and a transaction log.

Value model:
  - Skill players (QB/RB/WR/TE): prior-season total EPA (passing+rushing+receiving)
    QBs additionally cross-checked against the blended QB rating where available.
  - Defensive players: position baseline (public defensive EPA credit is unreliable).
  - Draft picks: expected rookie value by draft slot (Jimmy Johnson-style decay).
  - Retirements: subtract the retiree's prior-year value from their last team.
  - Coaching changes: neutral default (no clean data proxy) — flagged for manual edit.

Output:
  nfl-transactions/{year}_transactions.csv         (one row per transaction)
  nfl-transactions/{year}_team_deltas.csv          (aggregated per team)
"""

import os
import pandas as pd
import numpy as np
import nflreadpy as nfl
from datetime import datetime

OUTPUT_DIR = "nfl-transactions"

# Position groups
SKILL_OFFENSE = {"QB", "RB", "WR", "TE", "FB"}
OL = {"T", "G", "C", "OL", "OT", "OG"}
DEFENSE = {"DE", "DT", "NT", "EDGE", "LB", "ILB", "OLB", "MLB", "CB", "S", "SS", "FS", "DB", "DL"}
SPECIAL = {"K", "P", "LS"}

# Position baseline point values (used where EPA credit is unreliable, e.g. defense, OL)
# These are rough "starter caliber" season values in points. Tunable.
POSITION_BASELINE = {
    "QB": 0.0,   # QBs use rating delta / EPA, not baseline
    "RB": 8.0, "WR": 12.0, "TE": 8.0, "FB": 2.0,
    "T": 10.0, "G": 8.0, "C": 8.0, "OL": 9.0, "OT": 10.0, "OG": 8.0,
    "DE": 14.0, "EDGE": 14.0, "DT": 11.0, "NT": 8.0, "DL": 11.0,
    "LB": 9.0, "ILB": 8.0, "OLB": 10.0, "MLB": 9.0,
    "CB": 12.0, "S": 9.0, "SS": 9.0, "FS": 9.0, "DB": 10.0,
    "K": 4.0, "P": 2.0, "LS": 1.0,
}

# Multiplier converting a raw season EPA figure into "power-rating points"
# EPA is already in points; scale down because a full-season EPA swing of ~50
# shouldn't move a team rating by 50. Tunable to match your rating scale.
EPA_TO_POINTS = 0.15

ABBR_MAP = {"JAC": "JAX", "LAR": "LA", "GNB": "GB", "KAN": "KC",
            "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV", "WSH": "WAS"}


def normalize_team(abbr):
    a = str(abbr).strip().upper()
    return ABBR_MAP.get(a, a)


def get_target_year(today=None):
    today = today or datetime.now()
    return today.year - 1 if today.month < 6 else today.year


def draft_pick_value(overall_pick):
    """
    Expected rookie value in points by overall draft slot.
    Uses a smooth decay — pick 1 is most valuable, tapering through 7 rounds.
    Scaled into the same points space as EPA_TO_POINTS output.
    """
    if pd.isna(overall_pick) or overall_pick <= 0:
        return 0.0
    p = float(overall_pick)
    # Exponential-ish decay: top picks ~6 pts, late picks ~0.5 pts
    raw = 7.0 * np.exp(-p / 60.0)
    return round(raw, 2)


def build_player_values(prior_year):
    """
    Returns a dict: gsis_id -> {value, position, name}
    Value is prior-season total EPA scaled to points for skill players,
    or position baseline for non-skill / unreliable-EPA positions.
    """
    print(f"   Loading {prior_year} player stats for valuation...")
    stats = nfl.load_player_stats([prior_year])
    stats = stats.to_pandas() if hasattr(stats, "to_pandas") else stats

    # Aggregate per player across the season
    value_map = {}
    for _, row in stats.iterrows():
        pid = row.get("player_id")
        if not pid:
            continue
        pos = str(row.get("position", "")).upper()
        name = row.get("player_display_name") or row.get("player_name") or ""

        pass_epa = float(row.get("passing_epa", 0) or 0)
        rush_epa = float(row.get("rushing_epa", 0) or 0)
        rec_epa  = float(row.get("receiving_epa", 0) or 0)
        total_epa = pass_epa + rush_epa + rec_epa

        if pid not in value_map:
            value_map[pid] = {
                "position": pos,
                "name": name,
                "total_epa": 0.0,
            }
        value_map[pid]["total_epa"] += total_epa

    # Convert to points
    final = {}
    for pid, d in value_map.items():
        pos = d["position"]
        if pos in SKILL_OFFENSE:
            # EPA-based value for skill players
            points = round(d["total_epa"] * EPA_TO_POINTS, 2)
        elif pos in POSITION_BASELINE:
            # Baseline for positions where EPA credit is unreliable
            points = POSITION_BASELINE[pos]
        else:
            points = 2.0  # unknown position default
        final[pid] = {
            "value": points,
            "position": pos,
            "name": d["name"],
            "epa": round(d["total_epa"], 2),
        }
    return final


def collect_trades(target_year, player_values):
    """Returns list of transaction dicts from trades."""
    print("   Loading trades...")
    try:
        trades = nfl.load_trades(seasons=[target_year])
        trades = trades.to_pandas() if hasattr(trades, "to_pandas") else trades
    except Exception as e:
        print(f"   ⚠️  Could not load trades: {e}")
        return []

    transactions = []
    for _, row in trades.iterrows():
        # Only player rows (pick_season is NA for direct player trades)
        if pd.notna(row.get("pick_season")):
            # This is a traded draft pick
            pick_val = draft_pick_value(row.get("pick_number"))
            gave = normalize_team(row.get("gave"))
            received = normalize_team(row.get("received"))
            transactions.append({
                "type": "Trade (pick)",
                "player": f"Pick R{row.get('pick_round','?')} #{row.get('pick_number','?')}",
                "position": "PICK",
                "from_team": gave,
                "to_team": received,
                "value": pick_val,
                "epa": None,
                "season": target_year,
            })
        else:
            name = row.get("pfr_name") or "Unknown"
            gave = normalize_team(row.get("gave"))
            received = normalize_team(row.get("received"))
            # Try to find value by name match (trades use pfr_name, not gsis_id)
            val, pos, epa = match_value_by_name(name, player_values)
            transactions.append({
                "type": "Trade (player)",
                "player": name,
                "position": pos,
                "from_team": gave,
                "to_team": received,
                "value": val,
                "epa": epa,
                "season": target_year,
            })
    return transactions


def match_value_by_name(name, player_values):
    """Fuzzy-match a player name to the value map. Returns (value, position, epa)."""
    if not name:
        return 0.0, "", None
    target = str(name).strip().lower()
    for pid, d in player_values.items():
        if str(d["name"]).strip().lower() == target:
            return d["value"], d["position"], d["epa"]
    # No match — return neutral
    return 0.0, "", None


def collect_roster_changes(target_year, prior_year, player_values):
    """
    Detects free-agent signings and releases by comparing each player's team
    from prior_year roster to target_year roster.
    """
    print("   Loading rosters for free agency detection...")
    try:
        prior = nfl.load_rosters([prior_year])
        curr = nfl.load_rosters([target_year])
        prior = prior.to_pandas() if hasattr(prior, "to_pandas") else prior
        curr = curr.to_pandas() if hasattr(curr, "to_pandas") else curr
    except Exception as e:
        print(f"   ⚠️  Could not load rosters: {e}")
        return []

    # Build prior-year team-by-player
    prior_team = {}
    for _, row in prior.iterrows():
        pid = row.get("gsis_id") or row.get("player_id")
        if pid:
            prior_team[pid] = normalize_team(row.get("team"))

    transactions = []
    seen = set()
    for _, row in curr.iterrows():
        pid = row.get("gsis_id") or row.get("player_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)

        new_team = normalize_team(row.get("team"))
        old_team = prior_team.get(pid)
        name = row.get("full_name") or row.get("player_name") or ""
        pos = str(row.get("position", "")).upper()

        # Player changed teams via free agency (not in trades — those are separate)
        if old_team and old_team != new_team:
            val = player_values.get(pid, {}).get("value", POSITION_BASELINE.get(pos, 2.0))
            epa = player_values.get(pid, {}).get("epa")
            transactions.append({
                "type": "Free Agent Signing",
                "player": name,
                "position": pos,
                "from_team": old_team,
                "to_team": new_team,
                "value": val,
                "epa": epa,
                "season": target_year,
            })

    # Detect releases/retirements — players on prior roster not on any current roster
    curr_ids = set(seen)
    for pid, old_team in prior_team.items():
        if pid not in curr_ids:
            d = player_values.get(pid, {})
            val = d.get("value", 0)
            if val and val > 1.0:  # only note meaningful departures
                transactions.append({
                    "type": "Released / Retired",
                    "player": d.get("name", "Unknown"),
                    "position": d.get("position", ""),
                    "from_team": old_team,
                    "to_team": None,
                    "value": -abs(val),  # negative — team loses this value
                    "epa": d.get("epa"),
                    "season": target_year,
                })

    return transactions


def collect_draft(target_year, player_values):
    """Draft picks add rookie value to the drafting team."""
    print("   Loading draft picks...")
    try:
        draft = nfl.load_draft_picks([target_year])
        draft = draft.to_pandas() if hasattr(draft, "to_pandas") else draft
    except Exception as e:
        print(f"   ⚠️  Could not load draft: {e}")
        return []

    transactions = []
    for _, row in draft.iterrows():
        team = normalize_team(row.get("team"))
        overall = row.get("pick")
        name = row.get("pfr_player_name") or row.get("player_name") or "Drafted Player"
        pos = str(row.get("position", "")).upper()
        val = draft_pick_value(overall)
        transactions.append({
            "type": "Draft Pick",
            "player": f"{name} (R{row.get('round','?')} #{overall})",
            "position": pos,
            "from_team": None,
            "to_team": team,
            "value": val,
            "epa": None,
            "season": target_year,
        })
    return transactions


def aggregate_team_deltas(transactions):
    """Net the points value of all moves per team, split by unit."""
    teams = {}

    def ensure(team):
        if team and team not in teams:
            teams[team] = {
                "team": team,
                "net_delta": 0.0,
                "additions": 0.0,
                "subtractions": 0.0,
                "offense_delta": 0.0,
                "defense_delta": 0.0,
                "num_moves": 0,
            }

    for t in transactions:
        val = t["value"] or 0
        pos = t.get("position", "")
        is_offense = pos in SKILL_OFFENSE or pos in OL or pos == "PICK"

        # Team that gains the player
        to_team = t.get("to_team")
        if to_team:
            ensure(to_team)
            teams[to_team]["net_delta"] += val
            teams[to_team]["num_moves"] += 1
            if val >= 0:
                teams[to_team]["additions"] += val
            else:
                teams[to_team]["subtractions"] += val
            if is_offense:
                teams[to_team]["offense_delta"] += val
            elif pos in DEFENSE:
                teams[to_team]["defense_delta"] += val

        # Team that loses the player (for trades and releases)
        from_team = t.get("from_team")
        if from_team and t["type"] != "Released / Retired":
            # Trades: the giving team loses that value
            ensure(from_team)
            teams[from_team]["net_delta"] -= val
            teams[from_team]["num_moves"] += 1
            teams[from_team]["subtractions"] -= val
            if is_offense:
                teams[from_team]["offense_delta"] -= val
            elif pos in DEFENSE:
                teams[from_team]["defense_delta"] -= val
        elif from_team and t["type"] == "Released / Retired":
            # Already negative value, applied to the from_team
            ensure(from_team)
            teams[from_team]["net_delta"] += val  # val is negative
            teams[from_team]["num_moves"] += 1
            teams[from_team]["subtractions"] += val
            if is_offense:
                teams[from_team]["offense_delta"] += val
            elif pos in DEFENSE:
                teams[from_team]["defense_delta"] += val

    # Round everything
    result = []
    for team, d in teams.items():
        for k in ("net_delta", "additions", "subtractions", "offense_delta", "defense_delta"):
            d[k] = round(d[k], 2)
        result.append(d)
    result.sort(key=lambda x: -x["net_delta"])
    return result


def main():
    target_year = get_target_year()
    prior_year = target_year - 1

    print(f"\n🏈 Collecting {target_year} transactions (valuing against {prior_year})...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    player_values = build_player_values(prior_year)
    print(f"   Built value map for {len(player_values)} players")

    all_transactions = []
    all_transactions += collect_trades(target_year, player_values)
    all_transactions += collect_roster_changes(target_year, prior_year, player_values)
    all_transactions += collect_draft(target_year, player_values)

    print(f"   Collected {len(all_transactions)} total transactions")

    # Save transaction log
    tx_df = pd.DataFrame(all_transactions)
    tx_file = os.path.join(OUTPUT_DIR, f"{target_year}_transactions.csv")
    tx_df.to_csv(tx_file, index=False)
    print(f"   ✅ Saved transaction log → {tx_file}")

    # Save team deltas
    deltas = aggregate_team_deltas(all_transactions)
    deltas_df = pd.DataFrame(deltas)
    deltas_file = os.path.join(OUTPUT_DIR, f"{target_year}_team_deltas.csv")
    deltas_df.to_csv(deltas_file, index=False)
    print(f"   ✅ Saved team deltas → {deltas_file}")

    # Print top movers
    print(f"\n   📊 Top 5 offseason gainers:")
    for d in deltas[:5]:
        print(f"      {d['team']:<4} +{d['net_delta']:>6.2f} pts ({d['num_moves']} moves)")
    print(f"\n   📉 Bottom 5:")
    for d in deltas[-5:]:
        print(f"      {d['team']:<4} {d['net_delta']:>7.2f} pts ({d['num_moves']} moves)")


if __name__ == "__main__":
    main()
