"""
weekly_2_collect_transactions.py  (Spotrac version)

Scrapes NFL transactions from Spotrac (players AND coaches), values each
move in points, and produces:
  nfl-transactions/{year}_transactions.csv   (one row per transaction)
  nfl-transactions/{year}_team_deltas.csv    (aggregated per team)

Valuation model (unchanged from prior design):
  - Skill players (QB/RB/WR/TE): prior-season total EPA scaled to points
  - Defense / OL / unknown: position baseline
  - Coaches: role-based baseline (HC > OC/DC > position coach)
  - Departures (release/retire/fired): negative value for losing team
"""

import os
import re
import time
import random
import pandas as pd
import numpy as np
import nflreadpy as nfl
from datetime import datetime
from bs4 import BeautifulSoup
import undetected_chromedriver as uc

OUTPUT_DIR = "nfl-transactions"
DEBUG_DUMP_HTML = True  # set False once selectors are confirmed working

# ── Valuation constants (same knobs as before) ────────────────────────────
EPA_TO_POINTS = 0.15

SKILL_OFFENSE = {"QB", "RB", "WR", "TE", "FB"}

POSITION_BASELINE = {
    "QB": 6.0, "RB": 8.0, "WR": 12.0, "TE": 8.0, "FB": 2.0,
    "T": 10.0, "G": 8.0, "C": 8.0, "OL": 9.0, "OT": 10.0, "OG": 8.0, "LT": 10.0, "RT": 10.0,
    "DE": 14.0, "EDGE": 14.0, "DT": 11.0, "NT": 8.0, "DL": 11.0,
    "LB": 9.0, "ILB": 8.0, "OLB": 10.0, "MLB": 9.0,
    "CB": 12.0, "S": 9.0, "SS": 9.0, "FS": 9.0, "DB": 10.0,
    "K": 4.0, "P": 2.0, "LS": 1.0,
}

# Coaching role values in points — tunable
COACH_VALUES = {
    "head coach": 12.0,
    "offensive coordinator": 6.0,
    "defensive coordinator": 6.0,
    "special teams coordinator": 2.0,
    "coordinator": 5.0,       # generic fallback
    "coach": 1.5,             # position coaches etc.
    "general manager": 4.0,
}

# Spotrac team names → your abbreviations
SPOTRAC_TEAM_MAP = {
    "arizona cardinals": "ARI", "atlanta falcons": "ATL", "baltimore ravens": "BAL",
    "buffalo bills": "BUF", "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE", "dallas cowboys": "DAL",
    "denver broncos": "DEN", "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND", "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC", "los angeles rams": "LA", "los angeles chargers": "LAC",
    "las vegas raiders": "LV", "miami dolphins": "MIA", "minnesota vikings": "MIN",
    "new england patriots": "NE", "new orleans saints": "NO", "new york giants": "NYG",
    "new york jets": "NYJ", "philadelphia eagles": "PHI", "pittsburgh steelers": "PIT",
    "seattle seahawks": "SEA", "san francisco 49ers": "SF", "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN", "washington commanders": "WAS",
}


def get_target_year(today=None):
    today = today or datetime.now()
    return today.year - 1 if today.month < 6 else today.year


def normalize_team(text):
    """Match a Spotrac team name (or fragment) to an abbreviation."""
    if not text:
        return None
    t = str(text).strip().lower()
    if t in SPOTRAC_TEAM_MAP:
        return SPOTRAC_TEAM_MAP[t]
    # Partial match — Spotrac sometimes shows just "Cardinals" or logo alt text
    for full, abbr in SPOTRAC_TEAM_MAP.items():
        if t in full or full.split()[-1] == t:
            return abbr
    # Already an abbreviation?
    up = t.upper()
    if up in SPOTRAC_TEAM_MAP.values():
        return up
    return None


# ── Transaction type classification from description text ─────────────────
def classify_transaction(description):
    """
    Returns (tx_type, direction) where direction is:
      +1 = team gains value, -1 = team loses value, 0 = neutral/unknown
    """
    d = str(description).lower()

    # Coaching moves
    if any(k in d for k in ("hired", "named", "promoted to")):
        if any(k in d for k in COACH_VALUES.keys()):
            return "Coaching Hire", +1
    if "fired" in d or "relieved" in d or ("parted ways" in d and "coach" in d):
        return "Coaching Departure", -1

    # Player moves
    if "signed" in d or "agreed to terms" in d or "claimed" in d:
        return "Signing", +1
    if "re-signed" in d or "extension" in d or "restructure" in d:
        return "Extension / Re-sign", 0   # keeps existing value, no delta
    if "traded" in d or "acquired" in d:
        return "Trade", +1                # direction handled by from/to teams
    if "released" in d or "waived" in d or "cut" in d:
        return "Release", -1
    if "retired" in d or "retirement" in d:
        return "Retirement", -1
    if "drafted" in d or "draft pick" in d:
        return "Draft Pick", +1
    if "suspended" in d:
        return "Suspension", -1
    if "franchise tag" in d or "transition tag" in d:
        return "Tag", 0

    return "Other", 0


def coach_value(description):
    """Value a coaching move from its description text."""
    d = str(description).lower()
    for role, val in COACH_VALUES.items():
        if role in d:
            return val
    return COACH_VALUES["coach"]


# ── Player valuation (same EPA model as before) ────────────────────────────
def build_player_values(prior_year):
    """name(lower) -> {value, position, epa}"""
    print(f"   Loading {prior_year} player stats for valuation...")
    stats = nfl.load_player_stats([prior_year])
    stats = stats.to_pandas() if hasattr(stats, "to_pandas") else stats

    agg = {}
    for _, row in stats.iterrows():
        name = row.get("player_display_name") or row.get("player_name")
        if not name:
            continue
        key = str(name).strip().lower()
        pos = str(row.get("position", "")).upper()
        epa = (
            float(row.get("passing_epa", 0) or 0)
            + float(row.get("rushing_epa", 0) or 0)
            + float(row.get("receiving_epa", 0) or 0)
        )
        if key not in agg:
            agg[key] = {"position": pos, "total_epa": 0.0}
        agg[key]["total_epa"] += epa

    values = {}
    for key, d in agg.items():
        pos = d["position"]
        if pos in SKILL_OFFENSE:
            pts = round(d["total_epa"] * EPA_TO_POINTS, 2)
        else:
            pts = POSITION_BASELINE.get(pos, 2.0)
        values[key] = {"value": pts, "position": pos, "epa": round(d["total_epa"], 2)}
    return values


def player_value(name, position, player_values):
    """Look up value; fall back to position baseline."""
    key = str(name).strip().lower()
    if key in player_values:
        d = player_values[key]
        return d["value"], d["position"] or position, d["epa"]
    return POSITION_BASELINE.get(str(position).upper(), 2.0), position, None


# ── Spotrac scraping ────────────────────────────────────────────────────────
def scrape_spotrac_transactions(year):
    """
    Scrapes all transactions for a calendar year from Spotrac using
    undetected_chromedriver (Spotrac blocks plain requests).
    Returns list of raw dicts: {date, team, player, position, description}
    """
    url = (
        f"https://www.spotrac.com/nfl/transactions/_/year/{year}"
        f"/start/{year}-01-01/end/{year}-12-31"
    )
    print(f"   Launching browser for: {url}")

    options = uc.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,3000")

    driver = uc.Chrome(options=options)
    raw_transactions = []

    try:
        driver.get(url)
        # Let JS render + any lazy loading settle
        time.sleep(5 + random.uniform(0, 2))

        # Scroll to bottom a few times to trigger lazy-loaded rows
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)

        html = driver.page_source

        if DEBUG_DUMP_HTML:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            debug_path = os.path.join(OUTPUT_DIR, f"spotrac_debug_{year}.html")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"   🐛 Dumped page HTML → {debug_path} (inspect if parsing misses rows)")

        soup = BeautifulSoup(html, "html.parser")

        # ── Strategy 1: standard Spotrac table rows ──
        # Spotrac transaction pages typically use <table> with rows containing
        # date, team logo/link, player link, and a description cell.
        rows = soup.select("table tbody tr")
        print(f"   Found {len(rows)} table rows")

        current_date = None
        for tr in rows:
            cells = tr.find_all("td")
            if not cells:
                continue

            text_cells = [c.get_text(" ", strip=True) for c in cells]
            row_text = " | ".join(text_cells)

            # Date detection — either a dedicated date cell or a date-only row
            date_match = re.search(
                r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})",
                row_text,
            )
            if date_match:
                current_date = date_match.group(1)

            # Team — from a team link href (/nfl/<team-slug>/) or logo alt
            team = None
            team_link = tr.select_one("a[href*='/nfl/']")
            if team_link:
                href = team_link.get("href", "")
                m = re.search(r"/nfl/([a-z\-]+)/", href)
                if m:
                    slug = m.group(1).replace("-", " ")
                    team = normalize_team(slug)
            if not team:
                img = tr.find("img")
                if img and img.get("alt"):
                    team = normalize_team(img["alt"])

            # Player — usually the first player-profile link
            player = None
            player_link = tr.select_one("a[href*='/player/'], a[href*='/redirect/player/']")
            if player_link:
                player = player_link.get_text(strip=True)

            # Description — longest text cell (the sentence describing the move)
            description = max(text_cells, key=len) if text_cells else ""

            # Position — Spotrac often prefixes player names like "QB John Smith"
            position = ""
            pos_match = re.match(
                r"^(QB|RB|WR|TE|FB|T|G|C|OT|OG|OL|LT|RT|DE|DT|NT|EDGE|LB|ILB|OLB|MLB|CB|S|SS|FS|DB|K|P|LS)\b",
                description,
            )
            if pos_match:
                position = pos_match.group(1)

            if not description or len(description) < 10:
                continue  # skip empty/nav rows

            raw_transactions.append({
                "date": current_date,
                "team": team,
                "player": player or extract_name_from_description(description),
                "position": position,
                "description": description,
            })

    finally:
        driver.quit()

    print(f"   ✅ Scraped {len(raw_transactions)} raw transactions")
    return raw_transactions


def extract_name_from_description(description):
    """Fallback: pull 'Firstname Lastname' from the description text."""
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z'\-\.]+)+)\b", description)
    return m.group(1) if m else "Unknown"


# ── Transform raw → valued transactions ────────────────────────────────────
def value_transactions(raw_transactions, player_values, target_year):
    transactions = []
    for raw in raw_transactions:
        tx_type, direction = classify_transaction(raw["description"])

        if tx_type in ("Other", "Tag", "Extension / Re-sign", "Suspension"):
            # Log it with zero delta so it appears in the feed but doesn't move ratings
            # (Suspension could be negative later — start neutral to avoid noise)
            value = 0.0
            epa = None
            pos = raw["position"]
        elif tx_type in ("Coaching Hire", "Coaching Departure"):
            value = coach_value(raw["description"]) * direction
            epa = None
            pos = "COACH"
        else:
            base_val, pos, epa = player_value(
                raw["player"], raw["position"], player_values
            )
            value = base_val * direction

        transactions.append({
            "date": raw["date"],
            "type": tx_type,
            "player": raw["player"],
            "position": pos,
            "team": raw["team"],
            "from_team": raw["team"] if direction < 0 else None,
            "to_team": raw["team"] if direction >= 0 else None,
            "value": round(value, 2),
            "epa": epa,
            "description": raw["description"][:200],
            "season": target_year,
        })
    return transactions


def aggregate_team_deltas(transactions):
    teams = {}

    def ensure(team):
        if team and team not in teams:
            teams[team] = {
                "team": team, "net_delta": 0.0, "additions": 0.0,
                "subtractions": 0.0, "offense_delta": 0.0,
                "defense_delta": 0.0, "coaching_delta": 0.0, "num_moves": 0,
            }

    OFFENSE_POS = SKILL_OFFENSE | {"T", "G", "C", "OL", "OT", "OG", "LT", "RT"}

    for t in transactions:
        team = t.get("to_team") or t.get("from_team") or t.get("team")
        if not team:
            continue
        ensure(team)
        val = t["value"] or 0
        pos = t.get("position", "")

        teams[team]["net_delta"] += val
        teams[team]["num_moves"] += 1
        if val >= 0:
            teams[team]["additions"] += val
        else:
            teams[team]["subtractions"] += val

        if pos == "COACH":
            teams[team]["coaching_delta"] += val
        elif pos in OFFENSE_POS:
            teams[team]["offense_delta"] += val
        else:
            teams[team]["defense_delta"] += val

    result = []
    for team, d in teams.items():
        for k in ("net_delta", "additions", "subtractions",
                  "offense_delta", "defense_delta", "coaching_delta"):
            d[k] = round(d[k], 2)
        result.append(d)
    result.sort(key=lambda x: -x["net_delta"])
    return result


def main():
    target_year = get_target_year()
    prior_year = target_year - 1
    # Spotrac URL uses the calendar year of the offseason
    scrape_year = datetime.now().year

    print(f"\n🏈 Collecting {scrape_year} transactions from Spotrac "
          f"(valuing against {prior_year} stats)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    player_values = build_player_values(prior_year)
    print(f"   Built value map for {len(player_values)} players")

    raw = scrape_spotrac_transactions(scrape_year)
    if not raw:
        print("   ⚠️  No transactions scraped — check the debug HTML dump")
        return

    transactions = value_transactions(raw, player_values, target_year)

    tx_df = pd.DataFrame(transactions)
    tx_file = os.path.join(OUTPUT_DIR, f"{target_year}_transactions.csv")
    tx_df.to_csv(tx_file, index=False)
    print(f"   ✅ Saved transaction log → {tx_file} ({len(tx_df)} rows)")

    deltas = aggregate_team_deltas(transactions)
    deltas_df = pd.DataFrame(deltas)
    deltas_file = os.path.join(OUTPUT_DIR, f"{target_year}_team_deltas.csv")
    deltas_df.to_csv(deltas_file, index=False)
    print(f"   ✅ Saved team deltas → {deltas_file}")

    print(f"\n   📊 Top 5 offseason gainers:")
    for d in deltas[:5]:
        print(f"      {d['team']:<4} +{d['net_delta']:>6.2f} pts ({d['num_moves']} moves)")
    print(f"   📉 Bottom 5:")
    for d in deltas[-5:]:
        print(f"      {d['team']:<4} {d['net_delta']:>7.2f} pts ({d['num_moves']} moves)")


if __name__ == "__main__":
    main()
