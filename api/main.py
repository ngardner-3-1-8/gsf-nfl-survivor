import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from api.models import OptimizeRequest, OptimizeResponse
from api.data_loader import load_current_data
from api.optimizer import run_optimizer
import math
from fastapi.responses import JSONResponse
import glob


def sanitize(obj):
    """Recursively replace nan/inf with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj

app = FastAPI(title="Circa Survivor API", version="1.0.0")


# Allow requests from local dev and your future GitHub Pages URL
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://ngardner-3-1-8.github.io",
    os.environ.get("FRONTEND_URL", ""),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in ALLOWED_ORIGINS if o],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Paths — Railway runs from the repo root
DATA_DIR = os.environ.get("DATA_DIR", ".")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN, inf and other non-JSON-compliant values."""
    # Replace inf values first
    df = df.replace([np.inf, -np.inf], None)
    # Convert to object dtype temporarily to handle mixed types
    for col in df.columns:
        df[col] = df[col].where(pd.notnull(df[col]), None)
    return df

def load_historical_data(data_dir: str, year: int):
    """
    Loads the most recent Season_*_Final_Data.csv for a given historical year.
    Returns the DataFrame and metadata.
    """
    final_data_dir = os.path.join(
        data_dir, f"nfl-power-ratings/final-data/{year}_final_data"
    )
    if not os.path.exists(final_data_dir):
        raise FileNotFoundError(f"No final data directory for {year}")

    # Find the file with the highest week number
    files = glob.glob(os.path.join(final_data_dir, f"Season_{year}_Through_Week_*_Final_Data.csv"))
    if not files:
        raise FileNotFoundError(f"No final data files found for {year}")

    def extract_week(path):
        try:
            name = os.path.basename(path)
            return int(name.split("_Week_")[1].split("_Final")[0])
        except (IndexError, ValueError):
            return 0

    latest_file = max(files, key=extract_week)
    latest_week = extract_week(latest_file)

    df = pd.read_csv(latest_file)
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        df[col] = df[col].where(pd.notnull(df[col]), None)

    return df, latest_week, latest_file


@app.get("/")
def root():
    return {"status": "ok", "message": "Circa Survivor API is running"}


@app.get("/api/schedule")
def get_schedule(week: int = Query(None), year: int = Query(None)):
    try:
        data = load_current_data(DATA_DIR)
        current_year = data["target_year"]

        if year and year != current_year:
            df, latest_week, source_file = load_historical_data(DATA_DIR, year)
            week_col = "Week_x" if "Week_x" in df.columns else "Week"
            if week is not None:
                df = df[df[week_col] == week]
            # Build week options from the data
            circa_col = "Circa Week" if "Circa Week" in df.columns else None
            weeks_df = df[[week_col] + ([circa_col] if circa_col else [])].drop_duplicates()
            week_options = []
            for _, r in weeks_df.iterrows():
                w = int(r[week_col])
                label = str(r[circa_col]) if circa_col and pd.notna(r.get(circa_col)) else f"Week {w}"
                week_options.append({"week": w, "label": label})

            result = {
                "upcoming_week": latest_week,
                "target_year": year,
                "source_file": source_file,
                "weeks": sorted(df[week_col].dropna().unique().tolist()),
                "total_games": len(df),
                "games": df.to_dict(orient="records"),
                "is_historical": True,
            }
            return JSONResponse(content=sanitize(result))

        # Existing live logic unchanged
        df = clean_df(data["sim_df"])
        if week is not None:
            df = df[df["Week_x"] == week]
        result = {
            "upcoming_week": data["upcoming_week"],
            "target_year": current_year,
            "source_file": data["sim_file"],
            "weeks": sorted(df["Week_x"].dropna().unique().tolist()),
            "total_games": len(df),
            "games": df.to_dict(orient="records"),
            "is_historical": False,
        }
        return JSONResponse(content=sanitize(result))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ev")
def get_ev(model: str = Query("consensus"), week: int = Query(None)):
    valid_models = ["consensus", "sportsbook", "mp", "gsf", "sim"]
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {valid_models}")
    try:
        data = load_current_data(DATA_DIR, model=model)
        df = clean_df(data["ev_df"])
        if week is not None:
            df = df[df["Week_x"] == week]
        result = {
            "model": model,
            "upcoming_week": data["upcoming_week"],
            "target_year": data["target_year"],
            "source_file": data["ev_file"],
            "total_rows": len(df),
            "games": df.to_dict(orient="records"),
        }
        return JSONResponse(content=sanitize(result))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/last-updated")
def get_last_updated():
    try:
        data = load_current_data(DATA_DIR)

        # Sim results timestamp
        sim_path = os.path.join(DATA_DIR, "last_updated.json")
        timestamps = {"sim_updated": "Unknown"}
        if os.path.exists(sim_path):
            with open(sim_path, "r") as f:
                timestamps = json.load(f)

        # MP rankings timestamp
        mp_path = os.path.join(DATA_DIR, "mp_ratings_last_updated.json")
        mp_timestamps = {"mp_updated": "Unknown"}
        if os.path.exists(mp_path):
            with open(mp_path, "r") as f:
                mp_timestamps = json.load(f)

        return {
            **timestamps,
            **mp_timestamps,
            "upcoming_week": data["upcoming_week"],
            "target_year": data["target_year"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pick-percentages")
def get_pick_percentages():
    try:
        data = load_current_data(DATA_DIR)
        df = data["sim_df"]
        records = []
        for _, row in df.iterrows():
            week = row.get("Week_x") or row.get("Week")
            records.append({
                "week": week,
                "team": row.get("Home Team"),
                "pick_pct": row.get("Home Pick %", 0) or 0,
            })
            records.append({
                "week": week,
                "team": row.get("Away Team"),
                "pick_pct": row.get("Away Pick %", 0) or 0,
            })
        return JSONResponse(content=sanitize({"picks": records}))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/weeks")
def get_available_weeks(year: int = Query(None)):
    try:
        data = load_current_data(DATA_DIR)
        current_year = data["target_year"]

        if year and year != current_year:
            df, latest_week, _ = load_historical_data(DATA_DIR, year)
            week_col = "Week_x" if "Week_x" in df.columns else "Week"
            circa_col = "Circa Week" if "Circa Week" in df.columns else None
            weeks_df = df[[week_col] + ([circa_col] if circa_col else [])].drop_duplicates()
            week_options = []
            for _, row in weeks_df.iterrows():
                w = int(row[week_col])
                label = str(row[circa_col]) if circa_col and pd.notna(row.get(circa_col)) else f"Week {w}"
                week_options.append({"week": w, "label": label})
            return {
                "upcoming_week": latest_week,
                "target_year": year,
                "weeks": sorted(week_options, key=lambda x: x["week"]),
                "is_historical": True,
            }

        # Existing live logic
        df = data["sim_df"]
        upcoming = data["upcoming_week"]
        week_col = "Week_x" if "Week_x" in df.columns else "Week"
        circa_col = "Circa Week" if "Circa Week" in df.columns else None
        weeks_df = df[[week_col] + ([circa_col] if circa_col else [])].drop_duplicates()
        week_options = []
        for _, row in weeks_df.iterrows():
            w = int(row[week_col])
            label = str(row[circa_col]) if circa_col and pd.notna(row.get(circa_col)) else f"Week {w}"
            week_options.append({"week": w, "label": label})
        return {
            "upcoming_week": upcoming,
            "target_year": current_year,
            "weeks": week_options,
            "is_historical": False,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize")
def optimize(request: OptimizeRequest):
    """
    Runs the OR-Tools SCIP optimizer with user-defined constraints.
    Returns up to N EV-optimized and N win%-optimized solutions.
    """
    try:
        data = load_current_data(DATA_DIR)
        sim_df = data["sim_df"]
        result = run_optimizer(sim_df, request)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rankings")
def get_rankings(year: int = Query(None)):
    try:
        data = load_current_data(DATA_DIR)
        current_year = data["target_year"]
        target = year if year else current_year
        upcoming_week = data["upcoming_week"] if target == current_year else None

        ratings_dir = os.path.join(DATA_DIR, "nfl-power-ratings")

        if target != current_year:
            # For historical years use the latest available rankings file
            files = glob.glob(os.path.join(
                ratings_dir, f"nfl_power_ratings_blended_week_*_{target}.csv"
            ))
            if not files:
                raise FileNotFoundError(f"No rankings file found for {target}")
            def extract_week(p):
                try:
                    return int(os.path.basename(p).split("_week_")[1].split(f"_{target}")[0])
                except:
                    return 0
            current_file = max(files, key=extract_week)
            upcoming_week = extract_week(current_file)
        else:
            current_file = os.path.join(
                ratings_dir,
                f"nfl_power_ratings_blended_week_{upcoming_week}_{target}.csv"
            )
            if not os.path.exists(current_file):
                import glob as g
                files = g.glob(os.path.join(ratings_dir, f"nfl_power_ratings_blended_week_*_{target}.csv"))
                if not files:
                    raise FileNotFoundError(f"No rankings file found for {target}")
                current_file = max(files, key=lambda p: int(os.path.basename(p).split("_week_")[1].split(f"_{target}")[0]))

        df = pd.read_csv(current_file)
        df = df.rename(columns={"Power Rating": "GSF Power Rating"})
        if "GSF Power Rating" in df.columns:
            df["GSF Rank"] = df["GSF Power Rating"].rank(ascending=False, method="min").astype("Int64")
        df = clean_df(df)

        preseason_file = os.path.join(ratings_dir, f"nfl_power_ratings_blended_week_1_{target}.csv")
        preseason_data = {}
        if os.path.exists(preseason_file) and current_file != preseason_file:
            pre_df = pd.read_csv(preseason_file)
            pre_df = pre_df.rename(columns={"Power Rating": "GSF Power Rating"})
            for _, row in pre_df.iterrows():
                team = str(row.get("Team", ""))
                preseason_data[team] = {
                    "Preseason GSF Power Rating": row.get("GSF Power Rating"),
                    "Preseason MP Rating": row.get("MP_Rating"),
                    "Preseason Rank": row.get("Rank"),
                }

        records = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            team = str(rec.get("Team", ""))
            rec.update(preseason_data.get(team, {
                "Preseason GSF Power Rating": None,
                "Preseason MP Rating": None,
                "Preseason Rank": None,
            }))
            records.append(rec)

        return JSONResponse(content=sanitize({
            "upcoming_week": upcoming_week,
            "target_year": target,
            "has_preseason": bool(preseason_data),
            "is_historical": target != current_year,
            "rankings": records,
        }))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommended-bets")
def get_recommended_bets(year: int = Query(None)):
    try:
        data = load_current_data(DATA_DIR)
        current_year = data["target_year"]
        is_historical = year and year != current_year

        if is_historical:
            df, latest_week, _ = load_historical_data(DATA_DIR, year)
            upcoming_week = latest_week
        else:
            df = data["sim_df"]
            upcoming_week = data["upcoming_week"]
            year = current_year

        df = df.replace([np.inf, -np.inf], np.nan)

        # Tier function unchanged
        def tier(bet_type, edge_val):
            e = float(edge_val) if edge_val is not None and not pd.isna(edge_val) else None
            if e is None: return None
            if bet_type == "mc_spread":
                if e >= 4.0: return "S"
                if 1.0 <= e < 2.0: return "A"
                if e >= 0 and not (2.0 <= e < 3.0): return "B"
                return None
            if bet_type == "mc_ml":
                if e >= 0.20: return "S"
                if e >= 0.15: return "S"
                if e >= 0.10: return "A"
                if e >= 0.05: return "B"
                return None
            if bet_type == "mc_total":
                if e >= 5.0: return "S"
                if e >= 3.0: return "A"
                if 1.0 <= e < 2.0: return "B"
                return None
            if bet_type == "gsf_spread":
                if 2.0 <= e < 3.0: return "A"
                return None
            if bet_type == "combined_spread":
                if e >= 0: return "A"
            return None

        bets = []
        for _, row in df.iterrows():
            game = {
                "away_team": row.get("Away Team"),
                "home_team": row.get("Home Team"),
                "week": row.get("Week_x") or row.get("Week"),
                "circa_week": row.get("Circa Week"),
                "date": str(row.get("Date_x", "")),
            }

            # Helper to add a bet with optional historical W/L data
            def make_bet(base, bet_label):
                if is_historical:
                    base["win_loss"] = row.get(f"{bet_label} Win/Loss")
                    base["pnl"]      = row.get(f"{bet_label} P/L")
                return base

            # MC Spread
            mc_spread_bet  = row.get("Monte Carlo Spread Bet")
            mc_spread_edge = row.get("Monte Carlo Spread Edge")
            if mc_spread_bet and not pd.isna(mc_spread_bet) and str(mc_spread_bet).strip():
                t = tier("mc_spread", mc_spread_edge)
                if t:
                    bets.append(make_bet({
                        **game,
                        "bet_type": "Spread", "model": "Monte Carlo",
                        "pick": str(mc_spread_bet),
                        "edge": round(float(mc_spread_edge), 2) if not pd.isna(mc_spread_edge) else None,
                        "tier": t,
                        "unit_wager":  row.get("MC Spread Unit Wager"),
                        "unit_to_win": row.get("MC Spread Unit to Win"),
                        "kelly_wager": row.get("MC Spread Kelly Wager"),
                        "kelly_to_win":row.get("MC Spread Kelly To Win"),
                    }, "Sim Spread"))

            # MC Moneyline
            mc_ml_bet  = row.get("Monte Carlo Moneyline Bet")
            mc_ml_edge = row.get("Monte Carlo Moneyline Edge")
            if mc_ml_bet and not pd.isna(mc_ml_bet) and str(mc_ml_bet).strip():
                t = tier("mc_ml", mc_ml_edge)
                if t:
                    bets.append(make_bet({
                        **game,
                        "bet_type": "Moneyline", "model": "Monte Carlo",
                        "pick": str(mc_ml_bet),
                        "edge": round(float(mc_ml_edge) * 100, 1) if not pd.isna(mc_ml_edge) else None,
                        "edge_unit": "%", "tier": t,
                        "unit_wager":  row.get("MC ML Unit Wager"),
                        "unit_to_win": row.get("MC ML Unit to Win"),
                        "kelly_wager": row.get("MC ML Kelly Wager"),
                        "kelly_to_win":row.get("MC ML Kelly To Win"),
                    }, "Sim Moneyline"))

            # MC Total
            mc_total_bet  = row.get("Monte Carlo Total Bet")
            mc_total_edge = row.get("Monte Carlo Total Edge")
            if mc_total_bet and not pd.isna(mc_total_bet) and str(mc_total_bet).strip():
                t = tier("mc_total", mc_total_edge)
                if t:
                    bets.append(make_bet({
                        **game,
                        "bet_type": "Total", "model": "Monte Carlo",
                        "pick": str(mc_total_bet),
                        "edge": round(float(mc_total_edge), 2) if not pd.isna(mc_total_edge) else None,
                        "tier": t,
                        "unit_wager":  row.get("MC Total Unit Wager"),
                        "unit_to_win": row.get("MC Total Unit to Win"),
                        "kelly_wager": row.get("MC Total Kelly Wager"),
                        "kelly_to_win":row.get("MC Total Kelly To Win"),
                        "direction":   row.get("MC Bet Direction"),
                    }, "Sim Total"))

            # GSF Spread
            gsf_spread_bet  = row.get("GSF Spread Bet")
            gsf_spread_edge = row.get("GSF Spread Edge")
            if gsf_spread_bet and not pd.isna(gsf_spread_bet) and str(gsf_spread_bet).strip():
                t = tier("gsf_spread", gsf_spread_edge)
                if t:
                    bets.append(make_bet({
                        **game,
                        "bet_type": "Spread", "model": "GSF",
                        "pick": str(gsf_spread_bet),
                        "edge": round(float(gsf_spread_edge), 2) if not pd.isna(gsf_spread_edge) else None,
                        "tier": t,
                    }, "GSF Spread"))

            # Combined Spread
            con_spread_bet  = row.get("Consensus Spread Bet")
            con_spread_edge = row.get("Consensus Spread Edge")
            if (con_spread_bet and not pd.isna(con_spread_bet) and str(con_spread_bet).strip()
                    and mc_spread_bet and gsf_spread_bet
                    and str(mc_spread_bet).strip() == str(gsf_spread_bet).strip()):
                t = tier("combined_spread", con_spread_edge)
                if t:
                    bets.append(make_bet({
                        **game,
                        "bet_type": "Spread", "model": "Combined (MC+GSF)",
                        "pick": str(con_spread_bet),
                        "edge": round(float(con_spread_edge), 2) if not pd.isna(con_spread_edge) else None,
                        "tier": t, "note": "MC and GSF agree",
                    }, "Consensus Spread"))

        tier_order = {"S": 0, "A": 1, "B": 2}
        bets.sort(key=lambda b: (tier_order.get(b["tier"], 9), -(b["edge"] or 0)))

        # Season summary P/L when historical
        season_summary = None
        if is_historical:
            bet_labels = [
                ("GSF Spread", "GSF Spread"),
                ("MP Spread", "MP Spread"),
                ("Sim Spread", "Sim Spread"),
                ("Sim Spread Kelly", "Sim Spread (Kelly)"),
                ("Sim Moneyline", "Sim Moneyline"),
                ("Sim Moneyline Kelly", "Sim Moneyline (Kelly)"),
                ("GSF Moneyline", "GSF Moneyline"),
                ("MP Moneyline", "MP Moneyline"),
                ("Consensus Spread", "Consensus Spread"),
                ("Consensus Moneyline", "Consensus Moneyline"),
                ("Sim Total", "Sim Total"),
                ("Sim Total Kelly", "Sim Total (Kelly)"),
            ]
            season_summary = {}
            for key, col_label in bet_labels:
                wl_col  = f"{col_label} Win/Loss"
                pnl_col = f"{col_label} P/L"
                if wl_col in df.columns:
                    season_summary[key] = {
                        "wins":    int((df[wl_col] == "Win").sum()),
                        "losses":  int((df[wl_col] == "Loss").sum()),
                        "pushes":  int((df[wl_col] == "Push").sum()),
                        "no_bets": int((df[wl_col] == "No Bet").sum()),
                        "total_pl": round(float(df[pnl_col].sum()), 2) if pnl_col in df.columns else 0,
                    }

        return JSONResponse(content=sanitize({
            "upcoming_week": upcoming_week,
            "target_year": year,
            "is_historical": is_historical,
            "bets": bets,
            "season_summary": season_summary,
            "counts": {
                "S": sum(1 for b in bets if b["tier"] == "S"),
                "A": sum(1 for b in bets if b["tier"] == "A"),
                "B": sum(1 for b in bets if b["tier"] == "B"),
            }
        }))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import uuid
from datetime import datetime as dt

BETS_FILE = os.path.join(DATA_DIR, "bets.json")

def load_bets() -> dict:
    if not os.path.exists(BETS_FILE):
        return {}
    with open(BETS_FILE, "r") as f:
        return json.load(f)

def save_bets(bets: dict):
    with open(BETS_FILE, "w") as f:
        json.dump(bets, f, indent=2)

@app.get("/api/bets/{username}")
def get_bets(username: str):
    try:
        all_bets = load_bets()
        user_bets = all_bets.get(username, [])
        return JSONResponse(content=sanitize({"bets": user_bets}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/bets/{username}")
def add_bet(username: str, bet: dict):
    try:
        all_bets = load_bets()
        if username not in all_bets:
            all_bets[username] = []
        new_bet = {
            **bet,
            "id": str(uuid.uuid4()),
            "created_at": dt.utcnow().isoformat(),
            "result": bet.get("result", "pending"),
        }
        all_bets[username].append(new_bet)
        save_bets(all_bets)
        return JSONResponse(content=sanitize({"bet": new_bet}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/bets/{username}/{bet_id}")
def update_bet(username: str, bet_id: str, updates: dict):
    try:
        all_bets = load_bets()
        user_bets = all_bets.get(username, [])
        for i, b in enumerate(user_bets):
            if b["id"] == bet_id:
                user_bets[i] = {**b, **updates}
                all_bets[username] = user_bets
                save_bets(all_bets)
                return JSONResponse(content=sanitize({"bet": user_bets[i]}))
        raise HTTPException(status_code=404, detail="Bet not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/bets/{username}/{bet_id}")
def delete_bet(username: str, bet_id: str):
    try:
        all_bets = load_bets()
        user_bets = all_bets.get(username, [])
        all_bets[username] = [b for b in user_bets if b["id"] != bet_id]
        save_bets(all_bets)
        return {"deleted": bet_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/contest/{year}")
def get_contest_data(year: int):
    try:
        picks_path = os.path.join(DATA_DIR, f"circa-pick-history/{year}_survivor_picks.csv")
        if not os.path.exists(picks_path):
            raise FileNotFoundError(f"No pick history for {year}")

        picks_df = pd.read_csv(picks_path)

        # Load current sim data for team strength values
        try:
            sim_data = load_current_data(DATA_DIR)
            sim_df = sim_data["sim_df"]
            upcoming_week = sim_data["upcoming_week"]

            # Build team strength lookup from sim data
            # Key: team abbreviation → {win_pct, ev, pick_pct}
            team_strength = {}
            for _, row in sim_df[sim_df["Week_x"] >= upcoming_week].iterrows():
                for side in [("Away Team", "Consensus Away Win Pct", "consensus_Away_EV", "Away Pick %"),
                             ("Home Team", "Consensus Home Win Pct", "consensus_Home_EV", "Home Pick %")]:
                    team_col, win_col, ev_col, pick_col = side
                    team = str(row.get(team_col, ""))
                    if not team:
                        continue
                    win = float(row.get(win_col, 0) or 0)
                    ev = float(row.get(ev_col, 0) or 0)
                    pick = float(row.get(pick_col, 0) or 0)
                    week = int(row.get("Week_x", 0))
                    if team not in team_strength:
                        team_strength[team] = []
                    team_strength[team].append({
                        "week": week,
                        "win_pct": win,
                        "ev": ev,
                        "pick_pct": pick,
                    })
        except Exception:
            sim_df = None
            upcoming_week = 1
            team_strength = {}

        # All 32 NFL teams
        ALL_TEAMS = {
            "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
            "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
            "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
            "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
        }
        # Handle common abbreviation variants
        ABBR_MAP = {"JAC": "JAX", "LAR": "LA", "GNB": "GB", "KAN": "KC",
                    "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV"}

        def normalize(team):
            t = str(team).strip().upper()
            return ABBR_MAP.get(t, t)

        week_cols = [c for c in picks_df.columns if c.startswith("Week_")]
        num_weeks = len(week_cols)
        total_entries = len(picks_df)

        import re
        def get_contestant(name):
            m = re.match(r'^(.+)-(\d+)$', str(name).strip())
            return m.group(1).strip() if m else str(name).strip()

        picks_df["Contestant"] = picks_df["EntryName"].apply(get_contestant)
        picks_df["Total_Wins"] = pd.to_numeric(picks_df["Total_Wins"], errors="coerce").fillna(0).astype(int)

        # ── Survival curve — cap at num_weeks, no week 21 ──
        survival_curve = []
        for week in range(0, num_weeks + 1):  # 0 = before season, 1 through num_weeks
            surviving = int((picks_df["Total_Wins"] >= week).sum())
            eliminated = int((picks_df["Total_Wins"] == week - 1).sum()) if week > 0 else 0
            survival_curve.append({
                "week": week,
                "surviving": surviving,
                "pct_remaining": round(surviving / total_entries * 100, 2),
                "eliminated": eliminated,
                "pct_eliminated": round(eliminated / total_entries * 100, 2),
            })
        # Remove week 0 from display but keep for reference — only send weeks 1+
        survival_curve = [c for c in survival_curve if c["week"] >= 1]

        # ── Weekly pick popularity ──
        from collections import Counter
        weekly_picks = {}
        for col in week_cols:
            week_num = int(col.replace("Week_", ""))
            
            # Only include entries that survived TO this week (were alive to make a pick)
            alive_this_week = picks_df[picks_df["Total_Wins"] >= week_num]
            picks_this_week = alive_this_week[col].dropna()
            picks_this_week = picks_this_week[picks_this_week.str.strip() != ""]
            
            from collections import Counter
            counts = Counter(normalize(t) for t in picks_this_week.tolist())
            # Total is entries alive that week — not just those who picked
            total_alive = len(alive_this_week)
            
            weekly_picks[week_num] = [
                {
                    "team": t,
                    "count": c,
                    "pct": round(c / total_alive * 100, 2) if total_alive > 0 else 0,
                    "total_alive": total_alive,
                }
                for t, c in counts.most_common(10)
                if t and t != ""  # exclude empty strings
            ]

        # ── Per-entry analysis with remaining teams ──
        def score_remaining_teams(used_teams):
            """
            Given a set of already-used team abbreviations, compute:
            - remaining_teams: list of teams not yet used
            - best_ev_path: sum of top-N future EVs from remaining teams
            - best_win_path: sum of top-N future win_pcts from remaining teams
            - pool_strength: average win_pct of remaining teams across future weeks
            """
            remaining = ALL_TEAMS - used_teams
            future_weeks_remaining = max(0, num_weeks - upcoming_week + 1)

            best_ev = 0.0
            best_win = 0.0
            pool_wins = []

            for team in remaining:
                team_data = team_strength.get(team, [])
                for week_data in team_data:
                    if week_data["week"] >= upcoming_week:
                        best_ev += week_data["ev"]
                        pool_wins.append(week_data["win_pct"])

            # Best EV = if this entry could optimally pick from remaining teams
            # Sort remaining teams by EV for each future week and pick the best
            future_ev_by_week = {}
            future_win_by_week = {}
            for team in remaining:
                for wd in team_strength.get(team, []):
                    w = wd["week"]
                    if w >= upcoming_week:
                        if w not in future_ev_by_week or wd["ev"] > future_ev_by_week[w]["ev"]:
                            future_ev_by_week[w] = {"team": team, "ev": wd["ev"]}
                        if w not in future_win_by_week or wd["win_pct"] > future_win_by_week[w]["win_pct"]:
                            future_win_by_week[w] = {"team": team, "win_pct": wd["win_pct"]}

            # Greedy optimal path — pick best available team each week
            # (teams can't repeat, so we track used teams within the path)
            ev_path_teams = set()
            win_path_teams = set()
            optimal_ev = 0.0
            optimal_win = 0.0

            for w in sorted(future_ev_by_week.keys()):
                # Find best remaining team for EV this week
                candidates = [
                    (t, d) for t, d in [
                        (td["team"], td)
                        for w2, td in future_ev_by_week.items()
                        if w2 == w
                    ]
                    if t not in ev_path_teams
                ]
                # Re-scan all remaining teams for this week
                week_options = [
                    (team, wd["ev"])
                    for team in (remaining - ev_path_teams)
                    for wd in team_strength.get(team, [])
                    if wd["week"] == w
                ]
                if week_options:
                    best_team, best_val = max(week_options, key=lambda x: x[1])
                    optimal_ev += best_val
                    ev_path_teams.add(best_team)

            for w in sorted(future_win_by_week.keys()):
                week_options = [
                    (team, wd["win_pct"])
                    for team in (remaining - win_path_teams)
                    for wd in team_strength.get(team, [])
                    if wd["week"] == w
                ]
                if week_options:
                    best_team, best_val = max(week_options, key=lambda x: x[1])
                    optimal_win += best_val
                    win_path_teams.add(best_team)

            avg_win = sum(pool_wins) / len(pool_wins) if pool_wins else 0

            return {
                "remaining_count": len(remaining),
                "remaining_teams": sorted(remaining),
                "optimal_ev_path": round(optimal_ev, 4),
                "optimal_win_path": round(optimal_win * 100 / max(1, future_weeks_remaining), 2),
                "pool_avg_win_pct": round(avg_win * 100, 2),
            }

        # Only compute for surviving entries (performance)
        entry_stats = []
        surviving_entries = picks_df[picks_df["Total_Wins"] >= upcoming_week - 1]

        for _, row in surviving_entries.iterrows():
            used = set()
            for col in week_cols:
                val = row.get(col, "")
                if val and str(val).strip():
                    used.add(normalize(str(val).strip()))

            scores = score_remaining_teams(used) if team_strength else {
                "remaining_count": len(ALL_TEAMS - used),
                "remaining_teams": sorted(ALL_TEAMS - used),
                "optimal_ev_path": 0,
                "optimal_win_path": 0,
                "pool_avg_win_pct": 0,
            }

            entry_stats.append({
                "entry": str(row["EntryName"]),
                "contestant": str(row["Contestant"]),
                "total_wins": int(row["Total_Wins"]),
                "teams_used": sorted(used),
                **scores,
            })

        # Sort by optimal EV path descending
        entry_stats.sort(key=lambda x: -x["optimal_ev_path"])

        # ── Contestant summary with remaining team aggregates ──
        from collections import defaultdict
        contestant_map = defaultdict(list)
        for e in entry_stats:
            contestant_map[e["contestant"]].append(e)

        contestant_stats = []
        for name, entries in contestant_map.items():
            surviving = [e for e in entries if e["total_wins"] >= upcoming_week - 1]
            all_entries = picks_df[picks_df["Contestant"] == name]
            contestant_stats.append({
                "contestant": name,
                "entries": len(all_entries),
                "surviving": len(surviving),
                "max_wins": int(all_entries["Total_Wins"].max()),
                "avg_wins": round(float(all_entries["Total_Wins"].mean()), 1),
                # Best path across all surviving entries
                "best_ev_path": round(max((e["optimal_ev_path"] for e in surviving), default=0), 4),
                "best_win_path": round(max((e["optimal_win_path"] for e in surviving), default=0), 2),
                "avg_remaining_teams": round(
                    sum(e["remaining_count"] for e in surviving) / len(surviving), 1
                ) if surviving else 0,
                "avg_pool_strength": round(
                    sum(e["pool_avg_win_pct"] for e in surviving) / len(surviving), 2
                ) if surviving else 0,
            })
        contestant_stats.sort(key=lambda x: (-x["surviving"], -x["best_ev_path"]))

        # ── Season summary ──
        biggest_elim = max(survival_curve[1:], key=lambda x: x["eliminated"])
        # ── Season summary — ALL contestants, not just survivors ──
        # Parse all contestants from full picks_df (before any filtering)
        all_contestants = picks_df["Contestant"].nunique()
        
        summary = {
            "year": year,
            "total_entries": total_entries,
            "total_contestants": all_contestants,  # ← was len(contestant_stats) which only counted survivors
            "num_weeks": num_weeks,
            "final_survivors": int((picks_df["Total_Wins"] >= num_weeks).sum()),
            "biggest_elimination_week": biggest_elim["week"] - 1,
            "biggest_elimination_pct": biggest_elim["pct_eliminated"],
            "median_survival_week": round(float(picks_df["Total_Wins"].median()), 1),
            "upcoming_week": upcoming_week,
        }


        # ── Full entry list (for the entries view) ──
        all_entries = []
        for _, row in picks_df.iterrows():
            picks_by_week = {}
            for col in week_cols:
                wk = int(col.replace("Week_", ""))
                val = str(row.get(col, "") or "").strip()
                if val:
                    picks_by_week[wk] = normalize(val)
            all_entries.append({
                "entry": str(row["EntryName"]),
                "contestant": str(row["Contestant"]),
                "total_wins": int(row["Total_Wins"]),
                "picks": picks_by_week,
            })
        
        # Add to return value
        return JSONResponse(content=sanitize({
            "summary": summary,
            "survival_curve": survival_curve,
            "weekly_picks": weekly_picks,
            "entry_stats": entry_stats[:500],
            "contestant_stats": contestant_stats[:200],
            "all_entries": all_entries,  # ← new
        }))


    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/contest/years/available")
def get_available_contest_years():
    try:
        import glob
        pattern = os.path.join(DATA_DIR, "circa-pick-history/*_survivor_picks.csv")
        files = glob.glob(pattern)
        years = []
        for f in sorted(files):
            base = os.path.basename(f)
            try:
                year = int(base.split("_")[0])
                years.append(year)
            except ValueError:
                pass
        return {"years": sorted(years, reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/available-years")
def get_available_years():
    """Returns all years that have historical final data available, plus the current year."""
    try:
        data = load_current_data(DATA_DIR)
        current_year = data["target_year"]

        # Find all years with final data
        pattern = os.path.join(DATA_DIR, "nfl-power-ratings/final-data/*_final_data")
        dirs = glob.glob(pattern)
        historical_years = []
        for d in sorted(dirs):
            try:
                year = int(os.path.basename(d).split("_")[0])
                # Verify at least one final data file exists
                files = glob.glob(os.path.join(d, "Season_*_Final_Data.csv"))
                if files:
                    historical_years.append(year)
            except ValueError:
                pass

        all_years = sorted(set(historical_years + [current_year]), reverse=True)

        return JSONResponse(content=sanitize({
            "current_year": current_year,
            "years": [
                {
                    "year": y,
                    "is_current": y == current_year,
                    "label": f"{y} (Live)" if y == current_year else str(y),
                }
                for y in all_years
            ]
        }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug-paths")
def debug_paths():
    import os
    base = DATA_DIR
    try:
        nfl_sched_files = os.listdir(os.path.join(base, "nfl-schedules"))
    except:
        nfl_sched_files = "DIRECTORY MISSING"
    return {
        "cwd": os.getcwd(),
        "DATA_DIR": base,
        "nfl_schedules_contents": nfl_sched_files,
        "schedule_2026_exists": os.path.exists(os.path.join(base, "nfl-schedules/schedule_2026.csv")),
        "schedule_2025_exists": os.path.exists(os.path.join(base, "nfl-schedules/schedule_2025.csv")),
    }


# ---------------------------------------------------------------
# Railway startup — reads PORT from environment (Railway sets this)
# Falls back to 8000 for local development
# ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
