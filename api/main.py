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


@app.get("/")
def root():
    return {"status": "ok", "message": "Circa Survivor API is running"}


@app.get("/api/schedule")
def get_schedule(week: int = Query(None)):
    try:
        data = load_current_data(DATA_DIR)
        df = clean_df(data["sim_df"])
        if week is not None:
            df = df[df["Week_x"] == week]
        result = {
            "upcoming_week": data["upcoming_week"],
            "target_year": data["target_year"],
            "source_file": data["sim_file"],
            "weeks": sorted(df["Week_x"].dropna().unique().tolist()),
            "total_games": len(df),
            "games": df.to_dict(orient="records"),
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
def get_available_weeks():
    try:
        data = load_current_data(DATA_DIR)
        df = data["sim_df"]

        upcoming = data["upcoming_week"]

        week_col = "Week_x" if "Week_x" in df.columns else "Week"
        circa_col = "Circa Week" if "Circa Week" in df.columns else None

        weeks_df = df[[week_col] + ([circa_col] if circa_col else [])].drop_duplicates()
####        weeks_df = weeks_df[weeks_df[week_col] >= upcoming].sort_values(week_col)

        week_options = []
        for _, row in weeks_df.iterrows():
            w = int(row[week_col])
            label = str(row[circa_col]) if circa_col and pd.notna(row.get(circa_col)) else f"Week {w}"
            week_options.append({"week": w, "label": label})

        return {
            "upcoming_week": upcoming,
            "target_year": data["target_year"],
            "weeks": week_options,
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
def get_rankings():
    try:
        data = load_current_data(DATA_DIR)
        upcoming_week = data["upcoming_week"]
        target_year = data["target_year"]
        ratings_dir = os.path.join(DATA_DIR, "nfl-power-ratings")

        # Load current week file
        current_file = os.path.join(
            ratings_dir,
            f"nfl_power_ratings_blended_week_{upcoming_week}_{target_year}.csv"
        )
        if not os.path.exists(current_file):
            # Fall back to closest available
            import glob
            files = glob.glob(os.path.join(ratings_dir, f"nfl_power_ratings_blended_week_*_{target_year}.csv"))
            if not files:
                raise FileNotFoundError(f"No rankings file found for {target_year}")
            def extract_week(p):
                try:
                    return int(os.path.basename(p).split("_week_")[1].split(f"_{target_year}")[0])
                except:
                    return 0
            files_with_weeks = [(extract_week(f), f) for f in files]
            valid = [(w, f) for w, f in files_with_weeks if w <= upcoming_week]
            current_file = max(valid if valid else files_with_weeks, key=lambda x: x[0])[1]

        df = pd.read_csv(current_file)
        df = clean_df(df)

        # Try to load Week 1 preseason file for comparison
        preseason_file = os.path.join(
            ratings_dir,
            f"nfl_power_ratings_blended_week_1_{target_year}.csv"
        )
        preseason_data = {}
        if os.path.exists(preseason_file) and current_file != preseason_file:
            pre_df = pd.read_csv(preseason_file)
            for _, row in pre_df.iterrows():
                team = str(row.get("Team", ""))
                preseason_data[team] = {
                    "Preseason Power Rating": row.get("Power Rating"),
                    "Preseason MP Rating":    row.get("MP_Rating"),
                    "Preseason Rank":         row.get("Rank"),
                }

        # Merge preseason data in
        records = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            team = str(rec.get("Team", ""))
            if team in preseason_data:
                rec.update(preseason_data[team])
            else:
                rec["Preseason Power Rating"] = None
                rec["Preseason MP Rating"] = None
                rec["Preseason Rank"] = None
            records.append(rec)

        return JSONResponse(content=sanitize({
            "upcoming_week": upcoming_week,
            "target_year": target_year,
            "has_preseason": bool(preseason_data),
            "rankings": records,
        }))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommended-bets")
def get_recommended_bets():
    try:
        data = load_current_data(DATA_DIR)
        df = data["sim_df"]
        df = df.replace([np.inf, -np.inf], np.nan)

        def tier(bet_type, edge_val, gsf_edge=None, mc_edge=None):
            """Assign S/A/B tier based on historical profitability."""
            e = float(edge_val) if edge_val is not None and not pd.isna(edge_val) else None
            if e is None:
                return None

            if bet_type == "mc_spread":
                if e >= 4.0: return "S"
                if 1.0 <= e < 2.0: return "A"
                if e >= 0 and not (2.0 <= e < 3.0): return "B"
                return None  # skip 2-3 band

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
                return None  # skip 2-3 band

            if bet_type == "gsf_spread":
                if 2.0 <= e < 3.0: return "A"
                return None  # all other GSF spread tiers lose money

            if bet_type == "combined_spread":
                # MC and GSF agree — strong signal
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
                "game_time": str(row.get("Time", "")),
            }

            # ── MC Spread ──
            mc_spread_bet = row.get("Monte Carlo Spread Bet")
            mc_spread_edge = row.get("Monte Carlo Spread Edge")
            if mc_spread_bet and not pd.isna(mc_spread_bet) and str(mc_spread_bet).strip():
                t = tier("mc_spread", mc_spread_edge)
                if t:
                    bets.append({
                        **game,
                        "bet_type": "Spread",
                        "model": "Monte Carlo",
                        "pick": str(mc_spread_bet),
                        "edge": round(float(mc_spread_edge), 2) if not pd.isna(mc_spread_edge) else None,
                        "tier": t,
                        "unit_wager": row.get("MC Spread Unit Wager"),
                        "unit_to_win": row.get("MC Spread Unit to Win"),
                        "kelly_wager": row.get("MC Spread Kelly Wager"),
                        "kelly_to_win": row.get("MC Spread Kelly To Win"),
                    })

            # ── MC Moneyline ──
            mc_ml_bet = row.get("Monte Carlo Moneyline Bet")
            mc_ml_edge = row.get("Monte Carlo Moneyline Edge")
            if mc_ml_bet and not pd.isna(mc_ml_bet) and str(mc_ml_bet).strip():
                t = tier("mc_ml", mc_ml_edge)
                if t:
                    bets.append({
                        **game,
                        "bet_type": "Moneyline",
                        "model": "Monte Carlo",
                        "pick": str(mc_ml_bet),
                        "edge": round(float(mc_ml_edge) * 100, 1) if not pd.isna(mc_ml_edge) else None,
                        "edge_unit": "%",
                        "tier": t,
                        "unit_wager": row.get("MC ML Unit Wager"),
                        "unit_to_win": row.get("MC ML Unit to Win"),
                        "kelly_wager": row.get("MC ML Kelly Wager"),
                        "kelly_to_win": row.get("MC ML Kelly To Win"),
                    })

            # ── MC Total ──
            mc_total_bet = row.get("Monte Carlo Total Bet")
            mc_total_edge = row.get("Monte Carlo Total Edge")
            if mc_total_bet and not pd.isna(mc_total_bet) and str(mc_total_bet).strip():
                t = tier("mc_total", mc_total_edge)
                if t:
                    bets.append({
                        **game,
                        "bet_type": "Total",
                        "model": "Monte Carlo",
                        "pick": str(mc_total_bet),
                        "edge": round(float(mc_total_edge), 2) if not pd.isna(mc_total_edge) else None,
                        "tier": t,
                        "unit_wager": row.get("MC Total Unit Wager"),
                        "unit_to_win": row.get("MC Total Unit to Win"),
                        "kelly_wager": row.get("MC Total Kelly Wager"),
                        "kelly_to_win": row.get("MC Total Kelly To Win"),
                        "direction": row.get("MC Bet Direction"),
                    })

            # ── GSF Spread (2.0-3.0 only) ──
            gsf_spread_bet = row.get("GSF Spread Bet")
            gsf_spread_edge = row.get("GSF Spread Edge")
            if gsf_spread_bet and not pd.isna(gsf_spread_bet) and str(gsf_spread_bet).strip():
                t = tier("gsf_spread", gsf_spread_edge)
                if t:
                    bets.append({
                        **game,
                        "bet_type": "Spread",
                        "model": "GSF",
                        "pick": str(gsf_spread_bet),
                        "edge": round(float(gsf_spread_edge), 2) if not pd.isna(gsf_spread_edge) else None,
                        "tier": t,
                        "unit_wager": None,
                        "kelly_wager": None,
                    })

            # ── Combined Spread (MC + GSF agree) ──
            con_spread_bet = row.get("Consensus Spread Bet")
            con_spread_edge = row.get("Consensus Spread Edge")
            if (con_spread_bet and not pd.isna(con_spread_bet) and str(con_spread_bet).strip()
                    and mc_spread_bet and gsf_spread_bet
                    and str(mc_spread_bet).strip() == str(gsf_spread_bet).strip()):
                t = tier("combined_spread", con_spread_edge)
                if t:
                    bets.append({
                        **game,
                        "bet_type": "Spread",
                        "model": "Combined (MC+GSF)",
                        "pick": str(con_spread_bet),
                        "edge": round(float(con_spread_edge), 2) if not pd.isna(con_spread_edge) else None,
                        "tier": t,
                        "unit_wager": None,
                        "kelly_wager": None,
                        "note": "MC and GSF agree",
                    })

        # Sort: S first, then A, then B; within tier sort by edge descending
        tier_order = {"S": 0, "A": 1, "B": 2}
        bets.sort(key=lambda b: (tier_order.get(b["tier"], 9), -(b["edge"] or 0)))

        return JSONResponse(content=sanitize({
            "upcoming_week": data["upcoming_week"],
            "target_year": data["target_year"],
            "bets": bets,
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


# ---------------------------------------------------------------
# Railway startup — reads PORT from environment (Railway sets this)
# Falls back to 8000 for local development
# ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
