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
    allow_methods=["*"],
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


# ---------------------------------------------------------------
# Railway startup — reads PORT from environment (Railway sets this)
# Falls back to 8000 for local development
# ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
