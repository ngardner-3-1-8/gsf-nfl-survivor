import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from api.models import OptimizeRequest, OptimizeResponse
from api.data_loader import load_current_data

app = FastAPI(title="Circa Survivor API", version="1.0.0")

# Allow requests from local dev and your future GitHub Pages URL
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    os.environ.get("FRONTEND_URL", ""),  # set this in Railway vars later
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
    """Replace NaN and inf values so the DataFrame serialises to JSON cleanly."""
    df = df.replace([np.inf, -np.inf], None)
    df = df.where(pd.notnull(df), None)
    return df


@app.get("/")
def root():
    return {"status": "ok", "message": "Circa Survivor API is running"}


# Updated schedule endpoint
@app.get("/api/schedule")
def get_schedule(week: int = Query(None)):
    try:
        data = load_current_data(DATA_DIR)
        df = clean_df(data["sim_df"])

        if week is not None:
            df = df[df["Week_x"] == week]

        return {
            "upcoming_week": data["upcoming_week"],
            "target_year": data["target_year"],
            "source_file": data["sim_file"],
            "weeks": sorted(df["Week_x"].dropna().unique().tolist()),
            "total_games": len(df),
            "games": df.to_dict(orient="records"),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Updated EV endpoint
@app.get("/api/ev")
def get_ev(
    model: str = Query("consensus"),
    week: int = Query(None),
):
    valid_models = ["consensus", "sportsbook", "mp", "gsf", "sim"]
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {valid_models}")
    try:
        data = load_current_data(DATA_DIR, model=model)
        df = clean_df(data["ev_df"])

        if week is not None:
            df = df[df["Week_x"] == week]

        return {
            "model": model,
            "upcoming_week": data["upcoming_week"],
            "target_year": data["target_year"],
            "source_file": data["ev_file"],
            "total_rows": len(df),
            "games": df.to_dict(orient="records"),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Updated weeks endpoint
@app.get("/api/weeks")
def get_available_weeks():
    try:
        data = load_current_data(DATA_DIR)
        df = data["sim_df"]
        weeks = sorted(df["Week_x"].dropna().unique().tolist())
        return {
            "upcoming_week": data["upcoming_week"],
            "target_year": data["target_year"],
            "weeks": weeks,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize")
def optimize(request: OptimizeRequest):
    """
    OR-Tools optimizer — stub for now.
    Returns a placeholder so the frontend can be built against this
    contract before the solver is written in Phase 2.
    """
    return OptimizeResponse(
        ev_solutions=[],
        ranking_solutions=[],
        total_ev=0.0,
        total_win_pct=0.0,
        feasible=False,
        message="Optimizer not yet implemented — coming in Phase 2."
    )

# ---------------------------------------------------------------
# Railway startup — reads PORT from environment (Railway sets this)
# Falls back to 8000 for local development
# ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
