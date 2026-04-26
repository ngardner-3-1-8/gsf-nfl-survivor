import os
import glob
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from api.models import OptimizeRequest, OptimizeResponse

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
RATINGS_DIR = os.path.join(DATA_DIR, "nfl-power-ratings")
EV_DIR = os.path.join(DATA_DIR, "circa-survivor-ev")


def find_latest_file(directory: str, pattern: str) -> str:
    """Find the most recently modified file matching a pattern."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")
    return max(files, key=os.path.getmtime)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN and inf values so the DataFrame serialises to JSON cleanly."""
    df = df.replace([np.inf, -np.inf], None)
    df = df.where(pd.notnull(df), None)
    return df


@app.get("/")
def root():
    return {"status": "ok", "message": "Circa Survivor API is running"}


@app.get("/api/schedule")
def get_schedule(
    week: int = Query(None, description="Filter to a specific week"),
    year: int = Query(None, description="Season year"),
):
    try:
        filepath = find_latest_file(RATINGS_DIR, "final_sim_results_*.csv")
        df = pd.read_csv(filepath)
        df = clean_df(df)

        if week is not None:
            df = df[df["Week_x"] == week]

        weeks = sorted(df["Week_x"].dropna().unique().tolist())

        return {
            "source_file": os.path.basename(filepath),
            "weeks": weeks,
            "total_games": len(df),
            "games": df.to_dict(orient="records"),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
