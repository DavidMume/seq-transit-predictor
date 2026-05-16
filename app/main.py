import logging
from contextlib import asynccontextmanager
from glob import glob
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
STATIC_DIR = Path(__file__).parent / "static"

# Global state populated at startup
_stops_lookup: dict = {}   # stop_id -> {name, lat, lon}
_od_probs: dict = {}        # (origin_id, time_period) -> [(dest_id, prob), ...]
_time_periods: list = []
_stops_with_trips: list = []


def _load_data() -> None:
    global _stops_lookup, _od_probs, _time_periods, _stops_with_trips

    # ── Stops ─────────────────────────────────────────────────────────────────
    log.info("Loading stops.txt …")
    stops_df = pd.read_csv(
        DATA_DIR / "stops.txt",
        usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"],
        dtype={"stop_id": str},
    )
    stops_df["stop_id"] = stops_df["stop_id"].str.strip()
    stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
    stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")
    stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])

    _stops_lookup = {
        row.stop_id: {"name": row.stop_name, "lat": float(row.stop_lat), "lon": float(row.stop_lon)}
        for row in stops_df.itertuples(index=False)
    }
    stops_set = set(_stops_lookup)
    log.info(f"  {len(_stops_lookup):,} stops loaded")

    # ── OD CSV files ──────────────────────────────────────────────────────────
    csv_files = sorted(glob(str(DATA_DIR / "*TL Org-Dest Trips.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No OD CSV files found in {DATA_DIR}")
    log.info(f"Found {len(csv_files)} OD files")

    frames = []
    for path in csv_files:
        log.info(f"  Reading {Path(path).name} …")
        df = pd.read_csv(
            path,
            usecols=["time", "origin_stop", "destination_stop", "quantity"],
            dtype={"origin_stop": str, "destination_stop": str},
            low_memory=False,
        )
        frames.append(df)

    log.info("Concatenating all months …")
    od = pd.concat(frames, ignore_index=True)

    # ── Clean ─────────────────────────────────────────────────────────────────
    od["origin_stop"] = od["origin_stop"].str.strip()
    od["destination_stop"] = od["destination_stop"].str.strip()
    od["quantity"] = pd.to_numeric(od["quantity"], errors="coerce")
    od = od[od["destination_stop"] != "n/a"]
    od = od.dropna(subset=["origin_stop", "destination_stop", "quantity"])

    # ── Aggregate across all 6 months ─────────────────────────────────────────
    log.info("Aggregating …")
    od = (
        od.groupby(["origin_stop", "time", "destination_stop"], as_index=False)["quantity"]
        .sum()
    )

    _time_periods = sorted(od["time"].dropna().unique().tolist())
    log.info(f"  Time periods: {_time_periods}")

    # ── Probabilities ─────────────────────────────────────────────────────────
    log.info("Computing probabilities …")
    totals = (
        od.groupby(["origin_stop", "time"])["quantity"]
        .sum()
        .rename("total")
        .reset_index()
    )
    od = od.merge(totals, on=["origin_stop", "time"])
    od["prob"] = od["quantity"] / od["total"]

    # Keep only rows where both origin and destination exist in stops.txt
    od_valid = od[
        od["origin_stop"].isin(stops_set) & od["destination_stop"].isin(stops_set)
    ]

    # ── Build probability index ────────────────────────────────────────────────
    log.info("Building probability index …")
    for (origin, time_period), grp in od_valid.groupby(["origin_stop", "time"]):
        top50 = grp.nlargest(50, "prob")
        _od_probs[(origin, time_period)] = [
            (row.destination_stop, float(row.prob))
            for row in top50.itertuples(index=False)
        ]

    # ── Stops that have at least one outgoing trip to a known stop ────────────
    origin_ids = od_valid["origin_stop"].unique()
    _stops_with_trips = [
        {
            "stop_id": sid,
            "stop_name": _stops_lookup[sid]["name"],
            "lat": _stops_lookup[sid]["lat"],
            "lon": _stops_lookup[sid]["lon"],
        }
        for sid in origin_ids
        if sid in stops_set
    ]
    log.info(f"  {len(_stops_with_trips):,} origin stops with valid trips")
    log.info("Data loading complete.")


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_data()
    yield


app = FastAPI(title="SEQ Transit Destination Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/stops")
def get_stops():
    return _stops_with_trips


@app.get("/time_periods")
def get_time_periods():
    return _time_periods


@app.get("/predict")
def predict(origin_stop_id: str, time_period: str):
    entries = _od_probs.get((origin_stop_id, time_period), [])
    return [
        {
            "stop_id": sid,
            "stop_name": _stops_lookup[sid]["name"],
            "lat": _stops_lookup[sid]["lat"],
            "lon": _stops_lookup[sid]["lon"],
            "probability": prob,
        }
        for sid, prob in entries
        if sid in _stops_lookup
    ]


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
