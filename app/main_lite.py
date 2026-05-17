# Esta es la versión lite de la app, diseñada para correr
# en servidores con poca memoria (512MB).
# Solo tiene el modelo empírico básico.

import gzip
import logging
import os
import pickle
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
PKL_PATH   = DATA_DIR / "precomputed_ultralite.pkl.gz"
STATIC_DIR = Path(__file__).parent / "static"

# Datos cargados al inicio — compartidos por todos los requests
_stops_lookup: dict = {}
_od_probs:     dict = {}
_time_periods: list = []


def _download_if_missing():
    if PKL_PATH.exists():
        return
    url = os.environ.get("DATA_DOWNLOAD_URL", "")
    if not url:
        raise RuntimeError(
            f"{PKL_PATH} no encontrado y DATA_DOWNLOAD_URL no está configurado."
        )
    log.info(f"Descargando datos desde {url} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, PKL_PATH)
    log.info(f"Descarga completa: {PKL_PATH.stat().st_size / 1024 / 1024:.1f} MB")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _stops_lookup, _od_probs, _time_periods

    _download_if_missing()

    log.info(f"Cargando {PKL_PATH} ...")
    with gzip.open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    _stops_lookup = data["stops_lookup"]
    _od_probs     = data["od_probs"]
    _time_periods = data["time_periods"]

    log.info(
        f"Listo: {len(_stops_lookup):,} paradas, "
        f"{len(_od_probs):,} pares OD, "
        f"{len(_time_periods)} periodos."
    )
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(STATIC_DIR / "index_lite.html"))


@app.get("/time_periods")
async def time_periods():
    return _time_periods


@app.get("/stops")
async def stops():
    return [
        {
            "stop_id":        sid,
            "stop_name":      s["name"],
            "lat":            s["lat"],
            "lon":            s["lon"],
            "total_activity": s.get("total_activity", 0),
        }
        for sid, s in _stops_lookup.items()
    ]


@app.get("/predict")
async def predict(origin_stop_id: str, time_period: str):
    key = (origin_stop_id, time_period)
    dests = _od_probs.get(key)
    if not dests:
        return []
    return [
        {
            "stop_id":     d["stop_id"],
            "stop_name":   _stops_lookup.get(d["stop_id"], {}).get("name", d["stop_id"]),
            "lat":         d["lat"],
            "lon":         d["lon"],
            "probability": d["probability"],
        }
        for d in dests
        if d["stop_id"] != origin_stop_id
    ]
