"""
preprocess.py — Genera el archivo de datos precompilados para producción.

Ejecutar UNA VEZ localmente antes de desplegar:
    python scripts/preprocess.py

Lee todos los CSV mensuales y stops.txt, aplica toda la lógica de agregación,
probabilidades, modelo de gravedad y series temporales, y guarda el resultado
en data/precomputed.pkl.gz.
"""

import gzip
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "precomputed.pkl.gz"


def main():
    print("=" * 60)
    print("SEQ Transit Predictor — preprocesamiento de datos")
    print("=" * 60)

    # ── Paradas ───────────────────────────────────────────────────
    print("\n[1/8] Cargando stops.txt ...")
    stops_df = pd.read_csv(
        DATA_DIR / "stops.txt",
        usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"],
        dtype={"stop_id": str},
    )
    stops_df["stop_id"] = stops_df["stop_id"].str.strip()
    stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
    stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")
    stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])

    stops_lookup = {
        row.stop_id: {"name": row.stop_name, "lat": float(row.stop_lat), "lon": float(row.stop_lon)}
        for row in stops_df.itertuples(index=False)
    }
    stops_set = set(stops_lookup)
    print(f"    {len(stops_lookup):,} paradas cargadas")

    # ── CSVs de viajes ────────────────────────────────────────────
    print("\n[2/8] Leyendo archivos CSV ...")
    csv_files = sorted(glob(str(DATA_DIR / "*TL Org-Dest Trips.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {DATA_DIR}")
    print(f"    {len(csv_files)} archivos encontrados")

    frames = []
    for path in csv_files:
        print(f"    Leyendo {Path(path).name} ...")
        df = pd.read_csv(
            path,
            usecols=["month", "time", "origin_stop", "destination_stop", "quantity"],
            dtype={"origin_stop": str, "destination_stop": str},
            low_memory=False,
        )
        frames.append(df)

    # ── Limpieza ──────────────────────────────────────────────────
    print("\n[3/8] Limpiando datos ...")
    od = pd.concat(frames, ignore_index=True)
    del frames

    od["origin_stop"]      = od["origin_stop"].str.strip()
    od["destination_stop"] = od["destination_stop"].str.strip()
    od["quantity"]         = pd.to_numeric(od["quantity"], errors="coerce")
    od = od[od["destination_stop"] != "n/a"]
    od = od.dropna(subset=["origin_stop", "destination_stop", "quantity"])
    print(f"    {len(od):,} filas limpias")

    # ── Series temporales y tendencias ───────────────────────────
    # Calculamos la actividad mes a mes ANTES de la agregación por periodo
    # de tiempo, porque esa agregación elimina la columna "month".
    # Una serie temporal registra cuántas personas subieron y bajaron en
    # cada parada en cada mes calendario (ej: enero 2022, febrero 2022...).
    print("\n[4/8] Calculando series temporales mensuales ...")
    monthly_board = (
        od.groupby(["origin_stop", "month"])["quantity"]
        .sum().reset_index()
        .rename(columns={"origin_stop": "stop_id", "quantity": "boardings"})
    )
    monthly_alight = (
        od.groupby(["destination_stop", "month"])["quantity"]
        .sum().reset_index()
        .rename(columns={"destination_stop": "stop_id", "quantity": "alightings"})
    )
    monthly = (
        monthly_board
        .merge(monthly_alight, on=["stop_id", "month"], how="outer")
        .fillna(0)
        .sort_values(["stop_id", "month"])
    )
    monthly["boardings"]  = monthly["boardings"].astype(int)
    monthly["alightings"] = monthly["alightings"].astype(int)
    monthly["total"]      = monthly["boardings"] + monthly["alightings"]

    timeseries: dict = {}
    for stop_id, grp in monthly.groupby("stop_id"):
        if stop_id not in stops_set:
            continue
        timeseries[stop_id] = [
            {"month": row.month, "boardings": int(row.boardings),
             "alightings": int(row.alightings), "total": int(row.total)}
            for row in grp.itertuples(index=False)
        ]
    print(f"    {len(timeseries):,} series temporales construidas")

    # ── Regresión lineal: tendencia por parada ────────────────────
    # La regresión lineal traza la línea que mejor se ajusta al historial.
    # Si la línea sube, la parada está creciendo. Si baja, pierde actividad.
    # R² nos dice qué tan confiable es esa línea: 1.0=perfecta, 0.0=sin patrón.
    print("\n[5/8] Calculando tendencias y proyecciones ...")
    trend: dict = {}
    for stop_id, data in timeseries.items():
        if len(data) < 3:
            continue
        x = np.arange(len(data), dtype=np.float64)
        y = np.array([d["total"] for d in data], dtype=np.float64)

        coeffs    = np.polyfit(x, y, 1)
        slope     = float(coeffs[0])
        y_pred    = np.polyval(coeffs, x)
        ss_res    = float(np.sum((y - y_pred) ** 2))
        ss_tot    = float(np.sum((y - np.mean(y)) ** 2))
        r_sq      = round(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)), 3) if ss_tot > 0 else 0.0

        mean_act  = float(np.mean(y))
        if mean_act > 0 and abs(slope) / mean_act < 0.02:
            direction = "stable"
        elif slope > 0:
            direction = "growing"
        else:
            direction = "declining"

        last_idx = len(data) - 1
        last_y, last_m = int(data[-1]["month"][:4]), int(data[-1]["month"][5:7])
        projection = []
        for i in range(1, 4):
            proj_val = max(0.0, float(np.polyval(coeffs, last_idx + i)))
            m = last_m + i
            y_off = (m - 1) // 12
            m = ((m - 1) % 12) + 1
            projection.append({"month": f"{last_y + y_off:04d}-{m:02d}",
                                "projected_total": round(proj_val)})

        trend[stop_id] = {
            "slope":         round(slope, 1),
            "direction":     direction,
            "projection_3m": projection,
            "r_squared":     r_sq,
        }
    print(f"    {len(trend):,} tendencias calculadas")

    # ── Agregación por periodo de tiempo ─────────────────────────
    print("\n[6/8] Agregando por periodo de tiempo ...")
    od = (
        od.groupby(["origin_stop", "time", "destination_stop"], as_index=False)["quantity"]
        .sum()
    )
    time_periods = sorted(od["time"].dropna().unique().tolist())
    print(f"    {len(od):,} filas agregadas, {len(time_periods)} periodos de tiempo")

    # ── Probabilidades ────────────────────────────────────────────
    totals = (
        od.groupby(["origin_stop", "time"])["quantity"]
        .sum().rename("total").reset_index()
    )
    od = od.merge(totals, on=["origin_stop", "time"])
    od["prob"] = od["quantity"] / od["total"]

    od_valid = od[
        od["origin_stop"].isin(stops_set) & od["destination_stop"].isin(stops_set)
    ]

    od_probs: dict = {}
    for (origin, time_period), grp in od_valid.groupby(["origin_stop", "time"]):
        top50 = grp.nlargest(50, "prob")
        od_probs[(origin, time_period)] = [
            (row.destination_stop, float(row.prob))
            for row in top50.itertuples(index=False)
        ]
    print(f"    {len(od_probs):,} pares (origen, periodo) indexados")

    # ── Actividad ─────────────────────────────────────────────────
    boardings_df = (
        od.groupby(["origin_stop", "time"])["quantity"]
        .sum().reset_index()
        .rename(columns={"origin_stop": "stop_id", "quantity": "boardings"})
    )
    alightings_df = (
        od.groupby(["destination_stop", "time"])["quantity"]
        .sum().reset_index()
        .rename(columns={"destination_stop": "stop_id", "quantity": "alightings"})
    )
    act_df = boardings_df.merge(alightings_df, on=["stop_id", "time"], how="outer").fillna(0)
    act_df["boardings"]  = act_df["boardings"].astype(int)
    act_df["alightings"] = act_df["alightings"].astype(int)
    act_df["total"]      = act_df["boardings"] + act_df["alightings"]

    activity: dict = {}
    for row in act_df[act_df["stop_id"].isin(stops_set)].itertuples(index=False):
        activity[(row.stop_id, row.time)] = {
            "boardings":  int(row.boardings),
            "alightings": int(row.alightings),
            "total":      int(row.total),
        }

    origin_ids = od_valid["origin_stop"].unique()
    stops_with_trips = [
        {"stop_id": sid, "stop_name": stops_lookup[sid]["name"],
         "lat": stops_lookup[sid]["lat"], "lon": stops_lookup[sid]["lon"]}
        for sid in origin_ids if sid in stops_set
    ]
    print(f"    {len(activity):,} registros de actividad, {len(stops_with_trips):,} paradas con viajes")

    # ── Modelo de gravedad ────────────────────────────────────────
    print(f"\n[7/8] Calculando modelo de gravedad ...")
    beta          = 1.5
    stop_ids_grav = [s["stop_id"] for s in stops_with_trips]
    n_grav        = len(stop_ids_grav)
    lats_g        = np.array([stops_lookup[sid]["lat"] for sid in stop_ids_grav], dtype=np.float64)
    lons_g        = np.array([stops_lookup[sid]["lon"] for sid in stop_ids_grav], dtype=np.float64)

    stop_weight: dict = {}
    for (sid, _tp), v in activity.items():
        stop_weight[sid] = stop_weight.get(sid, 0) + v["total"]
    weights_g = np.array([float(stop_weight.get(sid, 0)) for sid in stop_ids_grav], dtype=np.float64)

    lats_r = np.radians(lats_g)
    lons_r = np.radians(lons_g)
    R_km   = 6371.0

    gravity_probs: dict = {}
    for i, origin_id in enumerate(stop_ids_grav):
        dlat   = lats_r - lats_r[i]
        dlon   = lons_r - lons_r[i]
        a      = np.sin(dlat/2)**2 + np.cos(lats_r[i]) * np.cos(lats_r) * np.sin(dlon/2)**2
        dists  = R_km * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        dists  = np.maximum(dists, 0.05)
        scores = weights_g / np.power(dists, beta)
        scores[i] = 0.0
        total_score = scores.sum()
        if total_score == 0.0:
            continue
        probs   = scores / total_score
        top_idx = np.argpartition(probs, -50)[-50:] if n_grav >= 50 else np.arange(n_grav)
        top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
        gravity_probs[origin_id] = [
            {"stop_id": stop_ids_grav[j], "stop_name": stops_lookup[stop_ids_grav[j]]["name"],
             "lat": float(lats_g[j]), "lon": float(lons_g[j]), "probability": float(probs[j])}
            for j in top_idx if probs[j] > 0
        ]
    print(f"    {len(gravity_probs):,} orígenes indexados en el modelo de gravedad")

    # ── Guardar ───────────────────────────────────────────────────
    print(f"\n[8/8] Guardando en {OUTPUT_PATH} ...")
    payload = {
        "stops_lookup":     stops_lookup,
        "od_probs":         od_probs,
        "activity":         activity,
        "time_periods":     time_periods,
        "stops_with_trips": stops_with_trips,
        "gravity_probs":    gravity_probs,
        "timeseries":       timeseries,
        "trend":            trend,
    }
    with gzip.open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n✓ Listo: {OUTPUT_PATH}")
    print(f"  Tamaño: {size_mb:.1f} MB")
    print(f"\nPróximo paso: subir este archivo como asset de un GitHub Release")
    print(f"  gh release upload v1-data data/precomputed.pkl.gz --clobber")


if __name__ == "__main__":
    main()
