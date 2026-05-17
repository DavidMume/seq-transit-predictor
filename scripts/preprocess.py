"""
preprocess.py — Genera el archivo de datos precompilados para producción.

Ejecutar UNA VEZ localmente antes de desplegar:
    python scripts/preprocess.py

Lee todos los CSV mensuales y stops.txt, aplica toda la lógica de agregación,
probabilidades, modelo de gravedad y series temporales, y guarda el resultado
en data/precomputed.pkl.gz.

Ajustar LITE_MODE = True para generar la versión reducida para Render.
"""

import gzip
import pickle
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

LITE_MODE = True

# ── Parámetros lite ───────────────────────────────────────────────────────────
LITE_OPERATORS = [
    'Gold Coast Light Rail',
    'Surfside Buslines',
]
LITE_MONTHS    = {'202601', '202602', '202603'}
LITE_TOP_N     = 5
LITE_MIN_TRIPS = 100

DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / ("precomputed_lite.pkl.gz" if LITE_MODE else "precomputed.pkl.gz")


def main():
    print("=" * 60)
    print("SEQ Transit Predictor — preprocesamiento de datos")
    print(f"Modo: {'LITE' if LITE_MODE else 'FULL'}")
    print("=" * 60)

    # ── Paradas ───────────────────────────────────────────────────
    print("\n[1] Cargando stops.txt ...")
    stops_df = pd.read_csv(
        DATA_DIR / "stops.txt",
        usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"],
        dtype={"stop_id": str},
    )
    stops_df["stop_id"]  = stops_df["stop_id"].str.strip()
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
    print("\n[2] Leyendo archivos CSV ...")
    all_csv = sorted(glob(str(DATA_DIR / "*TL Org-Dest Trips.csv")))
    if not all_csv:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {DATA_DIR}")

    if LITE_MODE:
        csv_files = [f for f in all_csv if any(m in Path(f).name for m in LITE_MONTHS)]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found for months {LITE_MONTHS} in {DATA_DIR}")
        print(f"    {len(csv_files)} archivo(s) seleccionado(s) para meses {sorted(LITE_MONTHS)}")
    else:
        csv_files = all_csv
        print(f"    {len(csv_files)} archivos encontrados")

    usecols = ["month", "time", "operator", "origin_stop", "destination_stop", "quantity"] \
              if LITE_MODE else \
              ["month", "time", "origin_stop", "destination_stop", "quantity"]

    frames = []
    for path in csv_files:
        print(f"    Leyendo {Path(path).name} ...")
        df = pd.read_csv(
            path,
            usecols=usecols,
            dtype={"origin_stop": str, "destination_stop": str},
            low_memory=False,
        )
        frames.append(df)

    # ── Limpieza ──────────────────────────────────────────────────
    print("\n[3] Limpiando datos ...")
    od = pd.concat(frames, ignore_index=True)
    del frames

    # Gold Coast operator filter (lite only)
    if LITE_MODE:
        before = len(od)
        od = od[od["operator"].isin(LITE_OPERATORS)]
        od = od.drop(columns=["operator"])
        print(f"    Filtrado por operador: {before:,} → {len(od):,} filas")

    od["origin_stop"]      = od["origin_stop"].str.strip()
    od["destination_stop"] = od["destination_stop"].str.strip()
    od["quantity"]         = pd.to_numeric(od["quantity"], errors="coerce")
    od = od[od["destination_stop"] != "n/a"]
    od = od.dropna(subset=["origin_stop", "destination_stop", "quantity"])
    print(f"    {len(od):,} filas limpias")

    # ── Series temporales y tendencias (solo modo FULL) ──────────
    timeseries: dict = {}
    trend:      dict = {}

    if not LITE_MODE:
        print("\n[4] Calculando series temporales mensuales ...")
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

        for stop_id, grp in monthly.groupby("stop_id"):
            if stop_id not in stops_set:
                continue
            timeseries[stop_id] = [
                {"month": row.month, "boardings": int(row.boardings),
                 "alightings": int(row.alightings), "total": int(row.total)}
                for row in grp.itertuples(index=False)
            ]
        print(f"    {len(timeseries):,} series temporales construidas")

        print("\n[5] Calculando tendencias y proyecciones ...")
        for stop_id, data in timeseries.items():
            if len(data) < 3:
                continue
            x = np.arange(len(data), dtype=np.float64)
            y = np.array([d["total"] for d in data], dtype=np.float64)

            coeffs  = np.polyfit(x, y, 1)
            slope   = float(coeffs[0])
            y_pred  = np.polyval(coeffs, x)
            ss_res  = float(np.sum((y - y_pred) ** 2))
            ss_tot  = float(np.sum((y - np.mean(y)) ** 2))
            r_sq    = round(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)), 3) if ss_tot > 0 else 0.0

            mean_act = float(np.mean(y))
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
    else:
        print("\n[4-5] Series temporales y tendencias: omitidas (modo LITE)")

    # ── Agregación por periodo de tiempo ─────────────────────────
    print("\n[6] Agregando por periodo de tiempo ...")
    od = (
        od.groupby(["origin_stop", "time", "destination_stop"], as_index=False)["quantity"]
        .sum()
    )
    time_periods = sorted(od["time"].dropna().unique().tolist())
    print(f"    {len(od):,} filas agregadas, {len(time_periods)} periodos de tiempo")

    # ── Probabilidades ────────────────────────────────────────────
    top_n = LITE_TOP_N if LITE_MODE else 50

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
        top_rows = grp.nlargest(top_n, "prob")
        if LITE_MODE:
            od_probs[(origin, time_period)] = [
                (row.destination_stop, float(np.float16(row.prob)))
                for row in top_rows.itertuples(index=False)
            ]
        else:
            od_probs[(origin, time_period)] = [
                (row.destination_stop, float(row.prob))
                for row in top_rows.itertuples(index=False)
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

    # ── Minimum trips filter (lite only) ─────────────────────────
    if LITE_MODE:
        origin_totals: dict = {}
        for (sid, _tp), v in activity.items():
            origin_totals[sid] = origin_totals.get(sid, 0) + v["boardings"]
        before_count = len(stops_with_trips)
        stops_with_trips = [s for s in stops_with_trips
                            if origin_totals.get(s["stop_id"], 0) >= LITE_MIN_TRIPS]
        valid_sids = {s["stop_id"] for s in stops_with_trips}
        od_probs   = {k: v for k, v in od_probs.items() if k[0] in valid_sids}
        activity   = {k: v for k, v in activity.items() if k[0] in valid_sids}
        print(f"    Filtro ≥{LITE_MIN_TRIPS} viajes: {before_count:,} → {len(stops_with_trips):,} paradas")

    print(f"    {len(activity):,} registros de actividad, {len(stops_with_trips):,} paradas con viajes")

    # ── Calibración de beta y modelo de gravedad (solo FULL) ──────
    gravity_probs:   dict  = {}
    beta_calibrated: float = 2.0  # fixed in lite mode

    if not LITE_MODE:
        print("\n[7a] Calibrando beta mediante búsqueda en cuadrícula ...")
        idx_of_cal = {s["stop_id"]: i for i, s in enumerate(stops_with_trips)}
        n_cal      = len(stops_with_trips)
        lats_cal_r = np.radians(np.array([stops_lookup[s["stop_id"]]["lat"] for s in stops_with_trips], dtype=np.float64))
        lons_cal_r = np.radians(np.array([stops_lookup[s["stop_id"]]["lon"] for s in stops_with_trips], dtype=np.float64))
        R_km_cal   = 6371.0

        sw_cal = np.zeros(n_cal, dtype=np.float64)
        for (sid, _tp), v in activity.items():
            if sid in idx_of_cal:
                sw_cal[idx_of_cal[sid]] += v["total"]

        BIN_MAX_CAL = 50
        obs_tld_cal = np.zeros(BIN_MAX_CAL, dtype=np.float64)
        cal_oi, cal_di, cal_t = [], [], []
        for (origin_id, tp), dests in od_probs.items():
            if origin_id not in idx_of_cal:
                continue
            b = activity.get((origin_id, tp), {}).get("boardings", 0)
            if b == 0:
                continue
            oi = idx_of_cal[origin_id]
            for dest_id, prob in dests:
                if dest_id not in idx_of_cal or dest_id == origin_id:
                    continue
                cal_oi.append(oi); cal_di.append(idx_of_cal[dest_id]); cal_t.append(float(prob) * b)

        best_rmse_c = float("inf")
        if cal_t:
            cal_oi_a = np.array(cal_oi); cal_di_a = np.array(cal_di); cal_t_a = np.array(cal_t, dtype=np.float64)
            dlat_c = lats_cal_r[cal_di_a] - lats_cal_r[cal_oi_a]
            dlon_c = lons_cal_r[cal_di_a] - lons_cal_r[cal_oi_a]
            a_c    = np.sin(dlat_c/2)**2 + np.cos(lats_cal_r[cal_oi_a]) * np.cos(lats_cal_r[cal_di_a]) * np.sin(dlon_c/2)**2
            obs_d  = R_km_cal * 2 * np.arcsin(np.sqrt(np.clip(a_c, 0.0, 1.0)))
            np.add.at(obs_tld_cal, np.clip(obs_d.astype(int), 0, BIN_MAX_CAL - 1), cal_t_a)
            if obs_tld_cal.sum() > 0:
                obs_tld_cal /= obs_tld_cal.sum()
                MAX_O = min(n_cal, 300)
                sidxs = np.round(np.linspace(0, n_cal - 1, MAX_O)).astype(int)
                dlat_m = lats_cal_r[np.newaxis, :] - lats_cal_r[sidxs, np.newaxis]
                dlon_m = lons_cal_r[np.newaxis, :] - lons_cal_r[sidxs, np.newaxis]
                a_m    = (np.sin(dlat_m/2)**2 + np.cos(lats_cal_r[sidxs, np.newaxis]) * np.cos(lats_cal_r[np.newaxis, :]) * np.sin(dlon_m/2)**2)
                dist_m = np.maximum(R_km_cal * 2 * np.arcsin(np.sqrt(np.clip(a_m, 0.0, 1.0))), 0.05)
                dbins_m = np.clip(dist_m.astype(int), 0, BIN_MAX_CAL - 1)
                ob_arr  = np.zeros(n_cal, dtype=np.float64)
                for (sid, _tp), v in activity.items():
                    if sid in idx_of_cal:
                        ob_arr[idx_of_cal[sid]] += v["boardings"]
                sboard = ob_arr[sidxs]
                for bi in range(50, 301, 10):
                    bt = bi / 100.0
                    sc  = sw_cal[np.newaxis, :] / np.power(dist_m, bt)
                    sc[np.arange(MAX_O), sidxs] = 0.0
                    ts  = sc.sum(axis=1, keepdims=True); vld = ts[:, 0] > 0
                    pm  = np.zeros_like(sc); pm[vld] = sc[vld] / ts[vld]
                    tri = pm * sboard[:, np.newaxis]
                    ptld = np.bincount(dbins_m.ravel(), weights=tri.ravel(), minlength=BIN_MAX_CAL)[:BIN_MAX_CAL]
                    if ptld.sum() > 0: ptld /= ptld.sum()
                    rmse_c = float(np.sqrt(np.mean((obs_tld_cal - ptld)**2)))
                    if rmse_c < best_rmse_c:
                        best_rmse_c = rmse_c; beta_calibrated = bt
        rmse_str = f"{best_rmse_c:.4f}" if cal_t else "n/a"
        print(f"    Beta calibrado: {beta_calibrated} (RMSE: {rmse_str})")

        print(f"\n[7] Calculando modelo de gravedad (beta={beta_calibrated}) ...")
        beta          = beta_calibrated
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
    else:
        print("\n[7] Modelo de gravedad: omitido (modo LITE)")

    # ── Estimación de memoria ─────────────────────────────────────
    total = sum(sys.getsizeof(v) for v in [
        stops_lookup, od_probs, activity,
        time_periods, stops_with_trips
    ])
    print(f"\nEstimated memory: {total / 1024 / 1024:.1f} MB")

    # ── Guardar ───────────────────────────────────────────────────
    print(f"\n[8] Guardando en {OUTPUT_PATH} ...")
    payload = {
        "stops_lookup":     stops_lookup,
        "od_probs":         od_probs,
        "activity":         activity,
        "time_periods":     time_periods,
        "stops_with_trips": stops_with_trips,
        "gravity_probs":    gravity_probs,
        "timeseries":       timeseries,
        "trend":            trend,
        "beta_calibrated":  beta_calibrated,
    }
    with gzip.open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n✓ Listo: {OUTPUT_PATH}")
    print(f"  Tamaño: {size_mb:.1f} MB")
    print(f"  Paradas: {len(stops_with_trips):,}")
    print(f"  Pares OD: {len(od_probs):,}")


if __name__ == "__main__":
    main()
