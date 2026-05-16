"""
preprocess.py — Genera el archivo de datos precompilados para producción.

Ejecutar UNA VEZ localmente antes de desplegar:
    python scripts/preprocess.py

Lee los 6 CSV mensuales y stops.txt, aplica toda la lógica de agregación
y probabilidades, y guarda el resultado en data/precomputed.pkl.gz.

El archivo resultante (~20 MB comprimido) reemplaza a los 420 MB de CSVs
en el servidor de producción. El servidor en producción lo descarga
automáticamente al arrancar si no lo tiene en disco.
"""

import gzip
import pickle
from glob import glob
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "precomputed.pkl.gz"


def main():
    print("=" * 60)
    print("SEQ Transit Predictor — preprocesamiento de datos")
    print("=" * 60)

    # ── Paradas ───────────────────────────────────────────────────
    print("\n[1/6] Cargando stops.txt ...")
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
    print("\n[2/6] Leyendo archivos CSV ...")
    csv_files = sorted(glob(str(DATA_DIR / "*TL Org-Dest Trips.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {DATA_DIR}")
    print(f"    {len(csv_files)} archivos encontrados")

    frames = []
    for path in csv_files:
        print(f"    Leyendo {Path(path).name} ...")
        df = pd.read_csv(
            path,
            usecols=["time", "origin_stop", "destination_stop", "quantity"],
            dtype={"origin_stop": str, "destination_stop": str},
            low_memory=False,
        )
        frames.append(df)

    # ── Limpieza y agregación ─────────────────────────────────────
    print("\n[3/6] Limpiando y agregando los 6 meses ...")
    od = pd.concat(frames, ignore_index=True)
    del frames  # Liberamos memoria

    od["origin_stop"] = od["origin_stop"].str.strip()
    od["destination_stop"] = od["destination_stop"].str.strip()
    od["quantity"] = pd.to_numeric(od["quantity"], errors="coerce")
    od = od[od["destination_stop"] != "n/a"]
    od = od.dropna(subset=["origin_stop", "destination_stop", "quantity"])

    od = (
        od.groupby(["origin_stop", "time", "destination_stop"], as_index=False)["quantity"]
        .sum()
    )
    time_periods = sorted(od["time"].dropna().unique().tolist())
    print(f"    {len(od):,} filas agregadas, {len(time_periods)} periodos de tiempo")

    # ── Probabilidades ────────────────────────────────────────────
    print("\n[4/6] Calculando probabilidades ...")
    totals = (
        od.groupby(["origin_stop", "time"])["quantity"]
        .sum()
        .rename("total")
        .reset_index()
    )
    od = od.merge(totals, on=["origin_stop", "time"])
    od["prob"] = od["quantity"] / od["total"]

    od_valid = od[
        od["origin_stop"].isin(stops_set) & od["destination_stop"].isin(stops_set)
    ]

    od_probs = {}
    for (origin, time_period), grp in od_valid.groupby(["origin_stop", "time"]):
        top50 = grp.nlargest(50, "prob")
        od_probs[(origin, time_period)] = [
            (row.destination_stop, float(row.prob))
            for row in top50.itertuples(index=False)
        ]
    print(f"    {len(od_probs):,} pares (origen, periodo) indexados")

    # ── Actividad ─────────────────────────────────────────────────
    print("\n[5/6] Calculando estadísticas de actividad ...")
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

    activity = {}
    for row in act_df[act_df["stop_id"].isin(stops_set)].itertuples(index=False):
        activity[(row.stop_id, row.time)] = {
            "boardings":  int(row.boardings),
            "alightings": int(row.alightings),
            "total":      int(row.total),
        }
    print(f"    {len(activity):,} registros de actividad")

    # ── Lista de paradas visibles en el mapa ──────────────────────
    origin_ids = od_valid["origin_stop"].unique()
    stops_with_trips = [
        {
            "stop_id":   sid,
            "stop_name": stops_lookup[sid]["name"],
            "lat":       stops_lookup[sid]["lat"],
            "lon":       stops_lookup[sid]["lon"],
        }
        for sid in origin_ids
        if sid in stops_set
    ]
    print(f"    {len(stops_with_trips):,} paradas con viajes válidos")

    # ── Guardar ───────────────────────────────────────────────────
    print(f"\n[6/6] Guardando en {OUTPUT_PATH} ...")
    payload = {
        "stops_lookup":    stops_lookup,
        "od_probs":        od_probs,
        "activity":        activity,
        "time_periods":    time_periods,
        "stops_with_trips": stops_with_trips,
    }
    with gzip.open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n✓ Listo: {OUTPUT_PATH}")
    print(f"  Tamaño: {size_mb:.1f} MB")
    print(f"\nPróximo paso: subir este archivo como asset de un GitHub Release")
    print(f"  gh release create v1-data data/precomputed.pkl.gz --title 'Precomputed data'")


if __name__ == "__main__":
    main()
