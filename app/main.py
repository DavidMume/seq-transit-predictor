# =============================================================================
# main.py — Servidor principal de la aplicación SEQ Transit Destination Predictor
#
# Este archivo hace tres cosas grandes:
#   1. Al arrancar, lee todos los datos (paradas y viajes) y los procesa
#      para que las respuestas del servidor sean instantáneas.
#   2. Define los "endpoints" — es decir, las URLs a las que el navegador
#      puede pedir información (como /stops, /predict, /activity).
#   3. Sirve el archivo HTML de la aplicación web al navegador.
# =============================================================================

# --- Importaciones -----------------------------------------------------------
# Cada "import" trae una herramienta externa que usamos en este archivo.

import gzip                             # Para leer el archivo de datos precomprimido (.pkl.gz)
import logging                          # Para imprimir mensajes informativos mientras el servidor trabaja
import os                               # Para leer variables de entorno (como DATA_DOWNLOAD_URL)
import pickle                           # Para deserializar el archivo de datos precompilados
import urllib.request                   # Para descargar el archivo de datos si no está en disco
from contextlib import asynccontextmanager  # Nos permite ejecutar código al iniciar y al cerrar el servidor
from glob import glob                   # Para buscar archivos usando patrones (como *.csv)
from pathlib import Path                # Para construir rutas de archivos de forma segura en cualquier sistema operativo

import numpy as np                      # Para operaciones vectorizadas rápidas sobre arrays (usado en el modelo de gravedad)
import pandas as pd                     # La herramienta principal para leer y manipular tablas de datos (como Excel pero en código)
from fastapi import FastAPI             # El framework (estructura base) con el que construimos el servidor web
from fastapi.middleware.cors import CORSMiddleware   # Permite que el navegador haga peticiones al servidor sin restricciones de seguridad de origen
from fastapi.responses import FileResponse           # Para enviar un archivo como respuesta (usamos esto para entregar el HTML)
from fastapi.staticfiles import StaticFiles          # Para servir archivos estáticos como imágenes o el index.html

# --- Configuración del sistema de logs ---------------------------------------
# "logging" es como un diario del servidor: imprime en la consola lo que
# está pasando en cada momento (cargando datos, errores, etc.).
# El formato muestra: hora  NIVEL  mensaje
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)  # Creamos un "diario" específico para este archivo

# --- Rutas de carpetas -------------------------------------------------------
# Path(__file__) es la ruta de este mismo archivo (main.py).
# .parent sube un nivel en la carpeta, como hacer "cd .." en la terminal.
DATA_DIR = Path(__file__).parent.parent / "data"    # Carpeta donde están los CSV y stops.txt
STATIC_DIR = Path(__file__).parent / "static"       # Carpeta donde está el archivo index.html

# Ruta del archivo precompilado. Si existe, el servidor arranca en segundos en vez de 30 segundos.
PRECOMPUTED_PATH = DATA_DIR / "precomputed.pkl.gz"

# URL de descarga del archivo precompilado. Se configura como variable de entorno en producción.
# Si no está configurada, el servidor intentará leer los CSV crudos desde DATA_DIR.
DATA_DOWNLOAD_URL = os.getenv("DATA_DOWNLOAD_URL", "")

# --- Variables globales ------------------------------------------------------
# Estas variables se llenan al arrancar el servidor y se quedan en memoria
# mientras el servidor está corriendo. Así, cada consulta del navegador
# recibe la respuesta en milisegundos en vez de tener que procesar datos de nuevo.

_stops_lookup: dict = {}    # Diccionario: stop_id → {name, lat, lon}. Nos dice el nombre y coordenadas de cada parada.
_od_probs: dict = {}         # Diccionario: (stop_origen, periodo) → lista de (stop_destino, probabilidad). El corazón de las predicciones.
_activity: dict = {}         # Diccionario: (stop_id, periodo) → {subidas, bajadas, total}. Estadísticas de actividad.
_time_periods: list = []     # Lista con los nombres de los periodos de tiempo disponibles (ej: "Weekday (8:30am-2:59:59pm)")
_stops_with_trips: list = [] # Lista de paradas que tienen al menos un viaje registrado — las que aparecen en el mapa
_gravity_probs: dict = {}    # Diccionario: stop_id → lista de {stop_id, stop_name, lat, lon, probability} según modelo de gravedad
_timeseries: dict = {}       # Diccionario: stop_id → lista cronológica de {month, boardings, alightings, total} por cada mes del dataset
_trend: dict = {}            # Diccionario: stop_id → {slope, direction, projection_3m, r_squared} de la regresión lineal

BETA_CALIBRATED: float = 1.5  # Exponente beta del modelo de gravedad, calibrado automáticamente al arrancar
_beta_rmse: float = 0.0       # RMSE obtenido durante la calibración (0 = aún no calibrado)


# =============================================================================
# _download_precomputed() — Descarga el archivo precompilado si no existe en disco
#
# Solo se ejecuta si DATA_DOWNLOAD_URL está configurado (en producción) y el
# archivo precomputed.pkl.gz no existe en la carpeta data/.
# =============================================================================
def _download_precomputed() -> None:
    if PRECOMPUTED_PATH.exists():
        return  # Ya tenemos el archivo, no hay que descargar nada
    if not DATA_DOWNLOAD_URL:
        return  # No hay URL configurada — asumimos que los CSV están disponibles localmente
    log.info(f"Descargando datos precompilados desde {DATA_DOWNLOAD_URL} ...")
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(DATA_DOWNLOAD_URL, PRECOMPUTED_PATH)
    size_mb = PRECOMPUTED_PATH.stat().st_size / 1024 / 1024
    log.info(f"  Descargados {size_mb:.1f} MB → {PRECOMPUTED_PATH}")


# =============================================================================
# _load_from_precomputed() — Carga datos desde el archivo pickle comprimido
#
# Si el archivo precomputed.pkl.gz existe, carga todas las estructuras en
# memoria en segundos. Devuelve True si tuvo éxito, False si no existe el archivo.
# =============================================================================
def _load_from_precomputed() -> bool:
    global _stops_lookup, _od_probs, _activity, _time_periods, _stops_with_trips, _gravity_probs, _timeseries, _trend
    if not PRECOMPUTED_PATH.exists():
        return False
    log.info(f"Cargando datos precompilados desde {PRECOMPUTED_PATH} ...")
    with gzip.open(PRECOMPUTED_PATH, "rb") as f:
        payload = pickle.load(f)
    _stops_lookup = payload["stops_lookup"]
    _od_probs = payload["od_probs"]
    _activity = payload["activity"]
    _time_periods = payload["time_periods"]
    _stops_with_trips = payload["stops_with_trips"]
    # Los siguientes campos son opcionales — archivos generados antes de estas versiones no los tienen
    _gravity_probs = payload.get("gravity_probs", {})
    _timeseries = payload.get("timeseries", {})
    _trend = payload.get("trend", {})
    log.info(
        f"  {len(_stops_lookup):,} paradas, "
        f"{len(_od_probs):,} pares OD, "
        f"{len(_gravity_probs):,} orígenes gravedad, "
        f"{len(_timeseries):,} series temporales cargadas."
    )
    return True


# =============================================================================
# _compute_monthly_series() — Calcula series temporales y tendencias por parada
#
# Una "serie temporal" es simplemente la actividad de una parada mes a mes,
# como un historial de cuánta gente la usó en enero 2022, febrero 2022, etc.
#
# Para detectar tendencias usamos "regresión lineal" (mínimos cuadrados ordinarios):
# trazamos la línea recta que mejor se ajusta a todos los puntos del historial.
# Si la línea sube, la parada está creciendo. Si baja, está perdiendo actividad.
#
# R² (coeficiente de determinación) nos dice qué tan confiable es esa línea:
#   R² = 1.0 → la línea explica perfectamente los datos (raro en datos reales)
#   R² = 0.5 → la línea explica la mitad de la variación
#   R² = 0.0 → la línea no explica nada — los datos son demasiado irregulares
#
# Recibe od_with_months: el DataFrame de viajes ANTES de la agregación por tiempo,
# con columna "month" en formato "YYYY-MM".
# =============================================================================
def _compute_monthly_series(od_with_months: pd.DataFrame) -> None:
    global _timeseries, _trend

    stops_with_coords = set(_stops_lookup.keys())  # Solo paradas con coordenadas en el mapa

    log.info("Calculando series temporales mensuales ...")

    # ── Subidas (boardings): cuántas personas SALIERON de cada parada cada mes ──
    monthly_board = (
        od_with_months.groupby(["origin_stop", "month"])["quantity"]
        .sum().reset_index()
        .rename(columns={"origin_stop": "stop_id", "quantity": "boardings"})
    )

    # ── Bajadas (alightings): cuántas personas LLEGARON a cada parada cada mes ──
    monthly_alight = (
        od_with_months.groupby(["destination_stop", "month"])["quantity"]
        .sum().reset_index()
        .rename(columns={"destination_stop": "stop_id", "quantity": "alightings"})
    )

    # Unimos subidas y bajadas. "outer" porque una parada puede tener bajadas pero
    # no subidas en algún mes específico, o viceversa.
    monthly = (
        monthly_board
        .merge(monthly_alight, on=["stop_id", "month"], how="outer")
        .fillna(0)
        .sort_values(["stop_id", "month"])
    )
    monthly["boardings"]  = monthly["boardings"].astype(int)
    monthly["alightings"] = monthly["alightings"].astype(int)
    monthly["total"]      = monthly["boardings"] + monthly["alightings"]

    # ── Construir el diccionario de series temporales ─────────────────────────
    for stop_id, grp in monthly.groupby("stop_id"):
        if stop_id not in stops_with_coords:
            continue  # Ignoramos paradas que no aparecen en el mapa
        _timeseries[stop_id] = [
            {
                "month":      row.month,
                "boardings":  int(row.boardings),
                "alightings": int(row.alightings),
                "total":      int(row.total),
            }
            for row in grp.itertuples(index=False)
        ]

    log.info(f"  {len(_timeseries):,} series temporales construidas")

    # ── Regresión lineal por parada ───────────────────────────────────────────
    # Para cada parada calculamos la pendiente (¿cuántos viajes gana o pierde
    # por mes?) y la proyección de los próximos 3 meses.
    log.info("Calculando tendencias y proyecciones ...")
    for stop_id, data in _timeseries.items():
        if len(data) < 3:
            continue  # Con menos de 3 puntos la regresión no es significativa

        x = np.arange(len(data), dtype=np.float64)         # Índice numérico del mes: 0, 1, 2, ...
        y = np.array([d["total"] for d in data], dtype=np.float64)

        # np.polyfit calcula los coeficientes de la línea recta que minimiza
        # la suma de errores al cuadrado. Devuelve [pendiente, intercepto].
        coeffs = np.polyfit(x, y, 1)
        slope     = float(coeffs[0])    # Viajes ganados (o perdidos) por mes
        intercept = float(coeffs[1])

        # ── Calcular R² ───────────────────────────────────────────────────────
        # R² = 1 - (suma de errores del modelo) / (suma de variación total)
        y_pred  = np.polyval(coeffs, x)
        ss_res  = float(np.sum((y - y_pred) ** 2))
        ss_tot  = float(np.sum((y - np.mean(y)) ** 2))
        r_sq    = round(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)), 3) if ss_tot > 0 else 0.0

        # ── Clasificar la dirección ───────────────────────────────────────────
        # "Estable" si el cambio mensual es menor al 2% de la media de actividad
        mean_act = float(np.mean(y))
        if mean_act > 0 and abs(slope) / mean_act < 0.02:
            direction = "stable"
        elif slope > 0:
            direction = "growing"
        else:
            direction = "declining"

        # ── Proyección de los próximos 3 meses ───────────────────────────────
        last_idx      = len(data) - 1
        last_month_str = data[-1]["month"]  # "YYYY-MM"
        last_y, last_m = int(last_month_str[:4]), int(last_month_str[5:7])
        projection = []
        for i in range(1, 4):
            proj_val = max(0.0, float(np.polyval(coeffs, last_idx + i)))
            m = last_m + i
            y_off = (m - 1) // 12
            m = ((m - 1) % 12) + 1
            projection.append({
                "month":            f"{last_y + y_off:04d}-{m:02d}",
                "projected_total":  round(proj_val),
            })

        _trend[stop_id] = {
            "slope":          round(slope, 1),
            "direction":      direction,
            "projection_3m":  projection,
            "r_squared":      r_sq,
        }

    log.info(f"  {len(_trend):,} tendencias calculadas")


# =============================================================================
# _load_from_csvs() — Carga y procesa todos los datos desde los CSV crudos
#
# Ruta de respaldo para desarrollo local cuando no existe precomputed.pkl.gz.
# Lee miles de registros de viajes y los convierte en estructuras rápidas
# de consultar. Puede tardar ~30 segundos para 6 meses de datos.
# =============================================================================
def _load_from_csvs() -> None:
    # "global" le dice a Python que queremos modificar las variables de arriba,
    # no crear variables locales nuevas con el mismo nombre.
    global _stops_lookup, _od_probs, _activity, _time_periods, _stops_with_trips, _timeseries, _trend

    # ── 1. Cargar paradas (stops.txt) ─────────────────────────────────────────
    # stops.txt es un archivo del formato GTFS (estándar de transporte público).
    # Contiene el ID, nombre y coordenadas (latitud/longitud) de cada parada del SEQ.
    log.info("Loading stops.txt …")
    stops_df = pd.read_csv(
        DATA_DIR / "stops.txt",
        usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"],  # Solo leemos las 4 columnas que necesitamos
        dtype={"stop_id": str},  # Forzamos que el ID sea texto, no número, para evitar errores al comparar
    )

    # El archivo tiene espacios en blanco al principio/final de algunos valores.
    # str.strip() los elimina, como recortar los bordes de un texto.
    stops_df["stop_id"] = stops_df["stop_id"].str.strip()

    # pd.to_numeric convierte el texto de latitud/longitud a número real.
    # errors="coerce" significa: si no se puede convertir, poner NaN (valor vacío) en vez de dar error.
    stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
    stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")

    # Eliminamos las filas donde la latitud o longitud no son válidas (quedaron como NaN).
    # No tiene sentido tener una parada sin coordenadas — no podríamos mostrarla en el mapa.
    stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])

    # Construimos el diccionario de paradas: { "1234": {"name": "...", "lat": -27.4, "lon": 153.0}, ... }
    # itertuples recorre cada fila de la tabla de forma eficiente.
    _stops_lookup = {
        row.stop_id: {"name": row.stop_name, "lat": float(row.stop_lat), "lon": float(row.stop_lon)}
        for row in stops_df.itertuples(index=False)
    }

    # Creamos un conjunto (set) con solo los IDs. Los conjuntos son mucho más rápidos
    # que las listas para preguntar "¿está este elemento aquí?".
    stops_set = set(_stops_lookup)
    log.info(f"  {len(_stops_lookup):,} stops loaded")

    # ── 2. Leer los archivos CSV de viajes (Origen-Destino) ───────────────────
    # Hay 6 archivos CSV, uno por mes (octubre 2025 a marzo 2026).
    # Usamos glob con el patrón "*TL Org-Dest Trips.csv" para encontrarlos todos
    # automáticamente, sin importar el nombre exacto de cada archivo.
    csv_files = sorted(glob(str(DATA_DIR / "*TL Org-Dest Trips.csv")))

    # Si no se encontró ningún archivo, lanzamos un error claro para que el usuario sepa qué falta.
    if not csv_files:
        raise FileNotFoundError(f"No OD CSV files found in {DATA_DIR}")
    log.info(f"Found {len(csv_files)} OD files")

    # Leemos cada archivo y lo guardamos en una lista de tablas.
    # Después las uniremos todas en una sola tabla grande.
    frames = []
    for path in csv_files:
        log.info(f"  Reading {Path(path).name} …")
        df = pd.read_csv(
            path,
            usecols=["month", "time", "origin_stop", "destination_stop", "quantity"],  # Leemos también "month" para la serie temporal
            dtype={"origin_stop": str, "destination_stop": str},  # IDs como texto, igual que en stops.txt
            low_memory=False,  # Evita una advertencia de pandas cuando hay tipos de datos mezclados en el CSV
        )
        frames.append(df)  # Agregamos esta tabla a la lista

    # pd.concat une todas las tablas mensuales en una sola tabla grande.
    # ignore_index=True reinicia la numeración de filas desde 0.
    log.info("Concatenating all months …")
    od = pd.concat(frames, ignore_index=True)

    # ── 3. Limpiar los datos ──────────────────────────────────────────────────
    # Los datos crudos pueden tener espacios, valores inválidos o destinos vacíos.
    # Es importante limpiar antes de calcular cualquier cosa.

    od["origin_stop"] = od["origin_stop"].str.strip()       # Quitamos espacios del ID de origen
    od["destination_stop"] = od["destination_stop"].str.strip()  # Quitamos espacios del ID de destino

    # quantity es la cantidad de viajes. La convertimos a número y descartamos filas donde no sea válido.
    od["quantity"] = pd.to_numeric(od["quantity"], errors="coerce")

    # Algunos registros tienen "n/a" como destino — son viajes sin destino registrado.
    # Los eliminamos porque no sirven para calcular probabilidades.
    od = od[od["destination_stop"] != "n/a"]

    # Eliminamos cualquier fila que tenga valores vacíos en las columnas clave.
    od = od.dropna(subset=["origin_stop", "destination_stop", "quantity"])

    # ── 3b. Calcular series temporales y tendencias ───────────────────────────
    # Hacemos esto ANTES de la agregación porque la columna "month" se pierde
    # en el groupby siguiente, que agrupa sin importar el mes del año.
    _compute_monthly_series(od)

    # ── 4. Agregar todos los meses en una sola tabla ──────────────────────────
    # En vez de tener filas separadas para octubre, noviembre, etc. del mismo viaje,
    # las sumamos todas. Así obtenemos el total de 6 meses para cada combinación
    # de (origen, periodo, destino).
    log.info("Aggregating …")
    od = (
        od.groupby(["origin_stop", "time", "destination_stop"], as_index=False)["quantity"]
        .sum()  # Sumamos la columna "quantity" dentro de cada grupo
    )

    # Extraemos la lista de periodos de tiempo únicos que existen en los datos.
    # sorted() los ordena alfabéticamente para que el desplegable se vea ordenado.
    _time_periods = sorted(od["time"].dropna().unique().tolist())
    log.info(f"  Time periods: {_time_periods}")

    # ── 5. Calcular probabilidades de destino ─────────────────────────────────
    # La probabilidad de ir de A a B en un periodo es:
    #   P(B | A, periodo) = viajes(A→B, periodo) / total_viajes(A, periodo)
    #
    # Primero calculamos el total de viajes por cada (origen, periodo).
    log.info("Computing probabilities …")
    totals = (
        od.groupby(["origin_stop", "time"])["quantity"]
        .sum()           # Total de viajes saliendo de ese origen en ese periodo
        .rename("total") # Renombramos la columna para que sea clara
        .reset_index()   # Convertimos el resultado de vuelta a tabla normal
    )

    # Unimos los totales con la tabla principal para tener el total en cada fila.
    # Así podemos dividir quantity / total en cada fila de forma vectorial (rápida).
    od = od.merge(totals, on=["origin_stop", "time"])
    od["prob"] = od["quantity"] / od["total"]  # Esta es la probabilidad: entre 0 y 1

    # Filtramos para quedarnos SOLO con filas donde tanto el origen como el destino
    # existen en stops.txt. Si no están en el mapa, no sirve mostrarlos.
    od_valid = od[
        od["origin_stop"].isin(stops_set) & od["destination_stop"].isin(stops_set)
    ]

    # ── 6. Construir el índice de probabilidades ──────────────────────────────
    # Recorremos cada combinación de (origen, periodo) y guardamos los 50 destinos
    # más probables en el diccionario _od_probs.
    # Esto es lo que hace rápido al endpoint /predict: ya no necesita calcular nada,
    # solo busca en este diccionario.
    log.info("Building probability index …")
    for (origin, time_period), grp in od_valid.groupby(["origin_stop", "time"]):
        # nlargest(50, "prob") devuelve las 50 filas con mayor probabilidad
        top50 = grp.nlargest(50, "prob")
        _od_probs[(origin, time_period)] = [
            (row.destination_stop, float(row.prob))  # Guardamos solo el ID y la probabilidad
            for row in top50.itertuples(index=False)
        ]

    # ── 7. Calcular estadísticas de actividad (subidas y bajadas) ─────────────
    # Para cada parada y periodo, calculamos cuántas personas suben (boardings)
    # y cuántas bajan (alightings). Esto es diferente a la probabilidad:
    # aquí nos interesa el volumen total, no la dirección probable.
    log.info("Computing activity stats …")

    # Boardings: agrupamos por parada de ORIGEN. Cada fila donde esta parada es
    # el origen representa personas que SUBEN en ella.
    boardings_df = (
        od.groupby(["origin_stop", "time"])["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"origin_stop": "stop_id", "quantity": "boardings"})  # Renombramos para claridad
    )

    # Alightings: agrupamos por parada de DESTINO. Cada fila donde esta parada es
    # el destino representa personas que BAJAN en ella.
    alightings_df = (
        od.groupby(["destination_stop", "time"])["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"destination_stop": "stop_id", "quantity": "alightings"})
    )

    # Unimos las subidas y bajadas en una sola tabla usando outer join.
    # "outer" significa: si una parada tiene subidas pero no bajadas (o viceversa),
    # igual aparece en el resultado, con 0 en el lado que falte.
    act_df = boardings_df.merge(alightings_df, on=["stop_id", "time"], how="outer").fillna(0)

    # Convertimos a entero porque no tiene sentido hablar de 142.0 viajes.
    act_df["boardings"] = act_df["boardings"].astype(int)
    act_df["alightings"] = act_df["alightings"].astype(int)
    act_df["total"] = act_df["boardings"] + act_df["alightings"]  # Total = subidas + bajadas

    # Guardamos en el diccionario global, pero solo para paradas que existen en stops.txt.
    for row in act_df[act_df["stop_id"].isin(stops_set)].itertuples(index=False):
        _activity[(row.stop_id, row.time)] = {
            "boardings": int(row.boardings),
            "alightings": int(row.alightings),
            "total": int(row.total),
        }
    log.info(f"  {len(_activity):,} activity records built")

    # ── 8. Lista de paradas que aparecerán en el mapa ─────────────────────────
    # Solo mostramos paradas que tienen al menos un viaje válido como origen,
    # Y que también existen en stops.txt (para tener coordenadas).
    origin_ids = od_valid["origin_stop"].unique()  # IDs únicos de paradas origen
    _stops_with_trips = [
        {
            "stop_id": sid,
            "stop_name": _stops_lookup[sid]["name"],
            "lat": _stops_lookup[sid]["lat"],
            "lon": _stops_lookup[sid]["lon"],
        }
        for sid in origin_ids
        if sid in stops_set  # Doble verificación: solo si tiene coordenadas
    ]
    log.info(f"  {len(_stops_with_trips):,} origin stops with valid trips")
    log.info("Data loading complete.")


# =============================================================================
# _calibrate_beta() — Calibra el exponente beta del modelo de gravedad
#
# El "beta" controla qué tan fuerte penaliza la distancia en el modelo.
# Un beta alto hace que los viajes largos sean mucho menos probables;
# un beta bajo permite destinos lejanos con más facilidad.
#
# Para encontrar el beta óptimo comparamos la "Distribución de Longitud de
# Viaje" (TLD) observada en los datos históricos contra la que predice el
# modelo de gravedad para cada valor de beta. El beta ganador es el que
# minimiza el RMSE entre ambas distribuciones.
#
# La TLD observada se construye a partir de _od_probs × subidas en _activity.
# Para la TLD predicha se usan hasta MAX_ORIGINS paradas muestreadas.
# =============================================================================
def _calibrate_beta() -> None:
    global BETA_CALIBRATED, _beta_rmse

    stop_ids = [s["stop_id"] for s in _stops_with_trips]
    n = len(stop_ids)
    if n < 10 or not _od_probs or not _activity:
        log.info("Calibración de beta omitida: datos insuficientes. Usando beta=1.5 por defecto.")
        return

    log.info("Calibrando beta del modelo de gravedad mediante búsqueda en cuadrícula ...")

    # Índice rápido: stop_id → posición en el array
    idx_of = {sid: i for i, sid in enumerate(stop_ids)}

    # Coordenadas en radianes para cálculo Haversine vectorizado
    lats_r = np.radians(np.array([_stops_lookup[sid]["lat"] for sid in stop_ids], dtype=np.float64))
    lons_r = np.radians(np.array([_stops_lookup[sid]["lon"] for sid in stop_ids], dtype=np.float64))
    R_km = 6371.0

    # Peso total de actividad por parada (suma de todos los periodos de tiempo)
    stop_weight = np.zeros(n, dtype=np.float64)
    for (sid, _tp), v in _activity.items():
        if sid in idx_of:
            stop_weight[idx_of[sid]] += v["total"]

    # ── Construir la TLD observada ────────────────────────────────────────────
    # Para cada par (origen, destino) registrado estimamos el número de viajes
    # como: viajes ≈ probabilidad_histórica × subidas_en_ese_periodo
    # Luego agrupamos por distancia en bins de 1 km (0–50 km).
    BIN_MAX = 50
    obs_tld = np.zeros(BIN_MAX, dtype=np.float64)

    od_oi_list, od_di_list, od_t_list = [], [], []
    for (origin_id, time_period), dests in _od_probs.items():
        if origin_id not in idx_of:
            continue
        boardings = _activity.get((origin_id, time_period), {}).get("boardings", 0)
        if boardings == 0:
            continue
        oi = idx_of[origin_id]
        for dest_id, prob in dests:
            if dest_id not in idx_of or dest_id == origin_id:
                continue
            od_oi_list.append(oi)
            od_di_list.append(idx_of[dest_id])
            od_t_list.append(float(prob) * boardings)

    if not od_t_list:
        log.info("Sin pares OD válidos para calibración. Usando beta=1.5 por defecto.")
        return

    od_oi = np.array(od_oi_list)
    od_di = np.array(od_di_list)
    od_t  = np.array(od_t_list, dtype=np.float64)

    dlat_obs = lats_r[od_di] - lats_r[od_oi]
    dlon_obs = lons_r[od_di] - lons_r[od_oi]
    a_obs    = (np.sin(dlat_obs / 2) ** 2
                + np.cos(lats_r[od_oi]) * np.cos(lats_r[od_di]) * np.sin(dlon_obs / 2) ** 2)
    obs_dists_km = R_km * 2 * np.arcsin(np.sqrt(np.clip(a_obs, 0.0, 1.0)))

    obs_bin_idx = np.clip(obs_dists_km.astype(int), 0, BIN_MAX - 1)
    np.add.at(obs_tld, obs_bin_idx, od_t)
    if obs_tld.sum() == 0:
        return
    obs_tld /= obs_tld.sum()

    # ── Matriz de distancias para la búsqueda en cuadrícula ───────────────────
    # Muestreamos hasta MAX_ORIGINS paradas para mantener el uso de memoria
    # razonable (matriz de ~23 MB para 300 × 12 000 paradas).
    MAX_ORIGINS = min(n, 300)
    sample_idxs = np.round(np.linspace(0, n - 1, MAX_ORIGINS)).astype(int)

    lats_s = lats_r[sample_idxs]
    lons_s = lons_r[sample_idxs]

    # dist_m shape: (MAX_ORIGINS, n) — distancia desde cada origen muestreado a todas las paradas
    dlat_m = lats_r[np.newaxis, :] - lats_s[:, np.newaxis]
    dlon_m = lons_r[np.newaxis, :] - lons_s[:, np.newaxis]
    a_m    = (np.sin(dlat_m / 2) ** 2
              + np.cos(lats_s[:, np.newaxis]) * np.cos(lats_r[np.newaxis, :]) * np.sin(dlon_m / 2) ** 2)
    dist_m = R_km * 2 * np.arcsin(np.sqrt(np.clip(a_m, 0.0, 1.0)))
    dist_m = np.maximum(dist_m, 0.05)

    # Índices de bin para cada par muestreado (precomputados, no cambian con beta)
    dist_bins_m = np.clip(dist_m.astype(int), 0, BIN_MAX - 1)  # (MAX_ORIGINS, n)

    # Subidas totales por parada muestreada (todos los periodos)
    origin_boardings = np.zeros(n, dtype=np.float64)
    for (sid, _tp), v in _activity.items():
        if sid in idx_of:
            origin_boardings[idx_of[sid]] += v["boardings"]
    sample_boardings = origin_boardings[sample_idxs]

    # Auto-bucles: cada origen no puede ser su propio destino
    self_rows = np.arange(MAX_ORIGINS)
    self_cols = sample_idxs

    # ── Búsqueda en cuadrícula: beta 0.5 → 3.0 en pasos de 0.1 ──────────────
    best_beta = 1.5
    best_rmse = float("inf")

    for beta_int in range(50, 301, 10):
        beta = beta_int / 100.0

        scores_m = stop_weight[np.newaxis, :] / np.power(dist_m, beta)
        scores_m[self_rows, self_cols] = 0.0

        total_s = scores_m.sum(axis=1, keepdims=True)
        valid   = total_s[:, 0] > 0
        probs_m = np.zeros_like(scores_m)
        probs_m[valid] = scores_m[valid] / total_s[valid]

        trips_m  = probs_m * sample_boardings[:, np.newaxis]
        flat_trips = trips_m.ravel()
        flat_bins  = dist_bins_m.ravel()
        pred_tld   = np.bincount(flat_bins, weights=flat_trips, minlength=BIN_MAX)[:BIN_MAX]

        if pred_tld.sum() > 0:
            pred_tld /= pred_tld.sum()

        rmse = float(np.sqrt(np.mean((obs_tld - pred_tld) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_beta = beta

    BETA_CALIBRATED = best_beta
    _beta_rmse = round(best_rmse, 4)
    log.info(f"Beta calibrado: {BETA_CALIBRATED} (RMSE: {best_rmse:.4f})")


# =============================================================================
# _compute_gravity_model() — Calcula probabilidades de destino con el modelo de gravedad
#
# El modelo de gravedad asume que la probabilidad de ir a una parada destino
# depende de dos cosas:
#   1. Qué tan "importante" es ese destino (cuánta gente lo usa en total)
#   2. Qué tan lejos está (más lejos = menos probable, con decaimiento beta)
#
# La fórmula es: gravedad(A→B) = actividad_total(B) / distancia(A,B)^beta
# La probabilidad se obtiene normalizando: P(B|A) = gravedad(A→B) / Σ gravedad(A→X)
#
# Es el mismo modelo que los economistas usan para predecir flujos de comercio
# entre países ("modelo de gravedad del comercio"), aplicado aquí a transporte.
# A diferencia del modelo empírico, NO depende del periodo de tiempo —
# usa el peso total de actividad de cada parada sumando todos los periodos.
# =============================================================================
def _compute_gravity_model(beta: float = 1.5) -> None:
    global _gravity_probs

    # Usamos todas las paradas con viajes registrados como candidatas de destino
    stop_ids = [s["stop_id"] for s in _stops_with_trips]
    n = len(stop_ids)
    if n == 0:
        log.warning("No hay paradas con viajes — el modelo de gravedad no puede calcularse.")
        return

    log.info(f"Calculando modelo de gravedad para {n:,} paradas (beta={beta}) ...")

    # Convertimos coordenadas a arrays numpy para aprovechar operaciones vectorizadas.
    # En vez de un bucle Python que procesa una parada por vez, numpy procesa
    # todo el array de una sola vez en código C compilado — mucho más rápido.
    lats = np.array([_stops_lookup[sid]["lat"] for sid in stop_ids], dtype=np.float64)
    lons = np.array([_stops_lookup[sid]["lon"] for sid in stop_ids], dtype=np.float64)

    # Calculamos el peso de cada parada: suma de actividad total a través de TODOS
    # los periodos de tiempo. Una parada "importante" es la que tiene muchos viajes.
    stop_weight: dict = {}
    for (sid, _tp), v in _activity.items():
        stop_weight[sid] = stop_weight.get(sid, 0) + v["total"]
    weights = np.array([float(stop_weight.get(sid, 0)) for sid in stop_ids], dtype=np.float64)

    # Pre-calculamos radianes una sola vez (la fórmula Haversine los necesita)
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    R_km = 6371.0  # Radio medio de la Tierra en kilómetros

    # Para cada parada origen, calculamos su distribución de probabilidades
    for i, origin_id in enumerate(stop_ids):
        # ── Distancia Haversine vectorizada ───────────────────────────────────
        # Haversine es la fórmula estándar para calcular distancias sobre una esfera
        # dadas dos pares de latitud/longitud. Aquí la calculamos desde el origen i
        # hasta TODAS las demás paradas al mismo tiempo usando numpy.
        dlat = lats_r - lats_r[i]
        dlon = lons_r - lons_r[i]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lats_r[i]) * np.cos(lats_r) * np.sin(dlon / 2) ** 2
        )
        dists = R_km * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

        # Distancia mínima de 0.05 km para evitar división por cero en paradas muy cercanas
        dists = np.maximum(dists, 0.05)

        # ── Puntuación de gravedad ────────────────────────────────────────────
        # gravedad(A→B) = peso(B) / distancia(A,B)^beta
        # Un beta mayor hace que la distancia "penalice" más a los destinos lejanos.
        scores = weights / np.power(dists, beta)
        scores[i] = 0.0  # El origen no puede ser su propio destino

        total_score = scores.sum()
        if total_score == 0.0:
            continue  # Parada sin actividad circundante — sin predicción posible

        # ── Normalizar a probabilidades ───────────────────────────────────────
        # Dividimos por la suma total para que todos los valores sumen 1.0
        probs = scores / total_score

        # ── Seleccionar los 50 destinos más probables ─────────────────────────
        # argpartition es más rápido que argsort completo cuando solo necesitamos
        # los k elementos más grandes de un array grande.
        if n >= 50:
            top_idx = np.argpartition(probs, -50)[-50:]
        else:
            top_idx = np.arange(n)
        top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]  # Ordenar de mayor a menor

        _gravity_probs[origin_id] = [
            {
                "stop_id":    stop_ids[j],
                "stop_name":  _stops_lookup[stop_ids[j]]["name"],
                "lat":        float(lats[j]),
                "lon":        float(lons[j]),
                "probability": float(probs[j]),
            }
            for j in top_idx
            if probs[j] > 0 and stop_ids[j] != origin_id  # Excluir auto-bucles explícitamente
        ]

    log.info(f"  Modelo de gravedad: {len(_gravity_probs):,} orígenes indexados.")


# =============================================================================
# lifespan — Controla qué pasa cuando el servidor arranca y cuando se cierra
#
# FastAPI necesita este patrón especial (asynccontextmanager) para ejecutar
# código al inicio. El "yield" separa lo que pasa al arrancar (antes)
# de lo que pasaría al cerrar (después, si hubiera algo que limpiar).
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    _download_precomputed()
    if not _load_from_precomputed():
        _load_from_csvs()
    # Calibrar beta siempre con los datos cargados, luego recalcular el modelo
    # con el exponente óptimo (aunque el pkl ya traía un modelo con beta=1.5).
    _calibrate_beta()
    _compute_gravity_model(BETA_CALIBRATED)
    yield


# --- Crear la aplicación FastAPI ---------------------------------------------
# Aquí nace el servidor. "lifespan=lifespan" le dice que use nuestra función
# de arriba para controlar el ciclo de vida.
app = FastAPI(title="SEQ Transit Destination Predictor", lifespan=lifespan)

# --- CORS: permitir peticiones desde el navegador ----------------------------
# CORS (Cross-Origin Resource Sharing) es una restricción de seguridad del navegador
# que bloquea peticiones entre dominios diferentes. Como el HTML y el servidor
# están en el mismo origen (localhost:8000), no es estrictamente necesario,
# pero lo activamos para facilitar pruebas o futuros usos externos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # "*" significa: aceptar peticiones de cualquier origen
    allow_methods=["GET"], # Solo permitimos el método GET (consultar datos, no modificar)
    allow_headers=["*"],   # Aceptar cualquier cabecera HTTP
)


# =============================================================================
# Endpoint: GET /stops
#
# Devuelve la lista de todas las paradas que tienen viajes registrados.
# El mapa del navegador llama a este endpoint al cargar para saber
# dónde colocar los marcadores azules.
#
# Devuelve: lista de objetos con stop_id, stop_name, lat, lon
# =============================================================================
@app.get("/stops")
def get_stops():
    # Calculamos la actividad total por parada sumando todos los periodos de tiempo.
    # Este valor se usa en el frontend para asignar colores y tamaños a los marcadores.
    stop_total: dict = {}
    for (sid, _tp), v in _activity.items():
        stop_total[sid] = stop_total.get(sid, 0) + v["total"]
    return [
        {**stop, "total_activity": stop_total.get(stop["stop_id"], 0)}
        for stop in _stops_with_trips
    ]


# =============================================================================
# Endpoint: GET /info
#
# Devuelve metadatos generales del dataset y el valor de beta calibrado.
# El frontend usa esto para mostrar "β = X.X (auto-calibrated)" en la tarjeta
# del modelo de gravedad.
#
# Devuelve: {beta_calibrated, total_months, total_stops, total_trips}
# =============================================================================
@app.get("/info")
def get_info():
    # Número de meses distintos en el dataset (contando las series temporales)
    all_months: set = set()
    for ts in _timeseries.values():
        for entry in ts:
            all_months.add(entry["month"])
    total_months = len(all_months)

    # Total de viajes registrados en todos los periodos y paradas
    total_trips = sum(v["total"] for v in _activity.values())

    return {
        "beta_calibrated": round(BETA_CALIBRATED, 2),
        "total_months":    total_months,
        "total_stops":     len(_stops_lookup),
        "total_trips":     int(total_trips),
    }


# =============================================================================
# Endpoint: GET /time_periods
#
# Devuelve los periodos de tiempo disponibles en los datos.
# El navegador usa esto para llenar el desplegable de selección de horario.
#
# Ejemplo de respuesta: ["Weekday (12:00am-8:29:59am)", "Weekend", ...]
# =============================================================================
@app.get("/time_periods")
def get_time_periods():
    return _time_periods  # Lista preparada al inicio, la devolvemos directamente


# =============================================================================
# Endpoint: GET /predict?origin_stop_id=X&time_period=Y
#
# Dados un ID de parada de origen y un periodo de tiempo, devuelve los
# 50 destinos más probables con su probabilidad.
# El mapa usa esta información para dibujar el mapa de calor (heatmap).
#
# Parámetros:
#   origin_stop_id — ID de la parada de origen (ej: "1234")
#   time_period    — Periodo de tiempo (ej: "Weekday (8:30am-2:59:59pm)")
#
# Devuelve: lista de hasta 50 objetos con stop_id, stop_name, lat, lon, probability
# =============================================================================
@app.get("/predict")
def predict(origin_stop_id: str, time_period: str):
    # Buscamos en el diccionario precalculado. Si no existe la combinación, devolvemos lista vacía.
    entries = _od_probs.get((origin_stop_id, time_period), [])
    return [
        {
            "stop_id": sid,
            "stop_name": _stops_lookup[sid]["name"],
            "lat": _stops_lookup[sid]["lat"],
            "lon": _stops_lookup[sid]["lon"],
            "probability": prob,  # Número entre 0 y 1 (ej: 0.27 significa 27%)
        }
        for sid, prob in entries
        if sid in _stops_lookup and sid != origin_stop_id  # Sin coordenadas ni auto-bucles
    ]


# =============================================================================
# Endpoint: GET /activity?origin_stop_id=X&time_period=Y
#
# Devuelve las estadísticas de actividad de una parada: cuántas personas
# suben (boardings), cuántas bajan (alightings) y el total.
# El panel lateral del mapa muestra estas cifras con barras de progreso.
#
# Parámetros:
#   origin_stop_id — ID de la parada
#   time_period    — Periodo de tiempo
#
# Devuelve: objeto con stop_id, stop_name, time_period, boardings, alightings, total
# =============================================================================
@app.get("/activity")
def get_activity(origin_stop_id: str, time_period: str):
    # Buscamos las estadísticas precalculadas para esta parada y periodo
    data = _activity.get((origin_stop_id, time_period))

    # Si no hay datos o la parada no existe en el mapa, devolvemos un objeto vacío
    if not data or origin_stop_id not in _stops_lookup:
        return {}

    # Combinamos la información de la parada con las estadísticas de actividad.
    # "**data" desempaqueta el diccionario {boardings, alightings, total} dentro del resultado.
    return {
        "stop_id": origin_stop_id,
        "stop_name": _stops_lookup[origin_stop_id]["name"],
        "time_period": time_period,
        **data,  # Equivale a escribir "boardings": data["boardings"], "alightings": data["alightings"], etc.
    }


# =============================================================================
# Endpoint: GET /predict/gravity?origin_stop_id=X
#
# Devuelve los 50 destinos más probables para una parada de origen según el
# modelo de gravedad. A diferencia de /predict, NO depende del periodo de tiempo:
# usa la actividad total acumulada de todos los periodos como peso de cada destino.
#
# Parámetros:
#   origin_stop_id — ID de la parada de origen
#
# Devuelve: lista de hasta 50 objetos con stop_id, stop_name, lat, lon, probability
# =============================================================================
# =============================================================================
# Endpoint: GET /trend?stop_id=X
#
# Devuelve la serie temporal completa (un registro por mes del dataset) y el
# resultado de la regresión lineal para una parada dada.
# El modo "Tendencia" del frontend usa esto para dibujar el gráfico SVG
# y mostrar la proyección de los próximos 3 meses.
#
# Parámetros:
#   stop_id — ID de la parada
#
# Devuelve: {stop_id, stop_name, timeseries: [...], trend: {...}}
# =============================================================================
@app.get("/trend")
def get_trend(stop_id: str):
    ts = _timeseries.get(stop_id)
    if not ts:
        return {}
    return {
        "stop_id":    stop_id,
        "stop_name":  _stops_lookup.get(stop_id, {}).get("name", ""),
        "timeseries": ts,
        "trend":      _trend.get(stop_id),  # Puede ser None si hay pocos meses de datos
    }


@app.get("/predict/gravity")
def predict_gravity(origin_stop_id: str):
    # El modelo de gravedad devuelve la misma lista sin importar el periodo de tiempo.
    # Si la parada no tiene datos de gravedad (ej: no tiene actividad registrada),
    # devolvemos lista vacía.
    return _gravity_probs.get(origin_stop_id, [])


# --- Archivos estáticos y página principal -----------------------------------

# Esto le dice al servidor que la carpeta "static" contiene archivos que
# el navegador puede pedir directamente (como imágenes o CSS adicional).
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# Endpoint: GET /
#
# Cuando el usuario abre http://localhost:8000 en el navegador,
# este endpoint envía el archivo HTML de la aplicación.
# Es la puerta de entrada a toda la interfaz visual.
# =============================================================================
@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))  # Enviamos el archivo HTML al navegador
