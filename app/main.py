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

import logging                          # Para imprimir mensajes informativos mientras el servidor trabaja
from contextlib import asynccontextmanager  # Nos permite ejecutar código al iniciar y al cerrar el servidor
from glob import glob                   # Para buscar archivos usando patrones (como *.csv)
from pathlib import Path                # Para construir rutas de archivos de forma segura en cualquier sistema operativo

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

# --- Variables globales ------------------------------------------------------
# Estas variables se llenan al arrancar el servidor y se quedan en memoria
# mientras el servidor está corriendo. Así, cada consulta del navegador
# recibe la respuesta en milisegundos en vez de tener que procesar datos de nuevo.

_stops_lookup: dict = {}    # Diccionario: stop_id → {name, lat, lon}. Nos dice el nombre y coordenadas de cada parada.
_od_probs: dict = {}         # Diccionario: (stop_origen, periodo) → lista de (stop_destino, probabilidad). El corazón de las predicciones.
_activity: dict = {}         # Diccionario: (stop_id, periodo) → {subidas, bajadas, total}. Estadísticas de actividad.
_time_periods: list = []     # Lista con los nombres de los periodos de tiempo disponibles (ej: "Weekday (8:30am-2:59:59pm)")
_stops_with_trips: list = [] # Lista de paradas que tienen al menos un viaje registrado — las que aparecen en el mapa


# =============================================================================
# _load_data() — Carga y procesa todos los datos al iniciar el servidor
#
# Esta función se ejecuta UNA SOLA VEZ cuando el servidor arranca.
# Lee miles de registros de viajes y los convierte en estructuras rápidas
# de consultar. Puede tardar 30 segundos pero vale la pena: después,
# cada pregunta del usuario se responde en menos de un milisegundo.
#
# No devuelve nada — guarda todo en las variables globales de arriba.
# =============================================================================
def _load_data() -> None:
    # "global" le dice a Python que queremos modificar las variables de arriba,
    # no crear variables locales nuevas con el mismo nombre.
    global _stops_lookup, _od_probs, _activity, _time_periods, _stops_with_trips

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
            usecols=["time", "origin_stop", "destination_stop", "quantity"],  # Solo las 4 columnas útiles
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

    # ── 4. Agregar los 6 meses en una sola tabla ──────────────────────────────
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
# lifespan — Controla qué pasa cuando el servidor arranca y cuando se cierra
#
# FastAPI necesita este patrón especial (asynccontextmanager) para ejecutar
# código al inicio. El "yield" separa lo que pasa al arrancar (antes)
# de lo que pasaría al cerrar (después, si hubiera algo que limpiar).
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_data()  # Cargamos todos los datos antes de aceptar cualquier petición
    yield         # El servidor corre normalmente entre el arranque y el cierre


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
    return _stops_with_trips  # Simplemente devolvemos la lista ya preparada al inicio


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
        if sid in _stops_lookup  # Verificamos que la parada tenga coordenadas antes de incluirla
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
