# SEQ Transit Destination Predictor

An interactive web app that shows — for any transit stop in South East Queensland — where travellers are likely to go, based on 6 months of Translink origin-destination trip data (October 2025 – March 2026).

## Setup

### 1. Prerequisites
Python 3.10+ is required.

### 2. Install dependencies
From the project root (`/Users/juanmunoz/Documents/Project RE`):

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

The server starts at **http://127.0.0.1:8000**.

On first startup, the backend will load and process all six monthly CSV files (~5 million rows). This takes **10–20 seconds**. A loading spinner is shown in the browser while it completes.

---

## How to use

1. Open **http://127.0.0.1:8000** in a browser.
2. The map loads centred on SEQ (Brisbane, Gold Coast, Sunshine Coast, Ipswich).
3. **Click any blue stop marker** on the map.
4. Select a **time period** from the dropdown (e.g. *Weekday (8:30am-2:59:59pm)*).
5. A **heatmap** appears showing where travellers from that stop are likely to go. Brighter = higher probability.
6. The **sidebar** lists the top 10 destination stops with probability percentages.

---

## Project structure

```
Project RE/
├── app/
│   ├── __init__.py
│   ├── main.py          ← FastAPI backend
│   └── static/
│       └── index.html   ← Single-file frontend (Leaflet + Leaflet.heat)
├── requirements.txt
├── README.md
├── stops.txt            ← GTFS stops file
├── 202510 (Oct) TL Org-Dest Trips.csv
├── 202511(Nov) TL Org-Dest Trips.csv
├── 202512(Dec) TL Org-Dest Trips.csv
├── 202601(Jan) TL Org-Dest Trips.csv
├── 202602(Feb) TL Org-Dest Trips.csv
└── 202603(Mar) TL Org-Dest Trips.csv
```

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /stops` | All stops that have at least one outgoing trip |
| `GET /time_periods` | List of available time period strings |
| `GET /predict?origin_stop_id=X&time_period=Y` | Top-50 destination predictions for a given origin and time |

## Notes

- Probabilities are computed as `P(dest | origin, time) = trips(origin→dest, time) / total_trips(origin, time)`, aggregated across all 6 months.
- Stops present in the OD data but absent from `stops.txt` are silently skipped.
- The heatmap shows up to 50 destinations; the sidebar shows the top 10.
