# DRIZZL / MonsoonAI - Integrated System

Regime-aware AI post-processing of monsoon rainfall forecasts. Combines a FastAPI backend (ML models) with a static frontend dashboard.

## Architecture

```
┌─────────────────┐     REST API      ┌─────────────────┐
│  Frontend       │ ◄────────────────► │  Backend API    │
│  (HTML/JS/Leaflet)│   localhost:8000  │  (FastAPI + ML) │
│  localhost:3000 │                    │  localhost:8000 │
└─────────────────┘                    └─────────────────┘
```

## Prerequisites

- Python 3.10+ with `venv`
- Linux/macOS/WSL
- Internet (for Leaflet/Chart.js CDN)

## Quick Start

### 1. Backend API (Terminal 1)

```bash
cd ~/DRIZZL-main
source venv/bin/activate
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Verify:** http://localhost:8000/health → `{"status":"healthy","models_loaded":true}`

### 2. Frontend (Terminal 2)

```bash
cd ~/DRIZZL-Frontend
python -m http.server 3000
```

**Open:** http://localhost:3000

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/districts` | List 15 Indian districts with coordinates |
| POST | `/predict/single` | Single forecast with custom weather input |
| POST | `/predict/district` | Forecast for a district (auto-fetches weather) |
| GET | `/predict/all-districts` | All districts forecast |
| GET | `/verification/metrics` | Model metrics (RMSE, ETS, CSI, POD, FAR, FSS) |
| GET | `/districts/geojson` | GeoJSON for mapping |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{"temperature":30,"humidity":80,"wind":10,"pressure":1000,"nwp_rainfall":50}'
```

---

## Frontend Features

| Section | Description |
|---------|-------------|
| **Dashboard** | KPI cards, rainfall comparison chart, regime distribution |
| **District Forecast** | Dropdown → live table + map for 15 districts |
| **Weather Regime** | Regime classifier with confidence percentages |
| **Heavy Rainfall** | Threshold probabilities + interactive Leaflet risk map |
| **Verification** | Model skill metrics vs raw NWP |

---

## Background Run (One-liner)

```bash
# Terminal 1
cd ~/DRIZZL-main && source venv/bin/activate && uvicorn api_server:app --host 0.0.0.0 --port 8000 &

# Terminal 2
cd ~/DRIZZL-Frontend && python -m http.server 3000 &
```

---

## Project Structure

```
DRIZZL-main/
├── api_server.py          # FastAPI backend
├── drizzlapp.py           # Original Streamlit app
├── train_models.py        # Model training script
├── district_forecast.py   # District-level forecasting
├── real_data.py           # Weather data fetchers (IMD/ERA5/NWP)
├── verification.py        # Verification metrics
├── historical_replay.py   # Time-series validation
├── risk.py                # Risk scoring
├── visualizations.py      # Plotly charts
├── models/                # Trained .pkl models
├── data/                  # Training/validation CSV
└── venv/                  # Python virtual environment

DRIZZL-Frontend/
├── index.html             # Dashboard HTML
├── script.js              # API integration + Leaflet map
└── style.css              # Styling
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: fastapi` | `cd ~/DRIZZL-main && source venv/bin/activate && pip install fastapi uvicorn` |
| Map not showing | Refresh page; map initializes on first visit to Heavy Rainfall tab |
| API connection failed | Ensure backend runs on port 8000; check CORS (allow_origins="*") |
| District not found | Use lowercase no-space keys: `mumbai`, `newdelhi`, `thiruvananthapuram` |

---

## Retrain Models

```bash
cd ~/DRIZZL-main
source venv/bin/activate
python train_models.py
```

---

## License

Prototype for Smart India Hackathon / Ministry of Earth Sciences.
