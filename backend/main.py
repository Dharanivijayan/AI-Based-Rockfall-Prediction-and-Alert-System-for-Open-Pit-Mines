"""
Backend API for the AI Rockfall Prediction & Alert System.

Endpoints:
  GET  /api/zones                 -> list of monitored zones
  GET  /api/risk/latest            -> latest risk score + tier per zone
  GET  /api/risk/history?zone=X     -> recent score history for charting
  GET  /api/alerts                 -> alert log
  POST /api/simulate/spike         -> manually inject a risk spike (for live demo)

Run with:
  uvicorn main:app --reload --port 8000
"""

import random
import sys
import os
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
from risk_model import MLRiskScorer, tier_for_score  # noqa: E402

app = FastAPI(title="Rockfall Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer = MLRiskScorer()

ZONES = ["Zone-A", "Zone-B", "Zone-C"]

state: Dict[str, dict] = {
    z: {
        "rainfall_mm_hr": round(random.uniform(0, 2), 2),
        "vibration_mm_s": round(random.uniform(0, 1.5), 2),
        "displacement_rate_mm_hr": round(random.uniform(0, 0.3), 2),
        "pore_pressure_kpa": round(random.uniform(45, 55), 2),
        "spike_ticks": 0,
    }
    for z in ZONES
}

history: Dict[str, List[dict]] = {z: [] for z in ZONES}
alerts: List[dict] = []
HISTORY_LIMIT = 200


class SpikeRequest(BaseModel):
    zone: str
    duration_ticks: int = 15


def step_zone(zone: str):
    s = state[zone]

    if s["spike_ticks"] > 0:
        s["rainfall_mm_hr"] += random.uniform(0.3, 1.0)
        s["vibration_mm_s"] += random.uniform(0.2, 0.8)
        s["displacement_rate_mm_hr"] += random.uniform(0.05, 0.2)
        s["pore_pressure_kpa"] += random.uniform(1, 3)
        s["spike_ticks"] -= 1
    else:
        s["rainfall_mm_hr"] = max(0, s["rainfall_mm_hr"] * 0.9 + random.uniform(-0.2, 0.2))
        s["vibration_mm_s"] = max(0, s["vibration_mm_s"] * 0.9 + random.uniform(-0.1, 0.1))
        s["displacement_rate_mm_hr"] = max(0, s["displacement_rate_mm_hr"] * 0.95)
        s["pore_pressure_kpa"] = 50 + (s["pore_pressure_kpa"] - 50) * 0.9

    score = scorer.score(s)
    tier = tier_for_score(score)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "zone": zone,
        "score": round(score, 1),
        "tier": tier,
    }
    history[zone].append(record)
    if len(history[zone]) > HISTORY_LIMIT:
        history[zone].pop(0)

    if tier in ("WARNING", "EVACUATE"):
        if not alerts or alerts[-1]["zone"] != zone or alerts[-1]["tier"] != tier:
            alerts.append(record)

    return record


@app.get("/api/zones")
def get_zones():
    return {"zones": ZONES}


@app.get("/api/risk/latest")
def get_latest():
    return {z: step_zone(z) for z in ZONES}


@app.get("/api/risk/history")
def get_history(zone: str = "Zone-A"):
    return {"zone": zone, "history": history.get(zone, [])}


@app.get("/api/alerts")
def get_alerts():
    return {"alerts": alerts[-50:]}


@app.post("/api/simulate/spike")
def simulate_spike(req: SpikeRequest):
    if req.zone not in state:
        return {"error": "unknown zone"}
    state[req.zone]["spike_ticks"] = req.duration_ticks
    return {"status": "spike injected", "zone": req.zone, "ticks": req.duration_ticks}


@app.get("/")
def root():
    return {"status": "ok", "service": "Rockfall Prediction API"}
