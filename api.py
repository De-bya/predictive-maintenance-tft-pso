"""
FastAPI inference endpoint for the TFT predictive maintenance model.
Serves real-time fault predictions from sensor data.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import torch
import time
import os
import sys
from typing import List
from datetime import datetime

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

sys.path.append(os.path.dirname(__file__))
from model.tft import TemporalFusionTransformer
from mlops.drift import cusum_detect

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="Real-time fault prediction for hydraulic systems using TFT + PSO",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



# ── Global model state ────────────────────────────────────────
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_TIME = time.time()
REQUEST_COUNT = 0
PREDICTION_LOG = []

def load_model():
    global MODEL
    checkpoint = "data/processed/best_model.pt"
    if not os.path.exists(checkpoint):
        raise RuntimeError(f"No model checkpoint at {checkpoint}. Run main.py first.")
    MODEL = TemporalFusionTransformer(
        input_dim=68, hidden_dim=96,
        num_heads=8, dropout=0.068, num_classes=2
    ).to(DEVICE)
    MODEL.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    MODEL.eval()
    print(f"✅ Model loaded on {DEVICE}")

@app.on_event("startup")
def startup_event():
    load_model()

# ── Schemas ───────────────────────────────────────────────────
class SensorWindow(BaseModel):
    """
    A 60-step window of 68 sensor features.
    Shape: [60 timesteps x 68 features] flattened to a list of lists.
    """
    window: List[List[float]] = Field(
        ...,
        description="60 timesteps x 68 features",
        example=[[0.1] * 68] * 60
    )

class PredictionResponse(BaseModel):
    prediction:    int       # 0 = fault, 1 = normal
    label:         str       # "FAULT" or "NORMAL"
    confidence:    float     # probability of predicted class
    fault_prob:    float     # probability of fault (class 0)
    normal_prob:   float     # probability of normal (class 1)
    latency_ms:    float
    timestamp:     str

class BatchRequest(BaseModel):
    windows: List[List[List[float]]] = Field(
        ..., description="List of sensor windows, each [60 x 68]"
    )

class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    uptime_sec:    float
    requests_served: int
    device:        str

class StatsResponse(BaseModel):
    total_requests:   int
    fault_count:      int
    normal_count:     int
    avg_confidence:   float
    avg_latency_ms:   float

# ── Routes ────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Predictive Maintenance API",
        "model":   "TFT + PSO",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        uptime_sec=round(time.time() - START_TIME, 2),
        requests_served=REQUEST_COUNT,
        device=str(DEVICE)
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(data: SensorWindow):
    global REQUEST_COUNT, PREDICTION_LOG
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    window = np.array(data.window, dtype=np.float32)
    if window.shape != (60, 68):
        raise HTTPException(
            status_code=422,
            detail=f"Expected shape (60, 68), got {window.shape}"
        )

    t0 = time.time()
    with torch.no_grad():
        x      = torch.tensor(window).unsqueeze(0).to(DEVICE)  # (1, 60, 68)
        logits = MODEL(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    latency_ms = round((time.time() - t0) * 1000, 2)
    pred       = int(np.argmax(probs))
    label      = "NORMAL" if pred == 1 else "FAULT"
    confidence = round(float(probs[pred]), 4)

    REQUEST_COUNT += 1
    PREDICTION_LOG.append({
        "pred": pred, "confidence": confidence, "latency_ms": latency_ms
    })

    return PredictionResponse(
        prediction=pred,
        label=label,
        confidence=confidence,
        fault_prob=round(float(probs[0]), 4),
        normal_prob=round(float(probs[1]), 4),
        latency_ms=latency_ms,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/predict/batch", tags=["Inference"])
def predict_batch(data: BatchRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for i, window in enumerate(data.windows):
        w = np.array(window, dtype=np.float32)
        if w.shape != (60, 68):
            raise HTTPException(
                status_code=422,
                detail=f"Window {i} has shape {w.shape}, expected (60, 68)"
            )
        t0 = time.time()
        with torch.no_grad():
            x      = torch.tensor(w).unsqueeze(0).to(DEVICE)
            logits = MODEL(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred  = int(np.argmax(probs))
        results.append({
            "index":       i,
            "prediction":  pred,
            "label":       "NORMAL" if pred == 1 else "FAULT",
            "confidence":  round(float(probs[pred]), 4),
            "fault_prob":  round(float(probs[0]), 4),
            "normal_prob": round(float(probs[1]), 4),
            "latency_ms":  round((time.time() - t0) * 1000, 2),
        })

    fault_count = sum(1 for r in results if r["prediction"] == 0)
    return {
        "total":       len(results),
        "fault_count": fault_count,
        "normal_count": len(results) - fault_count,
        "results":     results
    }

@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
def stats():
    if not PREDICTION_LOG:
        return StatsResponse(
            total_requests=0, fault_count=0,
            normal_count=0, avg_confidence=0.0, avg_latency_ms=0.0
        )
    faults    = sum(1 for p in PREDICTION_LOG if p["pred"] == 0)
    avg_conf  = round(np.mean([p["confidence"] for p in PREDICTION_LOG]), 4)
    avg_lat   = round(np.mean([p["latency_ms"]  for p in PREDICTION_LOG]), 2)
    return StatsResponse(
        total_requests=len(PREDICTION_LOG),
        fault_count=faults,
        normal_count=len(PREDICTION_LOG) - faults,
        avg_confidence=avg_conf,
        avg_latency_ms=avg_lat
    )

@app.post("/reload", tags=["Admin"])
def reload_model():
    """Hot-reload the model checkpoint without restarting the server"""
    try:
        load_model()
        return {"status": "reloaded", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ui", tags=["UI"])
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)