"""
FastAPI inference service — the deployment target of the MLOps pipeline.
Serves fault predictions and exposes the paper's operational metrics:
service availability (Eq. 14), throughput (Eq. 15) and latency (Eq. 16).
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops.registry import load_trained_model, rollback_model
from resilience.backoff import with_exponential_backoff

MODEL = None
START_TIME = time.time()
STATS = {"requests": 0, "failed": 0, "faults": 0, "total_latency": 0.0}

def load_model():
    """Load the production model; if it is corrupted, roll back (SRE)"""
    global MODEL
    try:
        MODEL, _ = with_exponential_backoff(load_trained_model,
                                            initial_delay=0.5, max_retries=3)
    except RuntimeError:
        try:
            MODEL, _ = rollback_model()
        except FileNotFoundError:
            print("⚠️  No model available — run main.py first")
            return
    print("✅ Model loaded")

@asynccontextmanager
async def lifespan(app):
    load_model()
    yield

app = FastAPI(
    title="Predictive Maintenance API",
    description="Fault prediction for hydraulic systems using TFT + PSO",
    lifespan=lifespan
)

class SensorWindow(BaseModel):
    """A 60-step window of 68 normalized sensor features."""
    window: List[List[float]] = Field(
        ...,
        description="60 timesteps x 68 features",
        json_schema_extra={"example": [[0.1] * 68] * 60}
    )

@app.get("/health")
def health():
    return {
        "status":       "healthy" if MODEL is not None else "down",
        "model_loaded": MODEL is not None,
        "uptime_sec":   round(time.time() - START_TIME, 2),
    }

@app.post("/predict")
def predict(data: SensorWindow):
    if MODEL is None:
        STATS["failed"] += 1
        raise HTTPException(status_code=503, detail="Model not loaded")

    window = np.array(data.window, dtype=np.float32)
    if window.shape != (60, 68):
        raise HTTPException(status_code=422,
                            detail=f"Expected shape (60, 68), got {window.shape}")

    t0 = time.time()
    with torch.no_grad():
        logits = MODEL(torch.tensor(window).unsqueeze(0))
        probs  = torch.softmax(logits, dim=1)[0].numpy()
    latency_ms = (time.time() - t0) * 1000

    pred = int(np.argmax(probs))
    STATS["requests"]      += 1
    STATS["faults"]        += int(pred == 0)
    STATS["total_latency"] += latency_ms

    return {
        "prediction": pred,                       # 0 = fault, 1 = normal
        "label":      "NORMAL" if pred == 1 else "FAULT",
        "fault_prob": round(float(probs[0]), 4),
        "latency_ms": round(latency_ms, 2),
        "timestamp":  datetime.utcnow().isoformat(),
    }

@app.get("/stats")
def stats():
    """Operational metrics from the paper (Eq. 14-16)"""
    served  = STATS["requests"]
    total   = served + STATS["failed"]
    elapsed = time.time() - START_TIME
    return {
        "requests_served":      served,
        "fault_predictions":    STATS["faults"],
        "availability_pct":     round(100 * served / total, 2) if total else 100.0,
        "throughput_req_per_s": round(served / elapsed, 4),
        "avg_latency_ms":       round(STATS["total_latency"] / served, 2) if served else 0.0,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
