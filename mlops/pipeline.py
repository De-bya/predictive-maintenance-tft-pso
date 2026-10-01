import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import yaml
import warnings
from mlops.drift import cusum_detect, needs_retraining
from mlops.registry import build_tft, save_checkpoint
from evaluation.metrics import compute_all_metrics, compute_throughput_latency
from resilience.backoff import with_exponential_backoff

METRICS_PATH = "data/processed/metrics.json"

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_processed():
    X_train = np.load("data/processed/X_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test  = np.load("data/processed/y_test.npy")
    print(f"✅ Data loaded — Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def default_params(config):
    return {
        'hidden_size':     config['model']['hidden_size'],
        'attention_heads': config['model']['attention_heads'],
        'num_layers':      config['model']['num_layers'],
        'dropout':         config['model']['dropout'],
        'learning_rate':   config['model']['learning_rate'],
    }

def train_model(model, X_train, y_train, config, params):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = model.to(device)
    epochs     = config['model']['max_epochs']
    batch_size = config['model']['batch_size']

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    print(f"\n🚀 Training TFT — {epochs} epochs on {device}")
    best_loss  = np.inf
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        # Keep the best epoch in memory (does not touch the production checkpoint)
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.4f}")

    model.load_state_dict(best_state)
    print(f"✅ Training complete — Best loss: {best_loss:.4f}")
    return model

def measure_scalability(model, X_test, n_requests=100):
    """Equations 15-16: throughput on the full test set (batch),
    latency as average time per single-window request"""
    device = next(model.parameters()).device
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        t0 = time.time()
        model(X_t)
        time_taken = time.time() - t0

        n = min(n_requests, len(X_t))
        t0 = time.time()
        for i in range(n):
            model(X_t[i:i + 1])
        response_time = time.time() - t0

    return compute_throughput_latency(len(X_t), time_taken, response_time, n)

def evaluate_model(model, X_test, y_test, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

    metrics = compute_all_metrics(y_test, preds, y_prob=probs)

    # Equation 6: CUSUM drift between actual labels and predicted probabilities
    drift_pts, drift_score = cusum_detect(
        y_test.astype(float), probs,
        delta=config['drift']['delta'],
        threshold=config['drift']['threshold']
    )
    metrics['drift_points']     = len(drift_pts)
    metrics['drift_score']      = drift_score
    metrics['needs_retraining'] = needs_retraining(drift_pts)

    throughput, latency = measure_scalability(model, X_test)
    metrics['Throughput (samples/s)'] = round(throughput, 2)
    metrics['Latency (ms/request)']   = round(latency * 1000, 2)

    return metrics, drift_pts

def run_pipeline(best_params=None):
    config = load_config()
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_processed()

        params = best_params or default_params(config)
        mlflow.log_params(params)

        model = build_tft(params, input_dim=X_train.shape[2])

        # Equation 7: retry training with exponential backoff on failure
        res = config['resilience']
        trained_model = with_exponential_backoff(
            lambda: train_model(model, X_train, y_train, config, params),
            initial_delay=res['initial_delay'],
            max_retries=res['max_retries']
        )
        metrics, drift_pts = evaluate_model(trained_model, X_test, y_test, config)

        mlflow.log_metrics({
            'accuracy':   metrics['Accuracy (%)'],
            'precision':  metrics['Precision (%)'],
            'recall':     metrics['Recall (%)'],
            'f1_score':   metrics['F1-Score (%)'],
            'rmse':       float(metrics['RMSE']),
            'mae':        float(metrics['MAE']),
            'throughput': metrics['Throughput (samples/s)'],
            'latency_ms': metrics['Latency (ms/request)'],
            'drift_score': metrics['drift_score'],
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.pytorch.log_model(
                trained_model,
                name="tft_model",
                serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE
            )

        save_checkpoint(trained_model, params)
        with open(METRICS_PATH, "w") as f:
            json.dump({k: v for k, v in metrics.items()
                       if isinstance(v, (int, float, str))}, f, indent=2)
        print("\n✅ Run logged to MLflow")

    # ── Drift → automated retraining (after the new model is in place) ──
    if metrics['needs_retraining'] and config['drift']['auto_retrain']:
        print(f"\n🚨 Drift threshold exceeded — triggering automated retraining")
        from mlops.retrain import run_automated_retraining
        metrics['retrain_result'] = run_automated_retraining(
            drift_score=metrics['drift_score'],
            drift_points=drift_pts,
            trigger_source="cusum_auto"
        )

    return trained_model, metrics
