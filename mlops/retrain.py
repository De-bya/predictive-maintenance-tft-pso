"""
Automated retraining pipeline.
Triggered when CUSUM drift detection fires.
Trains a new model and keeps it only if it beats the current production
model; otherwise the previous model is restored (SRE rollback).
"""
import os
import sys
import warnings
from datetime import datetime
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def run_automated_retraining(drift_score: float, drift_points: list,
                             trigger_source: str = "cusum"):
    from mlops.pipeline import train_model, evaluate_model, default_params
    from mlops.registry import (build_tft, load_trained_model, save_checkpoint,
                                backup_checkpoint, rollback_model, CHECKPOINT)
    from resilience.backoff import with_exponential_backoff

    config    = load_config()
    timestamp = datetime.utcnow().isoformat()

    print("\n" + "="*55)
    print("  🔄 AUTOMATED RETRAINING TRIGGERED")
    print(f"  Trigger:     {trigger_source}")
    print(f"  Drift score: {drift_score:.4f}")
    print(f"  Drift points:{len(drift_points)}")
    print("="*55)

    X_train = np.load("data/processed/X_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test  = np.load("data/processed/y_test.npy")

    # ── Current production model = baseline to beat ───────────
    if os.path.exists(CHECKPOINT):
        current_model, params = load_trained_model()
        current_accuracy = _accuracy(current_model, X_test, y_test)
        backup_checkpoint()
    else:
        params, current_accuracy = default_params(config), 0.0
    print(f"📊 Current model accuracy: {current_accuracy:.2f}%")

    mlflow.set_experiment(config['mlflow']['experiment_name'])
    with mlflow.start_run(run_name=f"retrain_{trigger_source}_{timestamp[:10]}"):
        mlflow.log_params({
            "trigger_source": trigger_source,
            "drift_score":    round(drift_score, 4),
            "drift_points":   len(drift_points),
            "prev_accuracy":  round(current_accuracy, 4),
        })

        # Retrain with the same (PSO-tuned) hyperparameters on the latest data
        model = build_tft(params, input_dim=X_train.shape[2])
        new_model = with_exponential_backoff(
            lambda: train_model(model, X_train, y_train, config, params)
        )
        metrics, _ = evaluate_model(new_model, X_test, y_test, config)
        new_accuracy = metrics['Accuracy (%)']

        # Deploy the new model, then verify it — roll back if it is worse
        save_checkpoint(new_model, params)
        promoted = new_accuracy > current_accuracy
        if not promoted and current_accuracy > 0:
            rollback_model()
        status = "PROMOTED" if promoted else "ROLLED_BACK"

        mlflow.log_metrics({
            'new_accuracy':   new_accuracy,
            'prev_accuracy':  current_accuracy,
            'accuracy_delta': new_accuracy - current_accuracy,
        })
        mlflow.log_param("promotion_status", status)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.pytorch.log_model(
                new_model, name="retrained_model",
                serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE
            )

    print("\n" + "="*55)
    print(f"  RETRAINING RESULT: {status}")
    print(f"  Previous accuracy: {current_accuracy:.2f}%")
    print(f"  New accuracy:      {new_accuracy:.2f}%")
    print("="*55)

    return {
        "status":        status,
        "timestamp":     timestamp,
        "trigger":       trigger_source,
        "drift_score":   drift_score,
        "prev_accuracy": current_accuracy,
        "new_accuracy":  new_accuracy,
        "promoted":      promoted,
    }

def _accuracy(model, X_test, y_test):
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()
    # Same 2-decimal precision as compute_all_metrics, so the comparison is fair
    return round(float((preds == y_test).mean() * 100), 2)

if __name__ == "__main__":
    # Manual trigger for testing
    print(run_automated_retraining(drift_score=6.5, drift_points=list(range(10)),
                                   trigger_source="manual_test"))
