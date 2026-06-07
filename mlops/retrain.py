"""
Automated retraining pipeline.
Triggered when CUSUM drift detection fires.
Trains a new model, evaluates it, and registers it in MLflow
only if it improves on the current best.
"""
import numpy as np
import torch
import mlflow
import mlflow.pytorch
import yaml
import os
import warnings
from datetime import datetime

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def run_automated_retraining(drift_score: float, drift_points: list,
                              trigger_source: str = "cusum"):
    """
    Full retraining pipeline triggered by drift detection.

    Args:
        drift_score:    CUSUM drift score that triggered retraining
        drift_points:   List of timesteps where drift was detected
        trigger_source: What triggered retraining ('cusum', 'manual', 'scheduled')

    Returns:
        dict with retraining results
    """
    from model.tft import TemporalFusionTransformer
    from mlops.pipeline import train_model, evaluate_model
    from mlops.drift import cusum_detect
    from resilience.backoff import with_exponential_backoff

    config    = load_config()
    timestamp = datetime.utcnow().isoformat()

    print("\n" + "="*55)
    print("  🔄 AUTOMATED RETRAINING TRIGGERED")
    print(f"  Trigger:     {trigger_source}")
    print(f"  Drift score: {drift_score:.4f}")
    print(f"  Drift points:{len(drift_points)}")
    print(f"  Timestamp:   {timestamp}")
    print("="*55)

    # ── Load data ─────────────────────────────────────────────
    X_train = np.load("data/processed/X_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test  = np.load("data/processed/y_test.npy")
    print(f"✅ Data loaded for retraining")

    # ── Get current best model accuracy for comparison ────────
    current_accuracy = _get_current_model_accuracy(
        X_test, y_test, config
    )
    print(f"📊 Current model accuracy: {current_accuracy:.2f}%")

    # ── Train new model ───────────────────────────────────────
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name=f"retrain_{trigger_source}_{timestamp[:10]}"):

        # Log retraining trigger metadata
        mlflow.log_params({
            "trigger_source":   trigger_source,
            "drift_score":      round(drift_score, 4),
            "drift_points":     len(drift_points),
            "retrain_timestamp": timestamp,
            "prev_accuracy":    round(current_accuracy, 4),
        })

        model = TemporalFusionTransformer(
            input_dim=X_train.shape[2],
            hidden_dim=config['model']['hidden_size'],
            num_heads=config['model']['attention_heads'],
            dropout=config['model']['dropout'],
            num_classes=2
        )

        print("\n🚀 Training new model...")

        def train_fn():
            return train_model(model, X_train, y_train, config)

        new_model = with_exponential_backoff(train_fn)
        metrics, preds, probs = evaluate_model(new_model, X_test, y_test)
        new_accuracy = metrics['Accuracy (%)']

        # ── Compare and decide whether to promote ─────────────
        improved = new_accuracy > current_accuracy
        status   = "PROMOTED" if improved else "REJECTED"

        mlflow.log_metrics({
            'new_accuracy':    new_accuracy,
            'prev_accuracy':   current_accuracy,
            'accuracy_delta':  new_accuracy - current_accuracy,
            'f1_score':        metrics['F1-Score (%)'],
            'recall':          metrics['Recall (%)'],
        })
        mlflow.log_param("promotion_status", status)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.pytorch.log_model(
                new_model,
                name=f"retrained_model_{timestamp[:10]}",
                serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE
            )

        print("\n" + "="*55)
        print(f"  RETRAINING RESULT: {status}")
        print(f"  Previous accuracy: {current_accuracy:.2f}%")
        print(f"  New accuracy:      {new_accuracy:.2f}%")
        print(f"  Delta:             {new_accuracy - current_accuracy:+.2f}%")
        print("="*55)

        if improved:
            # Save new model as the production checkpoint
            torch.save(new_model.state_dict(), "data/processed/best_model.pt")
            _save_retrain_log(timestamp, drift_score, current_accuracy,
                              new_accuracy, status)
            print("✅ New model promoted to production checkpoint")
        else:
            _save_retrain_log(timestamp, drift_score, current_accuracy,
                              new_accuracy, status)
            print("⏭️  Keeping existing model (no improvement)")

        return {
            "status":           status,
            "timestamp":        timestamp,
            "trigger":          trigger_source,
            "drift_score":      drift_score,
            "prev_accuracy":    current_accuracy,
            "new_accuracy":     new_accuracy,
            "delta":            new_accuracy - current_accuracy,
            "promoted":         improved,
        }


def _get_current_model_accuracy(X_test, y_test, config):
    """Evaluate the current checkpoint to get baseline accuracy"""
    from model.tft import TemporalFusionTransformer
    from sklearn.metrics import accuracy_score

    checkpoint = "data/processed/best_model.pt"
    if not os.path.exists(checkpoint):
        return 0.0

    try:
        model = TemporalFusionTransformer(
            input_dim=X_test.shape[2],
            hidden_dim=config['model']['hidden_size'],
            num_heads=config['model']['attention_heads'],
            dropout=config['model']['dropout'],
            num_classes=2
        )
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)
        )
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test, dtype=torch.float32))
            preds  = logits.argmax(dim=1).numpy()
        return accuracy_score(y_test, preds) * 100
    except Exception as e:
        print(f"⚠️  Could not evaluate current model: {e}")
        return 0.0


def _save_retrain_log(timestamp, drift_score, prev_acc, new_acc, status):
    """Append retraining event to a local log file"""
    os.makedirs("mlruns", exist_ok=True)
    log_path = "mlruns/retrain_log.txt"
    line = (f"{timestamp} | trigger=cusum | drift={drift_score:.4f} | "
            f"prev_acc={prev_acc:.2f}% | new_acc={new_acc:.2f}% | "
            f"status={status}\n")
    with open(log_path, "a") as f:
        f.write(line)
    print(f"📝 Retraining event logged to {log_path}")


if __name__ == "__main__":
    # Manual trigger for testing
    print("Running manual retraining trigger...")
    result = run_automated_retraining(
        drift_score=6.5,
        drift_points=list(range(10)),
        trigger_source="manual_test"
    )
    print(f"\nResult: {result}")