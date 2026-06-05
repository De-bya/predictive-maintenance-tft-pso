import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import yaml
import os
from model.tft import TemporalFusionTransformer
from mlops.drift import cusum_detect, needs_retraining
from evaluation.metrics import compute_all_metrics
from resilience.backoff import with_exponential_backoff

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

def train_model(model, X_train, y_train, config, best_params=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    lr         = best_params['learning_rate'] if best_params else config['model']['learning_rate']
    epochs     = config['model']['max_epochs']
    batch_size = config['model']['batch_size']

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 Training TFT — {epochs} epochs on {device}")
    best_loss = np.inf

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

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "data/processed/best_model.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.4f}")

    model.load_state_dict(torch.load("data/processed/best_model.pt"))
    print(f"✅ Training complete — Best loss: {best_loss:.4f}")
    return model

def evaluate_model(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

    metrics = compute_all_metrics(y_test, preds, y_prob=probs)

    # Check for drift using CUSUM (Equation 6)
    drift_pts = cusum_detect(y_test.astype(float), probs)
    metrics['drift_points'] = len(drift_pts)
    metrics['needs_retraining'] = needs_retraining(drift_pts)

    return metrics, preds, probs

def run_pipeline(best_params=None):
    config  = load_config()
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_processed()

        # Use PSO-tuned params if available, else use config defaults
        params = best_params or {
            'hidden_size':     config['model']['hidden_size'],
            'attention_heads': config['model']['attention_heads'],
            'dropout':         config['model']['dropout'],
            'learning_rate':   config['model']['learning_rate'],
        }

        mlflow.log_params(params)

        model = TemporalFusionTransformer(
            input_dim=X_train.shape[2],
            hidden_dim=params['hidden_size'],
            num_heads=params['attention_heads'],
            dropout=params['dropout'],
            num_classes=2
        )

        def train_fn():
            return train_model(model, X_train, y_train, config, params)

        trained_model = with_exponential_backoff(train_fn)
        metrics, preds, probs = evaluate_model(trained_model, X_test, y_test)

        # Log all metrics
        mlflow.log_metrics({
            'accuracy':  metrics['Accuracy (%)'],
            'precision': metrics['Precision (%)'],
            'recall':    metrics['Recall (%)'],
            'f1_score':  metrics['F1-Score (%)'],
            'rmse':      float(metrics['RMSE']),
            'mae':       float(metrics['MAE']),
        })

        # Save model with proper format (fixes serialization warning)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.pytorch.log_model(
                trained_model,
                name="tft_model",
                serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_TORCHSCRIPT
            )

        # Save local checkpoint
        from mlops.registry import save_checkpoint
        save_checkpoint(trained_model)

        print("\n✅ Run logged to MLflow — no warnings")

        if metrics['needs_retraining']:
            print("⚠️  Drift detected — scheduling retraining...")

    return trained_model, metrics