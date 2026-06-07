import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import yaml
import os
import warnings
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

def compute_class_weights(y_train, device, n_classes=2):
    """
    Compute inverse-frequency class weights from training labels.
    Minority class gets higher weight → model penalized more for missing it.
    Binary:     y_train shape (N,)
    Multi-label: y_train shape (N, 5)
    """
    import torch
    from sklearn.utils.class_weight import compute_class_weight

    if y_train.ndim == 1:
        # Binary mode
        classes = np.arange(n_classes)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        w = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"✅ Class weights (binary): fault={w[0]:.4f}, normal={w[1]:.4f}")
        return w

    else:
        # Multi-label mode — one weight tensor per target
        target_order = ['cooler','valve','pump','accumulator','stable_flag']
        n_classes_per_target = [3, 4, 3, 4, 2]
        all_weights = {}
        for i, (name, nc) in enumerate(zip(target_order, n_classes_per_target)):
            classes = np.arange(nc)
            col     = y_train[:, i]
            present = np.unique(col)
            weights = compute_class_weight(
                class_weight='balanced',
                classes=present,
                y=col
            )
            # Pad to full size if some classes missing
            full_w = np.ones(nc, dtype=np.float32)
            for cls, w in zip(present, weights):
                full_w[cls] = w
            all_weights[name] = torch.tensor(full_w, dtype=torch.float32).to(device)
            print(f"  {name:<15} weights: {[round(x,3) for x in full_w.tolist()]}")
        print("✅ Class weights computed for all 5 targets")
        return all_weights

# ══════════════════════════════════════════════════════════════
#  BINARY PIPELINE (original)
# ══════════════════════════════════════════════════════════════
def train_model(model, X_train, y_train, config, best_params=None):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = model.to(device)
    use_weighted = config['model'].get('use_weighted_loss', False)

    lr         = best_params['learning_rate'] if best_params else config['model']['learning_rate']
    epochs     = config['model']['max_epochs']
    batch_size = config['model']['batch_size']

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True)

    # ── Weighted loss ─────────────────────────────────────────
    if use_weighted:
        class_weights = compute_class_weights(y_train, device, n_classes=2)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
        print(f"⚖️  Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"Using standard CrossEntropyLoss")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

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

    model.load_state_dict(
        torch.load("data/processed/best_model.pt", weights_only=False)
    )
    print(f"✅ Training complete — Best loss: {best_loss:.4f}")
    return model

def evaluate_model(model, X_test, y_test, auto_retrain=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

    metrics = compute_all_metrics(y_test, preds, y_prob=probs)

    # CUSUM now returns (drift_points, drift_score)
    drift_pts, drift_score = cusum_detect(y_test.astype(float), probs)
    metrics['drift_points']     = len(drift_pts)
    metrics['drift_score']      = drift_score
    metrics['needs_retraining'] = needs_retraining(drift_pts)

    # ── Auto-retrain if drift detected ────────────────────────
    if auto_retrain and metrics['needs_retraining']:
        print(f"\n🚨 Drift threshold exceeded — triggering automated retraining")
        from mlops.retrain import run_automated_retraining
        retrain_result = run_automated_retraining(
            drift_score=drift_score,
            drift_points=drift_pts,
            trigger_source="cusum_auto"
        )
        metrics['retrain_result'] = retrain_result

    return metrics, preds, probs

def run_pipeline(best_params=None):
    config = load_config()
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_processed()

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
        # metrics, preds, probs = evaluate_model(trained_model, X_test, y_test)
        auto_retrain = config.get('drift', {}).get('auto_retrain', False)
        metrics, preds, probs = evaluate_model(
            trained_model, X_test, y_test, auto_retrain=auto_retrain
        )

        mlflow.log_metrics({
            'accuracy':  metrics['Accuracy (%)'],
            'precision': metrics['Precision (%)'],
            'recall':    metrics['Recall (%)'],
            'f1_score':  metrics['F1-Score (%)'],
            'rmse':      float(metrics['RMSE']),
            'mae':       float(metrics['MAE']),
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.pytorch.log_model(
                trained_model,
                name="tft_model",
                # serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_TORCHSCRIPT
                serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE
            )

        from mlops.registry import save_checkpoint
        save_checkpoint(trained_model)
        print("\n✅ Run logged to MLflow")

        if metrics['needs_retraining']:
            print("⚠️  Drift detected — scheduling retraining...")

    return trained_model, metrics

# ══════════════════════════════════════════════════════════════
#  MULTI-LABEL PIPELINE
# ══════════════════════════════════════════════════════════════
def compute_multilabel_loss(outputs, y_batch, class_weights=None):
    target_order = ['cooler', 'valve', 'pump', 'accumulator', 'stable_flag']
    total_loss   = 0.0
    for i, name in enumerate(target_order):
        logits = outputs[name]
        labels = y_batch[:, i]
        w      = class_weights[name] if class_weights else None
        total_loss += nn.CrossEntropyLoss(weight=w)(logits, labels)
    return total_loss / len(target_order)

def train_multilabel_model(model, X_train, y_train, config, best_params=None):
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model        = model.to(device)
    use_weighted = config['model'].get('use_weighted_loss', False)

    lr         = best_params['learning_rate'] if best_params else config['model']['learning_rate']
    epochs     = config['model']['max_epochs']
    batch_size = config['model']['batch_size']

    X_t    = torch.tensor(X_train, dtype=torch.float32)
    y_t    = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    # ── Weighted loss per target ──────────────────────────────
    if use_weighted:
        class_weights = compute_class_weights(y_train, device)
        print(f"⚖️  Using weighted CrossEntropyLoss for all 5 targets")
    else:
        class_weights = None
        print(f"Using standard CrossEntropyLoss")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_loss = np.inf

    print(f"\n🚀 Training MultiLabelTFT — {epochs} epochs on {device}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs    = model(xb)
            loss       = compute_multilabel_loss(outputs, yb, class_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "data/processed/best_model_multilabel.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.4f}")

    model.load_state_dict(
        torch.load("data/processed/best_model_multilabel.pt", weights_only=False)
    )
    print(f"✅ Training complete — Best loss: {best_loss:.4f}")
    return model

def evaluate_multilabel_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, f1_score
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_order = ['cooler', 'valve', 'pump', 'accumulator', 'stable_flag']
    model.eval()

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_t)

    print("\n===== Multi-Label Evaluation =====")
    all_metrics = {}
    for i, name in enumerate(target_order):
        logits = outputs[name].cpu().numpy()
        preds  = np.argmax(logits, axis=1)
        truths = y_test[:, i]
        acc    = accuracy_score(truths, preds) * 100
        f1     = f1_score(truths, preds, average='weighted', zero_division=0) * 100
        print(f"  {name:<15} Acc: {acc:.2f}%  F1: {f1:.2f}%")
        all_metrics[name] = {'accuracy': acc, 'f1': f1}

    avg_acc = np.mean([m['accuracy'] for m in all_metrics.values()])
    avg_f1  = np.mean([m['f1']       for m in all_metrics.values()])
    print(f"  {'AVERAGE':<15} Acc: {avg_acc:.2f}%  F1: {avg_f1:.2f}%")
    print("==================================\n")

    all_metrics['avg_accuracy'] = avg_acc
    all_metrics['avg_f1']       = avg_f1
    return all_metrics