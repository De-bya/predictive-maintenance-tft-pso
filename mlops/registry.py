"""
Model management: production checkpoint, backup, and SRE rollback.
The hyperparameters are saved next to the weights so every consumer
(API, retraining, chaos tests, plots) rebuilds the exact same model.
"""
import json
import os
import shutil
import torch
from model.tft import TemporalFusionTransformer

CHECKPOINT  = "data/processed/best_model.pt"
PARAMS      = "data/processed/best_params.json"
BACKUP      = "data/processed/last_good_model.pt"
BACKUP_PARAMS = "data/processed/last_good_params.json"

def build_tft(params, input_dim=68):
    return TemporalFusionTransformer(
        input_dim=input_dim,
        hidden_dim=params['hidden_size'],
        num_heads=params['attention_heads'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        num_classes=2
    )

def save_checkpoint(model, params):
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    with open(PARAMS, "w") as f:
        json.dump(params, f, indent=2)
    print(f"✅ Checkpoint saved to {CHECKPOINT}")

def load_trained_model(path=CHECKPOINT, params_path=PARAMS):
    """Rebuild the production TFT from its saved params + weights"""
    with open(params_path) as f:
        params = json.load(f)
    model = build_tft(params)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model, params

def backup_checkpoint():
    """Keep a copy of the current production model before replacing it"""
    if os.path.exists(CHECKPOINT):
        shutil.copy(CHECKPOINT, BACKUP)
        shutil.copy(PARAMS, BACKUP_PARAMS)
        print(f"💾 Backed up production model to {BACKUP}")

def rollback_model():
    """SRE rollback — restore last known good model"""
    print("🔄 Rolling back to last good checkpoint...")
    shutil.copy(BACKUP, CHECKPOINT)
    shutil.copy(BACKUP_PARAMS, PARAMS)
    return load_trained_model()
