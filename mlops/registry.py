import mlflow
import mlflow.pytorch
import torch
import os

def register_model(model, model_name="TFT_PredictiveMaintenance"):
    """Save model to MLflow registry with versioning"""
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/tft_model"
    mlflow.register_model(model_uri, model_name)
    print(f"✅ Model registered as '{model_name}'")

def save_checkpoint(model, path="data/processed/best_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"✅ Checkpoint saved to {path}")

def load_checkpoint(model, path="data/processed/best_model.pt"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"✅ Checkpoint loaded from {path}")
    else:
        print(f"⚠️  No checkpoint found at {path}")
    return model

def rollback_model(model, path="data/processed/best_model.pt"):
    """SRE rollback — restore last known good model"""
    print("🔄 Rolling back to last good checkpoint...")
    return load_checkpoint(model, path)