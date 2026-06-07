import os
import numpy as np
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def model_checkpoint_exists():
    """Steady-state probe: verify model checkpoint exists"""
    exists = os.path.exists("data/processed/best_model.pt")
    print(f"  [probe] Checkpoint exists: {exists}")
    return exists

def model_can_predict():
    """Steady-state probe: verify model can still make predictions after chaos"""
    try:
        from model.tft import TemporalFusionTransformer
        model = TemporalFusionTransformer(
            input_dim=68, hidden_dim=128,
            num_heads=8, dropout=0.1, num_classes=2
        )
        model.load_state_dict(
            torch.load("data/processed/best_model.pt", map_location="cpu")
        )
        model.eval()
        dummy = torch.randn(1, 60, 68)
        with torch.no_grad():
            out = model(dummy)
        ok = out.shape == (1, 2)
        print(f"  [probe] Model prediction shape {out.shape}: {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"  [probe] Model prediction FAILED: {e}")
        return False