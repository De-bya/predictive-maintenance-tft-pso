import argparse
import random
import numpy as np
import torch
from preprocessing.preprocess import run_preprocessing
from model.pso import run_pso
from mlops.pipeline import run_pipeline

def set_seed(seed=42):
    """Fix all random sources so runs are reproducible"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(skip_pso=False):
    set_seed()
    print("=" * 50)
    print("  Predictive Maintenance — TFT + PSO + MLOps")
    print("=" * 50)

    # ── Step 1: Preprocessing (Eq. 1-2, rolling windows) ──────
    print("\n📦 Step 1: Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = run_preprocessing()

    # ── Step 2: PSO hyperparameter tuning (Eq. 4) ─────────────
    best_params = None
    if not skip_pso:
        print("\n🔍 Step 2: PSO Hyperparameter Tuning...")
        best_params = run_pso(X_train, y_train)
    else:
        print("\n⏭️  Skipping PSO — using config.yaml defaults")

    # ── Step 3: Train + evaluate + CUSUM drift (MLflow) ───────
    print("\n🚀 Step 3: Training + Evaluation...")
    model, metrics = run_pipeline(best_params)

    print(f"\n🎉 Pipeline complete!")
    print(f"   Accuracy:   {metrics['Accuracy (%)']:.2f}%")
    print(f"   Recall:     {metrics['Recall (%)']:.2f}%")
    print(f"   F1-Score:   {metrics['F1-Score (%)']:.2f}%")
    print(f"   Throughput: {metrics['Throughput (samples/s)']} samples/s")
    print(f"   Latency:    {metrics['Latency (ms/request)']} ms/request")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pso", action="store_true",
                        help="Skip PSO and use config defaults")
    args = parser.parse_args()
    main(skip_pso=args.skip_pso)
