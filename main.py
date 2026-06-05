import argparse
from preprocessing.preprocess import run_preprocessing
from model.pso import run_pso
from mlops.pipeline import run_pipeline
import numpy as np

def main(skip_pso=False):
    print("=" * 50)
    print("  Predictive Maintenance — TFT + PSO + MLOps")
    print("=" * 50)

    # 1. Preprocess
    print("\n📦 Step 1: Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = run_preprocessing()

    # 2. PSO hyperparameter tuning (optional, takes time)
    best_params = None
    if not skip_pso:
        print("\n🔍 Step 2: PSO Hyperparameter Tuning...")
        best_params = run_pso(X_train, y_train)
    else:
        print("\n⏭️  Skipping PSO — using config defaults")

    # 3. Train + evaluate via MLOps pipeline
    print("\n🚀 Step 3: Training + Evaluation...")
    model, metrics = run_pipeline(best_params)

    print("\n🎉 Pipeline complete!")
    print(f"   Accuracy:  {metrics['Accuracy (%)']:.2f}%")
    print(f"   Recall:    {metrics['Recall (%)']:.2f}%")
    print(f"   F1-Score:  {metrics['F1-Score (%)']:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pso", action="store_true",
                        help="Skip PSO and use config defaults (faster)")
    args = parser.parse_args()
    main(skip_pso=args.skip_pso)