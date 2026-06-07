import argparse
import yaml
import numpy as np
from preprocessing.preprocess import run_preprocessing
from model.pso import run_pso

def main(skip_pso=False):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    multi_label = config['model'].get('multi_label', False)

    print("=" * 50)
    print("  Predictive Maintenance — TFT + PSO + MLOps")
    print(f"  Mode: {'Multi-label (5 targets)' if multi_label else 'Binary'}")
    print("=" * 50)

    # ── Step 1: Preprocessing ─────────────────────────────────
    print("\n📦 Step 1: Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = run_preprocessing()

    # ── Step 2: PSO (binary mode only) ────────────────────────
    best_params = None
    if not skip_pso and not multi_label:
        print("\n🔍 Step 2: PSO Hyperparameter Tuning...")
        best_params = run_pso(X_train, y_train)
    else:
        print("\n⏭️  Skipping PSO")

    # ── Step 3: Train + Evaluate ──────────────────────────────
    print("\n🚀 Step 3: Training + Evaluation...")

    if multi_label:
        from model.tft import MultiLabelTFT
        from mlops.pipeline import train_multilabel_model, evaluate_multilabel_model
        import mlflow

        mlflow.set_experiment("predictive_maintenance_multilabel")
        with mlflow.start_run():
            model = MultiLabelTFT(
                input_dim=X_train.shape[2],
                hidden_dim=config['model']['hidden_size'],
                num_heads=config['model']['attention_heads'],
                dropout=config['model']['dropout']
            )
            model   = train_multilabel_model(model, X_train, y_train, config)
            metrics = evaluate_multilabel_model(model, X_test, y_test)
            mlflow.log_metrics({
                'avg_accuracy': metrics['avg_accuracy'],
                'avg_f1':       metrics['avg_f1']
            })
            print("✅ Multi-label run logged to MLflow")

        print(f"\n🎉 Multi-label pipeline complete!")
        print(f"   Avg Accuracy: {metrics['avg_accuracy']:.2f}%")
        print(f"   Avg F1-Score: {metrics['avg_f1']:.2f}%")

    else:
        from mlops.pipeline import run_pipeline
        model, metrics = run_pipeline(best_params)
        print(f"\n🎉 Pipeline complete!")
        print(f"   Accuracy:  {metrics['Accuracy (%)']:.2f}%")
        print(f"   Recall:    {metrics['Recall (%)']:.2f}%")
        print(f"   F1-Score:  {metrics['F1-Score (%)']:.2f}%")

        # ── Auto-export to ONNX after training ─────────────────
        print("\n📦 Exporting model to ONNX...")
        try:
            from model.export_onnx import export_to_onnx, benchmark_pytorch_vs_onnx
            export_to_onnx()
            benchmark_pytorch_vs_onnx(n_runs=30)
        except Exception as e:
            print(f"⚠️  ONNX export failed (non-critical): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pso", action="store_true",
                        help="Skip PSO and use config defaults")
    args = parser.parse_args()
    main(skip_pso=args.skip_pso)