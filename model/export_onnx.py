"""
ONNX Export for the TFT model.
Exports the trained PyTorch model to ONNX format for
faster CPU inference and cross-platform deployment.
"""
import torch
import numpy as np
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def export_to_onnx(
    checkpoint_path="data/processed/best_model.pt",
    onnx_path="data/processed/tft_model.onnx",
    input_dim=68,
    hidden_dim=128,
    num_heads=8,
    dropout=0.068,
    seq_len=60,
    batch_size=1
):
    """
    Export trained TFT to ONNX format.

    Args:
        checkpoint_path: path to .pt checkpoint
        onnx_path:       where to save .onnx file
        input_dim:       number of input features (68)
        hidden_dim:      TFT hidden size (PSO best: 96)
        num_heads:       attention heads (PSO best: 8)
        dropout:         dropout rate
        seq_len:         sequence length (60)
        batch_size:      export batch size (1 for inference)

    Returns:
        onnx_path if successful
    """
    from model.tft import TemporalFusionTransformer

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run main.py first."
        )

    print(f"Loading model from {checkpoint_path}...")
    model = TemporalFusionTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_classes=2
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    model.eval()

    # Dummy input for tracing — shape: (batch, seq_len, features)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    print(f"Exporting to ONNX: {onnx_path}")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,       # optimize constants
        input_names=["sensor_window"],
        output_names=["logits"],
        dynamic_axes={
            "sensor_window": {0: "batch_size"},  # dynamic batch
            "logits":        {0: "batch_size"}
        }
    )

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✅ ONNX export successful")
    print(f"   Path:    {onnx_path}")
    print(f"   Size:    {size_mb:.2f} MB")
    print(f"   Opset:   17")
    print(f"   Inputs:  sensor_window (batch, {seq_len}, {input_dim})")
    print(f"   Outputs: logits (batch, 2)")

    return onnx_path


def benchmark_pytorch_vs_onnx(
    checkpoint_path="data/processed/best_model.pt",
    onnx_path="data/processed/tft_model.onnx",
    n_runs=50
):
    """
    Compare inference latency: PyTorch vs ONNX Runtime.
    Runs n_runs predictions and reports p50/p95/mean latency.
    """
    import onnxruntime as ort
    from model.tft import TemporalFusionTransformer

    print(f"\n{'='*55}")
    print(f"  LATENCY BENCHMARK: PyTorch vs ONNX Runtime")
    print(f"  Runs: {n_runs} predictions | Input: (1, 60, 68)")
    print(f"{'='*55}")

    dummy_np = np.random.randn(1, 60, 68).astype(np.float32)
    dummy_pt = torch.tensor(dummy_np)

    # ── PyTorch benchmark ─────────────────────────────────────
    model = TemporalFusionTransformer(
        input_dim=68, hidden_dim=128,
        num_heads=8, dropout=0.068, num_classes=2
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    model.eval()

    pt_latencies = []
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            model(dummy_pt)
        # Measure
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy_pt)
            pt_latencies.append((time.perf_counter() - t0) * 1000)

    # ── ONNX Runtime benchmark ────────────────────────────────
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

    onnx_latencies = []
    # Warmup
    for _ in range(5):
        session.run(None, {"sensor_window": dummy_np})
    # Measure
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {"sensor_window": dummy_np})
        onnx_latencies.append((time.perf_counter() - t0) * 1000)

    # ── Results ───────────────────────────────────────────────
    pt_mean   = np.mean(pt_latencies)
    pt_p50    = np.percentile(pt_latencies, 50)
    pt_p95    = np.percentile(pt_latencies, 95)

    onnx_mean = np.mean(onnx_latencies)
    onnx_p50  = np.percentile(onnx_latencies, 50)
    onnx_p95  = np.percentile(onnx_latencies, 95)

    speedup   = pt_mean / onnx_mean

    print(f"\n  {'Metric':<12} {'PyTorch':>12} {'ONNX Runtime':>14} {'Speedup':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Mean (ms)':<12} {pt_mean:>12.2f} {onnx_mean:>14.2f} {speedup:>9.2f}x")
    print(f"  {'p50  (ms)':<12} {pt_p50:>12.2f} {onnx_p50:>14.2f}")
    print(f"  {'p95  (ms)':<12} {pt_p95:>12.2f} {onnx_p95:>14.2f}")
    print(f"\n  🚀 ONNX Runtime is {speedup:.2f}x {'faster' if speedup>1 else 'slower'} than PyTorch")
    print(f"{'='*55}\n")

    return {
        "pytorch":  {"mean": pt_mean,   "p50": pt_p50,   "p95": pt_p95},
        "onnx":     {"mean": onnx_mean, "p50": onnx_p50, "p95": onnx_p95},
        "speedup":  speedup
    }


def verify_outputs_match(
    checkpoint_path="data/processed/best_model.pt",
    onnx_path="data/processed/tft_model.onnx"
):
    """Verify PyTorch and ONNX produce identical outputs"""
    import onnxruntime as ort
    from model.tft import TemporalFusionTransformer

    dummy_np = np.random.randn(1, 60, 68).astype(np.float32)
    dummy_pt = torch.tensor(dummy_np)

    model = TemporalFusionTransformer(
        input_dim=68, hidden_dim=128,
        num_heads=8, dropout=0.068, num_classes=2
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    model.eval()
    with torch.no_grad():
        pt_out = model(dummy_pt).numpy()

    session  = ort.InferenceSession(onnx_path,
                   providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"sensor_window": dummy_np})[0]

    max_diff = np.abs(pt_out - onnx_out).max()
    match    = max_diff < 1e-4

    print(f"Output verification:")
    print(f"  PyTorch logits: {pt_out[0].tolist()}")
    print(f"  ONNX logits:    {onnx_out[0].tolist()}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Match: {'✅ YES' if match else '❌ NO'} (threshold: 1e-4)")
    return match


if __name__ == "__main__":
    print("="*55)
    print("  TFT → ONNX Export Pipeline")
    print("="*55)

    # Step 1: Export
    export_to_onnx()

    # Step 2: Verify outputs match
    print("\n📋 Verifying outputs match PyTorch...")
    verify_outputs_match()

    # Step 3: Benchmark
    benchmark_pytorch_vs_onnx(n_runs=50)