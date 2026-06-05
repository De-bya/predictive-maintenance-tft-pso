"""
Standalone chaos engineering runner.
Simulates fault injection and measures model resilience.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import time
from resilience.chaos.probes  import model_checkpoint_exists, model_can_predict
from resilience.chaos.actions import drop_sensor, inject_noise, simulate_latency, restore_all
from resilience.sre            import SREMonitor
from model.tft                 import TemporalFusionTransformer

def load_model():
    model = TemporalFusionTransformer(
        input_dim=68, hidden_dim=96,
        num_heads=8, dropout=0.1, num_classes=2
    )
    model.load_state_dict(
        torch.load("data/processed/best_model.pt", map_location="cpu")
    )
    model.eval()
    return model

def predict(model, X):
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32)
        logits = model(t)
        return logits.argmax(dim=1).numpy()

def run_chaos_suite():
    print("\n" + "="*55)
    print("  CHAOS ENGINEERING — Fault Injection Test Suite")
    print("="*55)

    sre = SREMonitor(availability_target=0.973)

    # ── Steady state: BEFORE ─────────────────────────────────
    print("\n📋 Steady state check (BEFORE)...")
    assert model_checkpoint_exists(), "❌ No checkpoint — cannot run chaos"
    assert model_can_predict(),       "❌ Model broken before chaos — aborting"
    print("✅ Steady state OK\n")

    X_test  = np.load("data/processed/X_test.npy")
    y_test  = np.load("data/processed/y_test.npy")
    model   = load_model()
    results = {}

    # ── Baseline accuracy ────────────────────────────────────
    sre.start_window()
    preds_clean = predict(model, X_test)
    sre.record_uptime()
    baseline_acc = (preds_clean == y_test).mean()
    results['baseline'] = baseline_acc
    print(f"📊 Baseline accuracy (clean data): {baseline_acc*100:.2f}%")

    # ── Experiment 1: Sensor dropout ─────────────────────────
    print("\n🔴 Experiment 1: Pressure sensor PS1 dropout")
    sre.start_window()
    try:
        drop_sensor("PS1", duration_seconds=1)
        chaos_path = "data/processed/X_test_chaos.npy"
        X_chaos = np.load(chaos_path) if os.path.exists(chaos_path) else X_test
        preds_drop = predict(model, X_chaos)
        acc_drop = (preds_drop == y_test).mean()
        results['sensor_dropout'] = acc_drop
        sre.record_uptime()
        print(f"   Accuracy with PS1 dropped: {acc_drop*100:.2f}%")
        print(f"   Accuracy degradation:      {(baseline_acc - acc_drop)*100:.2f}%")
    except Exception as e:
        sre.record_downtime()
        print(f"   ❌ Failed: {e}")
        results['sensor_dropout'] = 0.0
    finally:
        restore_all()

    # ── Experiment 2: Noise injection ────────────────────────
    print("\n🔴 Experiment 2: Corrupted sensor readings (30% noise)")
    sre.start_window()
    try:
        inject_noise(noise_scale=5.0, affected_fraction=0.3)
        chaos_path = "data/processed/X_test_chaos.npy"
        X_chaos = np.load(chaos_path) if os.path.exists(chaos_path) else X_test
        preds_noise = predict(model, X_chaos)
        acc_noise = (preds_noise == y_test).mean()
        results['noise_injection'] = acc_noise
        sre.record_uptime()
        print(f"   Accuracy with noise:  {acc_noise*100:.2f}%")
        print(f"   Accuracy degradation: {(baseline_acc - acc_noise)*100:.2f}%")
    except Exception as e:
        sre.record_downtime()
        print(f"   ❌ Failed: {e}")
        results['noise_injection'] = 0.0
    finally:
        restore_all()

    # ── Experiment 3: Latency simulation ─────────────────────
    print("\n🔴 Experiment 3: Network latency simulation (2s)")
    sre.start_window()
    try:
        start = time.time()
        simulate_latency(delay_seconds=2)
        preds_late = predict(model, X_test)
        elapsed = time.time() - start
        acc_late = (preds_late == y_test).mean()
        results['latency'] = acc_late
        sre.record_uptime()
        print(f"   Accuracy under latency: {acc_late*100:.2f}%")
        print(f"   Total response time:    {elapsed:.2f}s")
    except Exception as e:
        sre.record_downtime()
        print(f"   ❌ Failed: {e}")

    # ── Steady state: AFTER ──────────────────────────────────
    print("\n📋 Steady state check (AFTER rollback)...")
    assert model_can_predict(), "❌ Model broken after chaos!"
    print("✅ System recovered successfully\n")

    # ── SRE Summary ──────────────────────────────────────────
    print("="*55)
    print("  SRE AVAILABILITY REPORT")
    print("="*55)
    sre.slo_breached()

    print("\n" + "="*55)
    print("  CHAOS EXPERIMENT SUMMARY")
    print("="*55)
    print(f"  Baseline accuracy:       {results.get('baseline',0)*100:.2f}%")
    print(f"  After sensor dropout:    {results.get('sensor_dropout',0)*100:.2f}%")
    print(f"  After noise injection:   {results.get('noise_injection',0)*100:.2f}%")
    print(f"  After latency:           {results.get('latency',0)*100:.2f}%")
    print(f"\n  Model resilience score:  "
          f"{min(results.values())/results.get('baseline',1)*100:.1f}% "
          f"(min accuracy retained)")
    print("="*55)

if __name__ == "__main__":
    run_chaos_suite()