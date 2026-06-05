import numpy as np
import time
import os

# Shared state so actions can communicate
_chaos_state = {
    "dropped_sensors": [],
    "noise_injected": False,
    "latency_active": False,
}

def drop_sensor(sensor_prefix="PS1", duration_seconds=5):
    """
    Simulate a sensor going offline by zeroing out its features
    in the processed test data.
    """
    print(f"  [chaos] Dropping sensor {sensor_prefix} for {duration_seconds}s...")
    _chaos_state["dropped_sensors"].append(sensor_prefix)

    X_test = np.load("data/processed/X_test.npy")
    original = X_test.copy()

    # Find columns that belong to this sensor
    import pandas as pd
    sensors_csv = "data/raw/sensors.csv"
    if os.path.exists(sensors_csv):
        cols = pd.read_csv(sensors_csv, nrows=0).columns.tolist()
        drop_indices = [i for i, c in enumerate(cols) if c.startswith(sensor_prefix)]
        X_test[:, :, drop_indices] = 0.0
        np.save("data/processed/X_test_chaos.npy", X_test)
        print(f"  [chaos] Zeroed {len(drop_indices)} features for {sensor_prefix}")

    time.sleep(duration_seconds)
    print(f"  [chaos] Sensor dropout period ended")

def inject_noise(noise_scale=10.0, affected_fraction=0.3):
    """Inject Gaussian noise into a fraction of test samples"""
    print(f"  [chaos] Injecting noise (scale={noise_scale}, fraction={affected_fraction})...")
    _chaos_state["noise_injected"] = True

    path = "data/processed/X_test_chaos.npy"
    if not os.path.exists(path):
        path = "data/processed/X_test.npy"

    X_test = np.load(path)
    n_affected = int(len(X_test) * affected_fraction)
    indices = np.random.choice(len(X_test), n_affected, replace=False)
    X_test[indices] += np.random.normal(0, noise_scale, X_test[indices].shape)
    np.save("data/processed/X_test_chaos.npy", X_test)
    print(f"  [chaos] Noise injected into {n_affected} samples")

def simulate_latency(delay_seconds=2):
    """Simulate network or inference latency"""
    print(f"  [chaos] Simulating {delay_seconds}s latency...")
    _chaos_state["latency_active"] = True
    time.sleep(delay_seconds)
    _chaos_state["latency_active"] = False
    print(f"  [chaos] Latency simulation complete")

def restore_all():
    """Rollback: remove chaos-modified files, restore clean state"""
    print("  [chaos] Rolling back — restoring clean data...")
    _chaos_state["dropped_sensors"] = []
    _chaos_state["noise_injected"]  = False

    if os.path.exists("data/processed/X_test_chaos.npy"):
        os.remove("data/processed/X_test_chaos.npy")
        print("  [chaos] Removed X_test_chaos.npy")

    print("  [chaos] System restored to clean state ✅")