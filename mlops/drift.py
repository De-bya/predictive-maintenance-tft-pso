import numpy as np

def cusum_detect(y_actual, y_pred_mean, delta=0.5, threshold=5.0):
    """
    Equation 6 from the paper:
    S_t = max(0, S_{t-1} + (y_t - mu) - delta)
    """
    S = 0.0
    drift_points = []

    for t, (y, mu) in enumerate(zip(y_actual, y_pred_mean)):
        S = max(0, S + (y - mu) - delta)
        if S > threshold:
            drift_points.append(t)
            S = 0.0  # reset after detecting drift

    if drift_points:
        print(f"⚠️  Drift detected at {len(drift_points)} points")
    else:
        print("✅ No drift detected")

    return drift_points

def needs_retraining(drift_points, min_drift_count=3):
    """Trigger retraining if drift is frequent enough"""
    return len(drift_points) >= min_drift_count