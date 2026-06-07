import numpy as np

def cusum_detect(y_actual, y_pred_mean, delta=0.5, threshold=5.0):
    """
    Equation 6 from the paper:
    S_t = max(0, S_{t-1} + (y_t - mu) - delta)

    Returns drift_points list AND final cumulative score
    """
    S = 0.0
    drift_points = []
    scores       = []

    for t, (y, mu) in enumerate(zip(y_actual, y_pred_mean)):
        S = max(0, S + (y - mu) - delta)
        scores.append(S)
        if S > threshold:
            drift_points.append(t)
            S = 0.0  # reset after detecting

    final_score = float(np.mean(scores)) if scores else 0.0

    if drift_points:
        print(f"⚠️  Drift detected at {len(drift_points)} points "
              f"| CUSUM score: {final_score:.4f}")
    else:
        print(f"✅ No drift detected | CUSUM score: {final_score:.4f}")

    return drift_points, final_score

def needs_retraining(drift_points, min_drift_count=3):
    return len(drift_points) >= min_drift_count