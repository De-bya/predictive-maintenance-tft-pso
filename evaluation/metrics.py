import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)

def compute_all_metrics(y_true, y_pred, y_prob=None):
    """Equations 8-13 from the paper"""

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None

    metrics = {
        "Accuracy (%)":  round(acc  * 100, 2),
        "Precision (%)": round(prec * 100, 2),
        "Recall (%)":    round(rec  * 100, 2),
        "F1-Score (%)":  round(f1   * 100, 2),
        "RMSE":          round(rmse, 4),
        "MAE":           round(mae,  4),
        "AUC-ROC":       round(auc,  4) if auc else "N/A",
    }

    print("\n===== Evaluation Metrics =====")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print("==============================\n")

    return metrics

def compute_availability(uptime, downtime):
    """Equation 14 from the paper"""
    availability = uptime / (uptime + downtime)
    print(f"Service Availability: {availability*100:.2f}%")
    return availability

def compute_throughput_latency(total_data, time_taken, response_time, n_requests):
    """Equations 15-16 from the paper"""
    throughput = total_data / time_taken
    latency    = response_time / n_requests
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Latency:    {latency:.4f} sec/request")
    return throughput, latency