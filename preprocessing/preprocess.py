import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import yaml
import pickle

# ── Label configs for multi-label mode ───────────────────────────────────────
LABEL_CONFIGS = {
    'cooler':      {'values': [3, 20, 100],        'n_classes': 3},
    'valve':       {'values': [73, 80, 90, 100],   'n_classes': 4},
    'pump':        {'values': [0, 1, 2],            'n_classes': 3},
    'accumulator': {'values': [90, 100, 115, 130], 'n_classes': 4},
    'stable_flag': {'values': [0, 1],               'n_classes': 2},
}

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_raw_data():
    X = pd.read_csv("data/raw/sensors.csv")
    y = pd.read_csv("data/raw/labels.csv")
    print(f"✅ Loaded sensors: {X.shape}, labels: {y.shape}")
    return X, y

def interpolate_missing(df):
    """Equation 1: linear interpolation for missing values"""
    missing_before = df.isnull().sum().sum()
    df = df.interpolate(method='linear', axis=0)
    df = df.bfill().ffill()
    missing_after = df.isnull().sum().sum()
    print(f"Missing values: {missing_before} → {missing_after}")
    return df

def zscore_normalize(df):
    """Equation 2: z-score normalization"""
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )
    print(f"✅ Z-score normalization applied — mean≈0, std≈1")
    return df_scaled, scaler

# ── Binary mode ───────────────────────────────────────────────────────────────
def encode_labels(y):
    """Binary: stable_flag → 1=normal, 0=fault"""
    y_binary = (y['stable_flag'] == 1).astype(int)
    print(f"Label distribution:\n{y_binary.value_counts().to_string()}")
    print(f"  (1=Normal, 0=Fault)")
    return y_binary, y

def create_sequences(X, y_binary, window_size=60):
    """Rolling windows for binary classification"""
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X.iloc[i:i + window_size].values)
        ys.append(y_binary.iloc[i + window_size])
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.int64)
    print(f"✅ Sequences — X: {Xs.shape}, y: {ys.shape}")
    return Xs, ys

# ── Multi-label mode ──────────────────────────────────────────────────────────
def encode_multilabels(y):
    """Encode each of the 5 targets as integer class indices"""
    y_encoded = pd.DataFrame()
    encoders  = {}
    for col in LABEL_CONFIGS:
        le = LabelEncoder()
        y_encoded[col] = le.fit_transform(y[col])
        encoders[col]  = le
        print(f"  {col}: {dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))}")
    print(f"\n✅ Multi-label encoding done — {len(LABEL_CONFIGS)} targets")
    return y_encoded, encoders

def create_multilabel_sequences(X, y_encoded, window_size=60):
    """Rolling windows for multi-label classification"""
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X.iloc[i:i + window_size].values)
        ys.append(y_encoded.iloc[i + window_size].values)  # (5,)
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.int64)
    print(f"✅ Multi-label sequences — X: {Xs.shape}, y: {ys.shape}")
    return Xs, ys

# ── Shared ────────────────────────────────────────────────────────────────────
def temporal_split(Xs, ys, test_split=0.2):
    """Time-aware split — NO shuffling"""
    split = int(len(Xs) * (1 - test_split))
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    if ys.ndim == 1:
        print(f"Train faults: {(y_train==0).sum()} | Test faults: {(y_test==0).sum()}")
    return X_train, X_test, y_train, y_test

def save_processed(X_train, X_test, y_train, y_test, scaler, encoders=None):
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy",  X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy",  y_test)
    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    if encoders:
        with open("data/processed/encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
    print("✅ All processed files saved to data/processed/")

def run_preprocessing():
    config     = load_config()
    window     = config['data']['window_size']
    test_split = config['data']['test_split']
    multi      = config['model'].get('multi_label', False)

    X, y      = load_raw_data()
    X         = interpolate_missing(X)
    X, scaler = zscore_normalize(X)

    if multi:
        print("\n🏷️  Multi-label mode")
        y_encoded, encoders = encode_multilabels(y)
        Xs, ys = create_multilabel_sequences(X, y_encoded, window)
    else:
        y_binary, _ = encode_labels(y)
        encoders    = None
        Xs, ys      = create_sequences(X, y_binary, window)

    X_train, X_test, y_train, y_test = temporal_split(Xs, ys, test_split)
    save_processed(X_train, X_test, y_train, y_test, scaler, encoders)
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    run_preprocessing()