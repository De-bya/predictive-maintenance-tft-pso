import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import yaml
import pickle

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
    # df = df.fillna(method='bfill').fillna(method='ffill')
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

def encode_labels(y):
    """
    Convert multi-condition labels into a single fault class.
    stable_flag=1 means normal. We create a combined fault label.
    """
    # Use stable_flag directly: 1 = normal, 0 = faulty
    # Also encode cooler/valve/pump/accumulator for multi-label use
    y_binary = (y['stable_flag'] == 1).astype(int)  # 1=normal, 0=fault
    print(f"Label distribution:\n{y_binary.value_counts().to_string()}")
    print(f"  (1=Normal, 0=Fault)")
    return y_binary, y

def create_sequences(X, y_binary, window_size=60):
    """
    Slide a window over cycles to create sequences.
    Each sequence captures temporal dependencies across cycles.
    """
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X.iloc[i:i + window_size].values)
        ys.append(y_binary.iloc[i + window_size])
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.int64)
    print(f"✅ Sequences — X: {Xs.shape}, y: {ys.shape}")
    return Xs, ys

def temporal_split(Xs, ys, test_split=0.2):
    """Time-aware split — NO shuffling to preserve temporal order"""
    split = int(len(Xs) * (1 - test_split))
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train faults: {(y_train==0).sum()} | Test faults: {(y_test==0).sum()}")
    return X_train, X_test, y_train, y_test

def save_processed(X_train, X_test, y_train, y_test, scaler):
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy",  X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy",  y_test)
    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✅ All processed files saved to data/processed/")

def run_preprocessing():
    config     = load_config()
    window     = config['data']['window_size']
    test_split = config['data']['test_split']

    X, y          = load_raw_data()
    X             = interpolate_missing(X)
    X, scaler     = zscore_normalize(X)
    y_binary, y_full = encode_labels(y)

    Xs, ys        = create_sequences(X, y_binary, window)
    X_train, X_test, y_train, y_test = temporal_split(Xs, ys, test_split)

    save_processed(X_train, X_test, y_train, y_test, scaler)
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    run_preprocessing()