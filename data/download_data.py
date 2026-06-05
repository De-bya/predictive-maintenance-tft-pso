import urllib.request
import zipfile
import os
import pandas as pd
import numpy as np

def download_hydraulic_data():
    os.makedirs("data/raw", exist_ok=True)

    url = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
    zip_path = "data/raw/hydraulic.zip"

    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("✅ Download complete")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("data/raw/")
    print("✅ Extracted")

    # Each sensor is a separate .txt file with 2205 rows x N columns
    # Each row = one measurement cycle
    sensor_files = {
        'PS1': 'PS1.txt', 'PS2': 'PS2.txt', 'PS3': 'PS3.txt',
        'PS4': 'PS4.txt', 'PS5': 'PS5.txt', 'PS6': 'PS6.txt',
        'EPS1': 'EPS1.txt', 'FS1': 'FS1.txt', 'FS2': 'FS2.txt',
        'TS1': 'TS1.txt', 'TS2': 'TS2.txt', 'TS3': 'TS3.txt',
        'TS4': 'TS4.txt', 'VS1': 'VS1.txt', 'CE': 'CE.txt',
        'CP': 'CP.txt', 'SE': 'SE.txt'
    }

    # Each sensor file has multiple readings per cycle
    # We take the mean across each cycle to get one row per cycle
    dfs = []
    for sensor, filename in sensor_files.items():
        filepath = f"data/raw/{filename}"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep='\t', header=None)
            # Summarize each cycle with mean, std, min, max
            summary = pd.DataFrame({
                f'{sensor}_mean': df.mean(axis=1),
                f'{sensor}_std':  df.std(axis=1),
                f'{sensor}_min':  df.min(axis=1),
                f'{sensor}_max':  df.max(axis=1),
            })
            dfs.append(summary)
            print(f"  Loaded {sensor}: {df.shape}")
        else:
            print(f"  ⚠️  {filename} not found, skipping")

    # Combine all sensors into one DataFrame
    X = pd.concat(dfs, axis=1)

    # Load the labels file (profile.txt)
    # Columns: cooler, valve, pump, accumulator (condition classes)
    profile_path = "data/raw/profile.txt"
    if os.path.exists(profile_path):
        y = pd.read_csv(profile_path, sep='\t', header=None,
                        names=['cooler', 'valve', 'pump', 'accumulator', 'stable_flag'])
    else:
        print("⚠️  profile.txt not found")
        y = pd.DataFrame()

    # Save
    X.to_csv("data/raw/sensors.csv", index=False)
    y.to_csv("data/raw/labels.csv", index=False)

    print(f"\n✅ sensors.csv saved — shape: {X.shape}")
    print(f"✅ labels.csv  saved — shape: {y.shape}")
    print(f"\nFeature columns ({len(X.columns)}):", list(X.columns[:8]), "...")
    print(f"Label  columns:", list(y.columns))

if __name__ == "__main__":
    download_hydraulic_data()