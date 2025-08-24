import pandas as pd
import numpy as np
import os
import pickle
import logging
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Setup logging
def get_logger(name):
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

# Version data snapshots
def version_snapshot(tag, paths, notes=""):
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "versioned_data")
    os.makedirs(base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = os.path.join(base, f"{tag}_{ts}")
    os.makedirs(snap_dir, exist_ok=True)
    manifest = {"tag": tag, "timestamp": ts, "notes": notes, "files": []}
    
    for p in paths:
        if os.path.exists(p):
            import shutil
            dst = os.path.join(snap_dir, os.path.basename(p))
            shutil.copy2(p, dst)
            manifest["files"].append({"src": p, "dst": dst})
    
    with open(os.path.join(snap_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    
    return snap_dir

# Initialize logger
logger = get_logger("transform")

def transform_data():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prep_dir = os.path.join(base_dir, "data_preparation")
    transform_dir = os.path.join(base_dir, "data_transformation")
    model_dir = os.path.join(base_dir, "model_building", "models")
    
    os.makedirs(transform_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Load prepared data
    logger.info("Loading prepared data")
    df_kaggle = pd.read_csv(os.path.join(prep_dir, "kaggle_prepared.csv"))
    df_hugging = pd.read_csv(os.path.join(prep_dir, "hugging_prepared.csv"))
    
    # Feature engineering
    logger.info("Performing feature engineering and transformation")
    
    # Initialize and fit scalers
    scaler_kaggle = StandardScaler()
    scaler_hugging = StandardScaler()
    
    # Transform numerical features
    df_kaggle_scaled = pd.DataFrame(
        scaler_kaggle.fit_transform(df_kaggle.select_dtypes(include=["number"])),
        columns=df_kaggle.select_dtypes(include=["number"]).columns
    )
    
    df_hugging_scaled = pd.DataFrame(
        scaler_hugging.fit_transform(df_hugging.select_dtypes(include=["number"])),
        columns=df_hugging.select_dtypes(include=["number"]).columns
    )
    
    # Save transformed data
    kaggle_output = os.path.join(transform_dir, "kaggle_transformed.csv")
    hugging_output = os.path.join(transform_dir, "hugging_transformed.csv")
    
    df_kaggle_scaled.to_csv(kaggle_output, index=False)
    df_hugging_scaled.to_csv(hugging_output, index=False)
    
    # Save scalers for future use
    scaler_kaggle_path = os.path.join(model_dir, "scaler_kaggle.pkl")
    scaler_hugging_path = os.path.join(model_dir, "scaler_hugging.pkl")
    
    with open(scaler_kaggle_path, 'wb') as f:
        pickle.dump(scaler_kaggle, f)
    
    with open(scaler_hugging_path, 'wb') as f:
        pickle.dump(scaler_hugging, f)
    
    # Version the transformed data
    version_snapshot("transformed", [kaggle_output, hugging_output], 
                    notes="Transformed data with StandardScaler")
    
    logger.info("Data transformation complete")
    return kaggle_output, hugging_output

if __name__ == "__main__":
    transform_data()