import pandas as pd
import numpy as np
import os
import pickle
import logging
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

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
logger = get_logger("prepare")

def prepare_data():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    storage_dir = os.path.join(base_dir, "data_storage")
    prep_dir = os.path.join(base_dir, "data_preparation")
    model_dir = os.path.join(base_dir, "model_building", "models")
    
    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Load raw data
    logger.info("Loading raw data")
    df_kaggle = pd.read_csv(os.path.join(storage_dir, "kaggle_curn.csv"))
    df_hugging = pd.read_csv(os.path.join(storage_dir, "huggingface_churn.csv"))
    
    # Data cleaning
    logger.info("Cleaning data")
    df_kaggle.dropna(inplace=True)
    df_hugging.dropna(inplace=True)
    
    # Encode categorical features
    logger.info("Encoding categorical features")
    le_kaggle = LabelEncoder()
    le_hugging = LabelEncoder()
    
    for col in df_kaggle.select_dtypes(include=["object"]).columns:
        if df_kaggle[col].nunique() < 10:
            df_kaggle[col] = le_kaggle.fit_transform(df_kaggle[col])
            
            # Save the encoder for future use
            encoder_path = os.path.join(model_dir, f"le_kaggle_{col}.pkl")
            with open(encoder_path, 'wb') as f:
                pickle.dump(le_kaggle, f)
    
    for col in df_hugging.select_dtypes(include=["object"]).columns:
        if df_hugging[col].nunique() < 10:
            df_hugging[col] = le_hugging.fit_transform(df_hugging[col])
            
            # Save the encoder for future use
            encoder_path = os.path.join(model_dir, f"le_hugging_{col}.pkl")
            with open(encoder_path, 'wb') as f:
                pickle.dump(le_hugging, f)
    
    # Save prepared data
    kaggle_output = os.path.join(prep_dir, "kaggle_prepared.csv")
    hugging_output = os.path.join(prep_dir, "hugging_prepared.csv")
    
    df_kaggle.to_csv(kaggle_output, index=False)
    df_hugging.to_csv(hugging_output, index=False)
    
    # Version the prepared data
    version_snapshot("prepared", [kaggle_output, hugging_output], 
                    notes="Prepared data with categorical encoding and NA removal")
    
    logger.info("Data preparation complete")
    return kaggle_output, hugging_output

if __name__ == "__main__":
    prepare_data()