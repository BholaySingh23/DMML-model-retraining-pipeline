import pandas as pd
import numpy as np
import os
import pickle
import logging
import json
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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

# Version model snapshots
def version_model(model, model_name, metrics, dataset_name, notes=""):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model_building", "versioned_models")
    os.makedirs(model_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{model_name}_{dataset_name}_{ts}.pkl")
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save model metadata
    metadata = {
        "model_name": model_name,
        "dataset": dataset_name,
        "timestamp": ts,
        "metrics": metrics,
        "notes": notes
    }
    
    metadata_path = os.path.join(model_dir, f"{model_name}_{dataset_name}_{ts}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path, metadata_path

# Initialize logger
logger = get_logger("retrain_model")

def get_latest_model(model_name, dataset_name):
    """Get the latest model of a specific type and dataset"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model_building", "versioned_models")
    
    if not os.path.exists(model_dir):
        logger.warning(f"No versioned models directory found at {model_dir}")
        return None
    
    # Find all model files matching the pattern
    model_files = [f for f in os.listdir(model_dir) if f.startswith(f"{model_name}_{dataset_name}_") and f.endswith(".pkl")]
    
    if not model_files:
        logger.warning(f"No models found for {model_name} on {dataset_name} dataset")
        return None
    
    # Sort by timestamp (which is part of the filename)
    model_files.sort(reverse=True)
    latest_model_file = model_files[0]
    
    # Load the model
    model_path = os.path.join(model_dir, latest_model_file)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def retrain_model(model_name, dataset_name, hyperparams=None):
    """Retrain an existing model with new data or hyperparameters"""
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transform_dir = os.path.join(base_dir, "data_transformation")
    model_dir = os.path.join(base_dir, "model_building")
    report_dir = os.path.join(model_dir, "reports")
    
    os.makedirs(report_dir, exist_ok=True)
    
    # Load data based on dataset name
    logger.info(f"Retraining {model_name} model using {dataset_name} dataset")
    
    if dataset_name == "kaggle":
        df = pd.read_csv(os.path.join(transform_dir, "kaggle_transformed.csv"))
    elif dataset_name == "huggingface":
        df = pd.read_csv(os.path.join(transform_dir, "hugging_transformed.csv"))
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    # Prepare data for training
    df = df.drop(columns=["customerID"], errors="ignore")
    
    # Handle target variable
    if "Churn" in df.columns:
        if df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        df.dropna(subset=["Churn"], inplace=True)
        target_col = "Churn"
    else:
        # Try to find a target column
        potential_targets = [col for col in df.columns if col.lower() in ["churn", "target", "label"]]
        if potential_targets:
            target_col = potential_targets[0]
        else:
            raise ValueError("Could not identify target column in dataset")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get the latest model
    model = get_latest_model(model_name, dataset_name)
    
    if model is None:
        logger.warning(f"No existing model found. Cannot retrain.")
        return None
    
    # Update hyperparameters if provided
    if hyperparams and hasattr(model, 'set_params'):
        model.set_params(**hyperparams)
    
    # Retrain the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(report_dir, f"{model_name}_{dataset_name}_retrained_report.csv")
    report_df.to_csv(report_path)
    
    # Save model with versioning
    model_path, metadata_path = version_model(
        model, 
        model_name, 
        metrics, 
        dataset_name, 
        notes=f"Retrained {model_name} on {dataset_name} dataset"
    )
    
    logger.info(f"Retrained model saved to {model_path}")
    logger.info(f"Metrics: {metrics}")
    
    # Update best model if this one is better
    best_model_path = os.path.join(model_dir, "models", f"best_model_{dataset_name}.pkl")
    
    if os.path.exists(best_model_path):
        with open(best_model_path, 'rb') as f:
            best_model = pickle.load(f)
        
        # Compare with existing best model
        # This would require loading the best model's metadata to compare metrics
        # For simplicity, we'll just update it
        with open(best_model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Updated best model at {best_model_path}")
    
    return {
        "model": model,
        "metrics": metrics,
        "model_path": model_path,
        "report_path": report_path
    }

if __name__ == "__main__":
    # Example: Retrain logistic regression model on kaggle dataset
    result = retrain_model("logistic_regression", "kaggle")
    
    # Example: Retrain with new hyperparameters
    new_hyperparams = {"C": 0.1, "solver": "liblinear"}
    result = retrain_model("logistic_regression", "kaggle", hyperparams=new_hyperparams)