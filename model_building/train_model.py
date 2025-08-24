import pandas as pd
import numpy as np
import os
import pickle
import logging
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
logger = get_logger("train_model")

def train_model(dataset_name="kaggle", retrain=False):
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prep_dir = os.path.join(base_dir, "data_preparation")
    transform_dir = os.path.join(base_dir, "data_transformation")
    model_dir = os.path.join(base_dir, "model_building")
    report_dir = os.path.join(model_dir, "reports")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    # Load data based on dataset name
    logger.info(f"Training model using {dataset_name} dataset")
    
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
    
    # Train models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name} model")
        
        # Train the model
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
        report_path = os.path.join(report_dir, f"{model_name}_{dataset_name}_report.csv")
        report_df.to_csv(report_path)
        
        # Save model with versioning
        model_path, metadata_path = version_model(
            model, 
            model_name, 
            metrics, 
            dataset_name, 
            notes=f"Trained on {dataset_name} dataset"
        )
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metrics: {metrics}")
        
        results[model_name] = {
            "model": model,
            "metrics": metrics,
            "model_path": model_path,
            "report_path": report_path
        }
    
    # Determine best model based on F1 score
    best_model_name = max(results.keys(), key=lambda k: results[k]["metrics"]["f1"])
    best_model_info = results[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with F1 score: {best_model_info['metrics']['f1']}")
    
    # Save best model separately
    best_model_path = os.path.join(model_dir, "models", f"best_model_{dataset_name}.pkl")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model_info["model"], f)
    
    return results

if __name__ == "__main__":
    # Train models on both datasets
    kaggle_results = train_model(dataset_name="kaggle")
    huggingface_results = train_model(dataset_name="huggingface")