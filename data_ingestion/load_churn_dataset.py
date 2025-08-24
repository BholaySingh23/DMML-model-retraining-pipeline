import os
import logging
import pandas as pd
from datasets import load_dataset
import requests
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

RAW_DIR = "raw_data"
extract_dir_kaggle = RAW_DIR + "/kaggle"
extract_dir_hugging = RAW_DIR + "/huggingface"
os.makedirs(RAW_DIR, exist_ok=True)


def fetch_kaggle_dataset():
    logging.info("Starting Kaggle data ingestion...")
    api_url = "http://localhost:9001/download_kaggle"
    response = requests.get(api_url, headers={"x-api-key": "demo-key"})
    if response.status_code == 200:
        print("Success:", response.json())
        logging.info(response.json()["message"])
    else:
        print("Error:", response.status_code, response.text)

def fetch_huggingface_dataset():
    logging.info("Starting Hugging Face data ingestion...")
    dataset_name = "aai510-group1/telco-customer-churn"  # Use a valid Hugging Face dataset name
    dataset = load_dataset(dataset_name)
    dataset["train"].to_csv(f"{extract_dir_hugging}/huggingface_churn.csv", index=False)
    logging.info(f"Hugging Face data successfully downloaded and stored in {extract_dir_hugging}")

def download_datasets():
    logging.info("Starting data ingestion...")
    # fetching kaggle dataset via API
    fetch_kaggle_dataset()
    fetch_huggingface_dataset()
    logging.info("Data ingestion complete!")

if __name__ == "__main__":
    download_datasets()
