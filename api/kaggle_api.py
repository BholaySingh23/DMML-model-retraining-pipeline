from fastapi import APIRouter, Header, HTTPException
import kaggle
import os
from utils.auth import auth_check
router = APIRouter()

RAW_DIR = "raw_data"
extract_dir_kaggle = RAW_DIR + "/kaggle"

@router.get("/download_kaggle")
def download_kaggle_dataset(x_api_key: str = Header(None)):
    auth_check(x_api_key)
    dataset = "blastchar/telco-customer-churn"
    os.makedirs(extract_dir_kaggle, exist_ok=True)
    try:
        kaggle.api.dataset_download_files(dataset, path=extract_dir_kaggle, unzip=True)
        # Rename the first CSV file to kaggle_curn.csv
        for file_name in os.listdir(extract_dir_kaggle):
            if file_name.endswith('.csv'):
                old_path = os.path.join(extract_dir_kaggle, file_name)
                new_path = os.path.join(extract_dir_kaggle, "kaggle_curn.csv")
                os.rename(old_path, new_path)
                break
        return {"status": "success", "message": f"Dataset downloaded and renamed to kaggle_curn.csv in {extract_dir_kaggle}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))