import shutil
import os

src_kaggle = "raw_data/kaggle/kaggle_curn.csv"
dst_kaggle = "data_storage/kaggle_curn.csv"

src_hf = "raw_data/huggingface/huggingface_churn.csv"
dst_hf = "data_storage/huggingface_churn.csv"

for src, dst in [(src_kaggle, dst_kaggle), (src_hf, dst_hf)]:
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Source file not found: {src}")