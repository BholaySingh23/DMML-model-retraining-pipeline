import shutil

shutil.copy("data_transformation/kaggle_transformed.csv", "versioned_data/kaggle_v1.csv")
shutil.copy("data_transformation/huggingface_transformed.csv", "versioned_data/huggingface_v1.csv")