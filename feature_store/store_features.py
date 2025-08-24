import pandas as pd
import sqlite3

df_kaggle = pd.read_csv("data_transformation/kaggle_transformed.csv")
df_hugging = pd.read_csv("data_transformation/huggingface_transformed.csv")

conn = sqlite3.connect("feature_store/feature_store.db")
df_kaggle.to_sql("kaggle_features", conn, if_exists="replace", index=False)
df_hugging.to_sql("huggingface_features", conn, if_exists="replace", index=False)
conn.close()