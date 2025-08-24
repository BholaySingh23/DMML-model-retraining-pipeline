import pandas as pd

df_kaggle = pd.read_csv("data_storage/kaggle_curn.csv")
df_hugging = pd.read_csv("data_storage/huggingface_churn.csv")

report = {}

for name, df in [("Kaggle", df_kaggle), ("HuggingFace", df_hugging)]:
    report[name] = {
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "negative_values": (df.select_dtypes(include=["number"]) < 0).sum().sum()
    }

report_df = pd.DataFrame.from_dict(report, orient="index")
report_df.to_excel("data_validation/data_validation_report.xlsx")