
# Churn ML Data Pipeline (with FastAPI source)

- End-to-end pipeline: ingestion → validation → preparation → transformation/warehouse → training → orchestration.
- Synthetic data in `sources/`; first run ingests to `data/raw/`.


## Pipeline run
# Data ingestion
python -m churn_pipeline.data_ingestion.load_churn_dataset

# Data storage
python -m churn_pipeline.data_storage.store_data

# Data validation
python -m churn_pipeline.data_validation.validate_data

# Data preparation
python -m churn_pipeline.data_preparation.prepare_data

# Data transformation
python -m churn_pipeline.data_transformation.transform_data

# Feature storage
python -m churn_pipeline.feature_store.store_features
