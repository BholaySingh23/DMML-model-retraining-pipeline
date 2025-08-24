doc = """
Customer Churn Prediction Pipeline

Steps:
1. Data Ingestion from Kaggle and Hugging Face
2. Local Storage
3. Data Validation
4. Data Preparation
5. Data Transformation
6. Feature Store
7. Data Versioning
8. Model Building
9. Pipeline Orchestration
10. Documentation

Business Objective:
Reduce addressable customer churn using predictive analytics.

Challenges:
- Data quality
- Feature drift
- API failures
- Model overfitting
- Dependency management
"""
with open("documentation/pipeline_documentation.txt", "w") as f:
    f.write(doc)