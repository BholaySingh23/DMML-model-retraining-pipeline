from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from datetime import datetime, timedelta
import subprocess
import os

def run_script(script_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, script_path)
    subprocess.run(["python", full_path], check=True)

def check_model_performance():
    """Check if model needs retraining based on performance metrics"""
    # This would typically involve checking model metrics against a threshold
    # For simplicity, we'll just return the retrain path
    return 'retrain_model'

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

with DAG(
    dag_id='churn_pipeline',
    default_args=default_args,
    schedule_interval='@daily',  # Run daily
    catchup=False,
    description='Customer churn prediction pipeline with model versioning and retraining'
) as dag:

    ingest_data = PythonOperator(
        task_id='ingest_data',
        python_callable=run_script,
        op_args=['data_ingestion/load_churn_dataset.py']
    )

    store_raw_data = PythonOperator(
        task_id='store_raw_data',
        python_callable=run_script,
        op_args=['data_storage/store_data.py']
    )

    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=run_script,
        op_args=['data_validation/validate_data.py']
    )

    prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=run_script,
        op_args=['data_preparation/prepare_data.py']
    )

    transform_data = PythonOperator(
        task_id='transform_data',
        python_callable=run_script,
        op_args=['data_transformation/transform_data.py']
    )

    store_features = PythonOperator(
        task_id='store_features',
        python_callable=run_script,
        op_args=['feature_store/store_features.py']
    )

    version_data = PythonOperator(
        task_id='version_data',
        python_callable=run_script,
        op_args=['versioned_data/version_data.py']
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=run_script,
        op_args=['model_building/train_model.py']
    )

    # Branch to decide if retraining is needed
    check_performance = BranchPythonOperator(
        task_id='check_performance',
        python_callable=check_model_performance,
    )

    # Retrain model if needed
    retrain_model = PythonOperator(
        task_id='retrain_model',
        python_callable=run_script,
        op_args=['model_building/retrain_model.py']
    )

    # Continue with the rest of the pipeline
    dag_simulation = PythonOperator(
        task_id='dag_simulation',
        python_callable=run_script,
        op_args=['pipeline_orchestration/dag_simulation.py']
    )

    pipeline_documentation = PythonOperator(
        task_id='pipeline_documentation',
        python_callable=run_script,
        op_args=['documentation/pipeline_documentation.py']
    )

    # Define task dependencies for the main pipeline
    ingest_data >> store_raw_data >> validate_data >> prepare_data >> transform_data >> store_features
    store_features >> version_data >> train_model >> check_performance
    
    # Define branching logic for model retraining
    check_performance >> retrain_model >> dag_simulation
    check_performance >> dag_simulation  # Skip retraining if not needed
    
    # Final documentation step
    dag_simulation >> pipeline_documentation
