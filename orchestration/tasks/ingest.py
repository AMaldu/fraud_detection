import os
import pandas as pd
import requests
from prefect import task, Flow
from datetime import timedelta


@task(retries=3, retry_delay=timedelta(seconds=10))
def ingest_files() -> pd.DataFrame:
    file_path = '/home/src/mlops/fraud_project/raw_data/fraud_dataset.csv'
    
    df = pd.read_csv(file_path)
    
    return df
