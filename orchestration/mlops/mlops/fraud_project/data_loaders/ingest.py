import requests
import pandas as pd
import os

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def ingest_files() -> pd.DataFrame:
    file_path = '/home/src/mlops/fraud_project/raw_data/fraud_dataset.csv'
    
    df = pd.read_csv(file_path, nrows = 10000)
    
    return df