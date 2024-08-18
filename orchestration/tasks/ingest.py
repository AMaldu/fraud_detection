import os
import pandas as pd
import requests
from prefect import task, Flow
from datetime import timedelta


@task(retries=3, retry_delay=timedelta(seconds=10))
def ingest_files() -> pd.DataFrame:
    """Read data into dataframe"""
    file_path = "data/bronze/PS_20174392719_1491204439457_log.csv"

    df = pd.read_csv(file_path)

    return df
