import datetime
import time
import random
import logging
import pandas as pd
import psycopg
import joblib
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from datetime import datetime, timedelta
import pytz

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists monitoring_metrics;
create table monitoring_metrics(
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_val float,
    date timestamp
)
"""

# Loading the data and model
data = pd.read_parquet("../data/gold/df_fraud_final.parquet", engine="fastparquet")

reference_data = data.iloc[:1000]
raw_data = data.iloc[-100:]

model_path = "../orchestration/models/random_forest_classifier.b"
with open(model_path, "rb") as f_in:
    model = joblib.load(f_in)

prep_path = "../orchestration/models/preprocessor_pipeline.b"
with open(prep_path, "rb") as f_in:
    preprocessor = joblib.load(f_in)

# Set the monitoring time and config of column mapping
begin = datetime.now()
num_features = ["diffbalanceOrig", "amount"]
cat_features = ["isFraud"]
column_mapping = ColumnMapping(
    target=None,
    prediction="isFraud",
    numerical_features=num_features,
    categorical_features=cat_features,
)

# Report config
report = Report(
    metrics=[
        ColumnDriftMetric(column_name="isFraud"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


# db preparation
def prep_db():
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE test;")
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(create_table_statement)


def generate_synthetic_dates(df, start_date, hours_interval=1):
    num_rows = len(df)
    date_range = [
        start_date + timedelta(hours=i * hours_interval) for i in range(num_rows)
    ]
    df["date"] = date_range
    return df


def calculate_metrics_postgresql(curr, step):
    try:
        logging.info(f"Processing step {step}")

        if raw_data.empty:
            logging.warning(f"No data found for step {step}")
            return

        logging.info("Data loaded, applying preprocessor")
        preprocessed_data = preprocessor.transform(raw_data)

        logging.info("Making predictions")
        raw_data["prediction"] = model.predict(preprocessed_data)

        # Genera fechas sintéticas cada hora en punto
        start_date = datetime.now().replace(
            minute=0, second=0, microsecond=0
        ) - timedelta(days=5)
        current_data = generate_synthetic_dates(raw_data, start_date)

        logging.info("Generating report")
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )
        result = report.as_dict()

        logging.info("Extracting results")
        prediction_drift = result["metrics"][0]["result"]["drift_score"]
        num_drifted_columns = result["metrics"][1]["result"][
            "number_of_drifted_columns"
        ]
        share_missing_values = result["metrics"][2]["result"]["current"][
            "share_of_missing_values"
        ]

        logging.info("Inserting data into database")
        for _, row in raw_data.iterrows():
            curr.execute(
                "INSERT INTO monitoring_metrics( prediction_drift, num_drifted_columns, share_missing_val, date) VALUES ( %s, %s, %s, %s)",
                (
                    prediction_drift,
                    num_drifted_columns,
                    share_missing_values,
                    row["date"],
                ),
            )

        logging.info(f"Data inserted for step {step}")

    except Exception as e:
        logging.error(f"Error processing step {step}: {e}")


def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.now() - timedelta(seconds=10)

    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        for time_value in range(1, 8):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, time_value)

            # Control del tiempo de envío
            new_send = datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                sleep_time = SEND_TIMEOUT - seconds_elapsed
                time.sleep(sleep_time)
            while last_send < new_send:
                last_send = last_send + timedelta(seconds=10)
            logging.info(f"Data sent for time {time_value}")


if __name__ == "__main__":
    batch_monitoring_backfill()
