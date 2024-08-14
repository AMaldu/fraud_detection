from prefect import Flow
from orchestration.tasks.ingest import ingest
from orchestration.utils.clean import clean
from orchestration.utils. feature_engineering import combine_features
from orchestration.utils.feature_selection import select_features
from orchestration.utils.preprocessing import preprocessor

with Flow("fraud_detection_pipeline") as flow:
    df = ingest_files()  
    df_cleaned = clean(df)
    df_combined = combine_features(df_cleaned)
    df_selected = select_features(df_combined, target='isFraud')
    X_train, X_test, y_train, y_test = preprocessor(df_selected, target='isFraud')


if __name__ == "__main__":
    flow.run()  