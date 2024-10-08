# pylint: disable=all
import os
import pickle
from typing import Tuple

import category_encoders as ce
import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from prefect import Flow, task
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@task
def ingest_files() -> pd.DataFrame:
    """Read data into dataframe"""
    file_path = "data/bronze/PS_20174392719_1491204439457_log.csv"

    df = pd.read_csv(file_path)

    step_counts = df["step"].value_counts()
    valid_steps = step_counts[step_counts >= 2].index
    filtered_df = df[df["step"].isin(valid_steps)]
    df = filtered_df.groupby("step").sample(n=2, random_state=42)

    return df


@task
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"oldbalanceOrg": "oldbalanceOrig"})

    df["step"] = df["step"].astype("int16")
    df[["type", "nameOrig", "nameDest"]] = df[["type", "nameOrig", "nameDest"]].astype(
        "category"
    )
    df[
        [
            "amount",
            "oldbalanceOrig",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
        ]
    ] = df[
        [
            "amount",
            "oldbalanceOrig",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
        ]
    ].astype(
        "float32"
    )

    return df


@task
def combine_features(df: DataFrame) -> DataFrame:
    df["diffbalanceOrig"] = df["newbalanceOrig"] - df["oldbalanceOrig"]
    df["diffbalanceDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df = df.drop(columns=["isFlaggedFraud"])

    return df


@task
def select_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df_copy = df.copy()

    numerical_cols = df_copy.select_dtypes(
        include=["float32", "int16", "int8", "int64", "float64"]
    ).columns
    categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns

    if target not in df_copy.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame columns.")

    if target in numerical_cols:
        numerical_cols = numerical_cols[numerical_cols != target]

    encoder = ce.TargetEncoder(cols=categorical_cols)
    df_encoded = encoder.fit_transform(df_copy, df_copy[target])

    correlation_matrix = df_encoded.corr()

    threshold = 0.9
    high_correlation_pairs = [
        (col1, col2)
        for col1 in correlation_matrix.columns
        for col2 in correlation_matrix.columns
        if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > threshold
    ]

    variables_to_remove = set()
    for col1, col2 in high_correlation_pairs:
        if col1 != target and col2 != target:
            if col1 not in variables_to_remove and col2 not in variables_to_remove:
                variables_to_remove.add(col2)

    df = df_copy.drop(columns=variables_to_remove, errors="ignore")

    return df


@task
def preprocessor(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series]:

    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_features = df.select_dtypes(include=["int", "float"]).columns.tolist()
    numeric_features.remove(target)

    print(f"Categorical features: {categorical_features}")
    print(f"Numeric features: {numeric_features}")

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X = df.drop(columns=target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train = preprocessor.fit_transform(X_train)

    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    X_test = preprocessor.transform(X_test)

    preprocessor_path = "orchestration/models/preprocessor_pipeline.b"
    with open(preprocessor_path, "wb") as f_out:
        pickle.dump(preprocessor, f_out)

    return X_train, X_test, y_train, y_test


@task(log_prints=True)
def train_best_model(
    X_train: csr_matrix,
    X_test: csr_matrix,
    y_train: pd.Series,
    y_test: pd.Series,
):
    model = RandomForestClassifier(
        n_estimators=190, max_depth=45, min_samples_split=7, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    os.makedirs("orchestration/models", exist_ok=True)
    with open("orchestration/models/random_forest_classifier.b", "wb") as f_out:
        pickle.dump(model, f_out)

    return None


@Flow
def main_flow():
    df = ingest_files()
    df_cleaned = clean(df)
    df_combined = combine_features(df_cleaned)
    df_selected = select_features(df_combined, target="isFraud")
    X_train, X_test, y_train, y_test = preprocessor(df_selected, target="isFraud")
    train_best_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main_flow()
