from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from prefect import task


def preprocessor(
    df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    categorical_features = ["type", "nameDest"]
    numeric_features = [
        "step",
        "amount",
        "oldbalanceOrig",
        "oldbalanceDest",
        "diffbalanceOrig",
        "diffbalanceDest",
    ]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    X = df.drop(columns=target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_preprocessed = pipeline.fit_transform(X_train)
    X_test_preprocessed = pipeline.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test
