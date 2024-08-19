from pandas import DataFrame


def combine_features(df: DataFrame) -> DataFrame:
    # Calculating balance differences
    df["diffbalanceOrig"] = df["newbalanceOrig"] - df["oldbalanceOrig"]
    df["diffbalanceDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Dropping unnecessary columns
    df = df.drop(
        columns=[
            "newbalanceOrig",
            "nameOrig",
            "newbalanceDest",
            "isFlaggedFraud",
        ]
    )

    return df
