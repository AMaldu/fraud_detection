from pandas import DataFrame
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:

    # Renaming columns
    df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})

    # Let's transform type to category to optimize memory and speed
    df['step'] = df['step'].astype('int16')
    df[['type', 'nameOrig', 'nameDest']] = df[['type', 'nameOrig', 'nameDest']].astype('category')
    df[['amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']] = df[['amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']].astype('float32')



    return df