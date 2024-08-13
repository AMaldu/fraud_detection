from typing import Tuple
import pandas as pd
from mlops.utils.data_preparation.cleaning import clean
from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.data_preparation.feature_selection import select_features

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    # Extraer parámetros
    target = kwargs.get('target')

    # Aplicar limpieza y ingeniería de características
    df = clean(df)
    df = combine_features(df)
    df = select_features(df, target = target)  # Ajustar si es necesario

    return df



