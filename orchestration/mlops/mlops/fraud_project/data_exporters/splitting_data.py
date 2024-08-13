from typing import Tuple
import pandas as pd
from scipy.sparse import csr_matrix
from pandas import Series
import numpy as np

from mlops.utils.data_preparation.preprocess import preprocessor


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test



@data_exporter
def preprocess(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Asegurarse de que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"El argumento df debe ser un DataFrame de pandas. Tipo recibido: {type(df)}")
    
    # Extraer parámetros
    target = kwargs.get('target', 'target')  # Nombre de la columna objetivo
    test_size = kwargs.get('test_size', 0.2)  # Tamaño del conjunto de prueba
    random_state = kwargs.get('random_state', 42)  # Semilla para aleatorización

    # Llamar a la función preprocessor
    X_train, X_test, y_train, y_test = preprocessor(
        df,
        target=target,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

@test
def test_training_set(
    X: csr_matrix,
    X_train: csr_matrix,
    y: np.ndarray,
    y_train: np.ndarray,
    *args,
) -> None:
    # Verificar que el tamaño del conjunto de entrenamiento es 2000
    expected_train_size = 2000
    assert (
        X_train.shape[0] == expected_train_size
    ), f'Training set should have {expected_train_size} examples, but has {X_train.shape[0]}'
    
    # Verificar que el número de características es el mismo en el conjunto de entrenamiento y en el total
    assert (
        X_train.shape[1] == X.shape[1]
    ), f'Training set should have {X.shape[1]} features, but has {X_train.shape[1]}'
    
    # Verificar que el tamaño de y_train coincide con el tamaño de X_train
    assert (
        len(y_train) == X_train.shape[0]
    ), f'Training set should have {X_train.shape[0]} examples, but has {len(y_train)}'


@test
def test_test_set(
    X: csr_matrix,
    X_test: csr_matrix,
    y: np.ndarray,
    y_test: np.ndarray,
    *args,
) -> None:
    # Verificar que el tamaño del conjunto de prueba es 2000
    expected_test_size = 2000
    assert (
        X_test.shape[0] == expected_test_size
    ), f'Test set should have {expected_test_size} examples, but has {X_test.shape[0]}'
    
    # Verificar que el número de características es el mismo en el conjunto de prueba y en el total
    assert (
        X_test.shape[1] == X.shape[1]
    ), f'Test set should have {X.shape[1]} features, but has {X_test.shape[1]}'
    
    # Verificar que el tamaño de y_test coincide con el tamaño de X_test
    assert (
        len(y_test) == X_test.shape[0]
    ), f'Test set should have {X_test.shape[0]} examples, but has {len(y_test)}'
