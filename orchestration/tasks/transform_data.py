from typing import Tuple
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from prefect import task

from orchestration.utils.data_preparation import preprocessor

@task
def preprocess(
    df: pd.DataFrame, 
    target: str = 'target',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = preprocessor(
        df,
        target=target,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# @task
# def test_training_set(
#     X: csr_matrix,
#     X_train: csr_matrix,
#     y: np.ndarray,
#     y_train: np.ndarray
# ) -> None:
#     """
#     Validate the training set to ensure it meets the expected properties.

#     Args:
#     - X (csr_matrix): The complete feature matrix.
#     - X_train (csr_matrix): The training feature matrix.
#     - y (np.ndarray): The complete target vector.
#     - y_train (np.ndarray): The training target vector.
#     """
#     # Verificar que el tamaño del conjunto de entrenamiento es 2000
#     expected_train_size = 2000
#     assert (
#         X_train.shape[0] == expected_train_size
#     ), f'Training set should have {expected_train_size} examples, but has {X_train.shape[0]}'
    
#     # Verificar que el número de características es el mismo en el conjunto de entrenamiento y en el total
#     assert (
#         X_train.shape[1] == X.shape[1]
#     ), f'Training set should have {X.shape[1]} features, but has {X_train.shape[1]}'
    
#     # Verificar que el tamaño de y_train coincide con el tamaño de X_train
#     assert (
#         len(y_train) == X_train.shape[0]
#     ), f'Training set should have {X_train.shape[0]} examples, but has {len(y_train)}'


# @task
# def test_test_set(
#     X: csr_matrix,
#     X_test: csr_matrix,
#     y: np.ndarray,
#     y_test: np.ndarray
# ) -> None:
#     """
#     Validate the test set to ensure it meets the expected properties.

#     Args:
#     - X (csr_matrix): The complete feature matrix.
#     - X_test (csr_matrix): The test feature matrix.
#     - y (np.ndarray): The complete target vector.
#     - y_test (np.ndarray): The test target vector.
#     """
#     # Verificar que el tamaño del conjunto de prueba es 2000
#     expected_test_size = 2000
#     assert (
#         X_test.shape[0] == expected_test_size
#     ), f'Test set should have {expected_test_size} examples, but has {X_test.shape[0]}'
    
#     # Verificar que el número de características es el mismo en el conjunto de prueba y en el total
#     assert (
#         X_test.shape[1] == X.shape[1]
#     ), f'Test set should have {X.shape[1]} features, but has {X_test.shape[1]}'
    
#     # Verificar que el tamaño de y_test coincide con el tamaño de X_test
#     assert (
#         len(y_test) == X_test.shape[0]
#     ), f'Test set should have {X_test.shape[0]} examples, but has {len(y_test)}'
