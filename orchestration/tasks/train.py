from typing import Callable, Dict, Tuple, Union
from pandas import Series
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from prefect import task

from mlops.utils.models.sklearn_classification import load_class, train_model

@task
def train(
    settings: Tuple[
        Dict[str, Union[bool, float, int, str]],  # Hiperparámetros
        csr_matrix,
        Series,
        Dict[str, Union[Callable[..., BaseEstimator], str]],  # Información del modelo
    ],
    **kwargs,
) -> Tuple[BaseEstimator, Dict[str, str]]:
    hyperparameters, X, y, model_info = settings


    # if 'criterion' in hyperparameters:
    #     criterion_values = ['gini', 'entropy']
    #     if isinstance(hyperparameters['criterion'], int):
    #         hyperparameters['criterion'] = criterion_values[hyperparameters['criterion']]
    
    # if 'bootstrap' in hyperparameters:
    #     bootstrap_values = [True, False]
    #     if isinstance(hyperparameters['bootstrap'], int):
    #         hyperparameters['bootstrap'] = bootstrap_values[hyperparameters['bootstrap']]
    
    # if 'max_features' in hyperparameters:
    #     max_features_values = ['sqrt', 'log2']
    #     if isinstance(hyperparameters['max_features'], int):
    #         if hyperparameters['max_features'] < len(max_features_values):
    #             hyperparameters['max_features'] = max_features_values[hyperparameters['max_features']]
    #         else:
    #             raise ValueError("Índice de 'max_features' fuera de rango.")
    
    model_class = model_info['cls']
    model = model_class(**hyperparameters)

    model, metrics, _ = train_model(model, X, y)

    return model, model_info
