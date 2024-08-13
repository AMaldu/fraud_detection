from typing import Callable, Dict, Tuple, Union
from pandas import Series
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from mlops.utils.models.sklearn_classification import load_class, train_model

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
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

    # Convertir índices en valores si es necesario
    if 'criterion' in hyperparameters:
        criterion_values = ['gini', 'entropy']
        if isinstance(hyperparameters['criterion'], int):
            hyperparameters['criterion'] = criterion_values[hyperparameters['criterion']]
    
    if 'bootstrap' in hyperparameters:
        bootstrap_values = [True, False]
        if isinstance(hyperparameters['bootstrap'], int):
            hyperparameters['bootstrap'] = bootstrap_values[hyperparameters['bootstrap']]
    
    if 'max_features' in hyperparameters:
        max_features_values = ['sqrt', 'log2']
        if isinstance(hyperparameters['max_features'], int):
            if hyperparameters['max_features'] < len(max_features_values):
                hyperparameters['max_features'] = max_features_values[hyperparameters['max_features']]
            else:
                raise ValueError("Índice de 'max_features' fuera de rango.")
    
    # Cargar la clase del modelo
    model_class = model_info['cls']
    model = model_class(**hyperparameters)
    model.fit(X, y)

    return model, model_info
