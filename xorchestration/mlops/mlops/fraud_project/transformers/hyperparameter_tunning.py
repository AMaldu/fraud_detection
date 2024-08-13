from typing import Dict, Union, Tuple, Callable
from pandas import Series
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from mlops.utils.models.sklearn_classification import load_class, tune_hyperparameters

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def hyperparameter_tuning(
    training_set_data: Dict[str, list],  # Recibe el diccionario que contiene splitting_data
    model_class_name: str,
    *args,
    **kwargs,
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series,
    Callable[..., BaseEstimator],
]:
    # Extraer splitting_data de training_set_data
    splitting_data = training_set_data.get('splitting_data', None)
    
    if splitting_data and len(splitting_data) == 4:
        X_train, X_test, y_train, y_test = splitting_data
    else:
        raise ValueError("El formato de splitting_data no es válido o faltan datos.")

    # Imprimir tipos y tamaños para verificar
    print("X_train type:", type(X_train), "Tamaño:", X_train.shape)
    print("X_test type:", type(X_test), "Tamaño:", X_test.shape)
    print("y_train type:", type(y_train), "Tamaño:", y_train.shape)
    print("y_test type:", type(y_test), "Tamaño:", y_test.shape)
    print("model_class_name:", model_class_name)

    # Cargar la clase del modelo
    model_class = load_class(model_class_name)

    # Ajustar hiperparámetros utilizando la función tune_hyperparameters
    hyperparameters = tune_hyperparameters(
        model_class=model_class,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        max_evaluations=kwargs.get('max_evaluations'),
        random_state=kwargs.get('random_state'),
    )

    # Devolver los hiperparámetros optimizados, junto con X_train, y_train, y la información del modelo
    return hyperparameters, X_train, y_train, dict(cls=model_class, name=model_class_name)
