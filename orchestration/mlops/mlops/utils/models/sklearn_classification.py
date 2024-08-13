from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import sklearn
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from mlops.utils.hyperparameters.shared import build_hyperparameters_space

HYPERPARAMETERS_WITH_CHOICE_INDEX = [
    'criterion',
    'max_features',
    'bootstrap',
]

def load_class(module_and_class_name: str) -> BaseEstimator:
    parts = module_and_class_name.split('.')
    cls = sklearn
    for part in parts:
        cls = getattr(cls, part)
    return cls

def train_model(
    model: BaseEstimator,
    X_train: csr_matrix,
    y_train: Series,
    X_test: Optional[csr_matrix] = None,
    fit_params: Optional[Dict] = None,
    y_test: Optional[Series] = None,
    **kwargs,
) -> Tuple[BaseEstimator, Optional[Dict], Optional[np.ndarray]]:
    model.fit(X_train, y_train, **(fit_params or {}))

    metrics = None
    y_pred = None
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred, average='binary') 
        recall = recall_score(y_test, y_pred, average='binary')
        metrics = dict(precision=precision, recall=recall)

    return model, metrics, y_pred

def tune_hyperparameters(
    model_class: Callable[..., BaseEstimator],
    X_train: csr_matrix,
    y_train: Series,
    X_test: csr_matrix,
    y_test: Series,
    callback: Optional[Callable[..., None]] = None,
    fit_params: Optional[Dict] = None,
    hyperparameters: Optional[Dict] = None,
    max_evaluations: int = 50,
    random_state: int = 42,
    precision_weight: float = 0.5,  
    recall_weight: float = 0.5,
) -> Dict:
    def __objective(
        params: Dict,
        X_train=X_train,
        X_test=X_test,
        callback=callback,
        fit_params=fit_params,
        model_class=model_class,
        y_train=y_train,
        y_test=y_test,
    ) -> Dict[str, Union[float, str]]:
        model, metrics, predictions = train_model(
            model_class(**params),
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
            fit_params=fit_params,
        )

        if callback:
            callback(
                hyperparameters=params,
                metrics=metrics,
                model=model,
                predictions=predictions,
            )

        loss = -(precision_weight * metrics['precision'] + recall_weight * metrics['recall'])  
        return dict(loss=loss, status=STATUS_OK)

    space, choices = build_hyperparameters_space(
        model_class,
        random_state=random_state,
        **(hyperparameters or {}),
    )

    best_hyperparameters = fmin(
        fn=__objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evaluations,
        trials=Trials(),
    )

    # Convert choice index to choice value
    for key in HYPERPARAMETERS_WITH_CHOICE_INDEX:
        if key in best_hyperparameters and key in choices:
            idx = int(best_hyperparameters[key])
            best_hyperparameters[key] = choices[key][idx]

    # Convert hyperparameter values to integers if necessary
    for key in [
        'max_depth',
        'min_samples_leaf',
        'min_samples_split',
        'n_estimators',
    ]:
        if key in best_hyperparameters:
            best_hyperparameters[key] = int(best_hyperparameters[key])

    # Ensure 'bootstrap' is a boolean
    if 'bootstrap' in best_hyperparameters:
        best_hyperparameters['bootstrap'] = bool(best_hyperparameters['bootstrap'])

    return best_hyperparameters
