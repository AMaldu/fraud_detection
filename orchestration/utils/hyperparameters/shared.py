from typing import Callable, Dict, List, Tuple, Union
from hyperopt import hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import LinearSVR
from xgboost import Booster
from prefect import task


def build_hyperparameters_space(
    model_class: Callable[
        ...,
        Union[
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
            RandomForestClassifier,
            Lasso,
            LinearRegression,
            LinearSVR,
            Booster,
        ],
    ],
    random_state: int = 42,
    **kwargs,
) -> Tuple[Dict, Dict[str, List]]:
    params = {}
    choices = {}

    if LinearSVR is model_class:
        params = dict(
            epsilon=hp.uniform("epsilon", 0.0, 1.0),
            C=hp.loguniform("C", -7, 3),
            max_iter=scope.int(hp.quniform("max_iter", 1000, 5000, 100)),
        )

    if RandomForestRegressor is model_class:
        params = dict(
            max_depth=scope.int(hp.quniform("max_depth", 5, 45, 5)),
            min_samples_leaf=scope.int(hp.quniform("min_samples_leaf", 1, 10, 1)),
            min_samples_split=scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
            n_estimators=scope.int(hp.quniform("n_estimators", 10, 60, 10)),
            random_state=random_state,
        )

    if RandomForestClassifier is model_class:
        params = dict(
            max_depth=scope.int(hp.quniform("max_depth", 5, 50, 5)),
            min_samples_leaf=scope.int(hp.quniform("min_samples_leaf", 1, 10, 1)),
            min_samples_split=scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
            n_estimators=scope.int(hp.quniform("n_estimators", 50, 200, 10)),
            criterion=hp.choice("criterion", ["gini", "entropy"]),
            max_features=hp.choice("max_features", ["sqrt", "log2"]),
            bootstrap=hp.choice("bootstrap", [True, False]),
            random_state=random_state,
        )

    if GradientBoostingRegressor is model_class:
        params = dict(
            learning_rate=hp.loguniform("learning_rate", -5, 0),
            max_depth=scope.int(hp.quniform("max_depth", 5, 40, 1)),
            min_samples_leaf=scope.int(hp.quniform("min_samples_leaf", 1, 10, 1)),
            min_samples_split=scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
            n_estimators=scope.int(hp.quniform("n_estimators", 10, 50, 10)),
            random_state=random_state,
        )

    if ExtraTreesRegressor is model_class:
        params = dict(
            max_depth=scope.int(hp.quniform("max_depth", 5, 30, 5)),
            min_samples_leaf=scope.int(hp.quniform("min_samples_leaf", 1, 10, 1)),
            min_samples_split=scope.int(hp.quniform("min_samples_split", 2, 20, 2)),
            n_estimators=scope.int(hp.quniform("n_estimators", 10, 40, 10)),
            random_state=random_state,
        )

    if Lasso is model_class:
        params = dict(
            alpha=hp.uniform("alpha", 0.0001, 1.0),
            max_iter=scope.int(hp.quniform("max_iter", 1000, 5000, 100)),
        )

    if LinearRegression is model_class:
        choices["fit_intercept"] = [True, False]

    if Booster is model_class:
        params = dict(
            colsample_bytree=hp.uniform("colsample_bytree", 0.5, 1.0),
            gamma=hp.uniform("gamma", 0.1, 1.0),
            learning_rate=hp.loguniform("learning_rate", -3, 0),
            max_depth=scope.int(hp.quniform("max_depth", 4, 100, 1)),
            min_child_weight=hp.loguniform("min_child_weight", -1, 3),
            num_boost_round=hp.quniform("num_boost_round", 500, 1000, 10),
            objective="reg:squarederror",
            random_state=random_state,
            reg_alpha=hp.loguniform("reg_alpha", -5, -1),
            reg_lambda=hp.loguniform("reg_lambda", -6, -1),
            subsample=hp.uniform("subsample", 0.1, 1.0),
        )

    for key, value in choices.items():
        params[key] = hp.choice(key, value)

    if kwargs:
        for key, value in kwargs.items():
            if value is not None:
                kwargs[key] = value

    return params, choices
