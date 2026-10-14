"""Model definitions: each model is a full pipeline from raw application data to a fraud score.

Every pipeline starts with the same two steps:
1. features   - turn negative "missing" codes into NaN and add <column>_missing flags
2. categories - cast categorical columns to a fixed set of categories, so category codes are
                identical in training and serving (where data arrives as JSON strings)

Then each model gets the preprocessing it needs:
- Logistic regression needs imputed, scaled numbers and one-hot encoded categories.
- LightGBM and XGBoost handle NaN and pandas categorical columns natively.
"""
from typing import Any, Callable

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from risk_platform.fraud.features import add_missing_flags
from risk_platform.fraud.schema import CATEGORIES


def cast_categories(X: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Cast columns to categoricals with fixed categories. Unknown values become NaN."""
    X = X.copy()
    for col, values in categories.items():
        X[col] = pd.Categorical(X[col], categories=values)
    return X


def _shared_steps(missing_columns: list[str]) -> list[tuple[str, FunctionTransformer]]:
    return [
        ("features", FunctionTransformer(add_missing_flags, kw_args={"columns": missing_columns})),
        ("categories", FunctionTransformer(cast_categories, kw_args={"categories": CATEGORIES})),
    ]


def _logistic_regression(params: dict[str, Any]) -> list[tuple[str, Any]]:
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                make_column_selector(dtype_include="number"),
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include="category")),
        ]
    )
    return [("preprocess", preprocess), ("model", LogisticRegression(**params))]


def _lightgbm(params: dict[str, Any]) -> list[tuple[str, Any]]:
    return [("model", LGBMClassifier(**params))]


def _xgboost(params: dict[str, Any]) -> list[tuple[str, Any]]:
    return [("model", XGBClassifier(enable_categorical=True, **params))]


BUILDERS: dict[str, Callable[[dict[str, Any]], list[tuple[str, Any]]]] = {
    "logistic_regression": _logistic_regression,
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
}


def build_pipeline(name: str, params: dict[str, Any], missing_columns: list[str]) -> Pipeline:
    if name not in BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose one of: {', '.join(BUILDERS)}")
    return Pipeline(_shared_steps(missing_columns) + BUILDERS[name](params))


# Optuna search spaces: each function returns the params to try in one trial

def _lightgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }


def _xgboost_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "tree_method": "hist",
        "random_state": 42,
    }


SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "lightgbm": _lightgbm_space,
    "xgboost": _xgboost_space,
}
