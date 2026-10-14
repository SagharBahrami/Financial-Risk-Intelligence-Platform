"""Optuna pruning callbacks for gradient boosting models, shared by all use cases.

Every `every` boosting rounds, the callback scores the model built so far on the validation
data, reports the score to Optuna, and stops training if the pruner decides the trial is
unlikely to beat earlier trials (optuna.TrialPruned is raised).
"""
from typing import Callable

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

ScoreFn = Callable[[np.ndarray], float]  # validation scores -> metric value (higher is better)


def _report(trial: optuna.Trial, value: float, step: int) -> None:
    trial.report(value, step)
    if trial.should_prune():
        raise optuna.TrialPruned(f"pruned at round {step} (value {value:.4f})")


def lightgbm_pruning_callback(trial: optuna.Trial, X_val: pd.DataFrame, score_fn: ScoreFn, every: int):
    """Callback for LGBMClassifier.fit(callbacks=[...]). X_val must already be preprocessed."""

    def callback(env) -> None:
        rounds = env.iteration + 1
        if rounds % every == 0:
            _report(trial, score_fn(env.model.predict(X_val, num_iteration=rounds)), rounds)

    return callback


class XGBoostPruningCallback(xgb.callback.TrainingCallback):
    """Callback for XGBClassifier(callbacks=[...]). X_val must already be preprocessed."""

    def __init__(self, trial: optuna.Trial, X_val: pd.DataFrame, score_fn: ScoreFn, every: int):
        super().__init__()
        self.trial = trial
        self.dval = xgb.DMatrix(X_val, enable_categorical=True)
        self.score_fn = score_fn
        self.every = every

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        rounds = epoch + 1
        if rounds % self.every == 0:
            _report(self.trial, self.score_fn(model.predict(self.dval, iteration_range=(0, rounds))), rounds)
        return False  # False = continue training
