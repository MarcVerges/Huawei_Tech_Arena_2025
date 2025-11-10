"""
Module Information
------------------
Author: Marc Vergés Sánchez
Created on: November 1, 2025

Description:
    This module was created by Marc Vergés Sánchez on November 1, 2025.
    For details about the project's purpose and usage, refer to the README file.

Dependencies:
    Dependencies are managed using UV.
    Check the following for dependency and environment information:
        - pyproject.toml
        - Python version (managed with UV)

Notes:
    Ensure that your environment matches the configurations specified in
    pyproject.toml and the README before running or modifying this module.
"""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge as SklearnRidge, Lasso as SklearnLasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd


def Ridge(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    alphas: list,
    train_cols: list,
) -> dict:
    results = {}

    for alpha in alphas:
        pipe = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            SklearnRidge(alpha=alpha, max_iter=10000),
        )

        pipe.fit(x_train[train_cols], y_train)
        y_pred = pipe.predict(x_val[train_cols])

        results[alpha] = (y_pred, y_val)

    return results


def Lasso(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    alphas: list,
    train_cols: list,
) -> dict:
    results = {}

    for alpha in alphas:
        pipe = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            SklearnLasso(alpha=alpha, max_iter=10000),
        )

        pipe.fit(x_train[train_cols], y_train)
        y_pred = pipe.predict(x_val[train_cols])

        results[alpha] = (y_pred, y_val)

    return results
