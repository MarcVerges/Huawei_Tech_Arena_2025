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
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def generate_rf_sample(
    x_train: pd.DataFrame, y_train: pd.Series, sample_size_tree: float
):
    n_samples = int(len(x_train) * sample_size_tree)
    sample_indices = x_train.sample(n=n_samples, replace=True, random_state=None).index
    return x_train.loc[sample_indices], y_train.loc[sample_indices]


def random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    number_of_trees: int,
    varaible_at_split: str,
    sample_size_tree: float,
    train_cols: list,
) -> dict:
    x_boot, y_boot = generate_rf_sample(x_train, y_train, sample_size_tree)
    start_time = time.time()

    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=number_of_trees,
            max_features=varaible_at_split,
            n_jobs=-1,
            random_state=42,
            bootstrap=True,
        ),
    )

    model.fit(x_boot[train_cols], y_boot)
    y_pred = pd.Series(model.predict(x_val[train_cols]), index=y_val.index)

    elapsed_time = time.time() - start_time
    print(
        f"Finished Random Forest: trees={number_of_trees}, features={varaible_at_split}, sample={sample_size_tree:.2f} | Time: {elapsed_time:.2f} s\n"
    )

    return (y_pred, y_val)


def gradient_boosting_trees(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    number_of_trees: int,
    learning_rate: float,
    tree_depth: int,
    regularization_lambda: float,
    train_cols: list,
) -> tuple:
    start_time = time.time()

    model = make_pipeline(
        StandardScaler(),
        HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_depth=tree_depth,
            max_iter=number_of_trees,
            l2_regularization=regularization_lambda,
            random_state=42,
        ),
    )

    model.fit(x_train[train_cols], y_train)
    y_pred = pd.Series(model.predict(x_val[train_cols]), index=y_val.index)

    elapsed_time = time.time() - start_time

    print(
        f"Finished HistGradientBoosting: trees={number_of_trees}, "
        f"lr={learning_rate}, depth={tree_depth}, reg={regularization_lambda:.4f} | "
        f"Time: {elapsed_time:.2f}s\n"
    )

    return y_pred, y_val
