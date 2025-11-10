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

import os
import pandas as pd
import numpy as np
from pathlib import Path
import src.const as const
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def save_dataframe_as_parquet(
    df: pd.DataFrame, directory: str, filename: str = "data.parquet"
) -> None:
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    df.to_parquet(file_path, engine="pyarrow", index=False)


def get_data(parquet_file: str) -> pd.DataFrame:
    parquet_path = Path(parquet_file)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"File not found: {parquet_file}")
    return pd.read_parquet(parquet_path)


def prepare_data_sets(df: pd.DataFrame) -> dict:
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    names = list(const.datasets_names_values.keys())
    values = list(const.datasets_names_values.values())

    number_rows_total = len(df_shuffled)
    number_of_rows = {}

    total_assigned = 0
    for i in range(len(values) - 1):
        count = round(values[i] * number_rows_total)
        number_of_rows[names[i]] = count
        total_assigned += count

    number_of_rows[names[-1]] = number_rows_total - total_assigned

    ret = {}
    base = 0
    for name, count in number_of_rows.items():
        ret[name] = df_shuffled.iloc[base : base + count].copy()
        base += count

    return ret


def feature_split(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(label_col, axis=1)
    label = df[label_col]

    return (features, label)


def evaluate_nested_results(models_results: dict) -> pd.DataFrame:
    rows = []

    for model_name, configs in models_results.items():
        for hyperparams, (y_pred, y_true) in configs.items():
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            rows.append(
                {
                    "model": model_name,
                    "hyperparams": hyperparams,
                    "RMSE": rmse,
                    "MAE": mae,
                }
            )

    return pd.DataFrame(rows)


def plot_rmse_mae_side_by_side(df_metrics: pd.DataFrame) -> None:
    df_metrics["label"] = df_metrics.apply(
        lambda row: f"{row['model']}\n{row['hyperparams']}", axis=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.barplot(x="label", y="RMSE", data=df_metrics, palette="viridis", ax=axes[0])
    axes[0].set_title("RMSE for Different Models")
    axes[0].set_xlabel("Model and Hyperparameters")
    axes[0].set_ylabel("RMSE (dB)")
    axes[0].tick_params(axis="x", rotation=45)

    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height + 0.02 * height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            rotation=0,
        )

    sns.barplot(x="label", y="MAE", data=df_metrics, palette="magma", ax=axes[1])
    axes[1].set_title("MAE for Different Models")
    axes[1].set_xlabel("Model and Hyperparameters")
    axes[1].set_ylabel("MAE (dB)")
    axes[1].tick_params(axis="x", rotation=45)

    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height + 0.02 * height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            rotation=0,
        )

    plt.tight_layout()
    plt.show()


def compute_predictive_baseline(
    train_df: pd.DataFrame, label_col: str
) -> tuple[pd.Series, pd.Series]:
    y_train = train_df[label_col]
    features_df = train_df.drop(columns=[label_col])

    corrs = features_df.corrwith(y_train)
    best_feature = corrs.abs().idxmax()

    print(
        f"Using feature '{best_feature}' as baseline predictor (corr={corrs[best_feature]:.3f})"
    )

    baseline_pred = train_df[best_feature]

    return baseline_pred, y_train


def save_results(results: dict, filename: str) -> None:
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Resultados guardados en {os.path.abspath(filename)}")


def select_best_model(df_metrics: pd.DataFrame) -> pd.Series:
    total_rank = df_metrics["RMSE"].rank(method="min") + df_metrics["MAE"].rank(
        method="min"
    )
    best_idx = total_rank.idxmin()
    best_model_row = df_metrics.loc[best_idx, ["model", "hyperparams", "RMSE", "MAE"]]

    return best_model_row


def check_linear_dependence(df, label_col, feature_cols):
    results = {}

    for col in feature_cols:
        if df[col].isna().any():
            X = df[col].fillna(df[col].mean()).values.reshape(-1, 1)
        else:
            X = df[col].values.reshape(-1, 1)
        y = df[label_col].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        results[col] = r2_score(y, y_pred)

    X_multi = df[feature_cols].copy()
    X_multi = X_multi.fillna(X_multi.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_multi)
    y = df[label_col].values
    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred_multi = model.predict(X_scaled)
    r2_multi = r2_score(y, y_pred_multi)

    print("R^2 per feature:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")
    print(f"\nMultivariate R^2: {r2_multi:.3f}")

    return results, r2_multi


def calculate_accuracy_from_results(model_row: pd.Series, results_dict: dict) -> float:
    key = model_row["hyperparams"]

    if key not in results_dict:
        raise ValueError(f"Key {key} not found in results_dict.")

    y_pred, y_actual = results_dict[key]
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)

    mae = np.mean(np.abs(y_pred - y_actual))
    range_of_target = y_actual.max() - y_actual.min()
    accuracy = 100 * (1 - mae / range_of_target)

    return accuracy


def plot_model_accuracies(best_models_info):
    model_names = []
    accuracies = []

    for model_name, best_row, results_dict in best_models_info:
        try:
            acc = calculate_accuracy_from_results(best_row, results_dict)
        except ValueError as e:
            print(f"Skipping {model_name}: {e}")
            continue

        model_names.append(model_name)
        accuracies.append(acc)
        print(f"{model_name} accuracy: {acc:.2f}%")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        model_names, accuracies, color=["red", "blue", "green", "orange", "purple"]
    )
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Best Models Accuracy")

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.show()


def plot_best_models_accuracy(best_models_info, tolerance: float = 0.10):
    plt.figure(figsize=(10, 8))

    model_colors = ["red", "blue", "green", "orange", "purple"]
    markers = ["o", "s", "D", "^", "v"]

    for i, (model_name, best_row, results_dict) in enumerate(best_models_info):
        key = best_row["hyperparams"]
        if key not in results_dict:
            print(
                f"Warning: {model_name} key {key} not found in results_dict. Skipping."
            )
            continue

        y_pred, y_actual = results_dict[key]
        y_pred = np.array(y_pred).flatten()
        y_actual = np.array(y_actual).flatten()
        within_tolerance = np.abs(y_pred - y_actual) <= tolerance * np.abs(y_actual)

        plt.scatter(
            y_actual,
            y_pred,
            color=model_colors[i % len(model_colors)],
            marker=markers[i % len(markers)],
            alpha=0.6,
            edgecolors=["black" if wt else "gray" for wt in within_tolerance],
            s=60,
            label=model_name,
        )

    all_y_actuals = np.concatenate(
        [
            np.array(results_dict[best_row["hyperparams"]][1]).flatten()
            for _, best_row, results_dict in best_models_info
            if best_row["hyperparams"] in results_dict
        ]
    )
    min_val, max_val = all_y_actuals.min(), all_y_actuals.max()
    plt.plot(
        [min_val, max_val], [min_val, max_val], "k--", lw=2, label="Perfect Prediction"
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual for Best Models (Tolerance {tolerance*100:.0f}%)")
    plt.legend()
    plt.show()


def plot_models_separately(best_models_info):
    num_models = len(best_models_info)
    fig, axes = plt.subplots(num_models, 1, figsize=(12, 4 * num_models), sharex=True)

    # Use y_actual from the first model (same for all)
    _, first_best_row, first_results_dict = best_models_info[0]
    key = first_best_row["hyperparams"]
    y_actual = np.array(first_results_dict[key][1]).flatten()
    x_axis = np.arange(len(y_actual))

    colors = ["red", "blue", "green", "orange", "purple"]
    markers = ["o", "s", "D", "^", "v"]

    for i, (model_name, best_row, results_dict) in enumerate(best_models_info):
        ax = axes[i] if num_models > 1 else axes
        key = best_row["hyperparams"]
        if key not in results_dict:
            print(
                f"Warning: {model_name} key {key} not found in results_dict. Skipping."
            )
            continue

        y_pred = np.array(results_dict[key][0]).flatten()

        ax.plot(x_axis, y_actual, "k-", lw=2, label="Actual")
        ax.plot(
            x_axis,
            y_pred,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            lw=1,
            alpha=0.8,
            label=f"Predicted ({model_name})",
        )

        ax.set_ylabel("RSRP Value")
        ax.set_title(f"{model_name}")
        ax.legend()

    axes[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    plt.show()
