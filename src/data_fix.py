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
import numpy as np


def remove_fixed_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        "Unnamed: 0",
        "Unnamed: 1",
        "Unnamed: 2",
        "Unnamed: 3",
        "Unnamed: 4",
        "Unnamed: 5",
        "Unnamed: 6",
        "Unnamed: 7",
        "Unnamed: 8",
        "Unnamed: 9",
        "Unnamed: 10",
        "Unnamed: 11",
        "Unnamed: 12",
        "Unnamed: 13",
        "Unnamed: 14",
        "Unnamed: 15",
        "Unnamed: 18",
        "Unnamed: 19",
        "Unnamed: 21",
        "Unnamed: 23",
        "Unnamed: 28",
    ]

    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]

    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
    else:
        print("No se encontraron columnas para eliminar.")

    return df


def add_derived_telecom_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()

    azimuth_rad = df_new["Azimuth"] * (np.pi / 180.0)
    df_new["Azimuth_sin"] = np.sin(azimuth_rad)
    df_new["Azimuth_cos"] = np.cos(azimuth_rad)

    min_dist = 1e-6
    safe_dis_serving = np.where(
        df_new["Dis_Serving"] == 0, min_dist, df_new["Dis_Serving"]
    )
    df_new["RSRQ_per_Distance"] = df_new["Serving Cell RSRQ"] / safe_dis_serving
    df_new["SINR_per_Distance"] = df_new["RS SINR Carrier 1"] / safe_dis_serving

    return df_new


def general_fix(df: pd.DataFrame) -> pd.DataFrame:
    fix1 = remove_fixed_empty_columns(df)
    fix2 = add_derived_telecom_features(fix1)

    return fix2
