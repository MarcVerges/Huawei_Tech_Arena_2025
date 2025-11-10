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
import pathlib as pth


def general_check(dataset: pd.DataFrame) -> list:
    """Return basic info about the dataset."""
    if dataset.empty:
        return [True]
    rows, columns = dataset.shape
    return [False, rows, columns, str(type(dataset))]


def type_NaN_data(dataset: pd.DataFrame) -> list[dict]:
    """Return [ {column: dtype}, {column: NaN_count} ]"""
    return [dataset.dtypes.astype(str).to_dict(), dataset.isna().sum().to_dict()]


def duplicate_columns(dataset: pd.DataFrame, sample: int = 1000) -> list[tuple]:
    """
    Return list of duplicate numeric columns.
    Uses a small random sample for large datasets to speed up comparison.
    """
    dup_cols = []
    numeric_cols = [
        col for col in dataset.columns if pd.api.types.is_numeric_dtype(dataset[col])
    ]

    # If dataset is very large, sample to avoid huge comparisons
    if len(dataset) > sample:
        data_to_check = dataset[numeric_cols].sample(sample, random_state=0)
    else:
        data_to_check = dataset[numeric_cols]

    # Hashing approach to detect duplicates fast
    col_hashes = data_to_check.apply(
        lambda s: pd.util.hash_pandas_object(s, index=False).sum()
    )
    duplicates = col_hashes[col_hashes.duplicated(keep=False)]

    if not duplicates.empty:
        hash_groups = duplicates.groupby(duplicates).groups
        for group in hash_groups.values():
            cols = list(group)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    dup_cols.append((cols[i], cols[j]))

    return dup_cols or [("no_match", "no_match")]


def calculation_min_max_str(dataset: pd.DataFrame) -> dict:
    """Vectorized min, max, mean per column."""

    ret = {}
    numeric_cols = dataset.select_dtypes(include=[np.number])
    date_cols = dataset.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"])

    # Numeric stats
    for col in numeric_cols:
        s = numeric_cols[col]
        ret[col] = [s.max(), s.min(), s.mean()]

    # Date stats
    for col in date_cols:
        s = date_cols[col]
        ret[col] = [s.max(), s.min(), "n/a"]

    # Non-numeric / non-date
    for col in dataset.columns:
        if col not in ret:
            ret[col] = ["not_numerical"] * 3

    return ret


def sanity_check(dataset: pd.DataFrame, name: str = "dataset") -> dict:
    """Run all checks efficiently on a single DataFrame."""
    list_res = general_check(dataset)

    if not list_res[0]:
        type_nan = type_NaN_data(dataset)
        dup = duplicate_columns(dataset)
        stats = calculation_min_max_str(dataset)

        list_res.extend(type_nan)
        list_res.append(dup)
        list_res.append(stats)

    return {name: list_res}


def show_data_quality(
    data_quality: dict, dataset: pd.DataFrame, dir: str, name: str
) -> None:
    """Generate Markdown report for a single dataset."""
    output_file = pth.Path(dir) / name

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Data Quality Report\n\n")

        for dataset_name, data in data_quality.items():
            f.write(f"## Dataset: {dataset_name}\n")
            f.write(f"{'-' * (9 + len(dataset_name))}\n\n")

            f.write("### General check\n")
            f.write(f"- Empty: {data[0]}\n")
            if not data[0]:
                f.write(f"- Rows: {data[1]}\n")
                f.write(f"- Columns: {data[2]}\n")
                f.write(f"- Type: {data[3]}\n\n")

                f.write("### Column types\n")
                for col, typ in data[4].items():
                    f.write(f"- {col}: {typ}\n")
                f.write("\n")

                f.write("### NaN counts\n")
                for col, cnt in data[5].items():
                    f.write(f"- {col}: {cnt}\n")
                f.write("\n")

                f.write("### Duplicated columns\n")
                for col_pair in data[6]:
                    f.write(f"- {col_pair}\n")
                f.write("\n")

                f.write("### Numeric stats (min, max, mean)\n")
                for col, vals in data[7].items():
                    f.write(f"- {col}: {vals}\n")
                f.write("\n")

                f.write("### Head of dataset\n")
                f.write(dataset.head().to_markdown())
                f.write("\n\n")
            else:
                f.write("Dataset is empty — no further analysis performed.\n\n")

    print(f"Data quality report saved to: {output_file}")
