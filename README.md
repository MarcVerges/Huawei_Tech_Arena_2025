# Huawei Tech Arena – Radio Signal Prediction Framework

## Project Overview

**Author:** Marc Vergés Sánchez  
**Creation Date:** November 1, 2025  

This project implements a complete framework for the analysis, preprocessing, and predictive modeling of radio signal data. It supports **linear, non-linear, and artificial neural network (ANN) models**, including evaluation and visualization of model performance.  

> ⚠️ **Note:** The raw dataset is **not included** in this repository to prevent security issues and reduce download size. Users should provide their own data in the expected format.

---

## Project Structure
├── data/ # Folder for processed datasets (not included)
├── data_raw/ # Folder for raw data (not included)
├── src/ # Source code
│ ├── init.py
│ ├── ann_models.py
│ ├── const.py
│ ├── data_EDA.py
│ ├── data_fix.py
│ ├── linear_models.py
│ ├── main.py
│ ├── non_linear_models.py
│ └── utils.py
├── Makefile # Automation for setup, linting, testing, and running
├── pyproject.toml # Dependency and environment configuration
├── setup.cfg # Optional setup configuration
├── uv.lock # UV environment lock file
└── README.md # Project documentation

---

## Dependencies

Dependencies are **managed through UV** and defined in `pyproject.toml`. The project requires **Python ≥3.11**.  

Key dependencies include:  

- `pandas`, `numpy` – data handling and numerical computation  
- `matplotlib`, `seaborn` – visualization  
- `scikit-learn` – linear and non-linear machine learning models  
- `torch`, `tensorflow`, `scikeras` – artificial neural networks  
- `black`, `flake8`, `mypy`, `pytest` – code formatting, linting, type checking, and testing  

Full dependency list is available in `pyproject.toml`.

---

## Features

1. **Data Handling and Preprocessing**
   - Automatic conversion of Excel inputs to Parquet for efficient processing.
   - Data cleaning, type validation, missing value imputation, and feature engineering.

2. **Exploratory Data Analysis (EDA)**
   - Generates Markdown reports with dataset quality, column types, missing values, and statistics.
   - Duplicate detection and basic dataset overview.

3. **Predictive Modeling**
   - **Linear Models:** Ridge and Lasso regression with hyperparameter tuning.
   - **Non-linear Models:** Random Forest and Gradient Boosting Trees with grid search.
   - **ANN Models:** Configurable multi-layer networks with early stopping and learning rate scheduling.

4. **Evaluation and Visualization**
   - Baseline predictor using the most correlated feature.
   - RMSE and MAE comparison of models, side-by-side plots, and accuracy evaluation.
   - Visualization of model predictions versus actual values and per-model analysis.

5. **Reproducibility**
   - Model outputs are saved using **pickle**, avoiding unnecessary retraining.
   - Hyperparameters and configurations are managed through `src/const.py`.

---

## Contact

**Author:** Marc Vergés Sánchez  
