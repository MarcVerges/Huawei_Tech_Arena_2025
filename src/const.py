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

from pathlib import Path

# Definition of constants refering data_trans file
BASE_DIR = Path(__file__).resolve().parent.parent
dir_of_data = BASE_DIR / "data_raw"
dir_of_save = BASE_DIR / "data"
file_data_name = "5G_campaign_interpolated.parquet"
file_input_name = "Data.xlsx"

# Definition of constants refering data_EDA file
file_report_name_1 = "Data_Report_EDA_1.txt"
file_report_name_2 = "Data_Report_EDA_2.txt"


# Definition of constants refering data partitions
datasets_names_values = {"train": 0.7, "valid": 0.2, "test": 0.1}


# Definition of constants refering linear_models file

alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
output_file_laso = "laso_values_predict_4.pkl"
output_file_ridge = "ridge_values_predict_4.pkl"  #

# Definition of constants refering non_linear_models file

number_of_trees_rf = [500, 750, 1000]
varaible_at_split_rf = ["sqrt", 0.2, 0.4]
sample_size_tree_rf = [0.6, 0.8, 0.9, 1.0]
output_file_rf = "RF_values_predict_4.pkl"

number_of_trees_gb = [300, 500, 800]
learning_rate_gb = [0.05, 0.08, 0.1]
tree_depth_gb = [5, 6, 8]
regularization_lambda_gb = [0.1, 0.2, 0.5]
output_file_gb = "GB_values_predict_4.pkl"

# Definition of constants refering ann_models file

epochs = 200
patience_es = 15
patience_lr = 7
HYPERPARAMS_LIST = [
    {"hidden_sizes": [128, 64, 32], "lr": 0.0005, "epochs": 200, "batch_size": 32},
    {"hidden_sizes": [256, 128], "lr": 0.001, "epochs": 200, "batch_size": 64},
    {"hidden_sizes": [64, 64, 32], "lr": 0.005, "epochs": 200, "batch_size": 128},
    {"hidden_sizes": [32, 16], "lr": 0.001, "epochs": 200, "batch_size": 16},
    {"hidden_sizes": [16, 32, 16], "lr": 0.001, "epochs": 200, "batch_size": 32},
]

output_file_ann = "ANN_values_predict_4.pkl"


# Definition of constants refering best preforming models

best_number_of_trees_rf = [750]
best_varaible_at_split_rf = ["sqrt"]
best_sample_size_tree_rf = [1.0]
best_output_file_rf = "RF_values_predict_best_model.pkl"

best_number_of_trees_gb = [
    300,
]
best_learning_rate_gb = [0.1]
best_tree_depth_gb = [8]
best_regularization_lambda_gb = [0.5]
best_output_file_gb = "GB_values_predict_best_model.pkl"
best_hyperparams_list = [
    {"hidden_sizes": [256, 128], "lr": 0.001, "epochs": 200, "batch_size": 64},
]

best_output_file_ann = "ANN_values_predict_best_model.pkl"
