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
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import src.linear_models as linear_models
import src.data_fix as data_fix
import src.non_linear_models as non_linear_models
import src.ann_models as ann_models
import src.data_EDA as data_EDA
import src.utils as utils
import src.const as const


def main():
    os.makedirs(const.dir_of_save, exist_ok=True)

    parquet_files = [
        f for f in os.listdir(str(const.dir_of_save)) if f.lower().endswith(".parquet")
    ]

    if len(parquet_files) == 0:
        data_out_full_path = const.dir_of_save / const.file_data_name
        data_in_full_path = const.dir_of_data / const.file_input_name

        df = pd.read_excel(data_in_full_path, decimal=",")
        df.to_parquet(data_out_full_path, index=False)

    else:
        parquet_path = os.path.join(str(const.dir_of_save), parquet_files[0])
        df = utils.get_data(parquet_path)

    # === 2. Preform EDA 1 ===.

    result_sanity_1 = data_EDA.sanity_check(df)
    data_EDA.show_data_quality(
        result_sanity_1, df, const.dir_of_save, const.file_report_name_1
    )

    # === 3. Preform data fixes ===.

    df_fixed_types = data_fix.general_fix(df)

    # === 4. Preform EDA 2 ===.

    result_sanity_2 = data_EDA.sanity_check(df_fixed_types)
    data_EDA.show_data_quality(
        result_sanity_2, df_fixed_types, const.dir_of_save, const.file_report_name_2
    )

    # === 5. Preform Columns definition===.

    label_col = "Serving Cell RSRP"
    numerical_cols = [
        "Serving Cell RSRQ",
        "Azimuth",
        "Dis_Serving",
        "Drive_Lat",
        "Drive_Lon",
        "Cell_lat",
        "Cell_lon",
        "RS SINR Carrier 1",
    ]
    derived_cols = [
        "Azimuth_sin",
        "Azimuth_cos",
        "RSRQ_per_Distance",
        "SINR_per_Distance",
    ]

    train_cols = numerical_cols + derived_cols

    # === 6. Preform data partition ===.

    segmented_datasets = {}
    segmented_datasets = utils.prepare_data_sets(df_fixed_types)

    train_df = segmented_datasets["train"]
    val_df = segmented_datasets["valid"]
    test_df = segmented_datasets["test"]

    x_train, y_train = utils.feature_split(train_df, label_col)
    x_val, y_val = utils.feature_split(val_df, label_col)
    x_test, y_test = utils.feature_split(test_df, label_col)

    # === 7. We generate the baseline ===.

    base_pred, base_act = utils.compute_predictive_baseline(df_fixed_types, label_col)

    # === 8. We Train with linear models ===.

    linear_res_ridge = {}
    linear_res_lasso = {}

    ridge_full_path = const.dir_of_save / const.output_file_ridge

    if os.path.exists(ridge_full_path):
        with open(ridge_full_path, "rb") as f:
            ridge_results = pickle.load(f)

    else:
        ridge_results = linear_models.Ridge(
            x_train, y_train, x_val, y_val, const.alphas, train_cols
        )
        utils.save_results(ridge_results, ridge_full_path)

    laso_full_path = const.dir_of_save / const.output_file_laso

    if os.path.exists(laso_full_path):
        with open(laso_full_path, "rb") as f:
            lasso_results = pickle.load(f)

    else:
        lasso_results = linear_models.Lasso(
            x_train, y_train, x_val, y_val, const.alphas, train_cols
        )
        utils.save_results(lasso_results, laso_full_path)

    linear_res_ridge["Base_Line"] = {1: (base_pred, base_act)}
    linear_res_lasso["Base_Line"] = {1: (base_pred, base_act)}
    linear_res_ridge["Ridge"] = ridge_results
    linear_res_lasso["Lasso"] = lasso_results

    linear_eval_ridge = utils.evaluate_nested_results(linear_res_ridge)
    utils.plot_rmse_mae_side_by_side(linear_eval_ridge)

    linear_eval_lasso = utils.evaluate_nested_results(linear_res_lasso)
    utils.plot_rmse_mae_side_by_side(linear_eval_lasso)

    # === 9. We Train with non-linear models ===.

    non_linear_res_rf = {}
    non_linear_res_gb = {}
    individual_results_rf = {}
    individual_results_gb = {}

    rf_full_path = const.dir_of_save / const.output_file_rf

    if os.path.exists(rf_full_path):
        with open(rf_full_path, "rb") as f:
            individual_results_rf = pickle.load(f)

    else:
        for num in const.number_of_trees_rf:
            for vari in const.varaible_at_split_rf:
                for samp in const.sample_size_tree_rf:
                    (
                        random_forest_pred,
                        random_forest_actual,
                    ) = non_linear_models.random_forest(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        num,
                        vari,
                        samp,
                        train_cols,
                    )
                    key = (num, vari, samp)
                    individual_results_rf[(num, vari, samp)] = (
                        random_forest_pred,
                        random_forest_actual,
                    )

        utils.save_results(individual_results_rf, rf_full_path)

    gb_full_path = const.dir_of_save / const.output_file_gb

    if os.path.exists(gb_full_path):
        with open(gb_full_path, "rb") as f:
            individual_results_gb = pickle.load(f)

    else:
        for num in const.number_of_trees_gb:
            for learn in const.learning_rate_gb:
                for depth in const.tree_depth_gb:
                    for regul in const.regularization_lambda_gb:
                        gb_pred, gb_val = non_linear_models.gradient_boosting_trees(
                            x_train,
                            y_train,
                            x_val,
                            y_val,
                            num,
                            learn,
                            depth,
                            regul,
                            train_cols,
                        )

                        key = (num, learn, depth, regul)
                        individual_results_gb[key] = (gb_pred, gb_val)

        utils.save_results(individual_results_gb, gb_full_path)

    non_linear_res_rf["Base_Line"] = {1: (base_pred, base_act)}
    non_linear_res_gb["Base_Line"] = {1: (base_pred, base_act)}
    non_linear_res_rf["Random_Forest"] = individual_results_rf
    non_linear_res_gb["Gradient_Boosting_trees"] = individual_results_gb

    non_linear_eval_rf = utils.evaluate_nested_results(non_linear_res_rf)
    utils.plot_rmse_mae_side_by_side(non_linear_eval_rf)
    non_linear_eval_gb = utils.evaluate_nested_results(non_linear_res_gb)
    utils.plot_rmse_mae_side_by_side(non_linear_eval_gb)

    # === 10. We Train with AAN - models ===.

    ann_results = {}
    individual_results_ann = {}

    ann_full_path = const.dir_of_save / const.output_file_ann

    if os.path.exists(ann_full_path):
        with open(ann_full_path, "rb") as f:
            individual_results_ann = pickle.load(f)

    else:
        for i, hyperparams in enumerate(const.HYPERPARAMS_LIST):
            if "epochs" not in hyperparams:
                hyperparams["epochs"] = const.epochs

            print(f"Training ANN with hyperparams set {i+1}: {hyperparams}")
            ann_pred, ann_val_actual = ann_models.train_ann(
                x_train,
                y_train,
                x_val,
                y_val,
                hyperparams,
                patience_es=const.patience_es,
                patience_lr=const.patience_lr,
            )

            key = str(hyperparams)
            individual_results_ann[key] = (ann_pred, ann_val_actual)

        utils.save_results(individual_results_ann, ann_full_path)

    ann_results["Base_Line"] = {1: (base_pred, base_act)}
    ann_results["ANN"] = individual_results_ann

    ann_eval_res = utils.evaluate_nested_results(ann_results)
    utils.plot_rmse_mae_side_by_side(ann_eval_res)

    # === 11. We Compare Best Preformance Models ===.

    best_ridge = utils.select_best_model(linear_eval_ridge)
    best_lasso = utils.select_best_model(linear_eval_lasso)
    best_rf = utils.select_best_model(non_linear_eval_rf)
    best_gb = utils.select_best_model(non_linear_eval_gb)
    best_ann = utils.select_best_model(ann_eval_res)

    baseline_row = {
        "model": "Baseline",
        "hyperparams": "(default)",
        "RMSE": np.sqrt(mean_squared_error(base_act, base_pred)),
        "MAE": mean_absolute_error(base_act, base_pred),
    }

    best_models_df = pd.DataFrame(
        [baseline_row, best_ridge, best_lasso, best_rf, best_gb, best_ann]
    )

    utils.plot_rmse_mae_side_by_side(best_models_df)

    # === 12. We Now compute Accuracy ===.

    best_models_info = [
        ("ANN", utils.select_best_model(ann_eval_res), individual_results_ann),
        (
            "Random Forest",
            utils.select_best_model(non_linear_eval_rf),
            individual_results_rf,
        ),
        (
            "Gradient Boosting",
            utils.select_best_model(non_linear_eval_gb),
            individual_results_gb,
        ),
        (
            "Lasso",
            utils.select_best_model(linear_eval_lasso),
            linear_res_lasso["Lasso"],
        ),
        (
            "Ridge",
            utils.select_best_model(linear_eval_ridge),
            linear_res_ridge["Ridge"],
        ),
    ]

    utils.plot_model_accuracies(best_models_info)

    for model_name, best_row, results_dict in best_models_info:
        acc = utils.calculate_accuracy_from_results(best_row, results_dict)

    utils.plot_best_models_accuracy(best_models_info, tolerance=0.10)
    utils.plot_models_separately(best_models_info)

    # === 13. Retrain the best performing models with train+val ===

    x_train_val = pd.concat([x_train, x_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    # Random Forest (best)

    best_individual_results_rf = {}
    best_rf_full_path = const.dir_of_save / const.best_output_file_rf

    if os.path.exists(best_rf_full_path):
        with open(best_rf_full_path, "rb") as f:
            best_individual_results_rf = pickle.load(f)
    else:
        for num in const.best_number_of_trees_rf:
            for vari in const.best_varaible_at_split_rf:
                for samp in const.best_sample_size_tree_rf:
                    y_pred, y_actual = non_linear_models.random_forest(
                        x_train_val,
                        y_train_val,
                        x_test,
                        y_test,
                        num,
                        vari,
                        samp,
                        train_cols,
                    )
                    key = (num, vari, samp)
                    best_individual_results_rf[key] = (y_pred, y_actual)
        utils.save_results(best_individual_results_rf, best_rf_full_path)

    best_non_linear_res_rf = {
        "Base_Line": {1: (base_pred, base_act)},
        "Random_Forest": best_individual_results_rf,
    }
    best_non_linear_eval_rf = utils.evaluate_nested_results(best_non_linear_res_rf)
    best_rf = utils.select_best_model(best_non_linear_eval_rf)

    # Gradient Boosting (best)

    best_individual_results_gb = {}
    best_gb_full_path = const.dir_of_save / const.best_output_file_gb

    if os.path.exists(best_gb_full_path):
        with open(best_gb_full_path, "rb") as f:
            best_individual_results_gb = pickle.load(f)
    else:
        for num in const.best_number_of_trees_gb:
            for lr in const.best_learning_rate_gb:
                for depth in const.best_tree_depth_gb:
                    for reg in const.best_regularization_lambda_gb:
                        y_pred, y_actual = non_linear_models.gradient_boosting_trees(
                            x_train_val,
                            y_train_val,
                            x_test,
                            y_test,
                            num,
                            lr,
                            depth,
                            reg,
                            train_cols,
                        )
                        key = (num, lr, depth, reg)
                        best_individual_results_gb[key] = (y_pred, y_actual)
        utils.save_results(best_individual_results_gb, best_gb_full_path)

    best_non_linear_res_gb = {
        "Base_Line": {1: (base_pred, base_act)},
        "Gradient_Boosting_trees": best_individual_results_gb,
    }
    best_non_linear_eval_gb = utils.evaluate_nested_results(best_non_linear_res_gb)
    best_gb = utils.select_best_model(best_non_linear_eval_gb)

    # ANN (best)

    best_individual_results_ann = {}
    best_ann_full_path = const.dir_of_save / const.best_output_file_ann

    if os.path.exists(best_ann_full_path):
        with open(best_ann_full_path, "rb") as f:
            best_individual_results_ann = pickle.load(f)
    else:
        for i, hyperparams in enumerate(const.best_hyperparams_list):
            if "epochs" not in hyperparams:
                hyperparams["epochs"] = const.epochs
            print(f"Training ANN with hyperparams set {i+1}: {hyperparams}")
            y_pred, y_actual = ann_models.train_ann(
                x_train_val,
                y_train_val,
                x_test,
                y_test,
                hyperparams,
                patience_es=const.patience_es,
                patience_lr=const.patience_lr,
            )
            key = str(hyperparams)
            best_individual_results_ann[key] = (y_pred, y_actual)
        utils.save_results(best_individual_results_ann, best_ann_full_path)

    best_ann_results = {
        "Base_Line": {1: (base_pred, base_act)},
        "ANN": best_individual_results_ann,
    }
    best_ann_eval_res = utils.evaluate_nested_results(best_ann_results)
    best_ann = utils.select_best_model(best_ann_eval_res)

    # RMSE/MAE Table

    baseline_row = {
        "model": "Baseline",
        "hyperparams": "(default)",
        "RMSE": np.sqrt(mean_squared_error(base_act, base_pred)),
        "MAE": mean_absolute_error(base_act, base_pred),
    }

    best_models_df = pd.DataFrame([baseline_row, best_rf, best_gb, best_ann])

    utils.plot_rmse_mae_side_by_side(best_models_df)

    # Best Models Info for Accuracy & Plotting

    best_models_info = [
        ("ANN", best_ann, best_individual_results_ann),
        ("Random Forest", best_rf, best_individual_results_rf),
        ("Gradient Boosting", best_gb, best_individual_results_gb),
    ]

    for model_name, best_row, results_dict in best_models_info:
        acc = utils.calculate_accuracy_from_results(best_row, results_dict)

    utils.plot_model_accuracies(best_models_info)
    utils.plot_best_models_accuracy(best_models_info, tolerance=0.10)
    utils.plot_models_separately(best_models_info)


if __name__ == "__main__":
    main()
