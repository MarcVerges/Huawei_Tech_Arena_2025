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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np  # Importar numpy para la manipulación de arrays


def train_ann(
    x_train, y_train, x_val, y_val, hyperparams, patience_es=15, patience_lr=7
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(x_train, "values"):
        x_train = x_train.values
    if hasattr(y_train, "values"):
        y_train = y_train.values
    if hasattr(x_val, "values"):
        x_val = x_val.values
    if hasattr(y_val, "values"):
        y_val = y_val.values

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    if len(y_train_tensor.shape) == 1:
        y_train_tensor = y_train_tensor.unsqueeze(1)
        y_val_tensor = y_val_tensor.unsqueeze(1)

    layers = []
    input_size = x_train_tensor.shape[1]
    for h in hyperparams["hidden_sizes"]:
        layers.append(nn.Linear(input_size, h))
        layers.append(nn.ReLU())
        input_size = h
    output_size = y_train_tensor.shape[1]
    layers.append(nn.Linear(input_size, output_size))
    model = nn.Sequential(*layers).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=patience_lr,
        min_lr=1e-6,
    )

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    best_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(hyperparams["epochs"]):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1

        scheduler.step(val_loss)

        if patience_counter >= patience_es:
            break

    if best_weights:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        y_pred = model(x_val_tensor).cpu().numpy()
        y_val_actual = y_val_tensor.cpu().numpy()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    y_pred = np.array(y_pred).flatten()
    y_val_actual = np.array(y_val_actual).flatten()

    return y_pred, y_val_actual
