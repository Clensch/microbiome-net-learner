# author: Michael Hüppe
# date: 15.12.2023
# project: biostat
import random
from typing import Union, List, Tuple

# external
# PY torch implementation
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math


def train_model(
        model: torch.nn.Module,
        train_loader: DataLoader, val_loader: DataLoader, num_epochs:
        int = 10, optimizer: optim.Optimizer = None,
        central_train_loader: DataLoader = None, central_val_loader: DataLoader = None):
    """
    Train a model with given data.
    :param model: Central model to train
    :param train_loader: Data loader for the training data
    :param val_loader: Data loader for the validation data
    :param num_epochs: Number of epochs to train for
    :param optimizer: Optimizer and Learning rate to use during training
    :param central_train_loader: 
    :param central_val_loader: 
    :return: train_losses, val_losses, val_accuracies
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    train_aucs = []
    val_losses = []
    val_accuracies = []
    val_aucs = []
    backcheck_central = not (central_train_loader is None or central_val_loader is None)
    if backcheck_central:
        train_losses_central = []
        train_accuracies_central = []
        train_aucs_central = []
        val_losses_central = []
        val_accuracies_central = []
        val_aucs_central = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_train_loss = running_loss / len(train_loader)
        train_loss, train_accuracy, train_auc = get_metric(model, train_loader, device, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_aucs.append(train_auc)
        # Validation
        val_loss, val_accuracy, val_auc = get_metric(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_aucs.append(val_auc)

        if backcheck_central:
            train_loss_c, train_accuracy_c, train_auc_c = get_metric(model, central_train_loader, device, criterion)
            train_losses_central.append(train_loss_c)
            train_accuracies_central.append(train_accuracy_c)
            train_aucs_central.append(train_auc_c)
            # Validation
            val_loss_c, val_accuracy_c, val_auc_c = get_metric(model, central_val_loader, device, criterion)
            val_losses_central.append(val_loss_c)
            val_accuracies_central.append(val_accuracy_c)
            val_aucs_central.append(val_auc_c)
    history = {
        "Loss": {"train": train_losses,
                 "test": val_losses},
        "Accuracy": {"train": train_accuracies,
                     "test": val_accuracies},
        "AUC": {"train": train_aucs,
                "test": val_aucs},
    }

    if not backcheck_central:
        return history, model, optimizer
    else:
        history_central = {
            "Loss": {"train": train_losses_central,
                     "test": val_losses_central},
            "Accuracy": {"train": train_accuracies_central,
                         "test": val_accuracies_central},
            "AUC": {"train": train_aucs_central,
                    "test": val_aucs_central},
        }
        return history, model, optimizer, history_central


def train_model_one_step(
        model: torch.nn.Module,
        train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer):
    """
    Train a model with given data.
    :param model: Central model to train
    :param train_loader: Data loader for the training data
    :param val_loader: Data loader for the validation data
    :param optimizer: Optimizer for training
    :return: train_losses, val_losses, val_accuracies
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Training
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_loss, train_accuracy, train_auc = get_metric(model, train_loader, device, criterion)
        # Validation
        val_loss, val_accuracy, val_auc = get_metric(model, val_loader, device, criterion)

    history = {
        "Loss": {"train": train_loss,
                 "test": val_loss},
        "Accuracy": {"train": train_accuracy,
                     "test": val_accuracy},
        "AUC": {"train": train_auc,
                "test": val_auc},
    }

    return history, model, optimizer


def get_metric(model: torch.nn.Module, data_loader: DataLoader, device, criterion) -> Tuple[float, float, float]:
    """
    Calculate the average loss and accuracy for the given model and data
    :param model: Model to evaluate
    :param data_loader: Data to evaluate on
    :param device: Device to perform the evaluation on
    :param criterion: Criterion for loss
    :return: loss, accuracy
    """
    model.eval()
    loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Calculate AUC using sklearn
            loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_loss = loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_preds)

    return float(average_loss.cpu()), float(accuracy), float(auc_score)
