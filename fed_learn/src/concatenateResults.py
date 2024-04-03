# author: Michael Hüppe
# date: 04.03.2024
# project: biostat/fed_learn/concatenateResults.py
import os
import json
from typing import Tuple, Any

from torch.utils.data import DataLoader

from biostat.fed_learn.src.train import get_metric


import torch
from biostat.fed_learn.src.genomDataset import GenomDataset

from biostat.fed_learn.src.central_model_feedForward import CentralModel

import zipfile


def load_model_from_path(input_size, output_size, path, device) -> Tuple[str, Any]:
    central_model = CentralModel(input_size, output_size, features=[30, 50], dropouts=[0.48, 0.49])
    # Now let's load the model
    central_model.load_state_dict(torch.load(path))
    central_model.to(device)
    central_model.eval()  # Put the model in evaluation mode
    name = os.path.basename(os.path.dirname(path))
    return name, central_model


def getFederatedModelPerformance(path):
    # Directory containing zip files
    genomeDataset_train = GenomDataset(r"../../materials/microbiome_project_train.csv")
    genomeDataset_test = GenomDataset(r"../../materials/microbiome_project_test.csv")
    train_loader = DataLoader(genomeDataset_train, batch_size=42, shuffle=False)
    test_loader = DataLoader(genomeDataset_test, batch_size=42, shuffle=False)

    input_size = genomeDataset_train.num_features
    output_size = genomeDataset_train.num_classes  # Binary classification
    directory = path
    results = {"federated": {}}
    # Iterate over each file in the directory
    i = 0
    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            continue
        filepath = os.path.join(directory, filename)
        # results["federated"][i] = json.load(open(os.path.join(filepath, "history.json")))
        modelpath = os.path.join(filepath, "model.pt")
        if os.path.isfile(modelpath):
            try:
                name, model = load_model_from_path(input_size, output_size, modelpath, "cpu")
                loss, accuracy, auc = get_metric(model, train_loader, "cpu", criterion=torch.nn.CrossEntropyLoss())
                loss_test, accuracy_test, auc_test = get_metric(model, test_loader, "cpu",
                                                                criterion=torch.nn.CrossEntropyLoss())
                print(f"Train: {accuracy, auc}, Test: {loss_test, accuracy_test, auc_test}")
                results["federated"][i] = {"Loss": {"train": [loss],
                                                    "test": [loss_test]},
                                           "Accuracy": {"train": [accuracy],
                                                        "test": [accuracy_test]},
                                           "AUC": {"train": [auc],
                                                   "test": [auc_test]}
                                           }
                i += 1
            except RuntimeError:
                continue

    return results


if __name__ == '__main__':
    # Directory containing zip files
    directory = "results"

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Check if the file is a zip file
        if filename.endswith(".zip"):
            # Create a directory with the same name as the zip file (without extension)
            dirname = os.path.splitext(filepath)[0]
            os.makedirs(dirname, exist_ok=True)

            # Extract the contents of the zip file to the created directory
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(dirname)
                print(f"Extracted {filename} to {dirname}")

                import zipfile

    results = getFederatedModelPerformance("../results")
    json.dump(results, open("../results/history_federated.json", "w"))

