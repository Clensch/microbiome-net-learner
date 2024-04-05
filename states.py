# author: Christopher Lensch, Michael Hüppe, Jannis Waller
# date: 03.03.2024
# project: biostat

import bios
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.optim as optim
from FeatureCloud.app.engine.app import AppState, app_state, Role
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.central_model_feedForward import CentralModel
from src.genomDataset import GenomDataset
from src.train import train_model

INPUT = '/mnt/input'
OUTPUT = '/mnt/output'

INITIAL_STATE = 'initial'
COMPUTE_STATE = 'compute'
AGGREGATE_STATE = 'aggregate'
SAVING_STATE = 'saving'
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_BATCH_SIZE = 26



@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE)

    def run(self):
        self.log('Initialization ...')
        self.store('current_iteration', 0)

        self.log('Reading config file...')
        config = bios.read(f'{INPUT}/config.yml')
        config = config['microbiome_net_learner']
        self.log('Done reading config file.')

        input_file = config['data']
        target_column = config['target']
        self.store('input_file', input_file)
        self.store('target_column', target_column)
        self.store('max_iterations', config.get("max_iterations", 10))
        self.store('epochs_per_iteration', config.get("epochs_per_iteration", 10))
        self.log('Done reading config file.')
        self.log('Reading training data ...')
        genomeDataset = GenomDataset(f'{INPUT}/{input_file}', sep=config.get("sep", ","))
        # Create train and validation datasets
        num_features = genomeDataset.num_features
        train_data, val_data = train_test_split(genomeDataset, test_size=config.get("test_size", 0.1), random_state=42)
        self.log('Done reading training data ...')
        self.log('Preparing initial model ...')
        # Create DataLoader instances
        batch_size = config.get("batch_size", 26)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        input_size = num_features
        output_size = genomeDataset.num_classes  # Binary classification
        model = CentralModel(input_size, output_size, features=[30, 50], dropouts=[0.48, 0.49])
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))

        self.store('model', model)
        self.store('model', optimizer)
        self.store("train_loader", train_loader)
        self.store("val_loader", val_loader)
        self.store("history", {})
        self.log('Done preparing initial model.')
        if self.is_coordinator:
            self.broadcast_data(model)

        self.log('Transition to compute state ...')
        return COMPUTE_STATE


@app_state(COMPUTE_STATE)
class ComputeState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE, role=Role.PARTICIPANT)
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)
        self.register_transition(SAVING_STATE)

    def run(self):
        self.log('Start computation...')

        model = self.await_data()
        train_loader = self.load("train_loader")
        val_loader = self.load("val_loader")
        optimizer = self.load("optimizer")
        self.log("Load loader")
        # TODO: get the train_loader, val_loader and train model for one epoch.
        new_history, model, optimizer = train_model(model, train_loader, val_loader,
                                                    num_epochs=self.load("epochs_per_iteration"),
                                                    optimizer=optimizer)
        self.store('model', model)
        self.store('optimizer', optimizer)
        history = self.load("history")
        if history is None:
            history = {}
        for metric, eval_performance in new_history.items():
            if metric not in history:
                history[metric] = {}
            for eval_mode, performance in eval_performance.items():
                if eval_mode not in history[metric]:
                    history[metric][eval_mode] = []
                history[metric][eval_mode] += performance
        self.store("history", history)

        self.send_data_to_coordinator(model)

        current_iteration = self.load('current_iteration')
        current_iteration += 1
        self.log(f'CURRENT ITERATION:{current_iteration}')
        self.store('current_iteration', current_iteration)
        if current_iteration >= self.load('max_iterations'):
            self.log('Transition to saving state ...')
            return SAVING_STATE

        if self.is_coordinator:
            self.log('Transition to aggregate state ...')
            return AGGREGATE_STATE
        else:
            self.log('Transition to compute state ...')
            return COMPUTE_STATE

    def loadData(self):
        return self.load("train_loader"), self.load("val_loader")


@app_state(AGGREGATE_STATE)
class AggregateState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE, role=Role.COORDINATOR)

    def run(self):
        self.log("Start aggregating ...")
        # Calculate gradient
        models = self.gather_data()
        state_dict_aggregated = {}
        layers = []
        for model in models:
            layers += list(model.state_dict().keys())
        layers = set(layers)
        for layer in layers:
            state_dict_aggregated[layer] = torch.from_numpy(np.asarray([model.state_dict()[layer].cpu().numpy() for model in models]).mean(axis=0))
        self.log("Update model weights ...")
        models[0].load_state_dict(state_dict_aggregated)
        self.broadcast_data(models[0])
        self.log('Transition to compute state ...')
        return COMPUTE_STATE


@app_state(SAVING_STATE)
class SavingState(AppState):

    def register(self):
        self.register_transition('terminal', label="Terminate the execution")

    def run(self):
        # Return Data
        self.log("Start saving ...")
        self.log("Saving Model to File")
        model = self.load("model")
        torch.save(model.state_dict(), os.path.join(OUTPUT, "model.pt"))
        self.saveHistory()
        self.savePlot()
        return 'terminal'

    def saveHistory(self):
        with open(os.path.join(OUTPUT, "history.json"), 'w', encoding='utf8') as json_file:
            json.dump(self.load("history"), json_file)

    def savePlot(self):
        history = self.load("history")
        fig, (loss_ax, accuracy_ax, auc_ax) = plt.subplots(3, sharex="col")
        fig.suptitle("Central Model Evaluation")

        loss_ax.set_title("Loss History")
        loss_ax.plot(history["Loss"]["train"], label="Train")
        loss_ax.plot(history["Loss"]["test"], label="Validation")
        loss_ax.legend()
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Categorical Cross Entropy")

        accuracy_ax.set_title("Accuracy History")
        accuracy_ax.plot(history["Accuracy"]["train"], label="Train")
        accuracy_ax.plot(history["Accuracy"]["test"], label="Validation")
        accuracy_ax.set_ylim([0.2, 1.1])
        accuracy_ax.legend()
        accuracy_ax.set_xlabel("Epoch")
        accuracy_ax.set_ylabel("Accuracy")

        auc_ax.set_title("AUC History")
        auc_ax.plot(history["AUC"]["train"], label="Train")
        auc_ax.plot(history["AUC"]["test"], label="Validation")
        auc_ax.set_ylim([0.2, 1.1])
        auc_ax.legend()
        auc_ax.set_xlabel("Epoch")
        auc_ax.set_ylabel("AUC")

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT, "history.png"))
