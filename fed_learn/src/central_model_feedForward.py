# author: Michael Hüppe
# date: 15.12.2023
# project: biostat
from typing import Union, List

import torch
import torch.nn as nn


# external
# PY torch implementation

class CentralModel(nn.Module):
    """
    Simple central model used for classification.
    """

    def __init__(self, input_size: int, output_size: int, features: Union[List[int], int] = 50,
                 dropouts: Union[List[float], float] = 0.2):
        super(CentralModel, self).__init__()

        # Define the layers
        self._features: List[int] = features if isinstance(features, list) else [features]
        self._dropouts: List[int] = dropouts if isinstance(dropouts, list) else [dropouts]
        self._nLayers: int = len(features)
        assert len(self._dropouts) == self._nLayers, "The number of dropouts and feature sizes has to be equal!"

        self._model = nn.Sequential(
            nn.Linear(input_size, self._features[0]),
            nn.ReLU(),
        )
        for i in range(self._nLayers):
            self._model.append(nn.Linear(self._features[i],
                                         self._features[i + 1] if i + 1 < self._nLayers else output_size))
            self._model.append(nn.Dropout(self._dropouts[i]))
            self._model.append(nn.ReLU())

        self._model.append(nn.Softmax(dim=1))

    def forward(self, x):
        """
        Define the forward step for the model. Since this is a pretty straight forward model
        and no particular preprocessing has to be used before calling it. The model is just called.
        :param x: data to call the model with
        :return: Prediction on that data
        """
        return self._model(x)


import torch.nn as nn


class ConvolutionalModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, features: Union[List[int], int] = 50,
                 dropouts: Union[List[float], float] = 0.2):
        super(ConvolutionalModel, self).__init__()

        # Define the layers
        self._features: List[int] = features if isinstance(features, list) else [features]
        self._dropouts: List[int] = dropouts if isinstance(dropouts, list) else [dropouts]
        self._nLayers: int = len(features)
        assert len(self._dropouts) == self._nLayers, "The number of dropouts and feature sizes has to be equal!"
        self.input = nn.Linear(input_size, features[0])

        self._model = nn.Sequential(
        )

        for i in range(self._nLayers):
            self._model.append(nn.Conv1d(1 if i == 0 else self._features[i],
                                         self._features[i + 1] if i + 1 < self._nLayers else output_size,
                                         kernel_size=10, padding=1),)
            self._model.append(nn.ReLU())
            self._model.append(nn.Dropout(self._dropouts[i]))
            self._model.append(nn.MaxPool1d(kernel_size=3, stride=2))

        self._model.append(nn.Flatten(),)
        self._model.append(nn.Softmax(dim=1))

    def forward(self, x):
        x = self.input(x)
        x = x.unsqueeze(1)  # Add a channel dimension for Conv1d
        return self._model(x)


import torch.nn as nn

# Define a basic block with residual connection for 1D input
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=2, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x

# Define the ResNet model for 1D input
class ResNet1D(nn.Module):
    def __init__(self, block, layers, input_size, feature_per_layer, num_classes=1000):
        super(ResNet1D, self).__init__()
        # Define the layers
        self._layers: List[int] = layers if isinstance(layers, list) else [layers]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_channels = feature_per_layer
        self.input = nn.Linear(input_size, feature_per_layer)
        self.conv1 = nn.Conv1d(1, feature_per_layer, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(feature_per_layer)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self._blocks = []
        for i, layer in enumerate(layers, start=1):
            self._blocks.append(self.make_layer(block, feature_per_layer * i, layer, stride=2 if i > 1 else 1).to(device))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(feature_per_layer*len(self._layers), num_classes)
        self.output = nn.Softmax(dim=1)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input(x)
        x = x.unsqueeze(1)  # Add a channel dimension for Conv1d
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for block in self._blocks:
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.output(x)
        return x
