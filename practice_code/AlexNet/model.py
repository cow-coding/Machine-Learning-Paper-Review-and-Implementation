"""
This model is based on
"""

import torch
import numpy as np
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet, self).__init__()
        self.output_dim = num_classes
        # 5 Conv Layers
        self.convolution = nn.Sequential(
            # First Conv
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),    # 11 x 11 x 3 x 96
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Second Conv
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),  # 5 x 5 x 96 x (128 x 2)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Third Conv
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),            # 3 x 3 x (128 x 2) x (192 x 2)
            nn.ReLU(inplace=True),
            # Fourth Conv
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),            # 3 x 3 x (192 x 2) x (192 x 2)
            nn.ReLU(inplace=True),
            # Fifth Conv
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),            # 3 x 3 x (192 x 2) x (128 x 2)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            # First FC
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            # Second FC
            nn.Linear(in_features=4096, out_features=4096),
            # Third FC
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

        self.init_bias()

    def init_bias(self):
        conv_count = 1

        for conv in self.convolution:
            if isinstance(conv, nn.Conv2d):
                nn.init.normal_(conv.weight, 0, 0.01)
                if conv_count in [2, 4, 5]:
                    nn.init.constant_(conv.bias, 1)
                else:
                    nn.init.constant_(conv.bias, 0)
                conv_count += 1

        for fc in self.classifier:
            if isinstance(fc, nn.Linear):
                nn.init.normal_(fc.weight, 0, 0.01)
                nn.init.constant_(fc.bias, 1)

    def forward(self, x):
        x = self.convolution(x)
        x = x.view(-1, 256*6*6)
        output = self.classifier(x)
        return output