"""Model scaffolding for segmentation and refinement."""
from __future__ import annotations

import torch
from torch import nn
import torchvision.models as models


class PointPromptHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, num_points: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_points * 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features.flatten(start_dim=1))


class RefineSegmentationHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 1):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.refine(features)


class SimpleSegmentationNet(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = False):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        mask = self.decoder(features)
        return mask
