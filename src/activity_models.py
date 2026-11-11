"""
Activity Classification Model Architectures
Project T253 — Escalator Misuse & Unsafe Riding Behaviour Detection

Provides modular deep-learning architectures for video action recognition:
  1. TemporalPoolingClassifier (2D CNN + Temporal Average Pooling)
  2. TemporalGRUClassifier (2D CNN + Temporal GRU)
  3. VideoResNetClassifier (3D / R(2+1)D Spatiotemporal Convolution)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class TemporalPoolingClassifier(nn.Module):
    """
    2D CNN Backbone with Temporal Average Pooling (Candidate A).
    Lightweight, highly efficient baseline for embedded CCTV deployments.
    Input shape: (Batch, Channels=3, Time=T, Height=H, Width=W)
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            in_features = base_model.fc.in_features
            # Remove classification head
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v3_small(weights=weights)
            in_features = base_model.classifier[0].in_features
            self.feature_extractor = nn.Sequential(
                base_model.features,
                base_model.avgpool
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, T, H, W)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        B, C, T, H, W = x.shape

        # Fold time into batch: (B * T, C, H, W)
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        # Extract features: (B * T, in_features, 1, 1)
        features = self.feature_extractor(x_reshaped)
        features = torch.flatten(features, 1)  # (B * T, in_features)

        # Unfold time: (B, T, in_features)
        features = features.view(B, T, -1)

        # Temporal average pooling across frames
        temporal_pooled = torch.mean(features, dim=1)  # (B, in_features)

        # Classification logits
        logits = self.classifier(temporal_pooled)  # (B, num_classes)
        return logits


class TemporalGRUClassifier(nn.Module):
    """
    2D CNN Backbone with Bidirectional GRU Temporal Aggregation (Candidate B).
    Explicitly models sequential transitions (e.g. standing -> slipping -> fallen).
    Input shape: (Batch, Channels=3, Time=T, Height=H, Width=W)
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = "resnet18",
        hidden_size: int = 128,
        num_layers: int = 1,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)
        in_features = base_model.fc.in_features
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        features = self.feature_extractor(x_reshaped)
        features = torch.flatten(features, 1).view(B, T, -1)  # (B, T, in_features)

        gru_out, _ = self.gru(features)  # (B, T, hidden_size * 2)

        # Temporal pooling over GRU representations
        temporal_pooled = torch.mean(gru_out, dim=1)  # (B, hidden_size * 2)

        logits = self.classifier(temporal_pooled)
        return logits


def build_activity_model(
    architecture: str = "resnet18_tap",
    num_classes: int = 5,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function to instantiate activity classification models.
    """
    arch = architecture.lower()
    if arch == "resnet18_tap":
        return TemporalPoolingClassifier(
            num_classes=num_classes, backbone_name="resnet18", pretrained=pretrained
        )
    elif arch == "mobilenetv3_tap":
        return TemporalPoolingClassifier(
            num_classes=num_classes, backbone_name="mobilenet_v3_small", pretrained=pretrained
        )
    elif arch == "resnet18_gru":
        return TemporalGRUClassifier(
            num_classes=num_classes, backbone_name="resnet18", pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
