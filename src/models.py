"""CNNs pré-treinadas na ImageNet com cabeça binária."""
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def build_model(name: str) -> nn.Module:
    """name: resnet18 | mobilenet_v2 | efficientnet_b0"""
    weights = "DEFAULT"
    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Modelo desconhecido: {name}")

    return model
