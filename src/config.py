"""Hiperparâmetros centralizados do experimento."""
from pathlib import Path

import torch

# Caminhos
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGS_DIR = RESULTS_DIR / "figs"
RESULTS_CSV = RESULTS_DIR / "results.csv"
STATS_TXT = RESULTS_DIR / "stats.txt"

# Experimento
N_RUNS = 30
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
IMG_SIZE = 64
NUM_CLASSES = 2
PATIENCE = 3  # early stopping (épocas sem melhoria na validação)

# Modelos
MODEL_H0 = "resnet18"
MODELS_ENSEMBLE = ("resnet18", "mobilenet_v2", "efficientnet_b0")

# Normalização ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
