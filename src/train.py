"""Treino com early stopping e inferência."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import config
from config import DEVICE, LR, PATIENCE
from metrics import compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval_loss(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).squeeze().long()
            logits = model(x)
            total += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


def train_model(model, train_loader, val_loader):
    """
    Treina o modelo; retorna (modelo com melhor peso na validação, tempo em segundos).
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    t0 = time.perf_counter()
    for _ in range(config.EPOCHS):  # leitura dinâmica: respeita --epochs do simulate.py
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).squeeze().long()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_loss = _eval_loss(model, val_loader, criterion)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0
    return model, elapsed


@torch.no_grad()
def predict_probs(model, loader):
    """Probabilidades da classe positiva (pneumonia), shape (N,)."""
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(DEVICE)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.cpu().numpy())
    return np.concatenate(probs)
