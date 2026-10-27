"""Combinação de modelos por votação suave."""
import numpy as np


def soft_voting(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Média das probabilidades de cada modelo."""
    stacked = np.stack(list_of_probs, axis=0)
    return stacked.mean(axis=0)
