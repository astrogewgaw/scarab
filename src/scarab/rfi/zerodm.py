import numpy as np


def zdm(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=0)


def zdot(data: np.ndarray) -> np.ndarray:
    mu = data.mean(axis=0)
    mumu = mu.mean()
    norm = np.sqrt(np.sum((mu - mumu) ** 2))
    norm = norm if norm > 0 else 1.0
    mu = (mu - mumu) / norm
    w = (data * mu).sum(axis=1)
    cleaned = data - w.reshape(-1, 1) * mu
    return cleaned
