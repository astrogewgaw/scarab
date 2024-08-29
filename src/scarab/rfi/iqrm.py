import numpy as np
from collections import defaultdict


def lags(radius: int):
    lag = 1
    geofactor = 1.5
    while lag <= radius:
        yield lag
        yield -lag
        lag = max(int(geofactor * lag), lag + 1)


def lagdiff(x: np.ndarray, lag: int):
    s = np.roll(x, lag)
    if lag >= 0:
        s[:lag] = x[0]
    else:
        s[lag:] = x[-1]
    return x - s


def tukey(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    stats = np.percentile(data, (25, 50, 75))
    sigma = stats[2] - stats[0] / 1.349
    return np.abs(data - stats[1]) > threshold * sigma


def iqrm(x: np.ndarray, radius: int = 5, threshold: float = 3.0) -> np.ndarray:
    n = x.size
    castvotes = defaultdict(set)
    receivedvotes = defaultdict(set)

    for lag in lags(radius):
        diff = lagdiff(x, lag)
        mask = tukey(diff, threshold)
        ii = np.where(mask)[0]
        jj = np.clip(ii - lag, 0, n - 1)
        for i, j in zip(ii, jj):
            castvotes[j].add(i)
            receivedvotes[i].add(j)

    finalmask = np.zeros_like(x, dtype=bool)
    for i, casters in castvotes.items():
        for j in casters:
            if j in castvotes and len(castvotes[j]) < len(receivedvotes[i]):
                finalmask[i] = True
                break
    return finalmask
