import numpy as np
from scarab.utils import madnorm
from collections import defaultdict

# TODO: See if we can absorb some or all of jess (https://github.com/josephwkania/jess).
# This is quite a large colection of RFI mitigation algorithms, and could prove helpful
# to have here.


def zdot(data: np.ndarray) -> np.ndarray:
    mu = data.mean(axis=0)
    mumu = mu.mean()
    norm = np.sqrt(np.sum((mu - mumu) ** 2))
    norm = norm if norm > 0 else 1.0
    mu = (mu - mumu) / norm
    w = (data * mu).sum(axis=1)
    cleaned = data - w.reshape(-1, 1) * mu
    return cleaned


def tukey(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    stats = np.percentile(data, (25, 50, 75))
    sigma = stats[2] - stats[0] / 1.349
    return np.abs(data - stats[1]) > threshold * sigma


def iqrm(x: np.ndarray, radius: int = 5, threshold: float = 3.0) -> np.ndarray:
    n = x.size
    castvotes = defaultdict(set)
    receivedvotes = defaultdict(set)

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


def automask(data: np.ndarray) -> np.ndarray:
    nf, _ = data.shape
    sigma = data.std(axis=1)
    radius = max(2, int(0.1 * nf))
    miqrm = iqrm(sigma, radius=radius, threshold=3.0)
    return miqrm


def autoclean(data: np.ndarray) -> np.ndarray:
    x = data.astype(np.float32)
    mask = automask(data)
    x[mask] = 0.0
    z = madnorm(x)
    z = zdot(z)
    z = np.clip(z, -3, +3)
    return z
