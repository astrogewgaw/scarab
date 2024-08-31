import numpy as np


def w50gauss(sigma: float) -> float:
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma


def w10gauss(sigma: float) -> float:
    return 2.0 * np.sqrt(2.0 * np.log(10.0)) * sigma


def normalise(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=1)
    stddev = data.std(axis=1)
    stddev[stddev == 0] = 1.0
    x = (data - mean.reshape(-1, 1)) / stddev.reshape(-1, 1)
    return x


def scrunch(data: np.ndarray, tf: int = 1, ff: int = 1) -> np.ndarray:
    nf, nt = data.shape
    nvalid = tf * (nt // tf)
    offset = (nt - nvalid) // 2
    x = data[:, offset : offset + nvalid]
    x = x.reshape(nf // ff, ff, nvalid // tf, tf).sum(axis=(1, 3))
    return x
