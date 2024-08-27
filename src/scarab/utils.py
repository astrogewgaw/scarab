import numpy as np


def w50gauss(sigma: float) -> float:
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma


def w10gauss(sigma: float) -> float:
    return 2.0 * np.sqrt(2.0 * np.log(10.0)) * sigma


def sqrsumnorm(data: np.ndarray) -> np.ndarray:
    return data * ((data**2).sum()) ** -0.5


def madnorm(data: np.ndarray):
    mu = data.mean(axis=1)
    sigma = data.std(axis=1)
    sigma[sigma == 0] = 1.0
    x = (data - mu.reshape(-1, 1)) / sigma.reshape(-1, 1)
    return x


def scrunch(data: np.ndarray, tf: int = 1, ff: int = 1):
    nf, nt = data.shape
    nvalid = tf * (nt // tf)
    offset = (nt - nvalid) // 2
    x = data[:, offset : offset + nvalid]
    x = x.reshape(nf // ff, ff, nvalid // tf, tf).sum(axis=(1, 3))
    return x
