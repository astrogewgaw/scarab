import numpy as np
from scipy.interpolate import make_interp_spline


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


def smoothen(
    x: np.ndarray,
    window: int = 1,
    interfact: int = 3,
    y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(x)
    indices = list(range(0, N))
    exindices = np.linspace(0, N, N * interfact)
    spline = make_interp_spline(indices, x, k=3)
    xnew = spline(exindices)

    ynew = []
    if y is not None:
        box = np.ones(window) / window
        y = np.convolve(y, box, mode="same")
        spline = make_interp_spline(indices, y, k=3)
        ynew = spline(exindices)
        ynew = np.asarray(ynew)
        return xnew, ynew
    else:
        return xnew
