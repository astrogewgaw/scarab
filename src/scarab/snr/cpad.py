import numpy as np


def ceilpow2(n: int) -> int:
    return 2 ** int(np.ceil(np.log2(n)))


def cpadpow2(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    N = ceilpow2(n)
    padding = (x.ndim - 1) * [(0, 0)] + [(0, N - n)]
    return np.pad(x, padding, mode="wrap")
