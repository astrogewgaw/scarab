import numpy as np

kdm: float = 1.0 / 2.41e-4


def delayperdm(f: float, fref: float) -> float:
    return kdm * (f**-2 - fref**-2)


def dm2delay(f: float, fref: float, dm: float) -> float:
    return kdm * dm * (f**-2 - fref**-2)


def dm2delays(freqs: np.ndarray, dm: float) -> np.ndarray:
    return np.vectorize(dm2delay, excluded=["fref", "dm"])(freqs, freqs[0], dm)


def dm2shifts(freqs: np.ndarray, dt: float, dm: float):
    return np.round(dm2delays(freqs, dm) / dt).astype(int)


def roll2d(data: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    rolled = data.copy()
    for ix, shift in enumerate(shifts):
        rolled[ix] = np.roll(rolled[ix], -shift)
    return rolled


def dedisperse(
    data: np.ndarray,
    freqs: np.ndarray,
    dt: float,
    dm: float,
    collapse: bool = False,
) -> np.ndarray:
    dedispersed = roll2d(data, dm2shifts(freqs, dt, dm))
    return np.sum(dedispersed, axis=0) if collapse else dedispersed
