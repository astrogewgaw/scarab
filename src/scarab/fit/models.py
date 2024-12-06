import numpy as np
from scipy.special import erfc
from scipy.signal import oaconvolve


def linear(
    x: np.ndarray,
    x0: float,
    m: float,
    c: float,
) -> np.ndarray:
    return m * (x - x0) + c


def boxcar(
    x: np.ndarray,
    width: float,
) -> np.ndarray:
    y = np.zeros_like(x)
    y[np.abs(x) <= 0.5 * width] = 1.0
    return y


def gauss(
    x: np.ndarray,
    fluence: float,
    center: float,
    sigma: float,
    dc: float,
):
    return dc + normgauss(x, fluence, center, sigma)


def normgauss(
    x: np.ndarray,
    fluence: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    S = fluence / sigma / np.sqrt(2 * np.pi)
    y = S * np.exp(-0.5 * np.power((x - center) / sigma, 2))
    return y


def pbfisotropic(x: np.ndarray, tau: float):
    nt = x.size
    dt = np.abs(x[0] - x[1])
    times = np.arange(nt) * dt
    times = times - times[nt // 2]

    t0 = -tau * np.log(0.001)
    if np.max(times) < t0:
        raise RuntimeError(f"The window is too short: {np.max(x)}, {t0}.")
    result = np.zeros_like(times)
    timemask = times >= 0.0
    result[timemask] = np.exp(-times[timemask] / tau) / tau
    return result


def scatanalytic(
    x: np.ndarray,
    fluence: float,
    center: float,
    sigma: float,
    tau: float,
    dc: float,
) -> np.ndarray:
    K = tau / sigma
    invK = 1.0 / K
    invsigma = 1.0 / sigma
    invsqrt = 1 / np.sqrt(2.0)

    y = (x - center) * invsigma
    if invK >= 10:
        return dc + normgauss(x, fluence, center + tau, sigma)
    else:
        argexp = 0.5 * invK**2 - y * invK
        argexp[argexp >= 300.0] = 0.0
        exgauss = 0.5 * invK * invsigma * np.exp(argexp) * erfc(-(y - invK) * invsqrt)
        return dc + fluence * exgauss


def scatconvolving(
    x: np.ndarray,
    fluence: float,
    center: float,
    sigma: float,
    tau: float,
    dc: float,
) -> np.ndarray:
    A = normgauss(x, fluence, center, sigma)
    B = pbfisotropic(x, tau)
    return dc + oaconvolve(A, B, mode="same") / np.sum(B)


def scatbandintmodel(
    x: np.ndarray,
    fluence: float,
    center: float,
    sigma: float,
    tau: float,
    dc: float,
    flow: float,
    fhigh: float,
    nf: int,
) -> np.ndarray:
    dt = np.abs(x[0] - x[1])
    f0 = 0.5 * (flow + fhigh)
    freqs = np.geomspace(flow, fhigh, num=nf)
    fluences = fluence * np.power(freqs / f0, -1.5)
    taus = tau * np.power(freqs / f0, -4.0)
    profiles = np.zeros((nf, x.size))
    for i in range(nf):
        profiles[i, :] = scatanalytic(x, fluences[i], center, sigma, taus[i], 0.0)
    result = np.sum(profiles, axis=0)
    totalfluence = np.sum(result) * dt
    result = dc + (fluence / totalfluence) * result
    return result


def scatgauss_afb_instrumental(
    x: np.ndarray,
    fluence: float,
    center: float,
    sigma: float,
    tau: float,
    taui: float,
    taud: float,
    dc: float,
) -> np.ndarray:
    A = np.power(tau, 2.0)
    A *= scatanalytic(x, fluence, center, sigma, tau, 0.0)
    A /= (tau - taui) * (tau - taud)

    B = np.power(taui, 2.0)
    B *= scatanalytic(x, fluence, center, sigma, taui, 0.0)
    B /= (tau - taui) * (taui - taud)

    C = np.power(taud, 2.0)
    C *= scatanalytic(x, fluence, center, sigma, taui, 0.0)
    C /= (tau - taud) * (taui - taud)

    return dc + A - B + C


def scatgauss_dfb_instrumental(
    x: np.ndarray,
    fluence: float,
    center: float,
    sigma: float,
    tau: float,
    taud: float,
    dc: float,
) -> np.ndarray:
    B = boxcar(x, taud)
    A = scatconvolving(x, fluence, center, sigma, tau, 0.0)
    return dc + oaconvolve(A, B, mode="same") / np.sum(B)


def rpl(
    x: np.ndarray,
    fluence: float,
    gamma: float,
    beta: float,
    xref: float,
    dc: float,
) -> np.ndarray:
    logx = np.log(x / xref)
    return fluence * np.exp(gamma * logx + beta * logx**2) + dc
