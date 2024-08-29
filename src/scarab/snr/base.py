import numpy as np

from scarab.snr.cpad import cpadpow2
from scarab.snr.noise import noisemean, noisestddev
from scarab.snr.templates import Template, TemplateBank


def snratio(
    X: np.ndarray,
    T: Template | TemplateBank,
    stddev: str | float = "iqr",
    mean: str | float = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nbins = X.shape[-1]
    XX = X.reshape(-1, nbins)
    nprofs, nbins = XX.shape
    ntemplates = 1 if isinstance(T, Template) else len(T)

    if isinstance(mean, float):
        means = np.full(nprofs, mean)
    elif isinstance(mean, str):
        means = np.asarray(noisemean(X, method=mean)).reshape(nprofs)
    else:
        raise ValueError(
            f"{mean} must be either a valid noise mean "
            "estimation method name specified as a string"
            ", or a float"
        )

    if isinstance(mean, float):
        stddevs = np.full(nprofs, stddev)
    elif isinstance(stddev, str):
        stddevs = np.asarray(noisestddev(X, method=stddev)).reshape(nprofs)
    else:
        raise ValueError(
            f"{stddev} must be either a valid noise stddev "
            "estimation method name specified as a string"
            ", or a float"
        )

    XX = cpadpow2((XX - means.reshape(-1, 1)) / stddevs.reshape(-1, 1))
    FX = np.fft.rfft(XX).reshape(nprofs, 1, -1)
    TT = T.prepare(XX.shape[-1])
    FT = np.fft.rfft(TT).reshape(1, ntemplates, -1)
    snr = np.fft.irfft(FX * FT)
    snr = snr[:, :, :nbins]

    models = np.zeros((nprofs, nbins))
    for iprof in range(nprofs):
        snrs = snr[iprof]
        itemplate, ibin = np.unravel_index(snrs.argmax(), snrs.shape)
        bestsnr = snrs[itemplate, ibin]
        bestT = T[itemplate] if isinstance(T, TemplateBank) else T
        models[iprof] = means[iprof]
        models[iprof, : bestT.size] += bestT.data * stddevs[iprof] * bestsnr
        shift = ibin - bestT.ref
        models[iprof] = np.roll(models[iprof], shift)
    return snr, means, stddevs, models
