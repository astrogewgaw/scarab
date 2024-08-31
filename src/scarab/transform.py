import numpy as np
from dataclasses import asdict

from scarab.base import Burst
from scarab.dm import dm2shifts, roll2d
from scarab.snr import Template, snratio
from scarab.utilities import normalise, scrunch


def normaliser(burst: Burst) -> Burst:
    normalised = normalise(burst.data)
    attrs = asdict(burst)
    attrs["data"] = normalised
    return Burst(**attrs)


def dedisperser(burst: Burst, dm: float) -> Burst:
    shifts = dm2shifts(burst.freqs, burst.dt, dm)
    prevshifts = dm2shifts(burst.freqs, burst.dt, burst.dm)
    dedispersed = roll2d(burst.data, shifts - prevshifts)
    attrs = asdict(burst)
    attrs["dm"] = dm
    attrs["data"] = dedispersed
    return Burst(**attrs)


def scruncher(burst: Burst, tf: int = 1, ff: int = 1) -> Burst:
    scrunched = scrunch(burst.data, tf, ff)
    nf, nt = scrunched.shape
    dt = burst.tobs / nt
    df = burst.bw / nf

    attrs = asdict(burst)
    attrs["nf"] = nf
    attrs["nt"] = nt
    attrs["df"] = df
    attrs["dt"] = dt
    attrs["data"] = scrunched
    return Burst(**attrs)


def clipper(burst: Burst, within: float = 50e-3) -> Burst:
    n0 = int(burst.nt // 2)
    dn = int(within / burst.dt)

    clipped = burst.data[:, n0 - dn : n0 + dn]
    profile = clipped.sum(axis=0)
    m = np.argmax(profile)
    n0 = n0 - dn + m

    clipped = burst.data[:, n0 - dn : n0 + dn]
    _, nt = clipped.shape

    attrs = asdict(burst)
    attrs["nt"] = nt
    attrs["data"] = clipped
    attrs["tobs"] = nt * burst.dt
    return Burst(**attrs)


def masker(burst: Burst, bcwidth: int = 10, threshold: float = 10.0) -> Burst:
    mask = burst.spectrum > 0.0

    runs = np.flatnonzero(
        np.diff(
            np.r_[
                np.int8(0),
                mask.view(np.int8),
                np.int8(0),
            ]
        )
    ).reshape(-1, 2)

    subbands = runs.copy()
    for i in np.arange(runs.shape[0] - 1):
        _, end = runs[i]
        nextstart, _ = runs[i + 1]
        if (nextstart - end) <= 1:
            subbands[i, 1] = -1
            subbands[i + 1, 0] = -1
    subbands = subbands.flatten()
    subbands = subbands[subbands != -1].reshape(-1, 2)
    subbands = subbands[~(np.diff(subbands) == 1).T[0], :]

    mask = np.asarray([False] * mask.size)
    for subband in subbands:
        mask[slice(*subband)] = True

    subsnrs = []
    for i, subband in enumerate(subbands):
        subdata = burst.data[slice(*subband), :]
        profile = subdata.sum(axis=0)
        profile = profile - np.median(profile)
        profile = profile / profile.std()
        boxcar = Template.gaussian(int(bcwidth))
        subsnrs.append(snratio(profile, boxcar)[0][0, 0, :].max())
    subsnrs = np.asarray(subsnrs)

    band = subbands[subsnrs >= threshold]
    mask = np.asarray([False] * mask.size)
    if len(band) <= 0:
        mask[slice(0, mask.size)] = True
    else:
        mask[slice(band[0][0], band[-1][-1])] = True

    masked = burst.data[mask, :]
    freqs = burst.freqs[mask]
    fh = freqs[0]
    fl = freqs[-1]
    nf = freqs.size
    bw = nf * burst.df
    fc = 0.5 * (fh + fl)

    attrs = asdict(burst)
    attrs["nf"] = nf
    attrs["fh"] = fh
    attrs["fc"] = fc
    attrs["fl"] = fl
    attrs["bw"] = bw
    attrs["data"] = masked
    return Burst(**attrs)
