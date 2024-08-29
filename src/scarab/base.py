from typing import Self
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np
import proplot as pplt
from priwo import readfil

from scarab.dm import dm2shifts, roll2d
from scarab.snr import Template, snratio
from scarab.utils import normalise, scrunch

# TODO: Flux esitmation via gmrtetc.

# TODO: Add routines for periodicity searches via:
#       * Folding?
#       * Lomb-Scargle periodogram,
#       * Pearson chi square method,
#       * Weighted wavelet transform,
#       * Information theory approaches (P4J; https://doi.org/10.3847/1538-4365/aab77c),
# and others. Take inspiration from frbpa (https://github.com/KshitijAggarwal/frbpa).

# TODO: Add code to estimate drift rate, via:
#           * dfdt (https://github.com/zpleunis/dfdt),
#           * subdriftlaw (https://github.com/mef51/subdriftlaw),
# or others.


@dataclass
class Burst:
    nf: int
    nt: int
    df: float
    dt: float
    fh: float
    fc: float
    fl: float
    bw: float
    dm: float
    mjd: float
    path: Path
    tobs: float
    data: np.ndarray

    @classmethod
    def new(cls, fn: str) -> Self:
        meta, data = readfil(fn)

        _, nt = data.shape
        fh = meta["fch1"]
        dt = meta["tsamp"]
        nf = meta["nchans"]
        df = np.abs(meta["foff"])

        dm = meta.get("refdm", 0.0)
        mjd = meta.get("tstart", 0.0)

        bw = nf * df
        tobs = nt * dt
        fl = fh - bw + (df * 0.5)
        fc = 0.5 * (fh + fl)

        burst = cls(
            nt=nt,
            nf=nf,
            dt=dt,
            df=df,
            fh=fh,
            fc=fc,
            fl=fl,
            bw=bw,
            dm=dm,
            mjd=mjd,
            tobs=tobs,
            data=data,
            path=Path(fn).resolve(),
        )

        burst.dm = dm
        return burst

    @property
    def freqs(self):
        return np.asarray([self.fh - i * self.df for i in range(self.nf)])

    @property
    def times(self):
        return np.asarray([self.dt * i * 1e3 for i in range(self.nt)])

    @property
    def mjds(self):
        return self.mjd + self.times

    @property
    def profile(self):
        return self.data.sum(axis=0)

    @property
    def spectrum(self):
        return self.data.sum(axis=1)

    def normalise(self) -> None:
        self.data = normalise(self.data)

    def dedisperse(self, dm: float) -> None:
        shifts = dm2shifts(self.freqs, self.dt, dm)
        prevshifts = dm2shifts(self.freqs, self.dt, self.dm)
        residuals = shifts - prevshifts
        self.data = roll2d(self.data, residuals)
        self.dm = dm

    def scrunch(self, tf: int = 1, ff: int = 1) -> None:
        self.data = scrunch(self.data, tf, ff)
        self.nf, self.nt = self.data.shape
        self.dt = self.tobs / self.nt
        self.df = self.bw / self.nf

    def zoommask(self, within: float = 100e-3) -> Self:
        n0 = int(self.nt // 2)
        dn = int(within / self.dt)

        zoomed = self.data[:, n0 - dn : n0 + dn]
        profile = zoomed.sum(axis=0)
        m = np.argmax(profile)
        n0 = n0 - dn + m

        zoomed = self.data[:, n0 - dn : n0 + dn]
        _, nt = zoomed.shape

        attrs = asdict(self)
        attrs["nt"] = nt
        attrs["data"] = zoomed
        attrs["tobs"] = nt * self.dt
        return type(self)(**attrs)

    def emitmask(self, threshold: float = 10.0) -> Self:
        mask = self.spectrum > 0.0

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
            subdata = self.data[slice(*subband), :]
            profile = subdata.sum(axis=0)
            profile = profile - np.median(profile)
            profile = profile / profile.std()
            boxcar = Template.gaussian(int(10))
            subsnrs.append(snratio(profile, boxcar)[0][0, 0, :].max())
        subsnrs = np.asarray(subsnrs)

        band = subbands[subsnrs >= threshold]
        mask = np.asarray([False] * mask.size)
        if len(band) <= 0:
            mask[slice(0, mask.size)] = True
        else:
            mask[slice(band[0][0], band[-1][-1])] = True

        data = self.data[mask, :]
        freqs = self.freqs[mask]
        fh = freqs[0]
        fl = freqs[-1]
        nf = freqs.size
        bw = nf * self.df
        fc = 0.5 * (fh + fl)

        attrs = asdict(self)
        attrs["nf"] = nf
        attrs["fh"] = fh
        attrs["fc"] = fc
        attrs["fl"] = fl
        attrs["bw"] = bw
        attrs["data"] = data
        return type(self)(**attrs)

    def plot(self):
        fig = pplt.figure(width=5, height=5)
        ax = fig.subplot()  # type: ignore
        paneltop = ax.panel_axes("top", width="5em", space=0)
        panelside = ax.panel_axes("right", width="5em", space=0)

        paneltop.axis("off")
        panelside.axis("off")
        paneltop.plot(self.times, self.profile)
        panelside.plot(self.spectrum, self.freqs)

        ax.imshow(
            self.data,
            cmap="batlow",
            aspect="auto",
            interpolation="none",
            extent=[
                self.times[0],
                self.times[-1],
                self.freqs[-1],
                self.freqs[0],
            ],
        )

        pplt.show()
