from typing import Self
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import proplot as pplt
from priwo import readfil

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

    @property
    def normprofile(self):
        normprofile = self.profile
        normprofile = normprofile - np.median(normprofile)
        normprofile = normprofile / normprofile.std()
        return normprofile

    @property
    def normspectrum(self):
        normspectrum = self.spectrum
        normspectrum = normspectrum - np.median(normspectrum)
        normspectrum = normspectrum / normspectrum.std()
        return normspectrum

    def plot(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "burst.png",
    ):
        def _(ax: pplt.Axes) -> None:
            paneltop = ax.panel_axes("top", width="5em", space=0)
            panelside = ax.panel_axes("right", width="5em", space=0)

            paneltop.set_yticks([])
            panelside.set_xticks([])

            paneltop.plot(self.times, self.normprofile)
            panelside.plot(self.freqs, self.normspectrum, orientation="horizontal")

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

            ax.format(xlabel="Time (ms)", ylabel="Frequency (MHz)")

        if ax is None:
            fig = pplt.figure(width=5, height=5)
            ax = fig.subplots(nrows=1, ncols=1)[0]
            assert ax is not None
            _(ax)
            if save:
                fig.savefig(saveto, dpi=dpi)
            if show:
                pplt.show()
        else:
            _(ax)
