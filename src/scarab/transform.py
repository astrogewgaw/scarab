from typing import Self
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np
import proplot as pplt

from scarab.base import Burst
from scarab.dm import dm2shifts, roll2d
from scarab.snr import Template, snratio
from scarab.utilities import normalise, scrunch


@dataclass
class Transformer:

    burst: Burst
    transformed: Burst
    inplace: bool = False
    istransformed: bool = False

    @classmethod
    def new(cls, burst: Burst, inplace: bool = False):
        return cls(burst=burst, transformed=burst, inplace=inplace)

    def normalise(self) -> Self:
        data = self.transformed.data
        normalised = normalise(data if self.inplace else data.copy())
        if not self.inplace:
            attrs = asdict(self.transformed)
            attrs["data"] = normalised
            self.transformed = Burst(**attrs)
        self.istransformed = True
        return self

    def dedisperse(self, dm: float) -> Self:
        data = self.transformed.data
        data = data if self.inplace else data.copy()
        shifts = dm2shifts(self.transformed.freqs, self.transformed.dt, dm)
        prevshifts = dm2shifts(
            self.transformed.freqs, self.transformed.dt, self.transformed.dm
        )
        dedispersed = roll2d(data, shifts - prevshifts)
        if self.inplace:
            self.transformed.dm = dm
            self.transformed.data = dedispersed
            self.transformed = self.transformed
        else:
            attrs = asdict(self.transformed)
            attrs["dm"] = dm
            attrs["data"] = dedispersed
            self.transformed = Burst(**attrs)
        self.istransformed = True
        return self

    def scrunch(self, tf: int = 1, ff: int = 1) -> Self:
        data = self.transformed.data
        data = data if self.inplace else data.copy()
        scrunched = scrunch(data, tf, ff)
        nf, nt = scrunched.shape
        dt = self.transformed.tobs / nt
        df = self.transformed.bw / nf

        if self.inplace:
            self.transformed.nf = nf
            self.transformed.nt = nt
            self.transformed.df = df
            self.transformed.dt = dt
            self.transformed.data = scrunched
            self.transformed = self.transformed
        else:
            attrs = asdict(self.transformed)
            attrs["nf"] = nf
            attrs["nt"] = nt
            attrs["df"] = df
            attrs["dt"] = dt
            attrs["data"] = scrunched
            self.transformed = Burst(**attrs)
        self.istransformed = True
        return self

    def clip(self, within: float = 50e-3) -> Self:
        n0 = int(self.transformed.nt // 2)
        dn = int(within / self.transformed.dt)
        data = self.transformed.data
        data = data if self.inplace else data.copy()

        clipped = data[:, n0 - dn : n0 + dn]
        profile = clipped.sum(axis=0)
        m = np.argmax(profile)
        n0 = n0 - dn + m

        clipped = self.transformed.data[:, n0 - dn : n0 + dn]
        _, nt = clipped.shape

        if self.inplace:
            self.transformed.nt = nt
            self.data = clipped
            self.transformed.tobs = self.transformed.nt * self.transformed.dt
            self.transformed = self.transformed
        else:
            attrs = asdict(self.transformed)
            attrs["nt"] = nt
            attrs["data"] = clipped
            attrs["tobs"] = nt * self.transformed.dt
            self.transformed = Burst(**attrs)
        self.istransformed = True
        return self

    def mask(self, boxwidth: int = 10, snrthres: float = 10.0) -> Self:
        data = self.transformed.data
        freqs = self.transformed.freqs
        mask = self.transformed.spectrum > 0.0
        data = data if self.inplace else data.copy()

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
            subdata = data[slice(*subband), :]
            profile = subdata.sum(axis=0)
            profile = profile - np.median(profile)
            profile = profile / profile.std()
            boxcar = Template.gaussian(int(boxwidth))
            subsnrs.append(snratio(profile, boxcar)[0][0, 0, :].max())
        subsnrs = np.asarray(subsnrs)

        band = subbands[subsnrs >= snrthres]
        mask = np.asarray([False] * mask.size)
        if len(band) <= 0:
            mask[slice(0, mask.size)] = True
        else:
            mask[slice(band[0][0], band[-1][-1])] = True

        masked = data[mask, :]
        freqs = self.transformed.freqs[mask]
        fh = freqs[0]
        fl = freqs[-1]
        nf = freqs.size
        bw = nf * self.transformed.df
        fc = 0.5 * (fh + fl)

        if self.inplace:
            self.transformed.nf = nf
            self.transformed.fh = fh
            self.transformed.fc = fc
            self.transformed.fl = fl
            self.transformed.bw = bw
            self.transformed.data = masked
            self.transformed = self.transformed
        else:
            attrs = asdict(self.transformed)
            attrs["nf"] = nf
            attrs["fh"] = fh
            attrs["fc"] = fc
            attrs["fl"] = fl
            attrs["bw"] = bw
            attrs["data"] = masked
            self.transformed = Burst(**attrs)
        self.istransformed = True
        return self

    def plot(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        saveto: str | Path = "transformed.png",
    ):
        fig = pplt.figure()
        if self.inplace:
            ax = fig.subplots(nrows=1, ncols=1)[0]
            self.transformed.plot(ax=ax)
        else:
            axs = fig.subplots(nrows=1, ncols=2)
            self.transformed.plot(ax=axs[0])
            self.transformed.plot(ax=axs[1])
        if save:
            fig.savefig(saveto, dpi=dpi)
        if show:
            pplt.show()
