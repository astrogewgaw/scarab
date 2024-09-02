from typing import Self
from dataclasses import asdict

import numpy as np

from scarab.base import Burst
from scarab.dm import dm2shifts, roll2d
from scarab.snr import Template, snratio
from scarab.utilities import normalise, scrunch


class Transform:

    def __init__(
        self,
        burst: Burst,
        inplace: bool = False,
    ) -> None:
        self.burst = burst
        self.inplace = inplace
        self.transformed = burst
        self.istransformed = False

    def normalise(self) -> Self:
        normalised = normalise(self.burst.data)
        if self.inplace:
            self.burst.data = normalised
            self.transformed = self.burst
        else:
            attrs = asdict(self.burst)
            attrs["data"] = normalised
            self.transformed = Burst(**attrs)
        self.istransformed = True
        return type(self)(burst=self.transformed, inplace=self.inplace)

    def dedisperse(self, dm: float) -> Self:
        shifts = dm2shifts(self.burst.freqs, self.burst.dt, dm)
        prevshifts = dm2shifts(self.burst.freqs, self.burst.dt, self.burst.dm)
        dedispersed = roll2d(self.burst.data, shifts - prevshifts)
        if self.inplace:
            self.burst.dm = dm
            self.burst.data = dedispersed
            self.transformed = self.burst
        else:
            attrs = asdict(self.burst)
            attrs["dm"] = dm
            attrs["data"] = dedispersed
            self.transformed = Burst(**attrs)
        self.istransformed = True
        return type(self)(burst=self.transformed, inplace=self.inplace)

    def scrunch(self, tf: int = 1, ff: int = 1) -> Self:
        scrunched = scrunch(self.burst.data, tf, ff)
        nf, nt = scrunched.shape
        dt = self.burst.tobs / nt
        df = self.burst.bw / nf

        if self.inplace:
            self.burst.nf = nf
            self.burst.nt = nt
            self.burst.df = df
            self.burst.dt = dt
            self.burst.data = scrunched
            self.transformed = self.burst
        else:
            attrs = asdict(self.burst)
            attrs["nf"] = nf
            attrs["nt"] = nt
            attrs["df"] = df
            attrs["dt"] = dt
            attrs["data"] = scrunched
            self.transformed = Burst(**attrs)
        self.istransformed = True
        return type(self)(burst=self.transformed, inplace=self.inplace)

    def clip(self, within: float = 50e-3) -> Self:
        n0 = int(self.burst.nt // 2)
        dn = int(within / self.burst.dt)

        clipped = self.burst.data[:, n0 - dn : n0 + dn]
        profile = clipped.sum(axis=0)
        m = np.argmax(profile)
        n0 = n0 - dn + m

        clipped = self.burst.data[:, n0 - dn : n0 + dn]
        _, nt = clipped.shape

        if self.inplace:
            self.burst.nt = nt
            self.data = clipped
            self.burst.tobs = self.burst.nt * self.burst.dt
            self.transformed = self.burst
        else:
            attrs = asdict(self.burst)
            attrs["nt"] = nt
            attrs["data"] = clipped
            attrs["tobs"] = nt * self.burst.dt
            self.transformed = Burst(**attrs)
        self.istransformed = True
        return type(self)(burst=self.transformed, inplace=self.inplace)

    def mask(self, boxwidth: int = 10, snrthres: float = 10.0) -> Self:
        mask = self.burst.spectrum > 0.0

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
            subdata = self.burst.data[slice(*subband), :]
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

        masked = self.burst.data[mask, :]
        freqs = self.burst.freqs[mask]
        fh = freqs[0]
        fl = freqs[-1]
        nf = freqs.size
        bw = nf * self.burst.df
        fc = 0.5 * (fh + fl)

        if self.inplace:
            self.burst.nf = nf
            self.burst.fh = fh
            self.burst.fc = fc
            self.burst.fl = fl
            self.burst.bw = bw
            self.burst.data = masked
            self.transformed = self.burst
        else:
            attrs = asdict(self.burst)
            attrs["nf"] = nf
            attrs["fh"] = fh
            attrs["fc"] = fc
            attrs["fl"] = fl
            attrs["bw"] = bw
            attrs["data"] = masked
            self.transformed = Burst(**attrs)
        self.istransformed = True
        return type(self)(burst=self.transformed, inplace=self.inplace)
