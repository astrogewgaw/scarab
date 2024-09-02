from typing import Self
from dataclasses import dataclass

import numpy as np
from scipy.linalg import norm
from scipy.fftpack import dct, idct

from scarab.base import Burst
from scarab.dm.fdmt import fdmt
from scarab.dm.base import dedisperse


def getKc(cdmt: np.ndarray, winsize: int = 5) -> int:
    _, nk = cdmt.shape
    noise_margin = int(nk / 2)
    cdmtT = np.abs(cdmt).T
    cdmtM = cdmtT.max(axis=1)
    noise_top = cdmtM[noise_margin:].mean()

    Kc = 0
    for i in range(winsize, len(cdmtT)):
        rollavg = cdmtM[i - winsize : i + 1].mean()
        if rollavg <= noise_top:
            Kc = i
            break
    else:
        raise RuntimeError("Could not find value for Kc.")
    return Kc


@dataclass
class DMOptimizer:

    burst: Burst
    dmt: np.ndarray
    ddmmin: float = -5
    ddmmax: float = +5
    ddmopt: float | None = None
    ddmoptmin: float | None = None
    ddmoptmax: float | None = None
    structparam: float | None = None
    uncertainty: np.ndarray | None = None
    relative_uncertainty: np.ndarray | None = None

    @classmethod
    def new(
        cls,
        burst: Burst,
        ddmmin: float = -5,
        ddmmax: float = +5,
    ) -> Self:
        zeroed = dedisperse(
            dt=burst.dt,
            dm=-burst.dm,
            data=burst.data,
            freqs=burst.freqs,
        )

        return cls(
            burst=burst,
            ddmmin=ddmmin,
            ddmmax=ddmmax,
            dmt=fdmt(
                zeroed,
                burst.fh,
                burst.fl,
                burst.df,
                burst.dt,
                burst.dm + ddmmin,
                burst.dm + ddmmax,
            ),
        )

    @property
    def nt(self):
        return self.dmt.shape[1]

    @property
    def ndm(self):
        return self.dmt.shape[0]

    @property
    def ddms(self):
        return np.linspace(self.ddmmin, self.ddmmax, self.ndm)

    def viasnr(self) -> None:
        cdmt = dct(self.dmt, norm="ortho")
        _, nk = cdmt.shape
        Kc = getKc(cdmt)

        fO = 3
        k = np.linspace(1, nk, nk)
        fL = 1 / (1 + (k / Kc) ** (2 * fO))

        fLdiag = np.diag(fL)
        lpfdata = fLdiag @ cdmt.T
        dmtsmooth = idct(lpfdata.T, norm="ortho")

        winmin = 1
        winstep = 1
        winmax = int(10e-3 / self.burst.dt)

        deltai = self.dmt - dmtsmooth
        noise_stddev = deltai.std()
        noise_const = dmtsmooth.min()
        zeroed = dmtsmooth - noise_const
        timeroots = noise_stddev * np.sqrt(np.arange(winmax))

        maxsnrs = []
        maxinits = []
        maxfinals = []
        for idm in zeroed:
            runsnr = -1
            runinit = None
            runfinal = None
            for winstart in range(0, len(idm) - winmin, winstep):
                S = np.cumsum(idm[winstart : winstart + winmax])
                snrs = S[winmin:] / timeroots[winmin : len(S)]
                maxsnr = snrs.max()
                if maxsnr > runsnr:
                    runsnr = maxsnr
                    runinit = winstart
                    runfinal = snrs.argmax()
            maxsnrs.append(runsnr)
            maxinits.append(runinit)
            maxfinals.append(runfinal)
        maxsnr = np.max(maxsnrs)
        binmax = np.argmax(maxsnrs)

        ix = binmax
        while ix < len(maxsnrs):
            ix = ix + 1
            if maxsnr - maxsnrs[ix] >= 1:
                break
        upperbound = ix

        ix = binmax
        while ix >= 0:
            ix = ix - 1
            if maxsnr - maxsnrs[ix] >= 1:
                break
        lowerbound = ix

        ddmopt = self.ddms[binmax]
        ddmoptmax = self.ddms[upperbound]
        ddmoptmin = self.ddms[lowerbound]

        self.ddmopt = ddmopt
        self.ddmoptmin = ddmoptmin
        self.ddmoptmax = ddmoptmax

    def viastructure(self) -> None:
        cdmt = dct(self.dmt, norm="ortho")
        _, nk = cdmt.shape
        Kc = getKc(cdmt)

        fO = 3
        k = np.linspace(1, nk, nk)
        fL = 1 / (1 + (k / Kc) ** (2 * fO))

        fLdiag = np.diag(fL)
        lpfdata = fLdiag @ cdmt.T
        dmtsmooth = idct(lpfdata.T, norm="ortho")
        fH = np.sqrt(2 - 2 * np.cos((k - 1) * np.pi / nk))

        combof = fH * fL
        combofdiag = np.diag(combof)
        cifiltered = combofdiag @ cdmt.T
        cifilteredN = np.asarray(norm(cifiltered, axis=0))

        deltai = self.dmt - dmtsmooth
        dbfilt = combofdiag @ lpfdata
        dbfiltN = norm(dbfilt, axis=0)
        cdeltai = dct(deltai, norm="ortho")
        cdeltaifilt = combofdiag @ cdeltai.T
        cdeltaifiltN = norm(cdeltaifilt, axis=0)
        uncertainty = np.asarray(cdeltaifiltN / dbfiltN)
        self.uncertainty = uncertainty

        maxbin = np.argmax(cifilteredN)
        deltadeltai = deltai - deltai[maxbin]
        cdeltadeltai = dct(deltadeltai, norm="ortho")
        cdeltadeltaifilt = combofdiag @ cdeltadeltai.T
        cdeltadeltaifiltN = norm(cdeltadeltaifilt, axis=0)
        relative_uncertainty = np.asarray(cdeltadeltaifiltN / dbfiltN)
        self.relative_uncertainty = relative_uncertainty

        ddmopt = self.ddms[maxbin]
        maxSP = cifilteredN[maxbin]
        adjustedSPs = cifilteredN + (cifilteredN * relative_uncertainty)

        maxranges = []
        continuous = False
        for i, value in enumerate(adjustedSPs):
            if not continuous and value >= maxSP:
                continuous = True
                maxranges.append([i])
            elif continuous and value < maxSP:
                continuous = False
                maxranges[-1].append(i)

        if len(maxranges) >= 1:
            ddmoptmin = float(self.ddms[maxranges[0][0]])
            ddmoptmax = (
                float(self.ddms[maxranges[-1][1]])
                if len(maxranges[-1]) == 2
                else self.ddms[-1]
            )
        else:
            ddmoptmin = self.ddms[0]
            ddmoptmax = self.ddms[-1]

        self.ddmopt = ddmopt
        self.structparam = maxSP
        self.ddmoptmin = ddmoptmin
        self.ddmoptmax = ddmoptmax
