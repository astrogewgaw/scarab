from typing import Self
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import proplot as pplt
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

    # Output variables
    ddmopt: float | None = None
    ddmoptmin: float | None = None
    ddmoptmax: float | None = None

    # Diagnostic variables
    Kc: float | None = None
    maxSP: float | None = None
    SPs: np.ndarray | None = None
    detrended: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    adjustedSPs: np.ndarray | None = None
    relative_detrended: np.ndarray | None = None
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

        self.Kc = Kc
        self.detrended = deltai

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

        maxbin = np.argmax(cifilteredN)
        deltadeltai = deltai - deltai[maxbin]
        cdeltadeltai = dct(deltadeltai, norm="ortho")
        cdeltadeltaifilt = combofdiag @ cdeltadeltai.T
        cdeltadeltaifiltN = norm(cdeltadeltaifilt, axis=0)
        relative_uncertainty = np.asarray(cdeltadeltaifiltN / dbfiltN)

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
        self.ddmoptmin = ddmoptmin
        self.ddmoptmax = ddmoptmax

        self.Kc = Kc
        self.maxSP = maxSP
        self.SPs = cifilteredN
        self.detrended = deltai
        self.adjustedSPs = adjustedSPs
        self.uncertainty = uncertainty
        self.relative_detrended = deltadeltai
        self.relative_uncertainty = relative_uncertainty

    def plot_dmt(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "dmt.png",
    ):
        def _(ax: pplt.Axes) -> None:
            ax.imshow(
                self.dmt,
                aspect="auto",
                cmap="batlow",
                origin="lower",
                interpolation="none",
                extent=[
                    self.burst.times[0],
                    self.burst.times[-1],
                    self.ddmmin,
                    self.ddmmax,
                ],
            )

            ax.format(
                xlabel="Time (ms)",
                ylabel=r"$\Delta$DM (pc cm$^{-3}$)",
                suptitle=f"DMT for {self.burst.path.name}",
            )

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

    def plot_maxts(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "maxts.png",
    ):
        if self.ddmopt is not None:
            zeroed = dedisperse(
                dt=self.burst.dt,
                dm=-self.burst.dm,
                data=self.burst.data,
                freqs=self.burst.freqs,
            )

            maxts = dedisperse(
                data=zeroed,
                collapse=True,
                dt=self.burst.dt,
                freqs=self.burst.freqs,
                dm=self.burst.dm + self.ddmopt,
            )

            def _(ax: pplt.Axes) -> None:
                ax.plot(self.burst.times, maxts)

                if self.ddmopt is not None:
                    ax.format(
                        xlabel="Time (ms)",
                        ylabel="Intensity (arbitrary units)",
                        suptitle=f"Time series at optimized DM of {self.burst.dm + self.ddmopt:.4f}",
                    )

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
        else:
            ValueError("Cannot plot since ddmopt is None.")

    def plot_dct(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "dct.png",
    ):
        def _(ax: pplt.Axes) -> None:
            ax.plot(np.abs(dct(self.dmt, norm="ortho")).T, ".")

            if self.Kc is not None:
                ax.axvline(self.Kc, ls="--")  # type: ignore

            ax.format(
                xscale="log",
                yscale="log",
                suptitle="DCT spectrum",
                xlabel=r"$\mathregular{k}$ Index",
                ylabel=r"DCT coefficients, $\mathregular{|C^Ti|}$",
            )

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

    def plot_SPs(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "SPs.png",
    ):
        def _(ax: pplt.Axes) -> None:
            if self.SPs is not None:
                ax.plot(self.ddms, self.SPs)

                ax.format(
                    ylabel="Structure Parameter",
                    xlabel=r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)",
                    suptitle=r"Variation of structure parameters with $\Delta$DM",
                )
            else:
                ValueError("Cannot plot since `SPs` is None.")

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

    def plot_detrended(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "detrended.png",
    ):
        def _(ax: pplt.Axes) -> None:
            if self.detrended is not None:
                nt = self.detrended.shape[1]
                tobs = nt * self.burst.dt

                ax.imshow(
                    self.detrended,
                    aspect="auto",
                    extent=[
                        0.0,
                        tobs,
                        self.ddmmin,
                        self.ddmmax,
                    ],
                )

                ax.format(
                    xlabel="Time (ms)",
                    suptitle="Detrended noise",
                    ylabel=r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)",
                )
            else:
                ValueError("Cannot plot since `detrended` is None.")

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

    def plot_relative_detrended(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "relative_detrended.png",
    ):
        def _(ax: pplt.Axes) -> None:
            if self.relative_detrended is not None:
                nt = self.relative_detrended.shape[1]
                tobs = nt * self.burst.dt

                ax.imshow(
                    self.relative_detrended,
                    aspect="auto",
                    extent=[
                        0.0,
                        tobs,
                        self.ddmmin,
                        self.ddmmax,
                    ],
                )

                ax.format(
                    xlabel="Time (ms)",
                    suptitle="Relative detrended noise",
                    ylabel=r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)",
                )
            else:
                ValueError("Cannot plot since `relative_detrended` is None.")

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

    def plot_uncertainity(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "uncertainty.png",
    ):
        def _(ax: pplt.Axes) -> None:
            if self.uncertainty is not None:
                ax.plot(self.ddms, 100 * self.uncertainty)

                ax.format(
                    ylabel="Uncertainty (%)",
                    suptitle=r"Uncertainity v/s $\Delta$DM",
                    xlabel=r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)",
                )
            else:
                ValueError("Cannot plot since `uncertainty` is None.")

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

    def plot_relative_uncertainity(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "relative_uncertainty.png",
    ):
        def _(ax: pplt.Axes) -> None:
            if self.relative_uncertainty is not None:
                ax.plot(self.ddms, 100 * self.relative_uncertainty)

                ax.format(
                    ylabel="Relative uncertainty (%)",
                    suptitle=r"Relative uncertainity v/s $\Delta$DM",
                    xlabel=r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)",
                )
            else:
                ValueError("Cannot plot since `relative_uncertainty` is None.")

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

    def plot_adjustedSPs(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "adjustedSPs.png",
    ):
        def _(ax: pplt.Axes) -> None:
            if self.adjustedSPs is not None:
                ax.plot(self.ddms, self.adjustedSPs)

                ax.format(
                    ylabel="Adjusted Structure Parameter",
                    xlabel=r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)",
                    suptitle="Relative Uncertainty Adjusted Structure Parameter",
                )
            else:
                ValueError("Cannot plot since `adjustedSPs` is None.")

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
