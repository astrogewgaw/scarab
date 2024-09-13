import inspect
from typing import Self
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
import proplot as pplt
from lmfit import Model
from rich.progress import track
from lmfit.model import ModelResult
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

from scarab.base import Burst
from scarab.dm import dm2delay

from scarab.fit.models import (
    rpl,
    gauss,
    normgauss,
    scatanalytic,
    scatconvolving,
    scatbandintmodel,
    scatgauss_afb_instrumental,
    scatgauss_dfb_instrumental,
)


# TODO: Absorb fitburst (https://github.com/CHIMEFRB/fitburst).
# This will allow us to fit the entire dynamic spectrum in one
# go, instead of fitting the profile and spectrum separately.


def peakfind(data: np.ndarray, winsize: int = 10, perthres: float = 0.05) -> list:
    smooth = median_filter(data, winsize)
    peaks, _ = find_peaks(smooth, distance=2 * winsize)
    peaks = peaks[smooth[peaks] >= perthres * smooth.max()]
    peaks = list(reversed([peak for _, peak in sorted(zip(smooth[peaks], peaks))]))
    return peaks


@dataclass
class ProfileFitter:

    burst: Burst
    multiple: bool
    result: ModelResult
    tries: list[ModelResult] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        burst: Burst,
        withmodel: str,
        multiple: bool = False,
    ) -> Self:
        modelfunc = {
            "unscattered": normgauss,
            "scattering_isotropic_analytic": scatanalytic,
            "scattering_isotropic_convolving": scatconvolving,
            "scattering_isotropic_bandintegrated": scatbandintmodel,
            "scattering_isotropic_afb_instrumental": scatgauss_afb_instrumental,
            "scattering_isotropic_dfb_instrumental": scatgauss_dfb_instrumental,
        }.get(withmodel, None)

        if modelfunc is None:
            raise NotImplementedError(f"Model {withmodel} is not implemented.")

        ixmax = np.argmax(burst.normprofile)
        tmax = burst.times[-1]
        tmin = burst.times[0]

        if multiple:
            peaks = peakfind(
                perthres=0.05,
                data=burst.normprofile,
                winsize=int(250e-6 / burst.dt),
            )
        else:
            peaks = [ixmax]

        components = []
        for i in range(len(peaks)):
            component = Model(modelfunc, prefix=f"P{i + 1}")
            components.append(component)
        models = np.cumsum(components)

        tries = []
        for M in track(models, description="Trying models..."):
            nc = len(M.components)
            for i, peak in enumerate(peaks[:nc]):
                tpeak = burst.times[peak]
                cmax = burst.normprofile[peak]
                cmin = np.min(burst.normprofile)
                coff = np.median(burst.normprofile[:100])

                M.set_param_hint(f"P{i + 1}dc", value=coff, min=cmin)
                M.set_param_hint(f"P{i + 1}center", min=tmin, max=tmax, value=tpeak)
                M.set_param_hint(f"P{i + 1}tau", value=1.0, min=burst.dt, max=np.inf)
                M.set_param_hint(f"P{i + 1}sigma", value=1.0, min=burst.dt, max=np.inf)

                M.set_param_hint(
                    f"P{i + 1}fluence",
                    min=0.0,
                    max=np.inf,
                    value=cmax - coff,
                )

                args = list(inspect.signature(modelfunc).parameters.keys())

                if "taui" in args:
                    M.set_param_hint(f"P{i + 1}taui", value=burst.dt, vary=False)

                if "taud" in args:
                    M.set_param_hint(
                        f"P{i + 1}taud",
                        value=dm2delay(
                            burst.fc - 0.5 * burst.df,
                            burst.fc + 0.5 * burst.df,
                            burst.dm,
                        ),
                        vary=False,
                    )

            if withmodel == "scattering_isotropic_bandintegrated":
                M.set_param_hint("flow", value=burst.fl)
                M.set_param_hint("fhigh", value=burst.fh)
                M.set_param_hint("nf", value=9, vary=False)

            params = M.make_params()
            for i in range(1, len(peaks[:nc])):
                params[f"P{i + 1}dc"].expr = "P1dc"

            tries.append(
                M.fit(
                    params=params,
                    x=burst.times,
                    method="leastsq",
                    data=burst.normprofile,
                )
            )

        result = tries[np.asarray([_.bic for _ in tries]).argmin()]

        return cls(
            burst=burst,
            tries=tries,
            result=result,
            multiple=multiple,
        )

    def plot_fit(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "bestfit.png",
    ):
        def _(ax: pplt.Axes) -> None:
            ax.plot(self.burst.times, self.burst.normprofile, lw=1, alpha=0.5)
            ax.plot(self.burst.times, self.result.best_fit, lw=2)

        if ax is None:
            fig = pplt.figure(width=5, height=2.5)
            ax = fig.subplots(nrows=1, ncols=1)[0]
            assert ax is not None
            _(ax)
            if save:
                fig.savefig(saveto, dpi=dpi)
            if show:
                pplt.show()
        else:
            _(ax)

    def plot_components(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "components.png",
    ):
        def _(ax: pplt.Axes) -> None:
            components = self.result.eval_components()
            for name, component in components.items():
                ax.plot(
                    lw=2,
                    label=name,
                    data=component,
                    x=self.burst.times,
                )

        if ax is None:
            fig = pplt.figure(width=5, height=2.5)
            ax = fig.subplots(nrows=1, ncols=1)[0]
            assert ax is not None
            _(ax)
            if save:
                fig.savefig(saveto, dpi=dpi)
            if show:
                pplt.show()
        else:
            _(ax)


@dataclass
class SpectrumFitter:

    burst: Burst
    result: ModelResult

    @classmethod
    def fit(cls, burst: Burst, withmodel: str) -> Self:
        modelfunc = {
            "gaussian": gauss,
            "running_power_law": rpl,
        }.get(withmodel, None)
        if modelfunc is None:
            raise NotImplementedError(f"Model {withmodel} is not implemented.")

        maxima = np.max(burst.normspectrum)
        minima = np.min(burst.normspectrum)
        ixmax = np.argmax(burst.normspectrum)

        model = Model(modelfunc)
        model.set_param_hint("dc", value=minima)
        model.set_param_hint("fluence", value=maxima - minima)

        if withmodel == "gaussian":
            model.set_param_hint("sigma", value=1.0)
            model.set_param_hint("center", value=burst.freqs[ixmax])
        elif withmodel == "running_power_law":
            model.set_param_hint("beta", value=0.0)
            model.set_param_hint("gamma", value=0.0)
            model.set_param_hint("xref", value=burst.fh, vary=False)

        params = model.make_params()

        result = model.fit(
            params=params,
            x=burst.freqs,
            method="leastsq",
            data=burst.normspectrum,
        )

        return cls(burst=burst, result=result)

    def plot_fit(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "bestfit.png",
    ):
        def _(ax: pplt.Axes) -> None:
            ax.plot(self.burst.times, self.burst.normprofile, lw=1, alpha=0.5)
            ax.plot(self.burst.times, self.result.best_fit, lw=2)

        if ax is None:
            fig = pplt.figure(width=5, height=2.5)
            ax = fig.subplots(nrows=1, ncols=1)[0]
            assert ax is not None
            _(ax)
            if save:
                fig.savefig(saveto, dpi=dpi)
            if show:
                pplt.show()
        else:
            _(ax)


@dataclass
class Fitter:

    burst: Burst
    multiple: bool
    models: dict[str, str]
    results: dict[str, ModelResult]
    fitters: dict[str, ProfileFitter | SpectrumFitter]

    @classmethod
    def fit(
        cls,
        burst,
        multiple: bool = False,
        withmodels: tuple[str, str] = ("unscattered", "gaussian"),
    ) -> Self:
        pm, sm = withmodels
        sf = SpectrumFitter.fit(burst, sm)
        pf = ProfileFitter.fit(burst, pm, multiple=multiple)

        pr = pf.result
        sr = sf.result
        if (pr is not None) and (sr is not None):
            return cls(
                burst=burst,
                multiple=multiple,
                models={"profile": pm, "spectrum": sm},
                fitters={"profile": pf, "spectrum": sf},
                results={"profile": pf.result, "spectrum": sf.result},
            )
        else:
            raise RuntimeError(
                {
                    (True, True): "Fit failed!",
                    (True, False): "Profile fit failed!",
                    (False, True): "Spectrum fit failed",
                }[(pf.result is None, sf.result is None)]
            )

    def plot(self):
        fig = pplt.figure(width=5, height=5)
        ax = fig.subplots(nrows=1, ncols=1)[0]
        pxtop = ax.panel_axes("top", width="5em", space=0)
        pxtoptop = ax.panel_axes("top", width="5em", space=0)
        pxside = ax.panel_axes("right", width="5em", space=0)

        pxtop.set_yticks([])
        pxside.set_xticks([])
        pxtoptop.set_yticks([])

        if isinstance(self.fitters["profile"], SpectrumFitter):
            self.fitters["spectrum"].plot_fit(pxside)
        if isinstance(self.fitters["profile"], ProfileFitter):
            self.fitters["profile"].plot_fit(pxtoptop)
            self.fitters["profile"].plot_components(pxtop)
        self.burst.plot(ax=ax, withprof=False, withspec=False)
        ax.format(suptitle=f"Best fit for {self.burst.path.name}")

        pplt.show()
