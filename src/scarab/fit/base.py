import inspect
from typing import Self
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


@dataclass
class Fitter:

    burst: Burst
    multiple: bool

    SM: str
    PM: str
    PR: ModelResult
    SR: ModelResult
    PF: ProfileFitter
    SF: SpectrumFitter
    results: dict[str, ModelResult]

    @classmethod
    def fit(
        cls,
        burst,
        multiple: bool = False,
        withmodels: tuple[str, str] = ("unscattered", "gaussian"),
    ) -> Self:
        PM, SM = withmodels
        SF = SpectrumFitter.fit(burst, SM)
        PF = ProfileFitter.fit(burst, PM, multiple=multiple)

        if (PF.result is not None) and (SF.result is not None):
            PR = PF.result
            SR = SF.result
            results = {"profile": PR, "spectrum": SR}
        else:
            raise RuntimeError(
                {
                    (True, True): "Fit failed!",
                    (True, False): "Profile fit failed!",
                    (False, True): "Spectrum fit failed",
                }[(PF.result is None, SF.result is None)]
            )

        return cls(
            PM=PM,
            SM=SM,
            PF=PF,
            SF=SF,
            PR=PR,
            SR=SR,
            burst=burst,
            results=results,
            multiple=multiple,
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

        pxtoptop.plot(self.burst.times, self.burst.normprofile, lw=1, alpha=0.5)
        pxtoptop.plot(self.burst.times, self.PR.best_fit, lw=2)
        for name, component in self.PR.eval_components().items():
            pxtop.plot(self.burst.times, component, lw=2, label=name)

        pxside.plot(
            self.burst.freqs,
            self.burst.normspectrum,
            orientation="horizontal",
            lw=1,
            alpha=0.5,
        )

        pxside.plot(self.burst.freqs, self.SR.best_fit, orientation="horizontal", lw=2)

        ax.imshow(
            self.burst.data,
            aspect="auto",
            cmap="batlow",
            interpolation="none",
            extent=[
                self.burst.times[0],
                self.burst.times[-1],
                self.burst.freqs[-1],
                self.burst.freqs[0],
            ],
        )

        ax.format(
            xlabel="Time (s)",
            ylabel="Frequency (MHz)",
            suptitle=f"Fit for {self.burst.path.name}",
        )

        pplt.show()
