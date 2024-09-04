import inspect
from typing import Self
from dataclasses import dataclass

import numpy as np
import proplot as pplt
from lmfit import Model
from lmfit.model import ModelResult

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


@dataclass
class ProfileFitter:

    burst: Burst
    result: ModelResult

    @classmethod
    def fit(cls, burst: Burst, withmodel: str) -> Self:
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

        maxima = np.max(burst.normprofile)
        minima = np.min(burst.normprofile)
        ixmax = np.argmax(burst.normprofile)

        model = Model(modelfunc)
        args = list(inspect.signature(modelfunc).parameters.keys())

        model.set_param_hint("tau", value=1.0)
        model.set_param_hint("sigma", value=1.0)
        model.set_param_hint("dc", value=minima)
        model.set_param_hint("center", value=burst.times[ixmax])
        model.set_param_hint("fluence", value=maxima - minima)

        if "taui" in args:
            model.set_param_hint("taui", value=burst.dt, vary=False)

        if "taud" in args:
            model.set_param_hint(
                "taud",
                value=dm2delay(
                    burst.fc - 0.5 * burst.df,
                    burst.fc + 0.5 * burst.df,
                    burst.dm,
                ),
                vary=False,
            )

        if withmodel == "scattering_isotropic_bandintegrated":
            model.set_param_hint("nf", value=9, vary=False)
            model.set_param_hint("flow", value=burst.fl)
            model.set_param_hint("fhigh", value=burst.fh)

        params = model.make_params()

        result = model.fit(
            params=params,
            x=burst.times,
            method="leastsq",
            data=burst.normprofile,
        )

        return cls(burst=burst, result=result)


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
        withmodels: tuple[str, str] = ("unscattered", "gaussian"),
    ) -> Self:
        PM, SM = withmodels
        PF = ProfileFitter.fit(burst, PM)
        SF = SpectrumFitter.fit(burst, SM)

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
        )

    def plot(self):
        fig = pplt.figure(width=5, height=5)
        ax = fig.subplots(nrows=1, ncols=1)[0]
        pxtop = ax.panel_axes("top", width="5em", space=0)
        pxside = ax.panel_axes("right", width="5em", space=0)

        pxtop.set_yticks([])
        pxside.set_xticks([])

        pxtop.plot(self.burst.times, self.burst.normprofile)
        pxtop.plot(self.burst.times, self.PR.best_fit)

        pxside.plot(self.burst.freqs, self.burst.normspectrum, orientation="horizontal")
        pxside.plot(self.burst.freqs, self.SR.best_fit, orientation="horizontal")

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
            suptitle=f"{self.burst.path.name}",
        )

        pplt.show()
