import inspect
from dataclasses import dataclass

import numpy as np
import proplot as pplt
from lmfit import Model
from lmfit.model import ModelResult

from scarab.base import Burst
from scarab.dm import dm2delay
from scarab.transform import clipper, masker

from scarab.fit.models import (
    gauss,
    normgauss,
    runpowlaw,
    scatanalytic,
    scatbandintmodel,
    scatconvolving,
    scatgauss_afb_instrumental,
    scatgauss_dfb_instrumental,
)

# TODO: Absorb fitburst (https://github.com/CHIMEFRB/fitburst).
# This will allow us to fit the entire dynamic spectrum in one
# go, instead of fitting the profile and spectrum separately.


@dataclass
class ProfileFitter:
    burst: Burst
    fitted: bool = False
    result: ModelResult | None = None

    def fit(self, withmodel: str):
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

        maxima = np.max(self.burst.profile)
        minima = np.min(self.burst.profile)
        ixmax = np.argmax(self.burst.profile)

        model = Model(modelfunc)
        args = list(inspect.signature(modelfunc).parameters.keys())

        model.set_param_hint("tau", value=1.0)
        model.set_param_hint("sigma", value=1.0)
        model.set_param_hint("dc", value=minima)
        model.set_param_hint("fluence", value=maxima - minima)
        model.set_param_hint("center", value=self.burst.times[ixmax])

        if "taui" in args:
            model.set_param_hint("taui", value=self.burst.dt, vary=False)

        if "taud" in args:
            model.set_param_hint(
                "taud",
                value=dm2delay(
                    self.burst.fc - 0.5 * self.burst.df,
                    self.burst.fc + 0.5 * self.burst.df,
                    self.burst.dm,
                ),
                vary=False,
            )

        if withmodel == "scattering_isotropic_bandintegrated":
            model.set_param_hint("flow", value=self.burst.fl)
            model.set_param_hint("fhigh", value=self.burst.fh)
            model.set_param_hint("nf", value=9, vary=False)

        params = model.make_params()
        params.add("w50i", expr="2.3548200*sigma")
        params.add("w10i", expr="4.2919320*sigma")

        self.result = model.fit(
            params=params,
            method="leastsq",
            x=self.burst.times,
            data=self.burst.profile,
        )


@dataclass
class SpectrumFitter:
    burst: Burst
    fitted: bool = False
    result: ModelResult | None = None

    def fit(self, withmodel: str) -> None:
        specfx = {
            "gaussian": gauss,
            "running_power_law": runpowlaw,
        }.get(withmodel, None)
        if specfx is None:
            raise NotImplementedError(f"Model {withmodel} is not implemented.")

        maxima = np.max(self.burst.spectrum)
        minima = np.min(self.burst.spectrum)
        ixmax = np.argmax(self.burst.spectrum)

        model = Model(specfx)
        model.set_param_hint("dc", value=minima)
        model.set_param_hint("fluence", value=maxima - minima)
        if withmodel == "gaussian":
            model.set_param_hint("sigma", value=1.0)
            model.set_param_hint("center", value=self.burst.freqs[ixmax])
        elif withmodel == "running_power_law":
            model.set_param_hint("beta", value=0.0)
            model.set_param_hint("gamma", value=0.0)
            model.set_param_hint("xref", value=self.burst.fh, vary=False)

        params = model.make_params()

        self.result = model.fit(
            params=params,
            method="leastsq",
            x=self.burst.freqs,
            data=self.burst.spectrum,
        )


class Fitter:

    def __init__(self, burst: Burst) -> None:
        self.burst = burst
        self.R = {"profile": None, "spectrum": None}

    @property
    def results(self):
        return self.R

    def result(self, which: str):
        return self.R[which]

    def fit(
        self,
        withmodels: tuple[str, str],
        bcwidth: int = 10,
        within: float = 100e-3,
        threshold: float = 10.0,
    ):
        self.C = clipper(self.burst, within=within)
        self.M = masker(self.C, bcwidth=bcwidth, threshold=threshold)

        self.PM, self.SM = withmodels
        self.PF = ProfileFitter(self.M)
        self.SF = SpectrumFitter(self.M)

        self.PF.fit(self.PM)
        self.SF.fit(self.SM)
        self.PR = self.PF.result
        self.SR = self.SF.result
        self.R = {"profile": self.PR, "spectrum": self.SR}

    def plot(self):
        fig = pplt.figure(width=5, height=5)

        ax = fig.subplot()  # type: ignore
        pxt = ax.panel_axes("top", width="5em", space=0)
        pxr = ax.panel_axes("right", width="5em", space=0)

        pxt.set_yticks([])
        pxr.set_xticks([])

        pxt.plot(self.M.times, self.M.profile)
        pxr.plot(self.C.freqs, self.C.spectrum, orientation="horizontal")
        pxr.plot(self.M.freqs, self.M.spectrum, orientation="horizontal")

        if (self.PR is not None) and (self.SR is not None):
            pxt.plot(self.M.times, self.PR.best_fit)
            pxr.plot(self.M.freqs, self.SR.best_fit, orientation="horizontal")

        ax.fill_betweenx(
            self.M.freqs,
            self.M.times[0],
            self.M.times[-1],
            alpha=0.20,
            color="grey",
        )

        ax.axhline(self.M.freqs[0], color="black", lw=1.5, ls="--")
        ax.axhline(self.M.freqs[-1], color="black", lw=1.5, ls="--")
        pxr.axhline(self.M.freqs[0], color="black", lw=1.5, ls="--")
        pxr.axhline(self.M.freqs[-1], color="black", lw=1.5, ls="--")

        ax.imshow(
            self.C.data,
            aspect="auto",
            cmap="batlow",
            interpolation="none",
            extent=[
                self.C.times[0],
                self.C.times[-1],
                self.C.freqs[-1],
                self.C.freqs[0],
            ],
        )

        ax.format(
            xlabel="Time (s)",
            ylabel="Frequency (MHz)",
            suptitle=f"{self.burst.path.name}",
        )

        pplt.show()
