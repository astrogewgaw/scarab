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
class Fitter:
    burst: Burst
    masked: Burst
    profres: ModelResult
    specres: ModelResult

    @classmethod
    def fit(
        cls,
        burst: Burst,
        withmodels: tuple[str, str],
        zoommaskwithin: float = 100e-3,
        emitmaskthreshold: float = 10.0,
    ) -> Self:
        zoomed = burst.zoommask(within=zoommaskwithin)
        masked = zoomed.emitmask(threshold=emitmaskthreshold)

        profmn, specmn = withmodels

        proffx = {
            "unscattered": normgauss,
            "scattering_isotropic_analytic": scatanalytic,
            "scattering_isotropic_convolving": scatconvolving,
            "scattering_isotropic_bandintegrated": scatbandintmodel,
            "scattering_isotropic_afb_instrumental": scatgauss_afb_instrumental,
            "scattering_isotropic_dfb_instrumental": scatgauss_dfb_instrumental,
        }.get(profmn, None)

        if proffx is None:
            raise NotImplementedError(f"Model {profmn} is not implemented.")

        binmax = np.argmax(masked.profile)
        profmax = np.max(masked.profile)
        profmin = np.min(masked.profile)
        profptp = profmax - profmin

        profmodel = Model(proffx)
        args = list(inspect.signature(proffx).parameters.keys())

        profmodel.set_param_hint("tau", value=1.0)
        profmodel.set_param_hint("sigma", value=1.0)
        profmodel.set_param_hint("dc", value=profmin)
        profmodel.set_param_hint("fluence", value=profptp)
        profmodel.set_param_hint("center", value=masked.times[binmax])

        if "taui" in args:
            profmodel.set_param_hint("taui", value=masked.dt, vary=False)

        if "taud" in args:
            profmodel.set_param_hint(
                "taud",
                value=dm2delay(
                    masked.fc - 0.5 * masked.df,
                    masked.fc + 0.5 * masked.df,
                    masked.dm,
                ),
                vary=False,
            )

        if profmn == "scattering_isotropic_bandintegrated":
            profmodel.set_param_hint("flow", value=masked.fl)
            profmodel.set_param_hint("fhigh", value=masked.fh)
            profmodel.set_param_hint("nf", value=9, vary=False)

        profparams = profmodel.make_params()
        profparams.add("w50i", expr="2.3548200*sigma")
        profparams.add("w10i", expr="4.2919320*sigma")

        profres = profmodel.fit(
            method="leastsq",
            params=profparams,
            x=masked.times,
            data=masked.profile,
        )

        specfx = {
            "gaussian": gauss,
            "running_power_law": runpowlaw,
        }.get(specmn, None)
        if specfx is None:
            raise NotImplementedError(f"Model {specmn} is not implemented.")

        binmax = np.argmax(masked.spectrum)
        specmax = np.max(masked.spectrum)
        specmin = np.min(masked.spectrum)
        specptp = specmax - specmin

        specmodel = Model(specfx)
        specmodel.set_param_hint("fluence", value=specptp)
        specmodel.set_param_hint("dc", value=specmin)
        if specmn == "gaussian":
            specmodel.set_param_hint("sigma", value=1.0)
            specmodel.set_param_hint("center", value=masked.freqs[binmax])
        elif specmn == "running_power_law":
            specmodel.set_param_hint("beta", value=0.0)
            specmodel.set_param_hint("gamma", value=0.0)
            specmodel.set_param_hint("xref", value=masked.fh, vary=False)

        specparams = specmodel.make_params()

        specres = specmodel.fit(
            method="leastsq",
            params=specparams,
            x=masked.freqs,
            data=masked.spectrum,
        )

        return cls(zoomed, masked, profres, specres)

    def plot(self):
        fig = pplt.figure(width=5, height=5)

        ax = fig.subplot()  # type: ignore
        pxt = ax.panel_axes("top", width="5em", space=0)
        pxr = ax.panel_axes("right", width="5em", space=0)

        pxt.set_yticks([])
        pxr.set_xticks([])

        pxt.plot(self.masked.times, self.masked.profile)
        pxt.plot(self.masked.times, self.profres.best_fit)
        pxr.plot(self.burst.freqs, self.burst.spectrum, orientation="horizontal")
        pxr.plot(self.masked.freqs, self.masked.spectrum, orientation="horizontal")
        pxr.plot(self.masked.freqs, self.specres.best_fit, orientation="horizontal")

        ax.fill_betweenx(
            self.masked.freqs,
            self.masked.times[0],
            self.masked.times[-1],
            color="grey",
            alpha=0.20,
        )

        ax.axhline(self.masked.freqs[0], color="black", lw=1.5, ls="--")
        ax.axhline(self.masked.freqs[-1], color="black", lw=1.5, ls="--")
        pxr.axhline(self.masked.freqs[0], color="black", lw=1.5, ls="--")
        pxr.axhline(self.masked.freqs[-1], color="black", lw=1.5, ls="--")

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
