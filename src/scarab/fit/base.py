import inspect
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
import pandas as pd
import proplot as pplt
from lmfit import Model
from rich.progress import track
from lmfit.model import ModelResult
from joblib import Parallel, delayed

from scarab.dm import dm2delay
from scarab.peaks import PeakFinder
from scarab.base import Burst, Bursts
from scarab.utilities import w10gauss, w50gauss

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


@dataclass
class FittedBurst(Burst):
    pass


@dataclass
class FittedBursts(Bursts):
    pass


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
        njobs: int = 4,
        multiple: bool = False,
        **kwargs,
    ):
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
            peaks = PeakFinder.find(burst.normprofile, **kwargs).peaks
        else:
            peaks = [ixmax]

        components = []
        for i in range(len(peaks)):
            component = Model(modelfunc, prefix=f"P{i + 1}")
            components.append(component)
        models = np.cumsum(components)

        def tryfit(M: Model) -> ModelResult:
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

            return M.fit(
                params=params,
                x=burst.times,
                method="leastsq",
                data=burst.normprofile,
            )

        tries: list = list(
            track(
                Parallel(
                    return_as="generator",
                    n_jobs=njobs if multiple else 1,
                )(delayed(tryfit)(M) for M in models),
                total=len(models),
            )
        )

        return cls(
            burst=burst,
            tries=tries,
            multiple=multiple,
            result=tries[np.asarray([_.bic for _ in tries]).argmin()],
        )

    @property
    def ncomps(self) -> int:
        comps = getattr(self.result, "components", None)
        numcomps = len(comps) if comps is not None else 1
        return numcomps

    @property
    def components(self) -> dict:
        return self.result.eval_components()

    @property
    def postfit(self) -> pd.DataFrame:
        allvals = {
            (suffix := f"P{i + 1}"): {
                key.replace(suffix, ""): val
                for key, val in self.result.best_values.items()
                if key.startswith(suffix)
            }
            for i in range(self.ncomps)
        }

        covar = np.asarray(self.result.covar)
        allerrs = np.sqrt(np.diag(covar))
        dcerr = allerrs[-1]
        allerrs = allerrs[:-1]

        allvalsnerrs = {}
        for i, (suffix, compvals) in enumerate(allvals.items()):
            compvalsnerrs = {}
            for j, key in enumerate(compvals.keys()):
                compvalsnerrs[key] = compvals[key]
                if key != "dc":
                    compvalsnerrs["".join([key, "_err"])] = allerrs[i * self.ncomps + j]
                else:
                    compvalsnerrs["".join([key, "_err"])] = dcerr
            allvalsnerrs[suffix] = compvalsnerrs

        return pd.DataFrame(
            [
                {
                    "t0": float(compvals["center"]),
                    "t0_err": float(compvals["center_err"]),
                    "W50": w50gauss(float(compvals["sigma"])),
                    "W10": w10gauss(float(compvals["sigma"])),
                    "W50_err": w50gauss(float(compvals["sigma_err"])),
                    "W10_err": w10gauss(float(compvals["sigma_err"])),
                    "tau": float(compvals["tau"]),
                    "tau_err": float(compvals["tau_err"]),
                }
                for _, compvals in allvalsnerrs.items()
            ]
        )

    def plotfit(
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

    def plotcomps(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "components.png",
    ):
        def _(ax: pplt.Axes) -> None:
            for name, component in self.components.items():
                ax.plot(
                    self.burst.times,
                    component,
                    lw=2,
                    label=name,
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
    def fit(cls, burst: Burst, withmodel: str):
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

    @property
    def postfit(self) -> pd.DataFrame:
        ferr = self.burst.df
        femitn = self.burst.freqs[0]
        femit0 = self.burst.freqs[-1]
        emitbw = self.burst.freqs[0] - self.burst.freqs[-1]
        femitmax = float(self.burst.freqs[self.result.best_fit.argmax()])
        return pd.DataFrame(
            [
                {
                    "femitmax": femitmax,
                    "femitmax_err": ferr,
                    "femit0": femit0,
                    "femit0_err": ferr,
                    "femitn": femitn,
                    "femitn_err": ferr,
                    "emitbw": emitbw,
                    "emitbw_err": ferr,
                }
            ]
        )

    def plotfit(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "bestfit.png",
    ):
        def _(ax: pplt.Axes, flip: bool = False) -> None:
            ax.plot(
                self.burst.freqs,
                self.burst.normspectrum,
                lw=1,
                alpha=0.5,
                orientation="horizontal" if flip else "vertical",
            )

            ax.plot(
                self.burst.freqs,
                self.result.best_fit,
                lw=2,
                orientation="horizontal" if flip else "vertical",
            )

        if ax is None:
            fig = pplt.figure(width=5, height=2.5)
            ax = fig.subplots(nrows=1, ncols=1)[0]
            assert ax is not None
            _(ax, flip=False)
            if save:
                fig.savefig(saveto, dpi=dpi)
            if show:
                pplt.show()
        else:
            _(ax, flip=True)


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
        njobs: int = 4,
        multiple: bool = False,
        withmodels: tuple[str, str] = ("unscattered", "gaussian"),
        **kwargs,
    ):
        pm, sm = withmodels

        sf = SpectrumFitter.fit(burst, sm)
        pf = ProfileFitter.fit(burst, pm, njobs=njobs, multiple=multiple, **kwargs)

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

    @property
    def ncomps(self) -> int:
        if isinstance(self.fitters["profile"], ProfileFitter):
            return self.fitters["profile"].ncomps
        else:
            raise RuntimeError("Something is terribly wrong! Exiting...")

    @property
    def components(self) -> dict:
        if isinstance(self.fitters["profile"], ProfileFitter):
            return self.fitters["profile"].components
        else:
            raise RuntimeError("Something is terribly wrong! Exiting...")

    @property
    def postfit(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.fitters["profile"].postfit,
                pd.concat(
                    [self.fitters["spectrum"].postfit] * self.ncomps,
                    ignore_index=True,
                ),
            ],
            axis=1,
            join="inner",
        )

    def plotfit(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "bestfit.png",
    ):
        def _(ax: pplt.Axes) -> None:
            pxtop = ax.panel_axes("top", width="5em", space=0)
            pxtoptop = ax.panel_axes("top", width="5em", space=0)
            pxside = ax.panel_axes("right", width="5em", space=0)

            pxtop.set_yticks([])
            pxside.set_xticks([])
            pxtoptop.set_yticks([])

            if isinstance(self.fitters["spectrum"], SpectrumFitter):
                self.fitters["spectrum"].plotfit(ax=pxside)
            if isinstance(self.fitters["profile"], ProfileFitter):
                self.fitters["profile"].plotfit(ax=pxtoptop)
                self.fitters["profile"].plotcomps(ax=pxtop)
            self.burst.plot(ax=ax, withprof=False, withspec=False)

            ax.format(suptitle=f"Best fit for {self.burst.path.name}")

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
