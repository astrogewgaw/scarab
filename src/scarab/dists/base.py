import time
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as st
from rich.progress import track
from joblib import Parallel, delayed

from scarab.utilities import smoothen
from scarab.dists.models import MODELS


@dataclass
class Distribution:
    table: pd.DataFrame
    tries: pd.DataFrame | None = None

    @classmethod
    def from_csv(cls, fn: str | Path):
        return cls(table=pd.read_csv(fn))

    def fit(
        self,
        field: str,
        njobs: int = 1,
        nboots: int = 100,
        alpha: float = 0.05,
        ntop: int | None = None,
        dist: str | list = "all",
        **kwargs,
    ) -> None:
        data = self.table[field]
        data = data.to_numpy()

        counts, edges = np.histogram(
            data,
            density=True,
            bins=kwargs.get("bins", "auto"),
        )

        edges = (edges + np.roll(edges, -1))[:-1] / 2.0

        window = kwargs.get("window", None)
        if window is not None:
            edges, counts = smoothen(
                x=edges,
                y=counts,
                window=window,
                interfact=kwargs.get("interfact", 1),
            )

        self.data = data
        self.bins = edges
        self.counts = counts
        self.histogram = (edges, counts)

        match dist:
            case "all":
                distributions = MODELS
            case "popular":
                distributions = [
                    MODELS[name]
                    for name in [
                        "norm",
                        "expon",
                        "pareto",
                        "dweibull",
                        "t",
                        "genextreme",
                        "gamma",
                        "lognorm",
                        "beta",
                        "uniform",
                        "loggamma",
                    ]
                ]
            case _:
                distributions = (
                    [MODELS[name] for name in dist]
                    if isinstance(dist, list)
                    else [MODELS[dist]]
                )
        self.distributions = distributions

        def tryfit(distribution) -> dict | None:
            try:
                tic = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    name = distribution.name
                    params = distribution.fit(data)

                    loc = params[-2]
                    scale = params[-1]
                    args = params[:-2]

                    scorestat = kwargs.get("scorestat", "RSS")
                    ypdf = distribution.pdf(edges, loc=loc, scale=scale, *args)
                    match scorestat:
                        case "energy":
                            score = st.energy_distance(counts, ypdf)
                        case "ks" | "KS":
                            score = -np.log10(st.ks_2samp(counts, ypdf).pvalue)  # type: ignore
                        case "wasserstein":
                            score = st.wasserstein_distance(counts, ypdf)
                        case "rss" | "RSS":
                            score = np.sum(np.power(counts - ypdf, 2.0))
                        case _:
                            raise
                    fitted = distribution(*args, loc=loc, scale=scale)

                    bootscore, bootpass = 0, None
                    try:
                        n = np.minimum(10000, len(data))
                        Dn = st.kstest(data, fitted.cdf)

                        def bootiter(_):
                            resamples = fitted.rvs(n)
                            params = distribution.fit(resamples)
                            fit = distribution(*params)
                            Dn_i = st.kstest(resamples, fit.cdf)
                            return Dn_i[0]

                        Dns = Parallel(n_jobs=njobs)(
                            delayed(bootiter)(i) for i in range(nboots)
                        )
                        Dn_alpha = np.quantile(Dns, 1 - alpha)  # type: ignore
                        bootpass = False if Dn[0] > Dn_alpha else True
                        bootscore = np.sum(Dns > Dn[0]) / nboots
                    except Exception:
                        pass
                toc = time.time()
                elapsed = toc - tic

                return {
                    "name": name,
                    "loc": loc,
                    "args": args,
                    "scale": scale,
                    "score": score,
                    "fitted": fitted,
                    "fittime": elapsed,
                    "bootpass": bootpass,
                    "bootscore": bootscore,
                    "scorestat": scorestat,
                }
            except Exception:
                pass

        tries = [
            _
            for _ in list(
                track(
                    Parallel(n_jobs=njobs, return_as="generator")(
                        delayed(tryfit)(distribution) for distribution in distributions
                    ),
                    total=len(distributions),
                )
            )
            if _ is not None
        ]
        tries = pd.DataFrame(tries[:ntop] if ntop is not None else tries)
        tries.sort_values(
            inplace=True,
            ascending=[False, True, True],
            by=["bootscore", "score", "bootpass"],
        )
        tries.reset_index(drop=True, inplace=True)

        self.tries = tries
        self.bestfit = self.tries.iloc[0, :].to_dict()
        self.bestmodel = self.bestfit["model"]

        try:
            ciiup = self.bestmodel.ppf(alpha)
        except ValueError:
            ciiup = np.max(self.counts)

        try:
            ciidown = self.bestmodel.ppf(1 - alpha)
        except ValueError:
            ciidown = np.min(self.counts)

        self.bestmodel["alpha"] = alpha
        self.bestmodel["ciiup"] = ciiup
        self.bestmodel["ciidown"] = ciidown
