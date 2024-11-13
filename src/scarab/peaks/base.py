from pathlib import Path
from dataclasses import dataclass

import numpy as np
import proplot as pplt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

from scarab.peaks.billauer import billauer
from scarab.peaks.topology import topology


@dataclass
class PeakFinder:

    data: np.ndarray
    peaks: np.ndarray
    original: np.ndarray

    window: int
    method: str
    threshold: float

    @classmethod
    def find(
        cls,
        data: np.ndarray,
        window: int = 10,
        threshold: float = 0.1,
        method: str = "billauer",
        **kwargs,
    ):
        peaks = []
        smooth = median_filter(data, window)
        match method:
            case "scipy":
                result = find_peaks(smooth, **kwargs)
                peaks, _ = result
            case "billauer":
                result = billauer(smooth, **kwargs)
                peaks = np.asarray([int(_[0]) for _ in result["peaks"]])
            case "topology":
                result = topology(smooth, **kwargs)
                peaks = np.asarray([int(_[0]) for _ in result["peaks"]])
            case _:
                raise ValueError(f'The "{method}" method is not implemented.')
        peaks = peaks[smooth[peaks] >= threshold * smooth.max()]
        peaks = [peak for _, peak in sorted(zip(smooth[peaks], peaks))]
        peaks = list(reversed(peaks))
        peaks = np.asarray(peaks)
        return cls(
            data=smooth,
            peaks=peaks,
            original=data,
            window=window,
            method=method,
            threshold=threshold,
        )

    def plot(
        self,
        dpi: int = 96,
        show: bool = True,
        save: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "peaks.png",
    ):
        def _(ax: pplt.Axes) -> None:
            ax.plot(self.data, lw=1, alpha=0.5)
            ax.plot(self.peaks, self.data[self.peaks], "o")

        if ax is None:
            fig = pplt.figure(width=10, height=5)
            ax = fig.subplots(nrows=1, ncols=1)[0]
            assert ax is not None
            _(ax)
            if save:
                fig.savefig(saveto, dpi=dpi)
            if show:
                pplt.show()
        else:
            _(ax)
