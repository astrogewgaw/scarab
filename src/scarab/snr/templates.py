import numpy as np
from typing import Self, Iterable
from dataclasses import dataclass
from scarab.utils import sqrsumnorm
from collections.abc import MutableSequence


@dataclass
class Template:
    ref: int
    wbin: int
    kind: str
    refto: str
    data: np.ndarray

    @property
    def size(self):
        return self.data.size

    @classmethod
    def boxcar(cls, wbin: int) -> Self:
        data = np.ones(wbin)
        return cls(
            ref=0,
            wbin=wbin,
            refto="start",
            kind="boxcar",
            data=sqrsumnorm(data),
        )

    @classmethod
    def gaussian(cls, wbin: int) -> Self:
        sigma = wbin / (2 * np.sqrt(2 * np.log(2)))
        xmax = int(np.ceil(3.5 * sigma))
        x = np.arange(-xmax, xmax + 1)
        data = np.exp(-(x**2) / (2 * sigma**2))
        return cls(
            wbin=wbin,
            refto="peak",
            ref=len(x) // 2,
            kind="gaussian",
            data=sqrsumnorm(data),
        )

    def prepare(self, nbins: int) -> np.ndarray:
        if not nbins > self.size:
            raise ValueError(
                f"Cannot pad template data to length n = {nbins}; this "
                f"is shorter than the template size ({self.size}). You "
                "are probably trying to use this template on data that "
                "is too short."
            )
        return np.roll(
            np.pad(
                self.data,
                (0, nbins - self.size),
                mode="constant",
                constant_values=(0.0, 0.0),
            ),
            -self.ref,
        )


@dataclass
class TemplateBank(MutableSequence):
    templates: list[Template]

    def __len__(self):
        return len(self.templates)

    def __getitem__(self, i):
        return self.templates[i]

    def __delitem__(self, i):
        del self.templates[i]

    def __setitem__(self, i, value):
        self.templates[i] = value

    def insert(self, index, value):
        self.templates.insert(index, value)

    @classmethod
    def boxcars(cls, wbins: Iterable) -> Self:
        return cls(templates=[Template.boxcar(wbin) for wbin in sorted(wbins)])

    @classmethod
    def gaussians(cls, wbins: Iterable) -> Self:
        return cls(templates=[Template.gaussian(wbin) for wbin in sorted(wbins)])

    def prepare(self, nbins: int) -> np.ndarray:
        return np.asarray([template.prepare(nbins) for template in self.templates])
