import numpy as np
from scarab.dm.base import delayperdm


def chanwidth(
    data: np.ndarray,
    fl: float,
    fh: float,
) -> float:
    nf, _ = data.shape
    return np.abs((fl - fh) / (nf - 1.0) if nf > 1 else 0.0)


def fdmt(
    data: np.ndarray,
    fh: float,
    fl: float,
    df: float,
    dt: float,
    dmi: float,
    dmf: float,
):
    def transform(
        data: np.ndarray,
        fh: float,
        fl: float,
        df: float,
        dt: float,
        ymin: int,
        ymax: int,
    ) -> np.ndarray:
        nf, nt = data.shape
        output = np.zeros((ymax - ymin + 1, nt), dtype=np.float32)
        if nf == 1:
            output[0] = data[0]
            return output

        i = int((((0.5 * (fl**-2 + fh**-2)) ** -0.5) - fh) / -df + 0.5)
        H, flH, fhH = data[:i], fh - (i - 1) * df, fh
        T, flT, fhT = data[i:], fl, fh - i * df
        dfH = chanwidth(H, flH, fhH)
        dfT = chanwidth(T, flT, fhT)

        yminH = int(ymin * delayperdm(flH, fhH) / delayperdm(fl, fh) + 0.5)
        ymaxH = int(ymax * delayperdm(flH, fhH) / delayperdm(fl, fh) + 0.5)
        yminT = int(ymin * delayperdm(flT, fhT) / delayperdm(fl, fh) + 0.5)
        ymaxT = int(ymax * delayperdm(flT, fhT) / delayperdm(fl, fh) + 0.5)

        HT = transform(H, fhH, flH, dfH, dt, ymin=yminH, ymax=ymaxH)
        TT = transform(T, fhT, flT, dfT, dt, ymin=yminT, ymax=ymaxT)

        for y in range(ymin, ymax + 1):
            yH = int(y * delayperdm(flH, fhH) / delayperdm(fl, fh) + 0.5)
            yT = int(y * delayperdm(flT, fhT) / delayperdm(fl, fh) + 0.5)
            yB = y - yH - yT

            ix = y - ymin
            ih = yH - yminH
            it = yT - yminT

            output[ix] = HT[ih] + np.roll(TT[it], -(yH + yB))

        return output

    ymin = int(dmi / (dt / delayperdm(fl, fh)))
    ymax = int(np.ceil(dmf / (dt / delayperdm(fl, fh))))
    return transform(
        data=data,
        fh=fh,
        fl=fl,
        df=df,
        dt=dt,
        ymin=ymin,
        ymax=ymax,
    )
