import numpy as np


def billauer(
    yaxis: np.ndarray,
    mindiff: float = 0,
    lookahead: int = 200,
    xaxis: np.ndarray | None = None,
):
    dump = []
    peaks = []
    valleys = []

    if xaxis is None:
        xaxis = np.asarray(range(len(yaxis)))
    if len(yaxis) != len(xaxis):
        raise ValueError("Input vectors yaxis and xaxis must be the same length.")
    if lookahead < 1:
        raise ValueError("Need lookahead to be >= 1.")
    if not (mindiff >= 0):
        raise ValueError("Need mindiff >= 0.")

    xmin = 0
    xmax = 0
    ymin = np.inf
    ymax = -np.inf
    for i, (x, y) in enumerate(zip(xaxis[:-lookahead], yaxis[:-lookahead])):
        if y > ymax:
            ymax = y
            xmax = x
        if y < ymin:
            ymin = y
            xmin = x
        if (y < ymax - mindiff) and (ymax != np.inf):
            if yaxis[i : i + lookahead].max() < ymax:
                dump.append(True)
                peaks.append([xmax, ymax])
                ymax, ymin = np.inf, np.inf
            if i + lookahead >= len(yaxis):
                break
            continue
        if (y > ymin + mindiff) and (ymin != -np.inf):
            if yaxis[i : i + lookahead].min() > ymin:
                dump.append(False)
                valleys.append([xmin, ymin])
                ymin, ymax = -np.inf, -np.inf
                if i + lookahead >= len(yaxis):
                    break
    try:
        if dump[0]:
            peaks.pop(0)
        else:
            valleys.pop(0)
        del dump
    except IndexError:
        pass
    return {"peaks": np.asarray(peaks), "valleys": np.asarray(valleys)}
