import numpy as np


def noisemean(data: np.ndarray, method="median") -> float | np.ndarray:
    if method == "median":
        return np.median(data, axis=-1)
    else:
        raise NotImplementedError(f"Mean estimation method {method} not implemented.")


def noisestddev(data: np.ndarray, method="iqr") -> float | np.ndarray:
    if method == "iqr":
        stats = np.percentile(data, (25, 75), axis=-1)
        return (stats[1] - stats[0]) / 1.3489795003921634
    elif method == "diffcov":

        def kernel(line: np.ndarray):
            y = np.diff(line)
            c = np.cov(y[:-1], y[1:])
            sw2 = -c[0, 1]
            return sw2**0.5

        try:
            return {
                1: kernel,
                2: lambda x: np.asarray([kernel(line) for line in x]),
            }[
                data.ndim
            ](data)
        except KeyError as Err:
            raise ValueError("Input cannot have more than 2 dimensions.") from Err

    else:
        raise NotImplementedError(f"Stddev estimation method {method} not implemented.")
