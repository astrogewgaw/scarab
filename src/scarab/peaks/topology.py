import numpy as np
import pandas as pd
from dataclasses import field, dataclass


@dataclass
class UnionFind:
    weights: dict = field(default_factory=dict)
    parents: dict = field(default_factory=dict)

    def __contains__(self, item):
        return item in self.parents

    def __getitem__(self, item):
        if item not in self.parents:
            self.weights[item] = 1
            self.parents[item] = item
            return item
        path = [item]
        root = self.parents[item]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def add(self, item, weight):
        if item not in self.parents:
            self.parents[item] = item
            self.weights[item] = weight

    def union(self, *items):
        roots = [self[x] for x in items]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.parents[r] = heaviest


def topology(
    x: np.ndarray,
    reverse: bool = True,
    limit: float | None = None,
):
    peaks = []
    valleys = []

    X = np.c_[x, x]

    if limit is None:
        limit = np.min(np.max(X)) - 1
    if X.max().max() < limit:
        limit = X.max().max() - 1
    if not reverse:
        clipped = np.clip(x, np.min(X), np.max(X))
        X = np.max(X) - clipped

    seen = set()
    uniqued = np.empty_like(X)
    iteration = np.nditer([X, uniqued], [], [["readonly"], ["writeonly", "allocate"]])
    with iteration:
        while not iteration.finished:
            A = iteration[0].item()
            while (A in seen) and np.isfinite(A):
                A = np.nextafter(A, -np.inf)
            iteration[1] = A
            if A not in seen:
                seen.add(A)
            iteration.iternext()
    X = uniqued

    h, w = X.shape
    indices = [
        (i, j)
        for i in range(h)
        for j in range(w)
        if (X[i][j] is not None) and (X[i][j] >= limit)
    ]
    indices.sort(key=lambda _: X[_[0]][_[1]], reverse=reverse)

    groups = {}
    UF = UnionFind()
    for i, pair in enumerate(indices):
        iy, ix = pair
        v = X[iy][ix]

        def iterneigh(y, x, leny, lenx):
            neighbours = [(y + j, x + i) for i in [-1, 0, 1] for j in [-1, 0, 1]]
            for j, i in neighbours:
                if j < 0 or j >= leny:
                    continue
                if i < 0 or i >= lenx:
                    continue
                if j == y and i == x:
                    continue
                yield j, i

        ni = [UF[Q] for Q in iterneigh(iy, ix, h, w) if Q in UF]
        nc = sorted([(X[UF[Q][0]][UF[Q][1]], Q) for Q in set(ni)], reverse=reverse)

        if i == 0:
            groups[pair] = (v, v, None)

        UF.add(pair, -i)

        if len(nc) > 0:
            oldpair = nc[0][1]
            UF.union(oldpair, pair)
            for bl, Q in nc[1:]:
                if UF[Q] not in groups:
                    groups[UF[Q]] = (float(bl), float(bl) - float(v), pair)
                UF.union(oldpair, Q)
    groups = [(k, groups[k][0], groups[k][1], groups[k][2]) for k in groups]
    groups.sort(key=lambda _: _[2], reverse=True)

    if limit is not None:
        ikeep = np.array(list(map(lambda _: _[2], groups))) > limit
        groups = np.array(groups, dtype="object")
        groups = groups[ikeep].tolist()
    if len(groups) > 0:
        peaks = np.array(list(map(lambda _: [_[0][0], _[1]], groups)))
        ixsort = np.argsort(peaks[:, 0])
        peaks = peaks[ixsort, :]

        valleys = np.array(
            list(
                map(
                    lambda _: [(_[3][0] if _[3] is not None else 0), _[2]],
                    groups,
                )
            )
        )
        ixsort = np.argsort(valleys[:, 0])
        valleys = valleys[ixsort, :]

    Xranked = np.zeros_like(X).astype(int)
    Xdetect = np.zeros_like(X).astype(float)
    for i, homclass in enumerate(groups):
        pbirth, bl, xdetect, _ = homclass
        y, x = pbirth
        Xranked[y, x] = i + 1
        Xdetect[y, x] = xdetect

    if X.shape[1] == 2:
        Xdetect = Xdetect[:, 0]
        Xranked = Xranked[:, 0]

    return {
        "peaks": peaks,
        "valleys": valleys,
        "groups": groups,
        "Xdetect": Xdetect,
        "Xranked": Xranked,
        "persistence": pd.DataFrame(
            {
                "x": np.array(list(map(lambda _: _[0][1], groups))),
                "y": np.array(list(map(lambda _: _[0][0], groups))),
                "score": np.array(list(map(lambda _: float(_[2]), groups))),
                "birth": np.array(list(map(lambda _: float(_[1]), groups))),
                "death": np.array(
                    list(map(lambda _: float(_[1]) - float(_[2]), groups))
                ),
            }
        ),
    }
