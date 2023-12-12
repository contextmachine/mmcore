import numpy as np
from more_itertools import flatten


def tess(shp, h):
    a2 = np.array(shp.mesh_data.vertices)
    a2[..., 2] += h
    *l, = zip(shp.mesh_data.vertices.tolist(), a2.tolist())
    *ll, = flatten(l)
    ixs = []

    def tess_bound(boundary):
        for i, v in enumerate(boundary):
            if i > len(boundary):
                yield boundary[i] + [0], boundary[i] + [h], boundary[i + 1] + [h], boundary[i] + [0], boundary[
                    i - 1] + [0], boundary[i - 1] + [h]
            else:
                yield boundary[i] + [0], boundary[i] + [h], boundary[i - 1] + [h], boundary[i] + [0], boundary[
                    i - 1] + [0], boundary[i - 1] + [h]

    for i in [tess_bound(bndr) for bndr in [shp.boundary] + shp.holes]:
        for j in i:
            for jj in j:
                ixs.append(ll.index(jj))

    return ixs, ll


