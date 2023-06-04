import dataclasses
import os

import multiprocess as mp
import numpy as np

from mmcore.geom.parametric import Circle, ParametricObject, PlaneLinear, add_crd, EvalPointTuple2D
from mmcore.geom.parametric.nurbs import NurbsCurve
from mmcore.geom.transform import remove_crd, WorldXY


@dataclasses.dataclass
class Pipe:
    """
    >>> nb2=NurbsCurve([[0, 0, 0        ] ,
    ...                 [-47, -315, 0   ] ,
    ...                 [-785, -844, 0  ] ,
    ...                 [-704, -1286, 0 ] ,
    ...                 [-969, -2316, 0 ] ] )
    >>> r=Circle(r=10.5)
    >>> oo=Pipe(nb2, r)

    """
    path: ParametricObject
    shape: ParametricObject

    def evalplane(self, t):
        pt = self.path.tan(t)
        return PlaneLinear(origin=pt.point, normal=pt.normal)

    def evaluate(self, t):
        u, v = t

        pln = self.evalplane(u)

        return remove_crd(
            add_crd(self.shape.evaluate(v), 1).reshape((4,)).tolist() @ pln.transform_from_other(WorldXY).matrix.T)

    def geval(self, uvs=(20, 20), bounds=((0, 1), (0, 1))):
        for i, u in enumerate(np.linspace(*bounds[0], uvs[0])):
            for j, v in enumerate(np.linspace(*bounds[1], uvs[1])):
                yield EvalPointTuple2D(i, j, u, v, *self.evaluate([u, v]))

    def veval(self, uvs=(20, 20), bounds=((0, 1), (0, 1))):
        data = np.zeros(uvs + (3,), dtype=float)
        for i, j, u, v, x, y, z in self.geval(uvs, bounds):
            data[i, j, :] = [x, y, z]
        return data

    def mpeval(self, uvs=(20, 20), bounds=((0, 1), (0, 1)), workers=-1):
        """
        >>> path = NurbsCurve([[0, 0, 0],
        ...               [-47, -315, 0],
        ...               [-785, -844, 0],
        ...               [-704, -1286, 0],
        ...               [-969, -2316, 0]])

        >>> profile = Circle(r=10.5)
        >>> pipe = Pipe(path, profile)
        >>> pipe.veval(uvs=(2000, 200)) # 400,000 points
        time 40.84727501869202 s
        >>> pipe.mpeval(uvs=(2000, 200)) # 400,000 points
        time 8.37929892539978 s # Yes it's also too slow, but it's honest work
        """

        def inner(u):
            return [(u, v, *self.evaluate([u, v])) for v in np.linspace(*bounds[1], uvs[1])]

        if workers == -1:
            workers = os.cpu_count()

        with mp.Pool(workers) as p:
            return p.map(inner, np.linspace(*bounds[0], uvs[0]))
