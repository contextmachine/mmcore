import copy
import dataclasses
import os

import geomdl.operations
import multiprocess as mp
import numpy as np
from more_itertools import chunked

from mmcore.base import AMesh
from mmcore.base.models.gql import MeshBasicMaterial
from mmcore.geom.materials import ColorRGB
from mmcore.geom.parametric import Circle, Linear, ParametricObject, PlaneLinear
from mmcore.geom.parametric.nurbs import NurbsCurve, NurbsSurface
from mmcore.geom.transform import WorldXY


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
        #print(self,t)
        if isinstance(self.path, NurbsCurve):

            pt = self.path.tan(t)
            point = pt.point
            normal = pt.normal
        elif isinstance(self.path, Linear):
            point = self.path.evaluate(t)
            normal = self.path.direction

        return PlaneLinear(origin=point, normal=normal, yaxis=[0, 0, 1])

    def evaluate(self, t):
        u, v = t

        return self.shape.evaluate(v).tolist() @ self.evalplane(u).transform_from_other(WorldXY)

    def evaluate_profile(self, t):
        T = self.evalplane(t).transform_from_other(WorldXY)

        if isinstance(self.shape, NurbsCurve):



            return self.shape.transform(T)
        elif isinstance(self.shape, Circle):
            return Circle(r=self.shape.r, plane=self.evalplane(t))

        elif hasattr(self.shape, "evaluate"):
            # s=[self.shape.evaluate(i).tolist() for i in np.linspace(0,1,6)]
            # l = []
            # for pt in s:
            #    l.append((pt @ T).tolist())
            # return l
            return self.shape.transform(T)

        else:
            s=copy.deepcopy(self.shape)
            l = []
            for pt in s:
                l.append(pt @ T)
            return l
            #s.transform(self.evalplane(t).transform_from_other(WorldXY))
        return s

    def normal(self, t):
        return geomdl.operations.normal(self.proxy, t)

    def geval(self, uvs=(20, 20), bounds=((0, 1), (0, 1))):
        for i, u in enumerate(np.linspace(*bounds[0], uvs[0])):
            for j, v in enumerate(np.linspace(*bounds[1], uvs[1])):
                yield i, j, u, v, *self.evaluate([u, v])

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

    @property
    def surface(self):
        ...


@dataclasses.dataclass
class NurbsPipe(Pipe):
    u_count: int = 16
    v_count: int = 4
    degree_u = 3
    degree_v = 1

    def generate_uv(self):
        return eval_pipe_uv(self, self.u_count, self.v_count)

    def surface(self):
        return NurbsSurface(list(self.generate_uv()), degree_u=self.degree_u, degree_v=self.degree_v)

    def mesh_data(self):
        return self.surface().mesh_data


def spline_pipe_mesh(points, uuid, name=None, thickness=1, color=(0, 0, 0), u_count=16, degree=3, **kwargs):
    return AMesh(uuid=uuid, name=name if name is not None else uuid,
                 geometry=NurbsPipe(NurbsCurve(points, degree=1 if len(points) <= 3 else degree), Circle(r=thickness),
                                    u_count=u_count,
                                    v_count=4).mesh_data().to_buffer(),
                 material=MeshBasicMaterial(color=ColorRGB(*color).decimal), **kwargs)


def spline_pipe_mesh_data(points, thickness=1, u_count=16, degree=3):
    return NurbsPipe(NurbsCurve(points, degree=degree), Circle(r=thickness), u_count=u_count, v_count=4).mesh_data()


def eval_pipe_uv(pipe, u_count, v_count):
    def wrapgen():
        for i in np.linspace(0, 1, u_count):
            for j in np.linspace(0, 1, v_count):
                yield pipe.evaluate_profile(i).evaluate(j).tolist()

    return chunked(wrapgen(), v_count)
