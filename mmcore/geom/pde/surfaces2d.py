"""
Module Description
====

This Python module implements multiple classes that represent 2D surfaces. Each class represents a specific type of
2D surface and associated operations. The key classes are

1.  `Surface2D`: Base class implementing common functionality for all 2D surfaces.
2.  `OffsetSurface2D`: Class representing a surface formed by moving away from an existing surface by a specified
distance.
3.  `PlaneSurface2D`: Class representing a planar surface defined by its origin and 'u' & 'v' directions.
4.  `PlaneLinePointSurface`: Class representing a planar surface defined by a line and a point.
5.  `PlaneLineDirectionSurface`: Class representing a planar surface defined by a line and a direction.
6.  `LineLineSurface2D`: Class representing a surface created by sweeping a line along another line.

# Class Examples

A few examples on how to use these classes are provided below.

```python
import numpy as np

# Assuming Line is a class with necessary properties and methods.
line1 = Line(start=[0, 0, 0], end=[1, 0, 0])
line2 = Line(start=[0, 1, 0], end=[1, 1, 0])
point = [0, 0, 0]
direction = np.array([0, 0, 1])

# 1. Surface2D
# No direct usage as it serves as a base class

# 2. OffsetSurface2D
offsetSurface = OffsetSurface2D(distance=2.5, parent=someOtherSurface2Dobject)

# 3. PlaneSurface
planeSurface = PlaneSurface2D()
planeSurface.origin = np.array([0, 0, 0])
planeSurface.u_direction = np.array([1, 0, 0])
planeSurface.v_direction = np.array([0, 1, 0])

# 4. PlaneLinePointSurface
planeLinePointSurface = PlaneLinePointSurface(line1, point)

# 5. PlaneLineDirectionSurface
planeLineDirectionSurface = PlaneLineDirectionSurface(line1, direction)

# 6. LineLineSurface2D
lineLineSurface = LineLineSurface2D(line1, line2)
"""

import numpy as np

from mmcore.base.models.gql import MeshPhongMaterial, MeshStandardVertexMaterial
from mmcore.geom.line import Line
from mmcore.geom.materials import ColorRGB
from mmcore.geom.pde import PDE2D
from mmcore.geom.vec import unit, cross
from mmcore.geom.mesh.mesh_tuple import create_mesh_tuple
from mmcore.func import vectorize


def ixs(u, v):
    """



    :param u:
    :type u:
    :param v:
    :type v:
    :return:
    :rtype:
    """
    for i in range(u):

        for j in range(v):

            a = i * (u + 1) + (j + 1)
            b = i * (v + 1) + j
            c = (i + 1) * (u + 1) + j
            d = (i + 1) * (v + 1) + (j + 1)

            # generate two faces (triangles) per iteration
            yield [a, b, d]
            yield [b, c, d]


class Surface2D:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.derivative = PDE2D(self)

    def tangent(self, u, v):
        return unit(self.derivative(u, v))

    def tan(self, u, v):
        return unit(self.derivative(u, v))

    def normal(self, u, v):
        return self.derivative.normal(u, v)

    def evaluate(self, u, v):
        ...

    def plane(self, u, v):
        return self.derivative.plane(u, v)

    def to_mesh_tuple(self, u=10, v=10, color=np.array([0.7, 0.3, 1.])):
        uv = np.meshgrid(np.linspace(0., 1.0, u), np.linspace(0., 1., v), indexing='ij')
        n = self.normal(*uv)
        print(n.shape)

        return create_mesh_tuple(
                attributes=dict(position=self(*uv).flatten(), normal=n.flatten(), uv=np.stack(uv, axis=-1).flatten()

                        ), indices=np.array(list(ixs(u - 1, v - 1)), int).flatten(), color=color
                )

    def amesh(self, uuid=None, u=10, v=10, color=np.array([0.7, 0.3, 1.]), name=None):
        return self.to_mesh_tuple(u, v, color=color).amesh(uuid, name=name, material=MeshStandardVertexMaterial(
            color=ColorRGB(*color).decimal, flatShading=False
            )
                                                           )

    def __call__(self, u, v): ...


class OffsetSurface2D(Surface2D):
    parent: 'Surface2D|OffsetSurface2D'
    distance: float

    def __init__(self, distance, parent):
        super().__init__()
        self.distance = distance
        self.parent = parent

    @vectorize(excluded=[0], signature='(),()->(i)')
    def evaluate(self, u, v):
        return self.parent.evaluate(u, v) + self.parent.normal(u, v) * self.distance

    def __call__(self, u, v):
        return self.evaluate(u, v)


class PlaneSurface(Surface2D):
    origin: np.ndarray
    u_direction: np.ndarray
    v_direction: np.ndarray

    @np.vectorize(excluded=[0], signature='(),()->(i)')
    def evaluate(self, u, v):
        return self.origin + self.u_direction * u + self.v_direction * v

    def __call__(self, u, v):
        return self.evaluate(u, v)


class PlaneLinePointSurface(PlaneSurface):

    def __init__(self, line, point):
        super().__init__()
        self._line = None
        self._point = np.array(point)
        self.line = line

    def solve(self):
        print(f'[solve] {self}')
        self.line2 = Line.from_ends(self.line.closest_point(self.point), self.point)

    @property
    def u_direction(self):
        return self._line.direction

    @property
    def v_direction(self):
        return self.line2.direction

    @property
    def origin(self):
        return self.line.start

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, v):
        self._point = np.array(v)
        self.solve()

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, v):
        self._line = v
        self.solve()


class PlaneLineDirectionSurface(PlaneSurface):
    def __init__(self, line, direction):
        super().__init__()
        self._v_direction = direction
        self._line = line

    @property
    def u_direction(self):
        return self._line.direction

    @property
    def v_direction(self):
        return self._v_direction

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, v):
        self._line = v


class LineLineSurface2D(Surface2D):

    def __init__(self, line1, line2):
        super().__init__()
        self._line1 = line1
        self._line2 = line2

    @property
    def line1(self):
        return self._line1

    @line1.setter
    def line1(self, v):
        self._line1 = v

    @property
    def line2(self):
        return self._line2

    @line2.setter
    def line2(self, v):
        self._line2 = v

    @vectorize(excluded=[0], signature='(),()->(i)')
    def evaluate(self, u, v):
        p = self.line1(u)

        return p + (self.line2(u) - p) * v

    def __call__(self, u, v):
        return self.evaluate(u, v)
