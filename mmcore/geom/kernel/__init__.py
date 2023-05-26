import abc
import inspect
from cmath import pi
from collections import namedtuple
from typing import Any

import gmsh
import numpy as np



gmsh.initialize()


def fuse(self, other):
    return gmsh.model.occ.fuse([self.gmsh_dim_tag], [other.gmsh_dim_tag])


class BooleanOperator:
    def __init__(self):
        super().__init__()

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        wrapper = lambda other: self(instance, owner)
        wrapper.__name__ = self.name
        return wrapper

    @abc.abstractmethod
    def __call__(self,
                 instance: 'KernelGeometry',
                 other: 'KernelGeometry',
                 remove_a: bool = False,
                 remove_b: bool = False,
                 gmsh_tag=None
                 ) -> 'KernelGeometry':  ...


class BooleanUnion(BooleanOperator):
    def __call__(self, instance, other, **kwargs):
        return instance.kernel_model.occ.fuse([instance.gmsh_dim_tag], [other.gmsh_dim_tag], **kwargs)


class BooleanIntersection(BooleanOperator):
    def __call__(self, instance, other, **kwargs):
        return instance.kernel_model.occ.intersect([instance.gmsh_dim_tag], [other.gmsh_dim_tag], **kwargs)


class BooleanIntersect(BooleanOperator):
    def __call__(self, instance, other, **kwargs):
        return instance.kernel_model.occ.intersect([instance.gmsh_dim_tag], [other.gmsh_dim_tag], **kwargs)


class BooleanCut(BooleanOperator):
    def __call__(self, instance, other, **kwargs):
        return instance.kernel_model.occ.cut([instance.gmsh_dim_tag], [other.gmsh_dim_tag], **kwargs)


class BooleanFragment(BooleanOperator):
    def __call__(self, instance, other, **kwargs):
        return instance.kernel_model.occ.fragment([instance.gmsh_dim_tag], [other.gmsh_dim_tag], **kwargs)


def intersect(self, other, gmsh_tag=None, objectDimTags=True, toolDimTags=True):
    return gmsh.model.occ.intersect([self.gmsh_dim_tag], [other.gmsh_dim_tag], tag=self.tag, objectDimTags=True,
                                    toolDimTags=True)


def centroid(self):
    return np.asarray(gmsh.model.occ.get_center_of_mass(self.gmsh_dim_tag))


def closest_point(obj, point):
    p, t = gmsh.model.getClosestPoint(*obj.gmsh_dim_tag, point)
    return p, t


GmshPointer = namedtuple("GmshPointer", ["dim", "tag"])
GmshPointer.stringify = lambda self: f'{self.dim}:{self.tag}'
DimTag = GmshPointer


class _KernType(type):
    single = None

    def __new__(mcs, classname, bases, attrs, **kwargs):
        if mcs.single is not None:
            return mcs.single
        else:
            return super().__new__(classname, bases, attrs, **kwargs)


class DelegateD:
    def __init__(self, kern):
        self.kern = kern

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name

    def __get__(self, insr, own):
        return lambda *args, **kwargs: getattr(self.kern, self.name)(insr, *args, **kwargs)


class Kernel:
    def __init__(self, *args, **kwargs):
        self._kern = gmsh
        self._kern.initialize(*args, **kwargs)
        self.model = self._kern.model

    def __getattr__(self, item):
        if inspect.ismethod(getattr(self._kern, item)):
            dlg = DelegateD(self._kern)
            dlg.__set_name__(item, self._kern)
            return dlg.__get__(item)
        else:
            return getattr(self._kern, item)


class GmshDescriptor:
    def __init__(self, dim):
        self.dim = dim

    def __get__(self, inst, own):
        return inst.gmsh_tag

    def get_name(self, inst):
        gmsh.model.get_entity_name(self.dim, inst.gmsh_tag)

    def __add__(self, inst):
        funk = eval("gmsh.model.occ.add" + inst.__class__.__name__.replace("Gmsh", ""))
        funk(inst)


class KernelGeometry:
    gmsh_dim: int
    gmsh_descriptor = GmshDescriptor(3)
    gmsh_tag: int | None = None
    kernel_model: Any
    __add__ = BooleanUnion()

    def __init__(self, *args, kernel_model=gmsh.model, gmsh_tag: int | None = None, **kwargs):
        super().__init__(*args, kernel_model=kernel_model, gmsh_tag=gmsh_tag, **kwargs)

        if self.gmsh_tag is None:
            self.gmsh_tag = int(self._uuid, 16)
        self.gmsh_tag = self.kernel_model.occ.addCylinder(*self.center, *self.direction, self.radius,
                                                          tag=self.gmsh_tag,
                                                          angle=self.angle)

    @property
    def gmsh_dim_tag(self): GmshPointer(self.gmsh_dim, self.gmsh_tag)

    @property
    def centroid(self): return centroid(self)

    def centroid_solve(self): return centroid(self)


class GmshCylinder(KernelGeometry):
    gmsh_dim = 3
    _angle = 2 * pi

    __match_args__ = "center", "direction", "radius"

    @property
    def angle(self): return self._angle

    @angle.setter
    def angle(self, value): self._angle = value


gmsh.model.occ.addPoint(0.2, -0.6, 0.1, 1.0)

gmsh.model.occ.addPoint(-1.6, 2.8, 0.3, 1.0)

gmsh.model.occ.addPoint(-5.4, 7.4, 0.9, 1.0)

gmsh.model.occ.addPoint(-14.8, 4.7, 1.2, 1.0)

gmsh.model.occ.addPoint(-0.8, -1.8, -5.4, 1.0)

gmsh.model.occ.addPoint(-5.4, 1.3, -3.8, 1.0)

gmsh.model.occ.addPoint(-11, 3.4, -3.8, 1.0)

gmsh.model.occ.addPoint(-15.3, 3.9, -3.8, 1.0)

gmsh.model.occ.addPoint(-6.7, 5.7, -3.8, 1.0)

gmsh.model.occ.addPoint(-2.2, -5.5, -3.8, 1.0)

a = gmsh.model.occ.addBSpline((1, 2, 3, 4))
b = gmsh.model.occ.addBSpline((5, 6, 7, 8))
c = gmsh.model.occ.addBSpline((8, 4))
d = gmsh.model.occ.addBSpline((1, 5))
