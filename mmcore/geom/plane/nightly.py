import numpy as np

from mmcore.geom.interfaces import ArrayInterface
from mmcore.geom.transform import rotate_2D
from mmcore.geom.vec import *

_ROT90 = rotate_2D(np.pi / 2)


class NCurve:
    _boundaries = np.array([0., 1.])

    @property
    def boundaries(self):
        return self._boundaries

    @boundaries.setter
    def boundaries(self, v):
        self._boundaries[:] = v

    def evaluate(self, t):
        ...

    @vectorize(excluded=[0], signature='()->(i)')
    def __call__(self, t):
        return self.evaluate(t)


from functools import lru_cache


@lru_cache(maxsize=None)
def _cached_line_direction(line):
    return line.end - line.start


@lru_cache(maxsize=None)
def _cached_line_unit(line):
    """
    !!!! Не используйте эту функцию. Запись в кеш обойдется дороже чем вычисление. Взятие из кеша в свою очередь
    сэкономит около 30 ms (10_000 объектов)

    Без кеширования:
execution: get unit 0.0 min. 0.43208909034729004 secs.
    С lru кешированием:
execution 0: get unit 0.0 min. 1.3732378482818604 secs.
execution 1: get unit 0.0 min. 0.4049699306488037 secs.

    !!!! На 100_000 все примерно также
execution: creating nlines 0.0 min. 0.14940190315246582 secs.
Без кеширования:
execution: get unit 0.0 min. 4.085143089294434 secs.

С lru кешированием:
execution: get unit 0.0 min. 13.715291976928711 secs.
execution: get unit 0.0 min. 3.980236053466797 secs.
    :param line:
    :type line:
    :return:
    :rtype:
    """
    return unit(_cached_line_direction(line))


class NLine(NCurve):
    def __init__(self, start, end):
        if isinstance(start, (tuple, list)):
            start = np.array(start, float)
        if isinstance(end, (tuple, list)):
            end = np.array(end, float)
        self._start, self._end = start, end
        self._boundaries = np.array([0.0, 1.0])

    def __hash__(self):
        return hash((repr(self.start), repr(self.end)))

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, v):
        self._start[:] = v

    @property
    def end(self):
        return self._end

    @property
    def origin(self):
        return self.start

    @origin.setter
    def origin(self, v):
        self.start = v

    @end.setter
    def end(self, v):
        self._end[:] = v

    @property
    def direction(self):
        return self.end - self.start

    @property
    def unit(self):
        return unit(self.end - self.start)

    def evaluate(self, t):
        return self.start + self.direction * t

    def tangent(self, t):
        return self.unit

    @property
    def xaxis(self):
        return self.unit

    @property
    def yaxis(self):
        return _ROT90.dot(self.unit)

    @property
    def length(self):
        return norm(self.direction)

    @xaxis.setter
    def xaxis(self, v):
        d = self.length
        self.end = self.start + np.array(v) * d

    @property
    def boundaries(self):
        return self._boundaries

    def closest_parameter(self, pt):
        return dot(self.xaxis, pt - self.start)

    def closest_point(self, pt):
        return self(self.closest_parameter(pt))

    def closest_distance(self, pt):
        return dist(self.closest_point(pt))


class NPlane(NLine):

    @vectorize(excluded=[0], signature='(i)->(i)')
    def local(self, t, pt):
        return dot([self.xaxis, self.yaxis, self.zaxis], pt - self.origin)


def relative_origin(parent, child):
    return parent.at_local(child.origin)


def relative_axis(parent, child):
    return parent.at_local(child.axis)


def absolute_origin(parent, child):
    return parent(child.origin)


def absolute_axis(parent, child):
    return parent(child.axis)


class Bundle:
    def __init__(self, bundle, parent):
        self._bundle = np.array(bundle) if isinstance(bundle, (list, tuple)) else bundle
        self.parent = parent

    @property
    def absolute(self):
        return self.bundle

    @property
    def bundle(self):

        return self._bundle

    @bundle.setter
    def bundle(self, v):
        self._bundle[:] = v

    @absolute.setter
    def absolute(self, v):
        self.bundle = v

    @property
    def relative(self):
        return self.parent.at_local(self.bundle)

    @relative.setter
    def relative(self, v):
        self.bundle = self.parent(v)

    def __mul__(self, other):

        if isinstance(self.bundle, Bundle):
            self.bundle.bundle.__mul__(other)
        else:
            return self.bundle.__mul__(other)

    def __rmul__(self, other):
        if isinstance(self.bundle, Bundle):
            self.bundle.bundle.__rmul__(other)
        else:
            return self.bundle.__rmul__(other)


cross_table = dict(x=lambda xyz: cross(xyz[1], xyz[2]), y=lambda xyz: cross(xyz[2], xyz[0]),
        z=lambda xyz: cross(xyz[0], xyz[1]), )

from mmcore.geom.plane.refine import PlaneRefine


class Pln:

    def __init__(self, origin=np.array((0., 0., 0.)), xaxis=np.array((0., 0., 1.0)), normal=np.array((0., 0., 1.0))):
        self.axis_names = {'xaxis', 'yaxis', 'normal'}
        self.axis_indices = {'xaxis': 0, 'yaxis': 1, 'normal': 2, 'zaxis': 2}

        self._origin = origin

        xaxis = np.array(xaxis) if isinstance(xaxis, (list, tuple)) else xaxis
        zaxis = np.array(normal) if isinstance(normal, (list, tuple)) else normal
        yaxis = cross(normal, xaxis)

        self._axis = np.array([xaxis, yaxis, zaxis]
                              )

    def refine(self, proprity_axis=('y', 'x')):
        refine = PlaneRefine(*proprity_axis)
        refine(self.local_axis, inplace=True)

    @property
    def local_origin(self):
        return self._origin

    @local_origin.setter
    def local_origin(self, v):
        self._origin = np.array(v) if isinstance(v, (list, tuple)) else v

    @property
    def origin(self):
        return self.local_origin

    @origin.setter
    def origin(self, v):
        self.local_origin = v

    @property
    def local_axis(self):
        return self._axis

    @property
    def axis(self):
        return self.local_axis

    @property
    def xaxis(self):
        return self.axis[0]

    @property
    def yaxis(self):
        return self.axis[1]

    @xaxis.setter
    def xaxis(self, v):
        self.axis[0] = v
        self.refine(('y', 'z'))

    @yaxis.setter
    def yaxis(self, v):
        self.axis[1] = v
        self.refine(('x', 'z'))

    @property
    def zaxis(self):
        return self.axis[2]

    @zaxis.setter
    def zaxis(self, v):
        self.axis[2] = v
        self.refine(('x', 'y'))

    @property
    def normal(self):
        return self.zaxis

    @normal.setter
    def normal(self, v):
        self.zaxis = v

    @vectorize(excluded=[0], signature='(i)->(i)')
    def __call__(self, uvh):
        return self.origin + self.axis[0] * uvh[0] + self.axis[1] * uvh[1] + self.axis[2] * uvh[2]

    @vectorize(excluded=[0], signature='(i)->(i)')
    def at_local(self, pt):
        return dot(self.axis, pt - self.origin)

    @vectorize(excluded=[0], signature='()->()')
    def other_plane_at_local(self, other: 'Pln'):
        return Pln(self.at_local(other.origin), self.xaxis.dot(other.xaxis), self.normal.dot(other.normal))

    def create_child(self, *args, **kwargs) -> 'ChildPln':
        return ChildPln(self, *args, **kwargs)

    def add_parent(self, parent) -> 'ChildPln':
        pln = ChildPln(parent)
        pln._axis = self._axis
        pln._origin = self._origin
        return pln


class WorldXYPlane(Pln):

    @property
    def axis(self):
        return self.local_axis

    @property
    def origin(self):
        return self.local_origin


class ChildPln(Pln):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent

    @property
    def axis(self):
        return self.parent(self._axis)

    @property
    def origin(self):
        return self.parent(self._axis)

    def to_plane(self):
        return Pln(self._origin, self._axis[0], self._axis[1])
