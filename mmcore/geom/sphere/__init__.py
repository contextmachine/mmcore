import numpy as np

from mmcore.func import vectorize
from mmcore.geom.plane import create_plane_from_xaxis_and_normal
from mmcore.geom.base_surfaces import ParametricSurface
from mmcore.geom.vec import unit


class Sphere(ParametricSurface, signature='(),()->(i)', match_args=('r',),
             param_range=((0.0, np.pi), (0.0, 2 * np.pi))):
    """
    A class representing a sphere in 3D space.

    :param r: The radius of the sphere.
    :type r: float
    :param origin: The coordinates of the origin of the sphere.
    :type origin: tuple(float, float, float)
    """
    def __new__(cls, r=1, origin=(0.0, 0.0, 0.0)):
        self = super().__new__(cls)
        self.r = r
        self.origin = np.array(origin)
        self.__evaluate__ = np.vectorize(self.evaluate, signature='(),()->(i)')
        return self

    def x(self, u, v):
        return self.r * np.sin(u) * np.cos(v) + self.origin[0]

    def y(self, u, v):
        return self.r * np.sin(u) * np.sin(v) + self.origin[1]

    def z(self, u, v):
        return self.r * np.cos(u) + self.origin[2]

    def __iter__(self):
        return iter((self.r, self.origin))

    def __array__(self):
        return np.array((self.r, *self.origin))

    def project(self, pt):
        return unit(pt - self.origin) * self.r + self.origin


@vectorize(excluded=[0], signature='(),()->()')
def evaluate_plane(sphere: Sphere, u, v):
    """
    :param sphere: The sphere object to evaluate the plane on.
    :type sphere: Sphere
    :param u: The parameter representing the u-coordinate.
    :type u: float
    :param v: The parameter representing the v-coordinate.
    :type v: float
    :return: The plane calculated based on the given sphere, u, and v parameters.
    :rtype: Plane

    """
    pt = sphere(u, v)

    return create_plane_from_xaxis_and_normal(
        xaxis=unit(sphere(u + 0.001, v) - sphere(u - 0.001, v)),
        normal=unit(pt - sphere.origin),
        origin=pt)


@vectorize(excluded=[0], signature='(),()->()')
def evaluate_normal(sphere: Sphere, u, v):
    """
    Evaluate the normal vector at a given point on the surface of a sphere.

    :param sphere: The sphere object on which to evaluate the normal.
    :type sphere: Sphere

    :param u: The u-coordinate of the point on the surface of the sphere.
    :type u: float

    :param v: The v-coordinate of the point on the surface of the sphere.
    :type v: float

    :return: The normal vector at the specified point on the surface of the sphere.
    :rtype: Vector
    """
    pt = sphere(u, v)
    return unit(pt - sphere.origin)
