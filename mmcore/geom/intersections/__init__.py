from collections import namedtuple

import numpy as np
from multipledispatch import dispatch
from scipy.optimize import fsolve

from mmcore.func import vectorize
from mmcore.geom.plane import Plane
from mmcore.geom.sphere import Sphere

Ray = namedtuple('Ray', ['origin', 'normal'])


@dispatch(Sphere, Ray)
@vectorize(excluded=[0], signature='(i,j)->(k)')
def ray_intersection(sphere: Sphere, ray: Ray):
    """

    Parameters
    ----------
    sphere :
    ray :

    Returns
    -------

    type: np.ndarray((n, 5), dtype=float)
    Array with shape: (n, [x, y, z, t, u, v]) where:
    1. n is the number of rays.
    2. x,y,z - intersection point cartesian coordinates.
    3. t - ray intersection param.
    4. u,v - sphere intersection params.

    """
    ray_start, ray_vector = ray

    def wrap(x):
        t, (u, v) = x[0], x[1:]
        return ray_start + ray_vector * t - sphere(u, v)

    t, u, v = fsolve(wrap, [0, 0, 0], full_output=False)
    return np.append(sphere.evaluate(u, v), [t, u, v])


@dispatch(Plane, Ray)
@vectorize(excluded=[0, 2], signature='(i,j)->(k)')
def ray_intersection(plane: Plane, ray: Ray, epsilon=1e-6):
    ray_origin, ray_direction = ray
    dotu = np.array(plane.normal).dot(ray_direction)
    if abs(dotu) < epsilon:
        return np.empty(6)
    w = ray_origin - plane.origin
    dotv = -np.array(plane.normal).dot(w)
    si = dotv / dotu
    Psi = w + si * ray_direction + plane.origin

    return np.array([*Psi, si, dotu, dotv])
