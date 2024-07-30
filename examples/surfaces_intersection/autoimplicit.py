import itertools
import numpy as np
from mmcore.geom.curves import NURBSpline
from mmcore.geom.curves.offset import OffsetOnSurface
from mmcore.geom.surfaces import PyRuled, Surface, CurveOnSurface
from mmcore.numeric import scalar_dot
from mmcore.numeric.closest_point import closest_points_on_surface
from mmcore.numeric.divide_and_conquer import (
    divide_and_conquer_min_2d,
    divide_and_conquer_min_2d_vectorized,
)
from mmcore.geom.bvh import contains_point, aabb
from mmcore.numeric.vectors import dot, norm


def create_ruled_from_points(points, degree=3):
    """
    Create a ruled surface from given points.

    :param points: A list of two lists or arrays representing the control points for the two curves.
                   Each list or array should contain points in the form of [x, y, z].
    :param degree: An integer specifying the degree of the NURBSpline curves. Defaults to 3.
    :return: A Ruled object representing a ruled surface created by connecting two NURBSpline curves.
    """
    return PyRuled(
        NURBSpline(np.array(points[0], dtype=float), degree=degree),
        NURBSpline(np.array(points[1], dtype=float), degree=degree),
    )


# Define control points for the ruled surface
# Define control points and create the ruled surface
cpts = np.array(
    [
        [
            [15.8, 10.1, 0.0],
            [-3.0, 13.0, 0.0],
            [-11.5, 7.7, 0.0],
            [-27.8, 12.8, 0.0],
            [-34.8, 9.3, 0.0],
        ],
        [
            [15.8, 3.6, 10.1],
            [-3.0, 8.6, 15.4],
            [-11.5, 12.3, 19.6],
            [-27.8, 6.3, 16.9],
            [-34.8, 5.0, 16.7],
        ],
    ]
)  # Control points definition
surface = create_ruled_from_points(cpts, degree=3)
ppt=np.array([[-0.37262838311109858, 19.712808549356065, 16.404873399828443], [-2.0414404128973751, -1.2964047904304496, 3.1698065065994614]])

from mmcore.geom.primitives import Cylinder

cc=Cylinder(*ppt,3.0)
from mmcore.numeric.marching import marching_intersection_curve_points

initial_point=np.array([-4.8015033570999908, 10.591609456337622, 9.7754043957510461])
res= marching_intersection_curve_points(cc.implicit,surface.implicit,cc.gradient,surface.gradient,initial_point,tol=1e-3 ,step=0.8).tolist()
print(res)
from mmcore.geom.curves.bspline import interpolate_nurbs_curve

uvs=closest_points_on_surface(surface,np.array(res))
curve=CurveOnSurface(surface, interpolate_nurbs_curve(uvs, degree=2))

offset_curve=OffsetOnSurface(curve, 0.3)

