from __future__ import annotations

from typing import Any

import numpy as np

from mmcore.numeric import curve_bound_points


def aabb_overlap(
    box1: np.ndarray[Any, np.dtype[float]], box2: np.ndarray[Any, np.dtype[float]]
) -> bool:
    """
    >>> from mmcore.geom.curves.bspline import NURBSpline    >>> from mmcore.numeric.aabb import aabb,curve_aabb,aabb_overlap
    >>> pts1=np.array([(-41.0, 143.0, 0.0), (563.0, -184.0, 0.0), (876.0, 594.0, 0.0), (1272.0, -104.0, 0.0), (1580.0, 604.0, 0.0), (2048.0, -462.0, 0.0)])
    >>> pts2=np.array([(211.0, -321.0, 0.0), (391.0, 632.0, 0.0), (942.0, -297.0, 0.0), (1183.0, 753.0, 0.0), (1507.0, -301.0, 0.0), (1921.0, 755.0, 0.0), (1921.0, -546.0, 0.0)])
    >>> n1,n2=NURBSpline(pts1),NURBSpline(pts2)
    >>> aabb_overlap(curve_aabb(n1), curve_aabb(n2))
    Out: True
        :param box1: First AABB
        :type box1: np.ndarray[(2, K), np.dtype[float]] *
        :param box2: Second AABB np.ndarray with shape (2, K) where K is the number of dims. For example in 3d case (x,y,z) K=3.
         :type box2: np.ndarray[(2, K), np.dtype[float]] *
        :return: Is box1 overlap with box2?
        :rtype: bool
        * where K is the number of dims. For example in 3d case (x,y,z) K=3.
    """
    return (
        box1[0][0] <= box2[1][0]
        and box1[1][0] >= box2[0][0]
        and box1[0][1] <= box2[1][1]
        and box1[1][1] >= box2[0][1]
        and box1[0][2] <= box2[1][2]
        and box1[1][2] >= box2[0][2]
    )


def aabb(points: np.ndarray):
    """
     AABB (Axis-Aligned Bounding Box) of a point collection.
    :param points: Points
    :rtype: np.ndarray[(N, K), np.dtype[float]] where:
        - N is a points count.
        - K is the number of dims. For example in 3d case (x,y,z) K=3.
    :return: AABB of a point collection.
    :rtype: np.ndarray[(2, K), np.dtype[float]] at [a1_min, a2_min, ... an_min],[a1_max, a2_max, ... an_max],
    """

    return np.array(
        (
            np.min(points, axis=len(points.shape) - 2),
            np.max(points, axis=len(points.shape) - 2),
        )
    )


def curve_aabb(curve, bounds=None, tol=1e-5):
    """
    >>> from mmcore.geom.curves.bspline import NURBSpline    >>> from mmcore.numeric.aabb import aabb,curve_aabb,aabb_overlap
    >>> pts1=np.array([(-41.0, 143.0, 0.0), (563.0, -184.0, 0.0), (876.0, 594.0, 0.0), (1272.0, -104.0, 0.0), (1580.0, 604.0, 0.0), (2048.0, -462.0, 0.0)])
    >>> pts2=np.array([(211.0, -321.0, 0.0), (391.0, 632.0, 0.0), (942.0, -297.0, 0.0), (1183.0, 753.0, 0.0), (1507.0, -301.0, 0.0), (1921.0, 755.0, 0.0), (1921.0, -546.0, 0.0)])
    >>> n1,n2=NURBSpline(pts1),NURBSpline(pts2)
    >>> aabb_overlap(curve_aabb(n1), curve_aabb(n2))
    :param curve: Any object supporting:
        - curve.interval() -> tuple[float,float],
        - curve.__call__(t:float) -> np.ndarray((K,), dtype=float) where K is the number of dims. For example in 3d case (x,y,z) K=3.

    :param tol: tolerance, default: 1e-5
    :return: AABB (Axis-Aligned Bounding Box) of curve object
    :rtype np.ndarray with shape (2, K).
    """
    return aabb(curve(curve_bound_points(curve, bounds=bounds, tol=tol)))


def curve_aabb_eager(curve, bounds=None, cnt=8):

    bounds = bounds if bounds is not None else curve.interval()

    vals = np.linspace(*bounds, cnt, dtype=float)
    return aabb(curve(vals))
