import math
from typing import Iterable, Optional, Tuple

import numpy as np

from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple,
    NURBSSurfaceTuple,
    from_homogeneous_2d,
    to_homogeneous_1d,
)

__all__ = ["elliptical_cone"]


def _orthonormal_frame(
        *,
        cplane: Optional[np.ndarray] = None,
        origin: Optional[Iterable[float]] = None,
        xaxis: Optional[Iterable[float]] = None,
        yaxis: Optional[Iterable[float]] = None,
        normal: Optional[Iterable[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a right-handed, orthonormal frame from assorted plane inputs.

    Priority:
      1) explicit ``cplane`` (shape (4,3) or (3,3))
      2) explicit ``xaxis`` and ``yaxis``
      3) explicit ``normal``
      4) defaults to the world XY plane.
    """

    if cplane is not None:
        P = np.asarray(cplane, dtype=float)
        if P.shape == (4, 3):
            origin, xaxis, yaxis, normal = P
        elif P.shape == (3, 3):
            origin, xaxis, yaxis = P
            normal = None
        elif P.shape == (2, 3):
            origin, normal = P
            xaxis = None
            yaxis = None
        else:
            raise ValueError("cplane must have shape (4,3), (3,3) or (2,3)")

    # Fall back to defaults when parts are missing.
    origin = np.array(origin if origin is not None else [0.0, 0.0, 0.0], dtype=float)

    if xaxis is not None and yaxis is not None:
        xaxis = np.array(xaxis, dtype=float)
        yaxis = np.array(yaxis, dtype=float)
    elif normal is not None:
        normal = np.array(normal, dtype=float)
        # pick a helper not parallel to normal
        helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        xaxis = np.cross(helper, normal)
        yaxis = np.cross(normal, xaxis)
    else:
        xaxis = np.array([1.0, 0.0, 0.0], dtype=float)
        yaxis = np.array([0.0, 1.0, 0.0], dtype=float)

    # Normalize and fix orthogonality.
    xaxis = xaxis / np.linalg.norm(xaxis)
    normal_vec = np.cross(xaxis, yaxis)
    if np.linalg.norm(normal_vec) < 1e-12:
        raise ValueError("xaxis and yaxis cannot be collinear")
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    yaxis = np.cross(normal_vec, xaxis)
    yaxis = yaxis / np.linalg.norm(yaxis)

    return origin, xaxis, yaxis, normal_vec


def _ellipse_curve(
        a: float = 1.0,
        b: float = 1.0,
        start_angle: float = 0.0,
        end_angle: float = 2 * math.pi,
        *,
        center=None,
        xaxis=None,
        yaxis=None,
        normal=None,
) -> NURBSCurveTuple:
    """
    Rational quadratic NURBS representation of an ellipse (or elliptical arc).

    Geometry is built exactly like the circle constructor but with anisotropic
    scaling by ``a`` and ``b`` in the local x/y directions.
    """

    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.array(center, dtype=float)

    O, xaxis, yaxis, _ = _orthonormal_frame(origin=center, xaxis=xaxis, yaxis=yaxis, normal=normal)

    delta = end_angle - start_angle
    if delta <= 0:
        raise ValueError("end_angle must be greater than start_angle")

    n_seg = int(math.ceil(delta / (math.pi / 2.0)))
    seg_angle = delta / n_seg

    control_points_local = []
    weights = []

    for i in range(n_seg):
        theta0 = start_angle + i * seg_angle
        theta2 = start_angle + (i + 1) * seg_angle
        theta_mid = 0.5 * (theta0 + theta2)
        w_mid = math.cos(seg_angle / 2.0)

        P0 = np.array([math.cos(theta0), math.sin(theta0)])
        P1 = np.array([math.cos(theta_mid), math.sin(theta_mid)]) / w_mid
        P2 = np.array([math.cos(theta2), math.sin(theta2)])

        if i == 0:
            control_points_local.extend([P0, P1, P2])
            weights.extend([1.0, w_mid, 1.0])
        else:
            control_points_local.extend([P1, P2])
            weights.extend([w_mid, 1.0])

    control_points_local = np.asarray(control_points_local)
    weights = np.asarray(weights, dtype=float)

    # Map to 3D with anisotropic scaling by a and b.
    control_points_global = []
    for u, v in control_points_local:
        cp = O + (a * u) * xaxis + (b * v) * yaxis
        control_points_global.append(cp)

    control_points_global = np.asarray(control_points_global, dtype=float)

    knot_vector = [0.0, 0.0, 0.0]
    for i in range(1, n_seg):
        u = i / n_seg
        knot_vector.extend([u, u])
    knot_vector.extend([1.0, 1.0, 1.0])

    return NURBSCurveTuple(3, np.asarray(knot_vector, dtype=float), control_points_global, weights)


def elliptical_cone(
        a: float = 1.0,
        b: float = 1.0,
        height: float = 1.0,
        *,
        cplane: Optional[np.ndarray] = None,
        origin: Optional[Iterable[float]] = None,
        xaxis: Optional[Iterable[float]] = None,
        yaxis: Optional[Iterable[float]] = None,
        apex: Optional[Iterable[float]] = None,
        start_angle: float = 0.0,
        end_angle: float = 2 * math.pi,
) -> NURBSSurfaceTuple:
    """
    Constructs an elliptical cone as a NURBS surface based on the given parameters. The method creates
    a cone surface using elliptical control curves and apex position, allowing customization of the
    cone's base shape, height, orientation, and angular extent.

    The resulting NURBS surface includes control points, weights, and parameters defining the
    surface's geometry within the specified constraints.

    :param a: Semi-major axis length of the elliptical base of the cone.
    :type a: float
    :param b: Semi-minor axis length of the elliptical base of the cone.
    :type b: float
    :param height: Height of the cone along its axis, extending from the base to the apex.
    :type height: float
    :param cplane: Coordinate plane matrix to define the base orientation. If specified, it is a
        3x3 matrix defining the coordinate system (optional).
    :type cplane: Optional[np.ndarray]
    :param origin: Origin point of the base ellipse. Overrides cplane origin if provided (optional).
    :type origin: Optional[Iterable[float]]
    :param xaxis: X-axis direction vector for the base ellipse. Overrides cplane x-axis if provided
        (optional).
    :type xaxis: Optional[Iterable[float]]
    :param yaxis: Y-axis direction vector for the base ellipse. Overrides cplane y-axis if provided
        (optional).
    :type yaxis: Optional[Iterable[float]]
    :param apex: Coordinates of the apex of the cone. If specified, the height parameter is ignored,
        and the apex is calculated directly (optional).
    :type apex: Optional[Iterable[float]]
    :param start_angle: Starting angle of the elliptical base (in radians), defining the segment of the
        ellipse to be used.
    :type start_angle: float
    :param end_angle: Ending angle of the elliptical base (in radians), defining the segment of the
        ellipse to be used.
    :type end_angle: float
    :return: A NURBSSurfaceTuple representing the elliptical cone surface, including orders, knots,
        control points, and weights.
    :rtype: NURBSSurfaceTuple
    """

    O, X, Y, N = _orthonormal_frame(cplane=cplane, origin=origin, xaxis=xaxis, yaxis=yaxis)

    if apex is not None:
        apex = np.array(apex, dtype=float)
        axis_vec = apex - O
        axis_len = np.linalg.norm(axis_vec)
        if axis_len < 1e-12:
            raise ValueError("apex must not coincide with the base origin")
        N = axis_vec / axis_len
        # Rebuild Y to stay orthogonal to the updated axis.
        X = X / np.linalg.norm(X)
        Y = np.cross(N, X)
        if np.linalg.norm(Y) < 1e-12:
            # pick an alternative X when provided axis is parallel to original X
            tmp = np.array([0.0, 1.0, 0.0]) if abs(N[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            X = np.cross(tmp, N)
            X = X / np.linalg.norm(X)
            Y = np.cross(N, X)
        Y = Y / np.linalg.norm(Y)
    else:
        apex = O + N * float(height)

    base_curve = _ellipse_curve(a, b, start_angle, end_angle, center=O, xaxis=X, yaxis=Y, normal=N)

    num_u = base_curve.control_points.shape[0]
    apex_row = np.tile(apex, (num_u, 1))

    control_points_h = np.zeros((num_u, 2, 4), dtype=float)
    control_points_h[:, 0, :] = to_homogeneous_1d(apex_row, base_curve.weights)
    control_points_h[:, 1, :] = to_homogeneous_1d(base_curve.control_points, base_curve.weights)

    cpts, weights = from_homogeneous_2d(control_points_h)

    v_knots = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    return NURBSSurfaceTuple(
        order_u=base_curve.order,
        order_v=2,
        knot_u=base_curve.knot,
        knot_v=v_knots,
        control_points=cpts,
        weights=weights,
    )


def circular_cone(radius: float = 1.0, height: float = 1.0, cplane=None, origin=None, xaxis=None, yaxis=None, apex=None,
                  start_angle: float = 0.0, end_angle: float = 2 * math.pi) -> NURBSSurfaceTuple:
    """
    Generates a circular cone as a NURBS (Non-Uniform Rational B-Spline) surface. The cone is defined by its dimensions and
    positioning parameters, including radius, height, and orientation. By default, the cone aligns with the Z-axis unless
    other specification parameters are provided. Users can define the extent of the cone surface by specifying start and end
    angles to create a partial circular cone segment.

    :param radius: The radius of the base of the cone. Default is 1.0.
    :type radius: float
    :param height: The height of the cone, measured from the center of the base to the apex. Default is 1.0.
    :type height: float
    :param cplane: A custom construction plane for the cone. If not provided, the default plane is used.
    :type cplane: Optional[Any]
    :param origin: The origin point for the base center of the cone, represented in the given plane coordinates.
    :type origin: Optional[Any]
    :param xaxis: The X-axis direction vector of the cone's construction plane.
    :type xaxis: Optional[Any]
    :param yaxis: The Y-axis direction vector of the cone's construction plane.
    :type yaxis: Optional[Any]
    :param apex: The apex point of the cone. If provided, overrides the height parameter.
    :type apex: Optional[Any]
    :param start_angle: The starting angle in radians for generating the surface of the cone. Default is 0.0.
    :type start_angle: float
    :param end_angle: The ending angle in radians for generating the surface of the cone. A full circular cone is created
        when this equals 2π. Default is 2π (full circle).
    :type end_angle: float
    :return: A NURBSSurfaceTuple representing the generated surface of the circular cone.
    :rtype: NURBSSurfaceTuple
    """
    return elliptical_cone(radius, radius, height, cplane=cplane, origin=origin, xaxis=xaxis, yaxis=yaxis, apex=apex,
                           start_angle=start_angle, end_angle=end_angle)
